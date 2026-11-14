"""charliegpt CLI. Pipeline: ingest → caption → index → profiles → build-dataset → train → serve → bot."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from charliegpt.config import Settings, detect_device, load_settings
from charliegpt.schema import Conversation, Message, Participant, Platform

app = typer.Typer(no_args_is_help=True, help="Train an LLM to text like you, then let it loose.")
ingest_app = typer.Typer(no_args_is_help=True, help="Convert raw exports into the uniform format.")
app.add_typer(ingest_app, name="ingest")
console = Console()


def S(ctx: typer.Context) -> Settings:
    return ctx.obj


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config (default: configs/default.yaml)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s", handlers=[RichHandler(console=console, show_path=False)])
    for noisy in ("httpx", "discord", "openai", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ctx.obj = load_settings(config)


# ---- ingest ----------------------------------------------------------------


def _save(settings: Settings, platform: Platform, convs) -> None:
    from charliegpt.store import save_threads

    out = settings.paths.processed / f"{platform.value}.jsonl"
    n = save_threads(convs, out)
    console.print(f"[green]✓[/] {n} {platform.value} threads → {out}")


@ingest_app.command("imessage")
def ingest_imessage(ctx: typer.Context, db: Optional[Path] = typer.Option(None, help="Path to chat.db (default from config)")):
    """Read iMessage history straight from chat.db (grant your terminal Full Disk Access)."""
    from charliegpt.ingest.imessage import IMessageParser

    s = S(ctx)
    _save(s, Platform.imessage, IMessageParser(db or s.imessage.db_path, s.identity.contacts, s.identity.name).parse())


@ingest_app.command("discord")
def ingest_discord(ctx: typer.Context, path: Path = typer.Argument(..., help="DiscordChatExporter JSON file or folder")):
    """Parse DiscordChatExporter JSON exports."""
    from charliegpt.ingest.discord import DiscordParser

    s = S(ctx)
    if not s.identity.discord_user_ids:
        console.print("[yellow]identity.discord_user_ids is empty: none of the messages will count as yours.[/]")
    _save(s, Platform.discord, DiscordParser(path, s.identity.discord_user_ids, s.identity.name).parse())


@ingest_app.command("instagram")
def ingest_instagram(ctx: typer.Context, path: Path = typer.Argument(..., help="Unzipped Instagram data export folder")):
    """Parse Instagram 'Download your information' JSON."""
    from charliegpt.ingest.instagram import InstagramParser

    s = S(ctx)
    if not s.identity.instagram_name:
        console.print("[yellow]identity.instagram_name is empty: none of the messages will count as yours.[/]")
    _save(s, Platform.instagram, InstagramParser(path, s.identity.instagram_name, s.identity.name).parse())


@app.command()
def stats(ctx: typer.Context):
    """Summarize what's been ingested."""
    from charliegpt.ingest.normalize import group_turns
    from charliegpt.store import load_sessions

    s = S(ctx)
    sessions = load_sessions(s.paths.processed, s.dataset.session_gap_hours)
    if not sessions:
        console.print("Nothing ingested yet. Run `charliegpt ingest ...` first.")
        raise typer.Exit(1)
    table = Table("platform", "threads", "sessions", "messages", "yours", "trainable replies")
    by_platform: dict[str, list[Conversation]] = {}
    for c in sessions:
        by_platform.setdefault(c.platform.value, []).append(c)
    contacts: Counter[str] = Counter()
    for p, convs in sorted(by_platform.items()):
        msgs = sum(len(c.messages) for c in convs)
        mine = sum(m.is_me for c in convs for m in c.messages)
        replies = 0
        for c in convs:
            turns = group_turns(c.messages, s.dataset.burst_gap_minutes)
            replies += sum(1 for a, b in zip(turns, turns[1:]) if b.is_me and not a.is_me)
            contacts[f"{p}: {c.title}"] += sum(m.is_me for m in c.messages)
        table.add_row(p, str(len({c.key for c in convs})), str(len(convs)), str(msgs), str(mine), str(replies))
    console.print(table)
    top = Table("chat", "your messages")
    for name, n in contacts.most_common(15):
        top.add_row(name, str(n))
    console.print(top)


# ---- enrichment ------------------------------------------------------------


@app.command()
def caption(ctx: typer.Context):
    """Caption every image in the ingested data with the vision model (cached; safe to re-run)."""
    from charliegpt.ingest.caption import caption_threads, make_captioner
    from charliegpt.store import save_threads
    from charliegpt.schema import Conversation as C

    s = S(ctx)
    captioner = make_captioner(s)
    if captioner is None:
        console.print("vision.backend is 'none'; nothing to do.")
        return
    for f in sorted(s.paths.processed.glob("*.jsonl")):
        convs = [C.model_validate_json(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = caption_threads(convs, captioner, on_error=lambda p, e: console.print(f"[yellow]skip {p}: {e}[/]"))
        save_threads(convs, f)
        console.print(f"[green]✓[/] {f.name}: {n} new captions")


@app.command()
def index(ctx: typer.Context):
    """Build the RAG index of your past (them → you) exchanges."""
    from charliegpt.memory.rag import Retriever, exchange_pairs
    from charliegpt.store import load_sessions

    s = S(ctx)
    sessions = load_sessions(s.paths.processed, s.dataset.session_gap_hours)
    pairs = exchange_pairs(sessions, s.dataset.burst_gap_minutes)
    Retriever(s.paths.memory / "rag", embed_model=s.rag.embed_model).build(pairs)
    console.print(f"[green]✓[/] indexed {len(pairs)} exchanges")


@app.command()
def profiles(ctx: typer.Context, min_messages: int = typer.Option(30, help="Skip chats smaller than this")):
    """Summarize each chat (relationship, topics, tone) with the helper model. Needs a running server."""
    from charliegpt.infer.client import LLMClient
    from charliegpt.memory.profiles import ProfileStore, build_profiles
    from charliegpt.store import load_sessions

    s = S(ctx)
    sessions = load_sessions(s.paths.processed, s.dataset.session_gap_hours)
    client = LLMClient(s.inference)
    store = ProfileStore(s.paths.memory / "profiles.json")
    n = asyncio.run(
        build_profiles(sessions, store, lambda msgs: client.helper(msgs, temperature=0.3), s.identity.name, min_messages=min_messages, on_progress=lambda k: console.print(f"  profiled {k}"))
    )
    console.print(f"[green]✓[/] {n} profiles → {store.path} (edit the `notes` fields freely)")


# ---- dataset + training ----------------------------------------------------


@app.command("build-dataset")
def build_dataset_cmd(ctx: typer.Context):
    """Turn sessions into train/valid JSONL in chat format."""
    from charliegpt.dataset.build import build_dataset
    from charliegpt.memory.profiles import ProfileStore
    from charliegpt.memory.rag import Retriever
    from charliegpt.store import load_sessions

    s = S(ctx)
    sessions = load_sessions(s.paths.processed, s.dataset.session_gap_hours)
    retriever = None
    if s.rag.enabled:
        r = Retriever(s.paths.memory / "rag", embed_model=s.rag.embed_model)
        retriever = r if r.ready else None
        if retriever is None:
            console.print("[yellow]no RAG index; building without examples (run `charliegpt index` first)[/]")
    st = build_dataset(sessions, s, s.paths.dataset, profiles=ProfileStore(s.paths.memory / "profiles.json"), retriever=retriever)
    console.print(f"[green]✓[/] {st.train} train / {st.valid} valid examples → {s.paths.dataset}")


@app.command()
def train(
    ctx: typer.Context,
    backend: str = typer.Option("auto", help="auto | cuda | mlx"),
    merge: bool = typer.Option(False, help="(cuda) also save a merged 16-bit model"),
):
    """Fine-tune: Unsloth QLoRA on CUDA, or mlx-lm LoRA on Apple Silicon."""
    s = S(ctx)
    if backend == "auto":
        backend = "cuda" if detect_device() == "cuda" else "mlx"
    if not (s.paths.dataset / "train.jsonl").exists():
        console.print("No dataset. Run `charliegpt build-dataset` first.")
        raise typer.Exit(1)
    if backend == "cuda":
        from charliegpt.train.cuda import train_cuda

        out = train_cuda(s, s.paths.dataset, s.paths.models, merge=merge)
    else:
        from charliegpt.train.mlx import train_mlx

        out = train_mlx(s, s.paths.dataset, s.paths.models)
    console.print(f"[green]✓[/] adapter saved → {out}")


# ---- serving ---------------------------------------------------------------


def serve_command(s: Settings, backend: str) -> list[str]:
    m, i = s.model, s.inference
    if backend == "cuda":
        cmd = ["vllm", "serve", m.serve_base, "--served-model-name", m.served_base_name, "--host", "0.0.0.0", "--port", str(i.port),
               "--max-model-len", str(m.max_seq_length * 2), "--gpu-memory-utilization", str(i.gpu_memory_utilization)]  # fmt: skip
        adapter = s.paths.models / m.adapter_name
        if adapter.exists():
            cmd += ["--enable-lora", "--max-lora-rank", str(s.train.lora_r), "--lora-modules", f"{m.adapter_name}={adapter}"]
        return cmd
    cmd = [sys.executable, "-m", "mlx_lm", "server", "--model", m.mlx_base, "--host", "0.0.0.0", "--port", str(i.port),
           "--chat-template-args", json.dumps({"enable_thinking": False})]  # fmt: skip
    adapter = s.paths.models / f"{m.adapter_name}-mlx"
    if adapter.exists():
        cmd += ["--adapter-path", str(adapter)]
    return cmd


@app.command()
def serve(
    ctx: typer.Context,
    backend: str = typer.Option("auto", help="auto | cuda | mlx"),
    print_only: bool = typer.Option(False, "--print", help="Print the command instead of running it"),
):
    """Run the OpenAI-compatible inference server (vLLM on CUDA, mlx_lm.server on Mac)."""
    s = S(ctx)
    if backend == "auto":
        backend = "cuda" if detect_device() == "cuda" else "mlx"
    cmd = serve_command(s, backend)
    console.print(f"[dim]$ {shlex.join(cmd)}[/]")
    if backend == "mlx":
        console.print("[dim]note: mlx_lm.server serves one model; persona and helper calls both use it.[/]")
    if not print_only:
        os.execvp(cmd[0], cmd)


# ---- bots + control --------------------------------------------------------


@app.command()
def bot(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="discord | imessage | instagram"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print replies instead of sending them"),
):
    """Run a bot in the foreground (Docker/launchd run exactly this)."""
    from charliegpt.infer.engine import ReplyEngine

    s = S(ctx)
    if dry_run:
        s.bots.dry_run = True
    if name == "discord":
        from charliegpt.bots.discord_bot import DiscordBot as Bot
    elif name == "imessage":
        from charliegpt.bots.imessage_bot import IMessageBot as Bot
    elif name == "instagram":
        from charliegpt.bots.instagram_bot import InstagramBot as Bot
    else:
        raise typer.BadParameter("expected discord, imessage or instagram")
    b = Bot(s, ReplyEngine.from_settings(s))
    if b.kill.paused:
        console.print("[yellow]kill switch is ON: bot will listen but not send. `charliegpt resume` to enable.[/]")
    asyncio.run(b.run())


@app.command()
def pause(ctx: typer.Context, reason: str = typer.Argument("", help="Optional note")):
    """Kill switch: every bot stops sending immediately."""
    from charliegpt.infer.guardrails import KillSwitch

    KillSwitch(S(ctx).paths.state).pause(reason)
    console.print("[red]⏸ paused[/] all bots")


@app.command()
def resume(ctx: typer.Context):
    """Turn the kill switch off."""
    from charliegpt.infer.guardrails import KillSwitch

    KillSwitch(S(ctx).paths.state).resume()
    console.print("[green]▶ resumed[/]")


@app.command()
def status(ctx: typer.Context):
    """Show kill switch, data and model state."""
    from charliegpt.infer.guardrails import KillSwitch

    s = S(ctx)
    k = KillSwitch(s.paths.state)
    console.print(f"kill switch: {'[red]PAUSED[/]' if k.paused else '[green]running[/]'}")
    console.print(f"device: {detect_device()}")
    for label, p in [
        ("processed", s.paths.processed),
        ("dataset", s.paths.dataset / "train.jsonl"),
        ("rag index", s.paths.memory / "rag" / "vectors.npy"),
        ("profiles", s.paths.memory / "profiles.json"),
        ("cuda adapter", s.paths.models / s.model.adapter_name),
        ("mlx adapter", s.paths.models / f"{s.model.adapter_name}-mlx"),
    ]:
        console.print(f"{label:>13}: {'✓' if p.exists() else '✗'} {p}")
    console.print(f"    inference: {s.inference.base_url}")


# ---- testing the model -----------------------------------------------------


def _msg(text: str, me: bool, name: str) -> Message:
    return Message(id=f"repl-{time.time_ns()}", sender_id="me" if me else "them", sender_name=name, is_me=me, timestamp=datetime.now(timezone.utc), text=text)


@app.command()
def chat(ctx: typer.Context, them: str = typer.Option("Alex", help="Who you're pretending to be"), platform: Platform = typer.Option(Platform.imessage)):
    """Text the model yourself in the terminal."""
    from charliegpt.infer.engine import ReplyEngine

    s = S(ctx)
    engine = ReplyEngine.from_settings(s)
    conv = Conversation(
        platform=platform,
        thread_id=f"repl-{them}",
        title=them,
        participants=[Participant(id="me", name=s.identity.name, is_me=True), Participant(id="them", name=them)],
    )
    console.print(f"[dim]You are {them}. Ctrl-C to quit.[/]")

    async def loop():
        while True:
            text = await asyncio.to_thread(console.input, f"[bold]{them}>[/] ")
            if not text.strip():
                continue
            conv.messages.append(_msg(text, False, them))
            t0 = time.monotonic()
            bubbles = await engine.reply(conv)
            for b in bubbles:
                console.print(f"[cyan]{s.identity.name}>[/] {b}")
                conv.messages.append(_msg(b, True, s.identity.name))
            console.print(f"[dim]({time.monotonic() - t0:.1f}s)[/]")

    try:
        asyncio.run(loop())
    except (KeyboardInterrupt, EOFError):
        pass


BENCH_PROMPTS = ["yo what's up", "did you see what happened in class today", "lmaooo look at this", "how was your weekend", "you're not gonna believe this", "ok be honest what do you think of her"]


@app.command()
def bench(ctx: typer.Context, n: int = typer.Option(12, "--n", "-n", help="Number of replies to time")):
    """Measure end-to-end reply latency (target: under 20s)."""
    from charliegpt.infer.engine import ReplyEngine

    s = S(ctx)
    engine = ReplyEngine.from_settings(s)

    async def run() -> list[float]:
        times = []
        for k in range(n):
            conv = Conversation(platform=Platform.imessage, thread_id="bench", title="Alex", participants=[Participant(id="them", name="Alex")])
            conv.messages.append(_msg(BENCH_PROMPTS[k % len(BENCH_PROMPTS)], False, "Alex"))
            t0 = time.monotonic()
            bubbles = await engine.reply(conv)
            times.append(time.monotonic() - t0)
            console.print(f"{times[-1]:5.1f}s  {' / '.join(bubbles)}")
        return times

    times = sorted(asyncio.run(run()))
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    console.print(f"p50 {statistics.median(times):.1f}s · p95 {p95:.1f}s · max {times[-1]:.1f}s")


if __name__ == "__main__":
    app()
