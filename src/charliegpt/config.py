"""Typed settings loaded from YAML (with `extends:` overlays) plus .env secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

DEFAULT_CONFIG = Path("configs/default.yaml")


class Identity(BaseModel):
    name: str = "Charlie"
    # Words that count as "talking to me" in group chats.
    aliases: list[str] = Field(default_factory=lambda: ["charlie"])
    discord_user_ids: list[str] = Field(default_factory=list)  # your user id(s) or username(s)
    instagram_name: str = ""
    # handle/user id -> display name (phone numbers, emails, usernames); overrides the vCard
    contacts: dict[str, str] = Field(default_factory=dict)
    # Contacts export; default: the first *.vcf in paths.messages
    contacts_vcf: Path | None = None


class Paths(BaseModel):
    # Raw exports, one folder per source; each parser only reads its own folder.
    messages: Path = Path("data/messages")  # imessage-exporter HTML + a contacts .vcf
    discord: Path = Path("data/discord")  # DiscordChatExporter JSON
    instagram: Path = Path("data/instagram")  # unzipped Instagram export
    # Uniform-format output (one JSONL per source).
    processed: Path = Path("features")
    dataset: Path = Path("data/dataset")
    memory: Path = Path("data/memory")
    captions: Path = Path("data/memory/captions.json")
    models: Path = Path("models")
    state: Path = Path("state")
    media: Path = Path("data/media")  # live-bot downloads


class DatasetCfg(BaseModel):
    session_gap_hours: float = 6.0
    burst_gap_minutes: float = 5.0
    history_turns: int = 12
    max_context_chars: int = 6000
    rag_k: int = 3
    rag_dropout: float = 0.3  # chance to train without examples, so the model doesn't depend on them
    val_fraction: float = 0.03
    min_history_turns: int = 1
    seed: int = 17


class ModelCfg(BaseModel):
    train_base: str = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    serve_base: str = "Qwen/Qwen3-14B-AWQ"
    served_base_name: str = "qwen3-14b"
    mlx_base: str = "mlx-community/Qwen3-8B-4bit"
    adapter_name: str = "charlie"
    max_seq_length: int = 4096


class TrainCfg(BaseModel):
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    learning_rate: float = 1e-4
    epochs: float = 2.0
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    eval_steps: int = 200
    mlx_iters: int = 1000
    mlx_num_layers: int = 16
    mlx_batch_size: int = 1


class InferenceCfg(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    fallback_url: str | None = None
    fallback_model: str | None = "default_model"  # mlx_lm.server's name for whatever it loaded
    api_key: str = "EMPTY"
    persona_model: str = "charlie"  # LoRA adapter name on the server
    helper_model: str = "qwen3-14b"  # plain base model: classifiers, excuses, profiles
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.1  # >1 discourages reusing tokens already in the chat (stops "y" loops)
    max_tokens: int = 200
    timeout_seconds: float = 60.0
    port: int = 8000
    gpu_memory_utilization: float = 0.70


class VisionCfg(BaseModel):
    backend: Literal["openai", "local", "none"] = "openai"
    base_url: str = "http://localhost:8001/v1"
    model: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    local_model_cuda: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    local_model_mlx: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    port: int = 8001
    # Share of GPU memory for `serve --vision`. 0.20 leaves room for the persona model next to it;
    # raise it (e.g. 0.6) when captioning with nothing else on the GPU.
    gpu_memory_utilization: float = 0.20
    # Path prefix rewrites for captioning on another machine, e.g.
    # {"/Users/you/Library/Messages/Attachments": "data/imessage_attachments"}
    path_map: dict[str, str] = Field(default_factory=dict)


class RagCfg(BaseModel):
    enabled: bool = True
    embed_model: str = "BAAI/bge-small-en-v1.5"


class BotsCfg(BaseModel):
    reply_to_all_dms: bool = True
    reply_in_groups_when_mentioned: bool = True
    ignore_chats: list[str] = Field(default_factory=list)  # conversation keys or thread ids
    debounce_seconds: float = 8.0  # wait for them to finish their burst
    typing_chars_per_second: float = 7.0
    min_typing_delay: float = 1.0
    max_typing_delay: float = 8.0
    typing_jitter: float = 0.6  # random extra seconds per bubble
    history_limit: int = 40
    takeover_minutes: float = 30.0  # go quiet in a chat after the real you texts in it
    dry_run: bool = False


class DiscordCfg(BaseModel):
    owner_user_id: str | None = None  # your real account; its messages count as "me"


class IMessageCfg(BaseModel):
    db_path: Path = Path("~/Library/Messages/chat.db")
    poll_seconds: float = 2.0


class InstagramCfg(BaseModel):
    i_understand_this_violates_tos: bool = False
    poll_seconds: float = 20.0
    session_file: Path = Path("state/instagram_session.json")


class GuardrailsCfg(BaseModel):
    # Plans, money, favors -> decline with an absurd excuse instead of agreeing.
    commitment_excuses: bool = True
    # Build each excuse around this many random English nouns/verbs (0 = let the model pick, which drifts to unicorns).
    excuse_random_words: int = 2
    # Tone references for the excuse writer (it's told never to reuse them).
    excuse_examples: list[str] = Field(
        default_factory=lambda: ["sorry i cant i have to ride my horse into the sunset", "i want to but the king of atlantis has sent me on a quest"]
    )
    # Used when generation fails, so a commitment never turns into a yes.
    fallback_excuses: list[str] = Field(
        default_factory=lambda: [
            "cant the council of geese has summoned me",
            "i would but im legally required to guard a lighthouse that day",
            "no can do my horse is riding me into the sunset this time",
            "sorry the king of atlantis moved my quest up",
            "cant im getting knighted by a raccoon",
        ]
    )
    # Regenerate (then drop) replies that talk like a bot ("as an AI", "chatbot", "my code"...).
    ai_leak_filter: bool = True
    # Regenerate a reply that's identical to one of your last 3 (stops loops like "y", "y", "y").
    no_repeats: bool = True
    # When they ask "are you an AI" and the model can't answer in character, brush it off with one of these.
    deflect_ai_questions: bool = True
    deflections: list[str] = Field(default_factory=lambda: ["lmao what", "bro what 💀", "what are you on", "huh", "no?? 💀"])


class Settings(BaseModel):
    identity: Identity = Field(default_factory=Identity)
    paths: Paths = Field(default_factory=Paths)
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    inference: InferenceCfg = Field(default_factory=InferenceCfg)
    vision: VisionCfg = Field(default_factory=VisionCfg)
    rag: RagCfg = Field(default_factory=RagCfg)
    bots: BotsCfg = Field(default_factory=BotsCfg)
    discord: DiscordCfg = Field(default_factory=DiscordCfg)
    imessage: IMessageCfg = Field(default_factory=IMessageCfg)
    instagram: InstagramCfg = Field(default_factory=InstagramCfg)
    guardrails: GuardrailsCfg = Field(default_factory=GuardrailsCfg)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent:
        data = _deep_merge(_read_yaml(path.parent / parent), data)
    return data


def load_settings(path: Path | str | None = None) -> Settings:
    load_dotenv()
    path = Path(path or os.environ.get("CHARLIEGPT_CONFIG", DEFAULT_CONFIG))
    raw = _read_yaml(path) if path.exists() else {}
    return Settings.model_validate(raw)


def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        import platform

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return "mps"
    return "cpu"
