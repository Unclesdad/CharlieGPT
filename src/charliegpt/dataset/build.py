"""Sessions -> chat-format JSONL ({"messages": [...]}) that Unsloth/TRL and mlx-lm both read.

Every one of your bursts that follows someone else's turn becomes one example:
(system prompt + recent history) -> your burst.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from charliegpt.config import Settings
from charliegpt.context import make_context
from charliegpt.ingest.normalize import group_turns
from charliegpt.memory.profiles import ProfileStore
from charliegpt.memory.rag import Retriever
from charliegpt.prompt import build_messages, format_target
from charliegpt.schema import Conversation


@dataclass
class BuildStats:
    train: int = 0
    valid: int = 0
    skipped: int = 0


def _is_valid_split(key: str, fraction: float) -> bool:
    """Split by thread so no chat leaks between train and validation."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return (h % 10_000) < fraction * 10_000


def examples_for(
    conv: Conversation,
    settings: Settings,
    profiles: ProfileStore | None,
    retriever: Retriever | None,
    rng: random.Random,
) -> list[dict]:
    ds = settings.dataset
    turns = group_turns(conv.messages, ds.burst_gap_minutes)
    out = []
    for i, turn in enumerate(turns):
        if not turn.is_me or i < ds.min_history_turns or turns[i - 1].is_me:
            continue
        target = format_target(turn)
        if not target:
            continue
        k = 0 if rng.random() < ds.rag_dropout else ds.rag_k
        ctx = make_context(conv, turns[:i], settings, now=turn.start, profiles=profiles, retriever=retriever, rag_k=k, exclude_thread=conv.key)
        out.append({"messages": build_messages(ctx) + [{"role": "assistant", "content": target}]})
    return out


def build_dataset(
    sessions: list[Conversation],
    settings: Settings,
    out_dir: Path,
    profiles: ProfileStore | None = None,
    retriever: Retriever | None = None,
) -> BuildStats:
    rng = random.Random(settings.dataset.seed)
    stats = BuildStats()
    train, valid = [], []
    for conv in sessions:
        exs = examples_for(conv, settings, profiles, retriever, rng)
        if not exs:
            stats.skipped += 1
        (valid if _is_valid_split(conv.key, settings.dataset.val_fraction) else train).extend(exs)
    if not valid and len(train) > 20:  # tiny datasets: carve out a few examples anyway
        rng.shuffle(train)
        cut = max(1, int(len(train) * settings.dataset.val_fraction))
        valid, train = train[:cut], train[cut:]
    rng.shuffle(train)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.train, stats.valid = len(train), len(valid)
    return stats
