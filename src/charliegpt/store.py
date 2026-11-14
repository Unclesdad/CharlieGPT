"""Read/write the uniform format: one JSONL of whole threads per platform."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from charliegpt.ingest.normalize import normalize
from charliegpt.schema import Conversation


def save_threads(convs: Iterable[Conversation], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in convs:
            f.write(c.model_dump_json() + "\n")
            n += 1
    return n


def load_threads(processed_dir: Path) -> list[Conversation]:
    convs: list[Conversation] = []
    for f in sorted(Path(processed_dir).glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            convs += [Conversation.model_validate_json(line) for line in fh if line.strip()]
    return convs


def load_sessions(processed_dir: Path, session_gap_hours: float) -> list[Conversation]:
    return normalize(load_threads(processed_dir), session_gap_hours)
