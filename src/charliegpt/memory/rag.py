"""Retrieval over past exchanges: "they said X, you replied Y" pairs, embedded on what they said."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from charliegpt.ingest.normalize import group_turns
from charliegpt.prompt import Example, format_target, render_turn
from charliegpt.schema import Conversation


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        from charliegpt.config import detect_device

        self.model = SentenceTransformer(model_name, device=detect_device())

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=len(texts) > 1000).astype(np.float32)


def exchange_pairs(sessions: list[Conversation], burst_gap_minutes: float, max_len: int = 400) -> list[dict]:
    pairs = []
    for s in sessions:
        turns = group_turns(s.messages, burst_gap_minutes)
        for prev, cur in zip(turns, turns[1:]):
            if cur.is_me and not prev.is_me:
                them, me = render_turn(prev, False), format_target(cur)
                if them and me and len(them) <= max_len and len(me) <= max_len:
                    pairs.append({"thread": s.key, "them": them, "me": me})
    return pairs


class Retriever:
    def __init__(self, index_dir: Path, embedder: Embedder | None = None, embed_model: str | None = None):
        self.index_dir = Path(index_dir)
        self._embedder = embedder
        self._embed_model = embed_model
        self.pairs: list[dict] = []
        self.vectors: np.ndarray | None = None
        if (self.index_dir / "pairs.jsonl").exists():
            self.pairs = [json.loads(l) for l in (self.index_dir / "pairs.jsonl").read_text(encoding="utf-8").split("\n") if l]  # not splitlines: texts can contain U+2028
            self.vectors = np.load(self.index_dir / "vectors.npy")

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self._embed_model or "BAAI/bge-small-en-v1.5")
        return self._embedder

    @property
    def ready(self) -> bool:
        return self.vectors is not None and len(self.pairs) > 0

    def build(self, pairs: list[dict]) -> None:
        self.pairs = pairs
        self.vectors = self.embedder([p["them"] for p in pairs]) if pairs else np.zeros((0, 1), np.float32)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "vectors.npy", self.vectors)
        (self.index_dir / "pairs.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")

    def search(self, query: str, k: int, exclude_thread: str | None = None) -> list[Example]:
        if not self.ready or not query or k <= 0:
            return []
        scores = self.vectors @ self.embedder([query])[0]
        # Only sort the best few candidates; fall back to a full sort if one thread hogs them all.
        top = min(len(scores), k * 20)
        cand = np.argpartition(-scores, top - 1)[:top]
        best = cand[np.argsort(-scores[cand])]

        def rest():  # full sort only if one thread hogs every top candidate
            yield from np.argsort(-scores)[top:]

        out: list[Example] = []
        for i in itertools.chain(best, rest()):
            p = self.pairs[i]
            if exclude_thread and p["thread"] == exclude_thread:
                continue
            out.append(Example(them=p["them"], me=p["me"]))
            if len(out) == k:
                break
        return out
