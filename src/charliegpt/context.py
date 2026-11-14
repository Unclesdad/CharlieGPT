"""Assemble a PromptContext from history + memory. Shared by dataset building and live inference."""

from __future__ import annotations

from datetime import datetime

from charliegpt.config import Settings
from charliegpt.memory.profiles import ProfileStore
from charliegpt.memory.rag import Retriever
from charliegpt.prompt import PromptContext, render_turn, truncate
from charliegpt.schema import Conversation, Turn


def make_context(
    conv: Conversation,
    turns: list[Turn],
    settings: Settings,
    now: datetime,
    profiles: ProfileStore | None = None,
    retriever: Retriever | None = None,
    rag_k: int = 0,
    exclude_thread: str | None = None,
) -> PromptContext:
    turns = turns[-settings.dataset.history_turns :]
    examples = []
    if retriever is not None and rag_k > 0:
        last_them = next((t for t in reversed(turns) if not t.is_me), None)
        if last_them:
            examples = retriever.search(render_turn(last_them, False), rag_k, exclude_thread=exclude_thread)
    ctx = PromptContext(
        me=settings.identity.name,
        platform=conv.platform,
        is_group=conv.is_group,
        title=conv.title,
        others=[p.name for p in conv.others()] or sorted({t.sender_name for t in turns if not t.is_me}),
        turns=turns,
        now=now,
        profile=profiles.get(conv) if profiles else None,
        examples=examples,
    )
    return truncate(ctx, settings.dataset.max_context_chars)
