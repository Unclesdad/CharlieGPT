"""history -> context -> persona model -> bubbles, with guardrails. Used by every bot and the REPL."""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from charliegpt.config import Settings
from charliegpt.context import make_context
from charliegpt.infer import guardrails
from charliegpt.infer.client import LLMClient
from charliegpt.ingest.caption import Captioner
from charliegpt.ingest.normalize import group_turns
from charliegpt.memory.profiles import ProfileStore
from charliegpt.memory.rag import Retriever
from charliegpt.prompt import build_messages, render_turn, split_bubbles
from charliegpt.schema import AttachmentKind, Conversation

log = logging.getLogger(__name__)
MAX_BUBBLES = 6


class ReplyEngine:
    def __init__(
        self,
        settings: Settings,
        client: LLMClient,
        profiles: ProfileStore | None = None,
        retriever: Retriever | None = None,
        captioner: Captioner | None = None,
    ):
        self.settings = settings
        self.client = client
        self.profiles = profiles
        self.retriever = retriever
        self.captioner = captioner

    @classmethod
    def from_settings(cls, settings: Settings) -> "ReplyEngine":
        from charliegpt.ingest.caption import make_captioner

        retriever = None
        if settings.rag.enabled:
            r = Retriever(settings.paths.memory / "rag", embed_model=settings.rag.embed_model)
            retriever = r if r.ready else None
        return cls(
            settings,
            LLMClient(settings.inference),
            profiles=ProfileStore(settings.paths.memory / "profiles.json"),
            retriever=retriever,
            captioner=make_captioner(settings),
        )

    async def caption_new_images(self, conv: Conversation) -> None:
        if not self.captioner:
            return
        for m in conv.messages[-10:]:
            for a in m.attachments:
                if a.kind == AttachmentKind.image and a.path and not a.caption:
                    try:
                        a.caption = await asyncio.to_thread(self.captioner.caption, Path(a.path))
                    except Exception as e:
                        log.warning("caption failed for %s: %s", a.path, e)

    async def reply(self, conv: Conversation, now: datetime | None = None) -> list[str]:
        """Bubbles to send in response to the latest messages, or [] to stay quiet."""
        s = self.settings
        await self.caption_new_images(conv)
        turns = group_turns(conv.messages, s.dataset.burst_gap_minutes)
        if not turns or turns[-1].is_me:
            return []
        ctx = make_context(
            conv,
            turns,
            s,
            now=now or datetime.now(timezone.utc),
            profiles=self.profiles,
            retriever=self.retriever,
            rag_k=s.dataset.rag_k if self.retriever else 0,
        )
        incoming = render_turn(turns[-1], conv.is_group)
        me = s.identity.name

        # Classify and generate concurrently; the classifier only costs latency when it fires.
        check = guardrails.needs_excuse(self.client, me, incoming) if s.guardrails.commitment_excuses else asyncio.sleep(0, False)
        excuse_needed, raw = await asyncio.gather(check, self.client.persona(build_messages(ctx)))
        if excuse_needed:
            style = [b for t in ctx.turns if t.is_me for b in t.bubbles()] + [e.me for e in ctx.examples]
            bubbles = await guardrails.absurd_excuse(self.client, me, incoming, style)
            if guardrails.breaks_character(bubbles):  # never fall through to a reply that might say yes
                bubbles = [random.choice(guardrails.FALLBACK_EXCUSES)]
            return bubbles

        bubbles = split_bubbles(raw, me)
        for _ in range(2):  # regenerate if empty (media-only) or out of character
            if bubbles and not guardrails.breaks_character(bubbles):
                return bubbles[:MAX_BUBBLES]
            bubbles = split_bubbles(await self.client.persona(build_messages(ctx)), me)
        if bubbles and not guardrails.breaks_character(bubbles):
            return bubbles[:MAX_BUBBLES]
        if guardrails.ASKS_IF_AI.search(incoming):
            return [random.choice(guardrails.DEFLECTIONS)]
        if bubbles:
            log.info("dropping out-of-character reply in %s", conv.key)
        return []
