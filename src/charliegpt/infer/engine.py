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
        self.last_why = ""  # set when a guardrail replaced the model's reply (shown by `charliegpt chat`)
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

        g = s.guardrails
        leaks = (lambda b: guardrails.breaks_character(b, incoming)) if g.ai_leak_filter else (lambda _: False)
        recent = [" ".join(t.bubbles()) for t in turns if t.is_me][-3:]
        bad = lambda b: leaks(b) or (g.no_repeats and guardrails.repeats(b, recent))  # noqa: E731

        # Classify and generate concurrently; the classifier only costs latency when it fires.
        check = guardrails.needs_excuse(self.client, me, incoming) if g.commitment_excuses else asyncio.sleep(0, False)
        excuse_needed, raw = await asyncio.gather(check, self.client.persona(build_messages(ctx)))
        if excuse_needed:
            style = [b for t in ctx.turns if t.is_me for b in t.bubbles()] + [e.me for e in ctx.examples]
            bubbles = await guardrails.absurd_excuse(self.client, me, incoming, style, g)
            if leaks(bubbles) and g.fallback_excuses:  # never fall through to a reply that might say yes
                bubbles = [random.choice(g.fallback_excuses)]
            self.last_why = "excuse: looked like a request for plans/money/a favor (guardrails.commitment_excuses)"
            return bubbles

        bubbles, brushoff = split_bubbles(raw, me), None
        for attempt in range(3):  # regenerate if empty (media-only), out of character, a repeat, or a non-answer
            if bubbles and not bad(bubbles):
                if not (g.no_brushoffs and guardrails.brushes_off(bubbles, incoming)):
                    return bubbles[:MAX_BUBBLES]
                brushoff = brushoff or bubbles
            if attempt < 2:
                bubbles = split_bubbles(await self.client.persona(build_messages(ctx)), me)
        if brushoff:
            self.last_why = "non-answer: all 3 tries dodged the question (guardrails.no_brushoffs)"
            return brushoff[:MAX_BUBBLES]
        if g.deflect_ai_questions and g.deflections and guardrails.ASKS_IF_AI.search(incoming):
            self.last_why = "deflection: asked if it's an AI and every reply broke character (guardrails.deflect_ai_questions)"
            return [random.choice(g.deflections)]
        if bubbles:
            log.info("dropping reply in %s (out of character or a repeat)", conv.key)
        return []
