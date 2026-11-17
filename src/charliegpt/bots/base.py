"""Platform-independent bot behavior.

- Replies to every DM, and in groups only when mentioned / named.
- Waits `debounce_seconds` for the other person to finish their burst before answering.
- Sends bubbles one at a time with typing delays proportional to length.
- If the real you texts in a chat, the bot goes quiet there for `takeover_minutes`.
- Checks the kill switch before every single send.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from charliegpt.config import Settings
from charliegpt.infer.engine import ReplyEngine
from charliegpt.infer.guardrails import KillSwitch
from charliegpt.schema import Conversation, Message, Platform


@dataclass
class ChatState:
    conv: Conversation
    ids: set[str] = field(default_factory=set)
    seeded: bool = False
    pending: asyncio.Task | None = None
    muted_until: float = 0.0


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)[-10:]  # "+1 (555) 123-4567" and "5551234567" are the same number


def chat_matches(conv: Conversation, entries: list[str]) -> bool:
    """An entry can be the chat key/id, its title, or someone in it (name, phone number, email, Discord id)."""
    if not entries:
        return False
    others = [p for p in conv.participants if not p.is_me]
    names = {conv.key, conv.thread_id, conv.title or ""} | {p.id for p in others} | {p.name for p in others}
    names = {n.lower() for n in names if n}
    phones = {_digits(p.id) for p in others if len(_digits(p.id)) >= 7}
    return any(e.lower() in names or (len(_digits(e)) >= 7 and _digits(e) in phones) for e in entries)


class BaseBot(ABC):
    platform: Platform

    def __init__(self, settings: Settings, engine: ReplyEngine, kill: KillSwitch | None = None):
        self.settings = settings
        self.cfg = settings.bots
        self.engine = engine
        self.kill = kill or KillSwitch(settings.paths.state)
        self.chats: dict[str, ChatState] = {}
        self.recent_sent: deque[str] = deque(maxlen=100)
        self.log = logging.getLogger(f"charliegpt.bots.{self.platform.value}")
        names = [settings.identity.name, *settings.identity.aliases]
        self.name_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in names if n) + r")\b", re.I)

    # ---- platform hooks -------------------------------------------------
    @abstractmethod
    async def run(self) -> None: ...

    @abstractmethod
    async def send(self, thread_id: str, text: str) -> None: ...

    async def typing(self, thread_id: str) -> None:  # optional indicator
        return None

    async def fetch_history(self, conv: Conversation, before: Message) -> list[Message]:
        return []

    # ---- shared logic ---------------------------------------------------
    def should_reply(self, conv: Conversation, msg: Message, mentioned: bool) -> bool:
        if msg.is_me or chat_matches(conv, self.cfg.ignore_chats):
            return False
        if self.cfg.only_chats and not chat_matches(conv, self.cfg.only_chats):
            return False
        if not conv.is_group:
            return self.cfg.reply_to_all_dms
        return self.cfg.reply_in_groups_when_mentioned and (mentioned or bool(self.name_re.search(msg.text)))

    def _state(self, conv: Conversation) -> ChatState:
        st = self.chats.get(conv.key)
        if st is None:
            st = self.chats[conv.key] = ChatState(conv=conv.model_copy(update={"messages": []}))
        else:  # titles/participants can change; keep the freshest metadata
            st.conv = conv.model_copy(update={"messages": st.conv.messages})
        return st

    def _append(self, st: ChatState, msg: Message) -> bool:
        if msg.id in st.ids:
            return False
        st.ids.add(msg.id)
        st.conv.messages.append(msg)
        st.conv.messages.sort(key=lambda m: m.timestamp)
        del st.conv.messages[: -self.cfg.history_limit]
        return True

    def _is_echo(self, msg: Message) -> bool:
        n = _norm(msg.text)
        if n in self.recent_sent:
            self.recent_sent.remove(n)
            return True
        return False

    async def handle(self, conv: Conversation, msg: Message, mentioned: bool = False, backfill: bool = False) -> None:
        st = self._state(conv)
        if not st.seeded:
            st.seeded = True
            try:
                for m in await self.fetch_history(conv, msg):
                    self._append(st, m)
            except Exception as e:
                self.log.warning("couldn't load history for %s: %s", conv.key, e)
        if not self._append(st, msg) or backfill:
            return
        if msg.is_me:
            if not self._is_echo(msg):  # the real you is here; back off
                self.log.info("human takeover in %s", conv.key)
                st.muted_until = time.monotonic() + self.cfg.takeover_minutes * 60
                if st.pending:
                    st.pending.cancel()
            return
        if not self.should_reply(st.conv, msg, mentioned):
            return
        if st.pending and not st.pending.done():
            st.pending.cancel()  # they're still typing; restart the wait
        st.pending = asyncio.create_task(self._respond_later(st))

    def typing_delay(self, text: str) -> float:
        d = len(text) / self.cfg.typing_chars_per_second
        return min(max(d, self.cfg.min_typing_delay), self.cfg.max_typing_delay) + random.uniform(0, self.cfg.typing_jitter)

    async def _respond_later(self, st: ChatState) -> None:
        try:
            await asyncio.sleep(self.cfg.debounce_seconds)
            if self.kill.paused or time.monotonic() < st.muted_until:
                return
            snapshot = st.conv.model_copy(update={"messages": list(st.conv.messages)})
            started = time.monotonic()
            bubbles = await self.engine.reply(snapshot)
            self.log.info("%s: %d bubble(s) in %.1fs", st.conv.key, len(bubbles), time.monotonic() - started)
            for b in bubbles:
                await self.typing(st.conv.thread_id)
                await asyncio.sleep(self.typing_delay(b))
                if self.kill.paused or time.monotonic() < st.muted_until:
                    return
                await self._send(st, b)
        except asyncio.CancelledError:
            pass
        except Exception:
            self.log.exception("reply failed in %s", st.conv.key)

    async def _send(self, st: ChatState, text: str) -> None:
        if self.cfg.dry_run:
            print(f"[dry-run → {st.conv.title or st.conv.thread_id}] {text}", flush=True)
            me = Message(
                id=f"dry-{time.time_ns()}",
                sender_id="me",
                sender_name=self.settings.identity.name,
                is_me=True,
                timestamp=datetime.now(timezone.utc),
                text=text,
            )
            self._append(st, me)
            return
        self.recent_sent.append(_norm(text))
        await self.send(st.conv.thread_id, text)
