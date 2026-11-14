"""Clean threads: dedupe, sort, resolve replies, split into sessions, group into turns."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

from charliegpt.schema import Conversation, Message, Turn


def clean(conv: Conversation) -> Conversation:
    seen: set[str] = set()
    msgs: list[Message] = []
    for m in sorted(conv.messages, key=lambda m: m.timestamp):
        if m.id in seen:
            continue
        seen.add(m.id)
        msgs.append(m)
    by_id = {m.id: m for m in msgs}
    for m in msgs:
        if m.reply_to and not m.reply_to_text and (target := by_id.get(m.reply_to)):
            m.reply_to_text = target.render()
    return conv.model_copy(update={"messages": msgs})


def split_sessions(conv: Conversation, gap_hours: float) -> list[Conversation]:
    gap = timedelta(hours=gap_hours)
    sessions: list[list[Message]] = []
    for m in conv.messages:
        if not sessions or m.timestamp - sessions[-1][-1].timestamp > gap:
            sessions.append([])
        sessions[-1].append(m)
    return [conv.model_copy(update={"session": i, "messages": s}) for i, s in enumerate(sessions)]


def group_turns(messages: Iterable[Message], burst_gap_minutes: float) -> list[Turn]:
    gap = timedelta(minutes=burst_gap_minutes)
    turns: list[Turn] = []
    for m in messages:
        last = turns[-1] if turns else None
        if last and last.sender_id == m.sender_id and m.timestamp - last.end <= gap:
            last.messages.append(m)
        else:
            turns.append(Turn(sender_id=m.sender_id, sender_name=m.sender_name, is_me=m.is_me, messages=[m]))
    return turns


def normalize(convs: Iterable[Conversation], session_gap_hours: float) -> list[Conversation]:
    out: list[Conversation] = []
    for c in convs:
        out += split_sessions(clean(c), session_gap_hours)
    return out
