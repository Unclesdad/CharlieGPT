"""THE prompt format. Training examples and live inference both go through `build_messages`,
so the model is always prompted exactly the way it was trained."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from charliegpt.schema import Platform, Turn

BUBBLE = "<|bubble|>"
BUBBLE_JOIN = f"\n{BUBBLE}\n"
GAP_MARKER_AFTER = timedelta(hours=1)
PLATFORM_NAMES = {Platform.imessage: "iMessage", Platform.discord: "Discord", Platform.instagram: "Instagram"}


@dataclass
class Example:
    them: str
    me: str


@dataclass
class PromptContext:
    me: str
    platform: Platform
    is_group: bool
    title: str | None
    others: list[str]
    turns: list[Turn]  # oldest first; the reply is to the end of this
    now: datetime
    profile: str | None = None
    examples: list[Example] = field(default_factory=list)


def _gap(delta: timedelta) -> str:
    hours = delta.total_seconds() / 3600
    if hours < 24:
        n = round(hours)
        return f"{n} hour{'s' if n != 1 else ''} later"
    n = round(hours / 24)
    return f"{n} day{'s' if n != 1 else ''} later"


def render_turn(turn: Turn, is_group: bool) -> str:
    bubbles = turn.bubbles()
    if turn.is_me:
        return BUBBLE_JOIN.join(bubbles)
    if is_group:
        return "\n".join(f"{turn.sender_name}: {b}" for b in bubbles)
    return "\n".join(bubbles)


def format_target(turn: Turn) -> str:
    return BUBBLE_JOIN.join(turn.bubbles())


def system_prompt(ctx: PromptContext) -> str:
    where = PLATFORM_NAMES.get(ctx.platform, "text")
    if ctx.is_group:
        chat = f'in the group chat "{ctx.title}" with {", ".join(ctx.others)}' if ctx.title else f"in a group chat with {', '.join(ctx.others)}"
    else:
        chat = f"in a DM with {ctx.others[0] if ctx.others else 'a friend'}"
    now = ctx.now.astimezone()
    lines = [
        f"You are {ctx.me}. You're texting on {where} {chat}.",
        f"It's {now.strftime('%A')}, {now.strftime('%I:%M %p').lstrip('0')}.",
        f"Write {ctx.me}'s next reply exactly the way {ctx.me} texts. Put {BUBBLE} between separate texts.",
    ]
    if ctx.profile:
        lines += ["", "About this chat:", ctx.profile.strip()]
    if ctx.examples:
        lines += ["", f"Things {ctx.me} has said in similar moments:"]
        for ex in ctx.examples:
            lines.append(f"- them: {ex.them}\n  {ctx.me}: {ex.me.replace(BUBBLE_JOIN, ' / ')}")
    return "\n".join(lines)


def build_messages(ctx: PromptContext) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt(ctx)}]
    prev_end: datetime | None = None
    for turn in ctx.turns:
        content = render_turn(turn, ctx.is_group)
        if not content:
            continue
        if prev_end and turn.start - prev_end > GAP_MARKER_AFTER:
            content = f"[{_gap(turn.start - prev_end)}]\n{content}"
        prev_end = turn.end
        role = "assistant" if turn.is_me else "user"
        if msgs[-1]["role"] == role:
            msgs[-1]["content"] += (BUBBLE_JOIN if role == "assistant" else "\n") + content
        else:
            msgs.append({"role": role, "content": content})
    return msgs


def truncate(ctx: PromptContext, max_chars: int) -> PromptContext:
    """Drop the oldest turns until the rendered prompt fits."""
    while len(ctx.turns) > 1 and sum(len(m["content"]) for m in build_messages(ctx)) > max_chars:
        ctx.turns = ctx.turns[1:]
    return ctx


_THINK = re.compile(r"<think>.*?</think>", re.S)


def split_bubbles(text: str, me: str) -> list[str]:
    text = _THINK.sub("", text).replace("<|im_end|>", "")
    prefix = re.compile(rf"^\s*{re.escape(me)}\s*:\s*", re.I)
    out = []
    for part in text.split(BUBBLE):
        part = prefix.sub("", part.strip()).strip()
        if part:
            out.append(part)
    return out
