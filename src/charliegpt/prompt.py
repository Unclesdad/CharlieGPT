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


def transcript_lines(ctx: PromptContext) -> list[str]:
    lines: list[str] = []
    prev_end: datetime | None = None
    for turn in ctx.turns:
        bubbles = turn.bubbles()
        if not bubbles:
            continue
        if prev_end and turn.start - prev_end > GAP_MARKER_AFTER:
            lines.append(f"[{_gap(turn.start - prev_end)}]")
        prev_end = turn.end
        name = ctx.me if turn.is_me else turn.sender_name
        lines += [f"{name}: {b}" for b in bubbles]
    return lines


def build_messages(ctx: PromptContext) -> list[dict[str, str]]:
    """system + one user message holding the chat so far; the model's reply is the only assistant turn.

    The history is a transcript, not alternating user/assistant turns: training only learns from
    assistant turns, so putting your past texts there would train on each one many times over.
    """
    chat = "\n".join(transcript_lines(ctx)) or "(no messages yet)"
    return [{"role": "system", "content": system_prompt(ctx)}, {"role": "user", "content": chat}]


def truncate(ctx: PromptContext, max_chars: int) -> PromptContext:
    """Drop the oldest turns until the rendered prompt fits."""
    while len(ctx.turns) > 1 and sum(len(m["content"]) for m in build_messages(ctx)) > max_chars:
        ctx.turns = ctx.turns[1:]
    return ctx


_THINK = re.compile(r"<think>.*?</think>", re.S)
# Placeholders the model learned from your sent media (see Attachment.render). A bot can't send them.
_MEDIA = re.compile(
    r'\[(?:image|video|gif|audio|voice memo|sticker|file|link|shared [^\]\s]+)(?::[^\]]*)?\]'
    r'|\(replying to "[^"]*"\)'
)


_EDITED = re.compile(r"\s*\(edited\)\s*$", re.I)


def split_bubbles(text: str, me: str) -> list[str]:
    text = _THINK.sub("", text).replace("<|im_end|>", "")
    prefix = re.compile(rf"^\s*{re.escape(me)}\s*:\s*", re.I)
    out = []
    # The model sometimes invents its own separator (<|bullet point|>, <|text|>); treat any <|...|> as a break.
    for part in re.split(r"<\|[^|<>\n]{1,30}\|>", text):
        part = _EDITED.sub("", _MEDIA.sub("", prefix.sub("", part.strip())))
        part = re.sub(r"[ \t]{2,}", " ", part).strip()
        if part:
            out.append(part)
    return out
