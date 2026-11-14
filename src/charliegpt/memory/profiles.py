"""Per-chat memory: a short summary of who you're talking to and how you talk to them.

Generated once from history by the helper (base) model; `notes` is yours to edit by hand.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

from charliegpt.schema import Conversation

PROFILE_PROMPT = """Below are text messages between {me} and {who}.
Write 3-5 short bullet points that would help someone continue this chat as {me}:
who {them_pronoun} is to {me}, recurring topics, inside jokes, and how {me} talks in this chat
(tone, slang, emoji). Be concrete. No preamble.

{transcript}"""


class ProfileStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, conv: Conversation) -> str | None:
        entry = self.data.get(conv.key)
        if entry is None and conv.title:
            # Live chats can have different ids than exports (Instagram); fall back to title.
            entry = next((e for e in self.data.values() if e.get("title") == conv.title and e.get("platform") == conv.platform.value), None)
        if not entry:
            return None
        parts = [entry.get("summary", "").strip(), entry.get("notes", "").strip()]
        return "\n".join(p for p in parts if p) or None

    def set(self, conv: Conversation, summary: str) -> None:
        entry = self.data.setdefault(conv.key, {"notes": ""})
        entry.update(title=conv.title, platform=conv.platform.value, summary=summary)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))


def transcript(sessions: list[Conversation], max_chars: int) -> str:
    lines: list[str] = []
    for s in sessions:
        lines += [f"{m.sender_name}: {r}" for m in s.messages if (r := m.render())]
    text = "\n".join(lines)
    return text[-max_chars:]  # most recent history matters most


async def build_profiles(
    sessions: list[Conversation],
    store: ProfileStore,
    complete: Callable[[list[dict]], "object"],
    me: str,
    max_chars: int = 8000,
    min_messages: int = 30,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    by_key: dict[str, list[Conversation]] = defaultdict(list)
    for s in sessions:
        by_key[s.key].append(s)
    n = 0
    for key, group in by_key.items():
        if sum(len(s.messages) for s in group) < min_messages:
            continue
        conv = group[-1]
        who = conv.title if conv.is_group else (conv.others()[0].name if conv.others() else "a friend")
        prompt = PROFILE_PROMPT.format(
            me=me,
            who=f"the group chat {who}" if conv.is_group else who,
            them_pronoun="this group" if conv.is_group else "this person",
            transcript=transcript(group, max_chars),
        )
        summary = await complete([{"role": "user", "content": prompt}])
        store.set(conv, str(summary).strip())
        n += 1
        if on_progress:
            on_progress(key)
    store.save()
    return n
