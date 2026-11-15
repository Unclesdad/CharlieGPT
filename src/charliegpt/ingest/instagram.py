"""Instagram: Meta "Download your information" export in JSON format.

Each thread is a folder with message_1.json (message_2.json, … for long chats) plus photos/,
videos/, audio/. Thread folders can sit anywhere under the root: the full export nests them in
`your_instagram_activity/messages/inbox/`, or they can be copied straight into data/instagram/.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from charliegpt.ingest.base import URL_RE, fix_mojibake
from charliegpt.schema import Attachment, AttachmentKind, Conversation, Message, Participant, Platform, Reaction

# Auto-generated texts that aren't really messages (matched after stripping whitespace).
SYSTEM_TEXT = re.compile(
    r"^(?:"
    r"(?:.+ )?[Ll]iked a message"
    r"|(?:.+ )?[Rr]eacted .+ to your message"
    r"|.+ sent an attachment\."
    r"|.+ shared a story\."
    r"|.+ (?:added .+ to|removed .+ from|joined|left) the group\."
    r"|.+ created the group(?: chat)?\."
    r"|.+ (?:named the group|(?:changed|removed) the group (?:name|photo)|changed the theme|set the (?:quick reaction|emoji)).*"
    r"|.+ set (?:their|your|the) .*nickname.*"
    r"|.+ (?:started|joined) (?:an audio call|a video chat)"
    r"|You missed (?:an audio call|a video chat)"
    r"|(?:Audio call|Video chat) (?:started|ended)"
    r"|.+ (?:unsent|pinned) a message\.?"
    r"|This poll is no longer available\.?"
    r"|.+ (?:voted for|changed (?:their|your) vote to) .+ in the poll\.?"
    r")$"
)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class InstagramParser:
    def __init__(self, root: Path, my_name: str, me_display: str):
        self.root = Path(root)
        self.my_name = my_name
        self.me_display = me_display

    def parse(self) -> Iterable[Conversation]:
        threads: dict[Path, list[Path]] = defaultdict(list)
        for f in self.root.rglob("message_*.json"):
            threads[f.parent].append(f)
        for folder, files in sorted(threads.items()):
            conv = self._thread(folder, sorted(files))
            if conv.messages:
                yield conv

    def _media(self, uri: str | None, folder: Path) -> str | None:
        """uris are relative to the original export root ("your_instagram_activity/messages/inbox/<thread>/photos/123"),
        which may no longer exist if threads were moved; anchor on the thread folder name instead."""
        if not uri:
            return None
        parts = Path(uri).parts
        candidates = [self.root / uri]
        if folder.name in parts:
            candidates.append(folder.joinpath(*parts[parts.index(folder.name) + 1 :]))
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def _thread(self, folder: Path, files: list[Path]) -> Conversation:
        raw_msgs: list[dict] = []
        title = None
        names: list[str] = []
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            title = title or fix_mojibake(data.get("title", "")) or None
            names = names or [fix_mojibake(p["name"]) for p in data.get("participants", [])]
            raw_msgs += data.get("messages", [])
        participants = [
            Participant(id=_slug(n), name=self.me_display if n == self.my_name else n, is_me=n == self.my_name)
            for n in names
        ]
        messages: list[Message] = []
        for i, m in enumerate(sorted(raw_msgs, key=lambda m: m["timestamp_ms"])):
            sender = fix_mojibake(m.get("sender_name", "unknown"))
            text = fix_mojibake(m.get("content", "") or "").strip()
            atts: list[Attachment] = []
            for key, kind in (("photos", AttachmentKind.image), ("videos", AttachmentKind.video), ("audio_files", AttachmentKind.audio), ("gifs", AttachmentKind.gif)):
                for a in m.get(key, []):
                    atts.append(Attachment(kind=kind, path=self._media(a.get("uri"), folder)))
            if sticker := m.get("sticker"):
                atts.append(Attachment(kind=AttachmentKind.sticker, path=self._media(sticker.get("uri"), folder)))
            if share := m.get("share"):
                about = fix_mojibake(share.get("share_text") or share.get("profile_share_name") or "").strip()
                owner = share.get("original_content_owner") or share.get("profile_share_username")
                label = f"@{owner}: {about}" if owner and about else (f"@{owner}" if owner else about)
                atts.append(Attachment(kind=AttachmentKind.link, url=share.get("link"), name=label or None))
            if atts and SYSTEM_TEXT.match(text):
                text = ""
            elif SYSTEM_TEXT.match(text):
                continue
            for url in URL_RE.findall(text):
                if not any(a.url == url for a in atts):
                    atts.append(Attachment(kind=AttachmentKind.link, url=url))
            reactions = [Reaction(emoji=fix_mojibake(r["reaction"]), sender_id=_slug(fix_mojibake(r.get("actor", "")))) for r in m.get("reactions", [])]
            is_me = sender == self.my_name
            msg = Message(
                id=f"{folder.name}:{m['timestamp_ms']}:{i}",
                sender_id=_slug(sender),
                sender_name=self.me_display if is_me else sender,
                is_me=is_me,
                timestamp=datetime.fromtimestamp(m["timestamp_ms"] / 1000, tz=timezone.utc),
                text=text,
                attachments=atts,
                reactions=reactions,
            )
            if not msg.is_empty:
                messages.append(msg)
        return Conversation(
            platform=Platform.instagram,
            thread_id=folder.name,
            title=title,
            is_group=len(participants) > 2,
            participants=participants,
            messages=messages,
        )
