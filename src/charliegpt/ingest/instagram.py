"""Instagram: Meta "Download your information" export in JSON format.

Files live at `[your_instagram_activity/]messages/inbox/<thread>/message_<n>.json`.
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

# Auto-generated texts that aren't really messages.
SYSTEM_TEXT = re.compile(
    r"^(Liked a message|Reacted .+ to your message|.+ sent an attachment\.|.+ shared a story\.|"
    r"You sent an attachment\.|.+ started a video chat|Video chat ended|.+ named the group .+|"
    r".+ added .+ to the group\.|.+ left the group\.|.+ changed the theme .+)$"
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
            if "inbox" in f.parts or "message_requests" in f.parts:
                threads[f.parent].append(f)
        for folder, files in sorted(threads.items()):
            conv = self._thread(folder, sorted(files))
            if conv.messages:
                yield conv

    def _media(self, uri: str | None) -> str | None:
        if not uri:
            return None
        # uris are relative to the export root (e.g. "your_instagram_activity/messages/inbox/.../photos/x.jpg")
        for base in (self.root, *self.root.parents):
            p = base / uri
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
            text = fix_mojibake(m.get("content", "") or "")
            atts: list[Attachment] = []
            for key, kind in (("photos", AttachmentKind.image), ("videos", AttachmentKind.video), ("audio_files", AttachmentKind.audio), ("gifs", AttachmentKind.gif)):
                for a in m.get(key, []):
                    atts.append(Attachment(kind=kind, path=self._media(a.get("uri"))))
            if sticker := m.get("sticker"):
                atts.append(Attachment(kind=AttachmentKind.sticker, path=self._media(sticker.get("uri"))))
            if share := m.get("share"):
                atts.append(Attachment(kind=AttachmentKind.link, url=share.get("link"), name=fix_mojibake(share.get("share_text", "")) or None))
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
