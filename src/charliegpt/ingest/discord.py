"""Discord: exports from DiscordChatExporter (https://github.com/Tyrrrz/DiscordChatExporter).

Both `--format Json` and `--format HtmlDark/HtmlLight` are read (HTML is parsed in discord_html.py).
`--media` is optional; it downloads attachments so images can be captioned.
Channels you never posted in are skipped: they can't teach your replies.
Discord's official data package only contains *your* messages with no replies around them,
so it can't teach conversational context and isn't supported.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from charliegpt.ingest.base import URL_RE, kind_from_name
from charliegpt.schema import Attachment, AttachmentKind, Conversation, Message, Participant, Platform, Reaction

DM_TYPES = {"DirectTextChat", "DirectGroupTextChat"}
SKIP_TYPES = {"ChannelPinnedMessage", "GuildMemberJoin", "RecipientAdd", "RecipientRemove", "Call", "ChannelNameChange", "ThreadCreated"}


class DiscordParser:
    def __init__(self, root: Path, my_ids: list[str], me_name: str, skip_without_me: bool = True):
        self.root = Path(root)
        self.my_ids = set(my_ids)  # user ids and/or usernames
        self.me_name = me_name
        self.skip_without_me = skip_without_me
        self.skipped: list[str] = []  # titles of channels dropped because you never posted

    def parse(self) -> Iterable[Conversation]:
        from charliegpt.ingest.discord_html import DiscordHTMLParser

        html = DiscordHTMLParser(self.my_ids, self.me_name)
        files = [self.root] if self.root.is_file() else sorted(self.root.rglob("*.json")) + sorted(self.root.rglob("*.html"))
        for f in files:
            if f.suffix == ".html":
                conv = html.parse_file(f)
            else:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "messages" not in data or "channel" not in data:
                    continue
                conv = self._conversation(data, f.parent)
            if not conv.messages:
                continue
            if self.skip_without_me and not any(m.is_me for m in conv.messages):
                self.skipped.append(conv.title or f.stem)
                continue
            yield conv

    def _resolve_media(self, url: str | None, base: Path) -> str | None:
        """With --media, DCE rewrites attachment urls to relative local paths."""
        if not url or url.startswith("http"):
            return None
        p = (base / url).resolve()
        return str(p) if p.exists() else None

    def _conversation(self, data: dict, base: Path) -> Conversation:
        ch = data["channel"]
        guild = data.get("guild") or {}
        is_dm = ch.get("type") in DM_TYPES
        is_group = ch.get("type") != "DirectTextChat"
        title = ch.get("name") if is_dm else f"{guild.get('name', '?')} #{ch.get('name')}"
        people: dict[str, Participant] = {}
        messages: list[Message] = []
        for m in data["messages"]:
            if m.get("type") in SKIP_TYPES:
                continue
            author = m.get("author") or {}
            aid = str(author.get("id"))
            is_me = aid in self.my_ids or author.get("name") in self.my_ids
            name = self.me_name if is_me else (author.get("nickname") or author.get("name") or aid)
            people.setdefault(aid, Participant(id=aid, name=name, is_me=is_me))
            text = m.get("content") or ""
            atts: list[Attachment] = []
            for a in m.get("attachments", []):
                fname = a.get("fileName")
                local = self._resolve_media(a.get("url"), base)
                atts.append(Attachment(kind=kind_from_name(fname), path=local, url=None if local else a.get("url"), name=fname))
            for s in m.get("stickers", []):
                atts.append(Attachment(kind=AttachmentKind.sticker, name=s.get("name")))
            for url in URL_RE.findall(text):
                kind = AttachmentKind.gif if "tenor.com" in url or "giphy.com" in url else AttachmentKind.link
                atts.append(Attachment(kind=kind, url=url))
            reactions = []
            for r in m.get("reactions", []):
                emoji = (r.get("emoji") or {}).get("name") or "?"
                users = r.get("users") or [{"id": "unknown"}] * int(r.get("count", 1))
                reactions += [Reaction(emoji=emoji, sender_id=str(u.get("id"))) for u in users]
            ref = m.get("reference") or {}
            msg = Message(
                id=str(m["id"]),
                sender_id=aid,
                sender_name=name,
                is_me=is_me,
                timestamp=datetime.fromisoformat(m["timestamp"]),
                text=text,
                attachments=atts,
                reactions=reactions,
                reply_to=str(ref["messageId"]) if ref.get("messageId") else None,
            )
            if not msg.is_empty:
                messages.append(msg)
        return Conversation(
            platform=Platform.discord,
            thread_id=str(ch["id"]),
            title=title,
            is_group=is_group,
            participants=list(people.values()),
            messages=messages,
        )
