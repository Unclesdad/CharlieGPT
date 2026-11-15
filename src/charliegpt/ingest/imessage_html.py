"""iMessage: HTML exports from imessage-exporter (https://github.com/ReagentX/imessage-exporter),
e.g. `imessage-exporter -f html -o data/messages`.

One file per chat, named by participant handles ("+15551234567, +15557654321.html") or by the
group's name ("Group Name - 123.html"). Threaded replies are nested inside their parent message.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Tag

from charliegpt.ingest.base import URL_RE, kind_from_name
from charliegpt.ingest.vcard import normalize_handle
from charliegpt.schema import Attachment, AttachmentKind, Conversation, Message, Participant, Platform, Reaction

ME_ID = "me"
TAPBACK_EMOJI = {"Loved": "❤️", "Liked": "👍", "Disliked": "👎", "Laughed": "😂", "Emphasized": "‼️", "Questioned": "❓"}
HANDLE_RE = re.compile(r"^(\+?\d[\d\s()-]{6,}|[^@\s]+@[^@\s]+\.\w+)$")
NAMED_GROUP_RE = re.compile(r"^(?P<name>.+) - \d+$")


def _classes(el: Tag) -> list[str]:
    return el.get("class") or []


def _child(el: Tag, tag: str, cls: str) -> Tag | None:
    return next((c for c in el.find_all(tag, recursive=False) if cls in _classes(c)), None)


def _children(el: Tag, tag: str, cls: str) -> list[Tag]:
    return [c for c in el.find_all(tag, recursive=False) if cls in _classes(c)]


def parse_timestamp(text: str) -> datetime:
    """'Nov 07, 2025  6:02:52 AM' in the exporting Mac's local time."""
    return datetime.strptime(" ".join(text.split()), "%b %d, %Y %I:%M:%S %p").astimezone()


class IMessageHTMLParser:
    def __init__(self, root: Path, contacts: dict[str, str], me_name: str):
        self.root = Path(root)
        self.contacts = contacts
        self.me_name = me_name

    def _person(self, label: str) -> tuple[str, str]:
        """Exporter sender label → (sender_id, display name). Labels are handles or contact names."""
        label = label.strip()
        if HANDLE_RE.match(label):
            handle = normalize_handle(label)
            return handle, self.contacts.get(handle, label)
        return label, label

    def parse(self) -> Iterable[Conversation]:
        for f in sorted(self.root.glob("*.html")):
            if f.name in ("orphaned.html",):
                continue
            conv = self.parse_file(f)
            if conv.messages:
                yield conv

    def parse_file(self, path: Path) -> Conversation:
        try:
            soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
        except Exception:  # lxml not installed
            soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        stem = path.stem
        named = NAMED_GROUP_RE.match(stem)
        handles = [] if named else [h.strip() for h in stem.split(",") if h.strip()]

        # Threaded replies appear twice: nested under their parent and again in the timeline.
        by_id: dict[str, Message] = {}
        for el in soup.find_all("div", class_="message"):
            if msg := self._message(el):
                if (seen := by_id.get(msg.id)) is None:
                    by_id[msg.id] = msg
                elif not seen.reply_to:
                    seen.reply_to = msg.reply_to
        messages = list(by_id.values())

        is_group = bool(named) or len(handles) != 1
        if not is_group:
            # DM: the other side is always the file's handle, however the exporter labeled them.
            hid, name = self._person(handles[0])
            for m in messages:
                for r in m.reactions:
                    if r.sender_id != ME_ID:
                        r.sender_id = hid
                if not m.is_me:
                    m.sender_id = hid
                    m.sender_name = name if name != handles[0] else m.sender_name
        people = {m.sender_id: m.sender_name for m in messages if not m.is_me}
        for h in handles:
            hid, name = self._person(h)
            people.setdefault(hid, name)
        participants = [Participant(id=ME_ID, name=self.me_name, is_me=True)] + [Participant(id=k, name=v) for k, v in people.items()]
        if named:
            title = named.group("name")
        elif not is_group:
            title = people[self._person(handles[0])[0]]
        else:
            title = ", ".join(people.get(self._person(h)[0], h) for h in handles) or stem
        return Conversation(platform=Platform.imessage, thread_id=stem, title=title, is_group=is_group, participants=participants, messages=messages)

    def _message(self, el: Tag) -> Message | None:
        body = next((c for c in el.find_all("div", recursive=False) if {"sent", "received"} & set(_classes(c))), None)
        if body is None:
            return None
        header = body.find("p", recursive=False)
        ts = header.find("span", class_="timestamp") if header else None
        link = ts.find("a") if ts else None
        if not link:
            return None
        guid = (re.search(r"message-guid=([^&\"]+)", link.get("href", "")) or [None, None])[1] or link.get_text()
        is_me = "sent" in _classes(body)
        sender_el = header.find("span", class_="sender")
        if is_me:
            sender_id, sender_name = ME_ID, self.me_name
        else:
            sender_id, sender_name = self._person(sender_el.get_text() if sender_el else "unknown")

        texts: list[str] = []
        atts: list[Attachment] = []
        for part in _children(body, "div", "message_part"):
            self._part(part, texts, atts)
        if loc := _child(body, "span", "shared_location"):
            atts.append(Attachment(kind=AttachmentKind.link, name="shared location"))
        text = "\n".join(t for t in texts if t)
        for url in URL_RE.findall(text):
            if not any(a.url == url for a in atts):
                atts.append(Attachment(kind=AttachmentKind.link, url=url))

        # Threaded reply: parent is the enclosing message.
        parent = el.find_parent("div", class_="message")
        parent_link = parent.find("span", class_="timestamp").find("a") if parent and parent.find("span", class_="timestamp") else None
        reply_to = None
        if parent_link:
            m = re.search(r"message-guid=([^&\"]+)", parent_link.get("href", ""))
            reply_to = m.group(1) if m else None

        msg = Message(
            id=guid,
            sender_id=sender_id,
            sender_name=sender_name,
            is_me=is_me,
            timestamp=parse_timestamp(link.get_text()),
            text=text,
            attachments=atts,
            reactions=self._tapbacks(body),
            reply_to=reply_to,
        )
        return None if msg.is_empty else msg

    def _part(self, part: Tag, texts: list[str], atts: list[Attachment]) -> None:
        if part.find("span", class_="unsent"):
            return
        if edited := part.find("div", class_="edited"):
            rows = edited.select("tfoot tr") or edited.select("tbody tr")
            if rows:
                cells = rows[-1].find_all("td")
                texts.append(cells[-1].get_text().strip())
            return
        for bubble in part.find_all("span", class_="bubble"):
            for img in bubble.find_all("img"):  # inline stickers
                img.decompose()
            texts.append(bubble.get_text().strip())
        for a in part.find_all("div", class_="attachment"):
            media = a.find(["img", "video", "audio"])
            if media is not None:
                src = media.get("src") or (media.find("source") or {}).get("src")
                kind = {"video": AttachmentKind.video, "audio": AttachmentKind.audio}.get(media.name) or kind_from_name(src)
                if kind == AttachmentKind.file:
                    kind = AttachmentKind.image
            else:
                href = (a.find("a") or {}).get("href")
                src, kind = href, kind_from_name(href)
            path = src if src and src.startswith("/") else None
            atts.append(Attachment(kind=kind, path=path, url=None if path else src, name=Path(src).name if src else None))
        if tr := part.find("span", class_="transcription"):
            audio = next((x for x in reversed(atts) if x.kind == AttachmentKind.audio), None)
            if audio:
                audio.caption = tr.get_text().removeprefix("Transcription:").strip()
        if st := part.find("div", class_="sticker"):
            name = st.find("div", class_="sticker_name")
            atts.append(Attachment(kind=AttachmentKind.sticker, name=name.get_text().strip() if name else None))
        if app := part.find("div", class_="app"):
            href = (app.find("a") or {}).get("href") or ""
            caption = app.find("div", class_="caption")
            name = app.find("div", class_="name")
            label = (caption or name).get_text().strip() if (caption or name) else None
            if href.startswith("http"):
                atts.append(Attachment(kind=AttachmentKind.link, url=href, name=label))
            elif label:
                atts.append(Attachment(kind=AttachmentKind.link, name=label))

    def _tapbacks(self, body: Tag) -> list[Reaction]:
        box = _child(body, "div", "tapbacks")
        if box is None:
            return []
        out = []
        for tb in box.find_all("span", class_="tapback"):
            verb = tb.find("b")
            emoji = TAPBACK_EMOJI.get(verb.get_text().strip(), verb.get_text().strip()) if verb else "🏷️"
            who = tb.get_text().replace(verb.get_text() if verb else "", "", 1).strip().removeprefix("by").strip()
            sender = ME_ID if who == "Me" else self._person(who)[0]
            out.append(Reaction(emoji=emoji, sender_id=sender))
        return out
