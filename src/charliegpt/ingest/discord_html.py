"""Discord: HTML exports from DiscordChatExporter (`--format HtmlDark`/`HtmlLight`), one file per channel,
named like "Server - Category - channel [channel_id].html".

Messages come in groups: the first container has the author header, follow-ups inherit it.
Timestamps are decoded from the message id (a Discord snowflake), which is exact and in UTC;
the rendered times are minute-resolution local time.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup, Tag

from charliegpt.ingest.base import URL_RE, kind_from_name
from charliegpt.schema import Attachment, AttachmentKind, Conversation, Message, Participant, Platform, Reaction

DISCORD_EPOCH_MS = 1420070400000
CHANNEL_ID_RE = re.compile(r"\[(\d+)\]$")
REPLY_ID_RE = re.compile(r"'(\d+)'")


def snowflake_time(message_id: str) -> datetime:
    return datetime.fromtimestamp(((int(message_id) >> 22) + DISCORD_EPOCH_MS) / 1000, tz=timezone.utc)


def _url_name(url: str) -> str:
    return unquote(Path(urlparse(url).path).name)


class DiscordHTMLParser:
    def __init__(self, my_ids: set[str], me_name: str):
        self.my_ids = my_ids  # user ids and/or usernames
        self.me_name = me_name

    def parse_file(self, path: Path) -> Conversation:
        html = path.read_text(encoding="utf-8")
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:  # lxml not installed
            soup = BeautifulSoup(html, "html.parser")

        entries = [e.get_text(" ", strip=True) for e in soup.select(".preamble__entry")]
        guild = entries[0] if entries else ""
        channel = entries[1] if len(entries) > 1 else path.stem
        is_dm = guild == "Direct Messages"
        title = channel.split(" / ")[-1] if is_dm else f"{guild} #{channel.split(' / ')[-1]}"
        m = CHANNEL_ID_RE.search(path.stem)
        thread_id = m.group(1) if m else path.stem

        people: dict[str, Participant] = {}
        messages: list[Message] = []
        for group in soup.select("div.chatlog__message-group"):
            author: tuple[str, str, str | None, bool] | None = None  # (id, display name, username, is_bot)
            for container in group.select("div.chatlog__message-container"):
                header = container.select_one(".chatlog__header")
                if header is not None:
                    a = header.select_one(".chatlog__author")
                    if a is not None:
                        uid = a.get("data-user-id") or a.get("title") or a.get_text(strip=True)
                        is_bot = header.select_one(".chatlog__author-tag") is not None
                        author = (uid, a.get_text(strip=True), a.get("title"), is_bot)
                if author is None or author[3] or container.select_one(".chatlog__system-notification-content"):
                    continue  # bots and "X pinned a message" notices
                msg = self._message(container, *author[:3])
                if msg is None:
                    continue
                people.setdefault(msg.sender_id, Participant(id=msg.sender_id, name=msg.sender_name, is_me=msg.is_me))
                messages.append(msg)

        return Conversation(
            platform=Platform.discord,
            thread_id=thread_id,
            title=title,
            is_group=not is_dm or len(people) > 2,
            participants=list(people.values()),
            messages=messages,
        )

    def _message(self, c: Tag, user_id: str, display: str, username: str | None) -> Message | None:
        mid = c.get("data-message-id")
        if not mid:
            return None
        is_me = user_id in self.my_ids or (username is not None and username in self.my_ids)
        text = ""
        content = c.select_one(".chatlog__content")
        if content is not None:
            for tag in content.select(".chatlog__edited-timestamp"):
                tag.decompose()
            for img in content.select("img.chatlog__emoji"):
                img.replace_with(img.get("alt") or "")
            for br in content.find_all("br"):
                br.replace_with("\n")
            for q in content.select(".chatlog__markdown-quote-content"):
                q.insert(0, "> ")
            text = content.get_text().strip()

        atts: list[Attachment] = []
        for a in c.select(".chatlog__attachment"):
            media = a.select_one("img, video, audio")
            href = (a.select_one("a") or {}).get("href") or (media.get("src") if media else None)
            title = media.get("title", "") if media is not None else ""
            name = title.split(": ", 1)[-1].rsplit(" (", 1)[0] if ": " in title else (_url_name(href) if href else None)
            kind = {"video": AttachmentKind.video, "audio": AttachmentKind.audio}.get(media.name if media else "", kind_from_name(name))
            if kind == AttachmentKind.file and media is not None and media.name == "img":
                kind = AttachmentKind.image
            atts.append(Attachment(kind=kind, url=href, name=name))
        for s in c.select(".chatlog__sticker"):
            atts.append(Attachment(kind=AttachmentKind.sticker, name=s.get("title")))
        for e in c.select(".chatlog__embed"):
            if gif := e.select_one(".chatlog__embed-generic-gifv source, .chatlog__embed-generic-gifv"):
                atts.append(Attachment(kind=AttachmentKind.gif, url=gif.get("src")))
                continue
            link = e.select_one(".chatlog__embed-title-link, .chatlog__embed-author-link")
            title = e.select_one(".chatlog__embed-title")
            url = link.get("href") if link is not None else None
            if url or title:
                atts.append(Attachment(kind=AttachmentKind.link, url=url, name=title.get_text(" ", strip=True) if title else None))
            elif img := e.select_one(".chatlog__embed-generic-image, .chatlog__embed-image"):
                atts.append(Attachment(kind=AttachmentKind.image, url=img.get("src")))
        linked = {a.url for a in atts}
        for url in URL_RE.findall(text):
            if url not in linked:
                kind = AttachmentKind.gif if "tenor.com" in url or "giphy.com" in url else AttachmentKind.link
                atts.append(Attachment(kind=kind, url=url))

        reactions = []
        for r in c.select(".chatlog__reaction"):
            img = r.select_one("img")
            emoji = (img.get("alt") if img is not None else None) or r.get("title") or "?"
            count_el = r.select_one(".chatlog__reaction-count")
            count = int(count_el.get_text(strip=True)) if count_el and count_el.get_text(strip=True).isdigit() else 1
            reactions += [Reaction(emoji=emoji, sender_id="unknown")] * count

        reply_to = reply_text = None
        if reply := c.select_one(".chatlog__reply"):
            link = reply.select_one(".chatlog__reply-link")
            if link is not None and (m := REPLY_ID_RE.search(link.get("onclick", ""))):
                reply_to = m.group(1)
                reply_text = link.get_text(" ", strip=True) or None

        msg = Message(
            id=mid,
            sender_id="me" if is_me else user_id,
            sender_name=self.me_name if is_me else display,
            is_me=is_me,
            timestamp=snowflake_time(mid),
            text=text,
            attachments=atts,
            reactions=reactions,
            reply_to=reply_to,
            reply_to_text=reply_text,
        )
        return None if msg.is_empty else msg
