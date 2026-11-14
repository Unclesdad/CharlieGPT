"""Instagram DM bot via instagrapi (the private API).

DEMO ONLY: automating a personal Instagram account violates Instagram's Terms of Use and can get
the account banned. The bot refuses to start unless `instagram.i_understand_this_violates_tos: true`.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timezone

import httpx

from charliegpt.bots.base import BaseBot
from charliegpt.schema import Attachment, AttachmentKind, Conversation, Message, Participant, Platform


class InstagramBot(BaseBot):
    platform = Platform.instagram

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.settings.instagram.i_understand_this_violates_tos:
            raise RuntimeError("Instagram bot is demo-only; set instagram.i_understand_this_violates_tos: true to run it.")
        from instagrapi import Client

        self.cl = Client()
        self.my_id = ""
        self.started = False

    def _login(self) -> None:
        sess = self.settings.instagram.session_file
        if sess.exists():
            self.cl.load_settings(sess)
        self.cl.login(os.environ["INSTAGRAM_USERNAME"], os.environ["INSTAGRAM_PASSWORD"])
        sess.parent.mkdir(parents=True, exist_ok=True)
        self.cl.dump_settings(sess)
        self.my_id = str(self.cl.user_id)

    def _download(self, url: str, name: str) -> str | None:
        dest = self.settings.paths.media / "instagram" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(httpx.get(url, timeout=30).content)
            return str(dest)
        except httpx.HTTPError:
            return None

    def _message(self, m, names: dict[str, str], download: bool) -> Message:
        uid = str(m.user_id)
        is_me = uid == self.my_id
        atts = []
        media = getattr(m, "media", None) or getattr(m, "visual_media", None)
        if media is not None and getattr(media, "thumbnail_url", None):
            path = self._download(str(media.thumbnail_url), f"{m.id}.jpg") if download else None
            atts.append(Attachment(kind=AttachmentKind.image, path=path))
        if getattr(m, "link", None):
            atts.append(Attachment(kind=AttachmentKind.link, url=str(getattr(m.link, "link_url", "") or "")))
        ts = m.timestamp if m.timestamp.tzinfo else m.timestamp.replace(tzinfo=timezone.utc)
        return Message(
            id=str(m.id),
            sender_id="me" if is_me else uid,
            sender_name=self.settings.identity.name if is_me else names.get(uid, uid),
            is_me=is_me,
            timestamp=ts,
            text=m.text or "",
            attachments=atts,
        )

    async def run(self) -> None:
        await asyncio.to_thread(self._login)
        self.log.info("logged in as %s", self.my_id)
        while True:
            try:
                threads = await asyncio.to_thread(self.cl.direct_threads, 20)
                for t in threads:
                    names = {str(u.pk): (u.full_name or u.username) for u in t.users}
                    conv = Conversation(
                        platform=Platform.instagram,
                        thread_id=str(t.id),
                        title=t.thread_title or ", ".join(names.values()),
                        is_group=bool(t.is_group),
                        participants=[Participant(id="me", name=self.settings.identity.name, is_me=True)]
                        + [Participant(id=k, name=v) for k, v in names.items()],
                    )
                    msgs = await asyncio.to_thread(self.cl.direct_messages, t.id, 20)
                    for m in sorted(msgs, key=lambda m: m.timestamp):
                        msg = await asyncio.to_thread(self._message, m, names, self.started)
                        await self.handle(conv, msg, backfill=not self.started)
                self.started = True  # first pass only records history
            except Exception:
                self.log.exception("poll failed")
            await asyncio.sleep(self.settings.instagram.poll_seconds)

    async def send(self, thread_id: str, text: str) -> None:
        await asyncio.to_thread(self.cl.direct_send, text, thread_ids=[int(thread_id)])
