"""Discord bot (official bot account). Needs the Message Content intent enabled in the dev portal."""

from __future__ import annotations

import os

import discord

from charliegpt.bots.base import BaseBot
from charliegpt.ingest.base import URL_RE, kind_from_name
from charliegpt.schema import Attachment, AttachmentKind, Conversation, Message, Participant, Platform

MAX_IMAGE_BYTES = 10 * 1024 * 1024


class DiscordBot(BaseBot):
    platform = Platform.discord

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        self.channels: dict[str, discord.abc.Messageable] = {}
        self.client.event(self.on_ready)
        self.client.event(self.on_message)

    async def run(self) -> None:
        token = os.environ.get("DISCORD_TOKEN")
        if not token:
            raise RuntimeError("DISCORD_TOKEN is not set (put it in .env)")
        await self.client.start(token)

    async def on_ready(self) -> None:
        self.log.info("logged in as %s", self.client.user)

    def _is_me(self, user: discord.abc.User) -> bool:
        owner = self.settings.discord.owner_user_id
        return user == self.client.user or (owner is not None and str(user.id) == str(owner))

    def _conversation(self, channel) -> Conversation:
        me = self.settings.identity.name
        if isinstance(channel, discord.DMChannel):
            other = channel.recipient
            return Conversation(
                platform=Platform.discord,
                thread_id=str(channel.id),
                title=other.display_name if other else None,
                is_group=False,
                participants=[Participant(id="me", name=me, is_me=True)] + ([Participant(id=str(other.id), name=other.display_name)] if other else []),
            )
        title = f"{channel.guild.name} #{channel.name}" if getattr(channel, "guild", None) else getattr(channel, "name", None)
        return Conversation(platform=Platform.discord, thread_id=str(channel.id), title=title, is_group=True)

    async def _message(self, m: discord.Message, download: bool) -> Message:
        is_me = self._is_me(m.author)
        atts: list[Attachment] = []
        for a in m.attachments:
            kind = kind_from_name(a.filename, a.content_type)
            path = None
            if download and kind == AttachmentKind.image and a.size <= MAX_IMAGE_BYTES:
                dest = self.settings.paths.media / "discord" / f"{a.id}_{a.filename}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                await a.save(dest)
                path = str(dest)
            atts.append(Attachment(kind=kind, path=path, url=None if path else a.url, name=a.filename))
        atts += [Attachment(kind=AttachmentKind.sticker, name=s.name) for s in m.stickers]
        atts += [Attachment(kind=AttachmentKind.link, url=u) for u in URL_RE.findall(m.content)]
        ref = m.reference
        resolved = ref.resolved if ref and isinstance(ref.resolved, discord.Message) else None
        return Message(
            id=str(m.id),
            sender_id="me" if is_me else str(m.author.id),
            sender_name=self.settings.identity.name if is_me else m.author.display_name,
            is_me=is_me,
            timestamp=m.created_at,
            text=m.clean_content,
            attachments=atts,
            reply_to=str(ref.message_id) if ref and ref.message_id else None,
            reply_to_text=resolved.clean_content if resolved else None,
        )

    async def on_message(self, m: discord.Message) -> None:
        if m.author.bot and m.author != self.client.user:
            return
        self.channels[str(m.channel.id)] = m.channel
        msg = await self._message(m, download=not self._is_me(m.author))
        mentioned = self.client.user in m.mentions
        await self.handle(self._conversation(m.channel), msg, mentioned=mentioned)

    async def fetch_history(self, conv: Conversation, before: Message) -> list[Message]:
        channel = self.channels.get(conv.thread_id)
        if channel is None:
            return []
        out = []
        async for m in channel.history(limit=self.cfg.history_limit, before=discord.Object(id=int(before.id))):
            if not m.author.bot or m.author == self.client.user:
                out.append(await self._message(m, download=False))
        return out[::-1]

    async def _channel(self, thread_id: str):
        return self.channels.get(thread_id) or self.client.get_channel(int(thread_id)) or await self.client.fetch_channel(int(thread_id))

    async def typing(self, thread_id: str) -> None:
        await (await self._channel(thread_id)).typing()

    async def send(self, thread_id: str, text: str) -> None:
        await (await self._channel(thread_id)).send(text, allowed_mentions=discord.AllowedMentions.none())
