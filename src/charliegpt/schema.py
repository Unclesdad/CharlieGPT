"""The uniform message format every platform is converted into."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Platform(str, Enum):
    imessage = "imessage"
    discord = "discord"
    instagram = "instagram"
    other = "other"


class AttachmentKind(str, Enum):
    image = "image"
    video = "video"
    audio = "audio"
    sticker = "sticker"
    gif = "gif"
    link = "link"
    file = "file"


class Attachment(BaseModel):
    kind: AttachmentKind
    path: str | None = None  # local file, if we have it
    url: str | None = None
    name: str | None = None
    mime: str | None = None
    caption: str | None = None  # filled in by the vision model

    def render(self) -> str:
        if self.kind == AttachmentKind.image:
            return f'[image: "{self.caption}"]' if self.caption else "[image]"
        if self.kind == AttachmentKind.audio:
            return f'[voice memo: "{self.caption}"]' if self.caption else "[voice memo]"
        if self.kind == AttachmentKind.link:
            if self.url and self.name:
                about = " ".join(self.name.split())
                about = about if len(about) <= 100 else about[:97] + "..."
                return f'[shared {self.url}: "{about}"]'
            if self.url:
                return f"[link: {self.url}]"
            return f"[{self.name}]" if self.name else "[link]"
        if self.kind == AttachmentKind.sticker:
            return f"[sticker: {self.name}]" if self.name else "[sticker]"
        if self.kind == AttachmentKind.file:
            return f"[file: {self.name}]" if self.name else "[file]"
        return f"[{self.kind.value}]"


class Reaction(BaseModel):
    emoji: str
    sender_id: str


class Participant(BaseModel):
    id: str
    name: str
    is_me: bool = False


class Message(BaseModel):
    id: str
    sender_id: str
    sender_name: str
    is_me: bool
    timestamp: datetime
    text: str = ""
    attachments: list[Attachment] = Field(default_factory=list)
    reactions: list[Reaction] = Field(default_factory=list)
    reply_to: str | None = None  # id of the message this replies to
    reply_to_text: str | None = None  # resolved during normalization

    def render(self) -> str:
        """Text the model sees for this single message (one bubble)."""
        parts: list[str] = []
        if self.reply_to_text:
            snippet = self.reply_to_text.replace("\n", " ")
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            parts.append(f'(replying to "{snippet}")')
        text = self.text.strip()
        if text:
            parts.append(text)
        for att in self.attachments:
            # Don't repeat a bare link that's already in the text (keep it if it has a title).
            if att.kind == AttachmentKind.link and att.url and att.url in text and not att.name:
                continue
            parts.append(att.render())
        return " ".join(parts)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.attachments


class Conversation(BaseModel):
    """One session of a thread (threads are split on long silences)."""

    platform: Platform
    thread_id: str  # stable id of the chat on its platform
    session: int = 0
    title: str | None = None
    is_group: bool = False
    participants: list[Participant] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.platform.value}:{self.thread_id}#{self.session}"

    @property
    def key(self) -> str:
        """Key for per-chat memory (stable across sessions)."""
        return f"{self.platform.value}:{self.thread_id}"

    def others(self) -> list[Participant]:
        return [p for p in self.participants if not p.is_me]


class Turn(BaseModel):
    """Consecutive messages from one sender: a 'burst' of bubbles."""

    sender_id: str
    sender_name: str
    is_me: bool
    messages: list[Message]

    @property
    def start(self) -> datetime:
        return self.messages[0].timestamp

    @property
    def end(self) -> datetime:
        return self.messages[-1].timestamp

    def bubbles(self) -> list[str]:
        return [r for m in self.messages if (r := m.render())]
