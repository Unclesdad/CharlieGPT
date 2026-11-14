"""Shared helpers for platform parsers."""

from __future__ import annotations

import mimetypes
import re
from typing import Iterable, Protocol

from charliegpt.schema import AttachmentKind, Conversation

URL_RE = re.compile(r"https?://\S+")


class Parser(Protocol):
    def parse(self) -> Iterable[Conversation]: ...


def kind_from_name(name: str | None, mime: str | None = None) -> AttachmentKind:
    mime = mime or (mimetypes.guess_type(name or "")[0] or "")
    lower = (name or "").lower()
    if lower.endswith(".gif") or "gif" in mime:
        return AttachmentKind.gif
    if lower.endswith((".heic", ".heif")) or mime.startswith("image/"):
        return AttachmentKind.image
    if lower.endswith((".caf", ".m4a", ".amr", ".opus")) or mime.startswith("audio/"):
        return AttachmentKind.audio
    if mime.startswith("video/"):
        return AttachmentKind.video
    return AttachmentKind.file


def fix_mojibake(s: str) -> str:
    """Meta exports UTF-8 bytes as latin-1 escaped text ("ð\u009f\u0098\u0082" for 😂)."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
