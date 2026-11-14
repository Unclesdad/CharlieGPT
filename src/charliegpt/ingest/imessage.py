"""iMessage: read ~/Library/Messages/chat.db directly (needs Full Disk Access for the terminal).

The query helpers here are shared with the live iMessage bot.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from charliegpt.ingest.base import URL_RE, kind_from_name
from charliegpt.schema import Attachment, AttachmentKind, Conversation, Message, Participant, Platform, Reaction

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
GROUP_STYLE = 43
TAPBACKS = {2000: "❤️", 2001: "👍", 2002: "👎", 2003: "😂", 2004: "‼️", 2005: "❓"}
ME_ID = "me"


def connect(db_path: Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def apple_time(value: int | None) -> datetime:
    value = value or 0
    # macOS 10.13+ stores nanoseconds; older versions store seconds.
    seconds = value / 1e9 if value > 1e11 else value
    return APPLE_EPOCH + timedelta(seconds=seconds)


def decode_attributed_body(blob: bytes | None) -> str | None:
    """Pull the plain string out of an NSAttributedString typedstream (newer macOS leaves `text` NULL)."""
    if not blob or b"NSString" not in blob:
        return None
    try:
        b = blob.split(b"NSString", 1)[1][5:]
        if b[0] == 0x81:
            length = int.from_bytes(b[1:3], "little")
            b = b[3 : 3 + length]
        else:
            length = b[0]
            b = b[1 : 1 + length]
        return b.decode("utf-8", errors="replace")
    except IndexError:
        return None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def message_sql(conn: sqlite3.Connection, where: str = "") -> str:
    cols = _columns(conn, "message")
    optional = [
        f"m.{c}" if c in cols else f"NULL AS {c}"
        for c in ("attributedBody", "associated_message_type", "associated_message_guid", "thread_originator_guid", "item_type", "associated_message_emoji")
    ]
    return f"""
        SELECT m.ROWID AS rowid, m.guid, m.text, m.is_from_me, m.date, {", ".join(optional)},
               h.id AS handle, cmj.chat_id
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        {where}
        ORDER BY m.ROWID
    """


def fetch_chats(conn: sqlite3.Connection) -> dict[int, sqlite3.Row]:
    return {r["ROWID"]: r for r in conn.execute("SELECT ROWID, guid, chat_identifier, display_name, style FROM chat")}


def fetch_chat_handles(conn: sqlite3.Connection) -> dict[int, list[str]]:
    out: dict[int, list[str]] = defaultdict(list)
    for r in conn.execute("SELECT chj.chat_id, h.id FROM chat_handle_join chj JOIN handle h ON h.ROWID = chj.handle_id"):
        out[r[0]].append(r[1])
    return out


def fetch_attachments(conn: sqlite3.Connection, rowids: list[int]) -> dict[int, list[Attachment]]:
    out: dict[int, list[Attachment]] = defaultdict(list)
    for i in range(0, len(rowids), 900):  # sqlite variable limit
        chunk = rowids[i : i + 900]
        q = f"""SELECT maj.message_id, a.filename, a.mime_type, a.transfer_name
                FROM message_attachment_join maj JOIN attachment a ON a.ROWID = maj.attachment_id
                WHERE maj.message_id IN ({",".join("?" * len(chunk))})"""
        for r in conn.execute(q, chunk):
            path = str(Path(r["filename"]).expanduser()) if r["filename"] else None
            name = r["transfer_name"] or (Path(path).name if path else None)
            out[r["message_id"]].append(Attachment(kind=kind_from_name(name, r["mime_type"]), path=path, name=name, mime=r["mime_type"]))
    return out


def rows_to_messages(
    rows: Iterable[sqlite3.Row],
    attachments: dict[int, list[Attachment]],
    contacts: dict[str, str],
    me_name: str,
) -> list[Message]:
    """Convert rows to Messages. Tapbacks become Reactions on their target (if it's in the batch)."""
    messages: list[Message] = []
    by_guid: dict[str, Message] = {}
    tapbacks: list[tuple[str, Reaction]] = []
    for r in rows:
        if r["item_type"] not in (None, 0):  # group renames, member changes, etc.
            continue
        sender_id = ME_ID if r["is_from_me"] else (r["handle"] or "unknown")
        assoc = r["associated_message_type"] or 0
        if 2000 <= assoc < 3000:
            emoji = TAPBACKS.get(assoc) or r["associated_message_emoji"] or "👍"
            target = (r["associated_message_guid"] or "").split("/")[-1].removeprefix("bp:")
            tapbacks.append((target, Reaction(emoji=emoji, sender_id=sender_id)))
            continue
        if assoc >= 3000:  # tapback removed
            continue
        text = r["text"] or decode_attributed_body(r["attributedBody"]) or ""
        text = text.replace("￼", "").strip()  # attachment placeholder char
        atts = list(attachments.get(r["rowid"], []))
        for url in URL_RE.findall(text):
            atts.append(Attachment(kind=AttachmentKind.link, url=url))
        msg = Message(
            id=r["guid"],
            sender_id=sender_id,
            sender_name=me_name if r["is_from_me"] else contacts.get(sender_id, sender_id),
            is_me=bool(r["is_from_me"]),
            timestamp=apple_time(r["date"]),
            text=text,
            attachments=atts,
            reply_to=r["thread_originator_guid"],
        )
        if msg.is_empty:
            continue
        messages.append(msg)
        by_guid[msg.id] = msg
    for target, reaction in tapbacks:
        if target in by_guid:
            by_guid[target].reactions.append(reaction)
    return messages


def chat_conversation(chat: sqlite3.Row, handles: list[str], contacts: dict[str, str], me_name: str) -> Conversation:
    is_group = chat["style"] == GROUP_STYLE
    participants = [Participant(id=ME_ID, name=me_name, is_me=True)] + [
        Participant(id=h, name=contacts.get(h, h)) for h in handles
    ]
    title = chat["display_name"] or (", ".join(p.name for p in participants[1:]) if is_group else contacts.get(chat["chat_identifier"], chat["chat_identifier"]))
    return Conversation(platform=Platform.imessage, thread_id=chat["guid"], title=title, is_group=is_group, participants=participants)


class IMessageParser:
    def __init__(self, db_path: Path, contacts: dict[str, str], me_name: str):
        self.db_path = db_path
        self.contacts = contacts
        self.me_name = me_name

    def parse(self) -> Iterable[Conversation]:
        conn = connect(self.db_path)
        chats = fetch_chats(conn)
        handles = fetch_chat_handles(conn)
        rows_by_chat: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for r in conn.execute(message_sql(conn)):
            rows_by_chat[r["chat_id"]].append(r)
        all_ids = [r["rowid"] for rows in rows_by_chat.values() for r in rows]
        attachments = fetch_attachments(conn, all_ids)
        for chat_id, rows in rows_by_chat.items():
            if chat_id not in chats:
                continue
            conv = chat_conversation(chats[chat_id], handles.get(chat_id, []), self.contacts, self.me_name)
            conv.messages = rows_to_messages(rows, attachments, self.contacts, self.me_name)
            if conv.messages:
                yield conv
        conn.close()
