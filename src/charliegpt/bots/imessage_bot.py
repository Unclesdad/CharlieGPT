"""iMessage bot (macOS only): polls chat.db for new messages, sends via Messages.app (AppleScript).

Requirements: Full Disk Access for the process (to read chat.db) and Automation permission for
Messages (macOS prompts the first time it sends).
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from collections import defaultdict

from charliegpt.bots.base import BaseBot
from charliegpt.ingest import imessage as im
from charliegpt.schema import Conversation, Message, Platform

SEND_SCRIPT = [
    "-e", "on run argv",
    "-e", 'tell application "Messages" to send (item 1 of argv) to chat id (item 2 of argv)',
    "-e", "end run",
]  # fmt: skip


class IMessageBot(BaseBot):
    platform = Platform.imessage

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sys.platform != "darwin":
            raise RuntimeError("the iMessage bot only runs on macOS")
        self.conn: sqlite3.Connection | None = None
        self._chat_rows: dict[int, sqlite3.Row] = {}
        self._chat_handles: dict[int, list[str]] = {}
        self._chat_by_guid: dict[str, int] = {}

    def _refresh_chats(self) -> None:
        assert self.conn
        self._chat_rows = im.fetch_chats(self.conn)
        self._chat_handles = im.fetch_chat_handles(self.conn)
        self._chat_by_guid = {r["guid"]: rid for rid, r in self._chat_rows.items()}

    def _conv(self, chat_id: int) -> Conversation:
        ident = self.settings.identity
        return im.chat_conversation(self._chat_rows[chat_id], self._chat_handles.get(chat_id, []), ident.contacts, ident.name)

    def _messages(self, where: str, params: tuple) -> dict[int, list[Message]]:
        assert self.conn
        rows = list(self.conn.execute(im.message_sql(self.conn, where), params))
        atts = im.fetch_attachments(self.conn, [r["rowid"] for r in rows])
        by_chat: dict[int, list[sqlite3.Row]] = defaultdict(list)
        for r in rows:
            by_chat[r["chat_id"]].append(r)
        ident = self.settings.identity
        return {cid: im.rows_to_messages(rs, atts, ident.contacts, ident.name) for cid, rs in by_chat.items()}

    async def run(self) -> None:
        self.conn = im.connect(self.settings.imessage.db_path)
        self._refresh_chats()
        last = self.conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message").fetchone()[0]
        self.log.info("watching chat.db from ROWID %s", last)
        while True:
            new_last = self.conn.execute("SELECT COALESCE(MAX(ROWID), 0) FROM message").fetchone()[0]
            if new_last > last:
                batch = await asyncio.to_thread(self._messages, "WHERE m.ROWID > ?", (last,))
                last = new_last
                if any(cid not in self._chat_rows for cid in batch):
                    self._refresh_chats()
                for cid, msgs in batch.items():
                    if cid not in self._chat_rows:
                        continue
                    conv = self._conv(cid)
                    for msg in msgs:
                        await self.handle(conv, msg)
            await asyncio.sleep(self.settings.imessage.poll_seconds)

    async def fetch_history(self, conv: Conversation, before: Message) -> list[Message]:
        cid = self._chat_by_guid.get(conv.thread_id)
        if cid is None:
            return []
        where = "WHERE m.ROWID IN (SELECT message_id FROM chat_message_join WHERE chat_id = ? ORDER BY message_id DESC LIMIT ?)"
        msgs = (await asyncio.to_thread(self._messages, where, (cid, self.cfg.history_limit))).get(cid, [])
        return [m for m in msgs if m.id != before.id]

    async def send(self, thread_id: str, text: str) -> None:
        proc = await asyncio.create_subprocess_exec("osascript", *SEND_SCRIPT, text, thread_id, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"osascript failed: {err.decode().strip()}")
