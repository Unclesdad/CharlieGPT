import json
import sqlite3
from pathlib import Path

from charliegpt.ingest.discord import DiscordParser
from charliegpt.ingest.imessage import IMessageParser, decode_attributed_body
from charliegpt.ingest.instagram import InstagramParser
from charliegpt.schema import AttachmentKind

NS = 1_000_000_000


def make_chat_db(path):
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, guid TEXT, chat_identifier TEXT, display_name TEXT, style INTEGER);
        CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
        CREATE TABLE message (ROWID INTEGER PRIMARY KEY, guid TEXT, text TEXT, attributedBody BLOB, is_from_me INTEGER,
            date INTEGER, handle_id INTEGER, associated_message_type INTEGER, associated_message_guid TEXT,
            thread_originator_guid TEXT, item_type INTEGER);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        CREATE TABLE attachment (ROWID INTEGER PRIMARY KEY, filename TEXT, mime_type TEXT, transfer_name TEXT);
        CREATE TABLE message_attachment_join (message_id INTEGER, attachment_id INTEGER);
        INSERT INTO handle VALUES (1, '+15550001111');
        INSERT INTO chat VALUES (1, 'iMessage;-;+15550001111', '+15550001111', NULL, 45);
        INSERT INTO chat_handle_join VALUES (1, 1);
        """
    )
    # A modern message has NULL text and the content in attributedBody.
    body = b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84\x12NSAttributedString\x00\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84\x08NSString\x01\x94\x84\x01+\x0bhello there\x86"
    rows = [
        (1, "g1", None, body, 0, 100 * NS, 1, 0, None, None, 0),
        (2, "g2", "lol ok", None, 1, 160 * NS, 0, 0, None, None, 0),
        (3, "g3", "Loved “hello there”", None, 1, 170 * NS, 0, 2000, "p:0/g1", None, 0),
        (4, "g4", "￼", None, 0, 200 * NS, 1, 0, None, "g2", 0),
        (5, "g5", "Alex named the conversation", None, 0, 210 * NS, 1, 0, None, None, 2),
    ]
    db.executemany("INSERT INTO message VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    db.executemany("INSERT INTO chat_message_join VALUES (1, ?)", [(r[0],) for r in rows])
    db.execute("INSERT INTO attachment VALUES (1, '~/Library/Messages/Attachments/x/IMG_1.HEIC', 'image/heic', 'IMG_1.HEIC')")
    db.execute("INSERT INTO message_attachment_join VALUES (4, 1)")
    db.commit()
    db.close()


def test_attributed_body():
    blob = b"xxNSString\x01\x94\x84\x01+\x05hello\x86"
    assert decode_attributed_body(blob) == "hello"
    assert decode_attributed_body(None) is None


def test_imessage(tmp_path):
    make_chat_db(tmp_path / "chat.db")
    [conv] = list(IMessageParser(tmp_path / "chat.db", {"+15550001111": "Alex"}, "Charlie").parse())
    assert not conv.is_group and conv.title == "Alex"
    texts = [m.text for m in conv.messages]
    assert texts == ["hello there", "lol ok", ""]  # tapback + group event removed
    first, _, photo = conv.messages
    assert first.sender_name == "Alex" and first.reactions[0].emoji == "❤️"
    assert photo.attachments[0].kind == AttachmentKind.image and photo.reply_to == "g2"


def test_discord(tmp_path):
    export = {
        "guild": {"id": "0", "name": "Direct Messages"},
        "channel": {"id": "42", "type": "DirectTextChat", "name": "sam"},
        "messages": [
            {"id": "1", "type": "Default", "timestamp": "2026-03-01T20:00:00+00:00", "content": "check this https://youtu.be/x",
             "author": {"id": "7", "name": "sam", "nickname": "Sam"}, "attachments": [], "reactions": []},
            {"id": "2", "type": "Default", "timestamp": "2026-03-01T20:01:00+00:00", "content": "bro 💀",
             "author": {"id": "99", "name": "charlie"}, "attachments": [{"id": "a", "url": "https://cdn/x.png", "fileName": "x.png"}],
             "reactions": [{"emoji": {"name": "😂"}, "count": 1, "users": [{"id": "7"}]}], "reference": {"messageId": "1"}},
            {"id": "3", "type": "ChannelPinnedMessage", "timestamp": "2026-03-01T20:02:00+00:00", "content": "", "author": {"id": "7"}},
        ],
    }  # fmt: skip
    (tmp_path / "dm.json").write_text(json.dumps(export))
    [conv] = list(DiscordParser(tmp_path, ["99"], "Charlie").parse())
    assert not conv.is_group and len(conv.messages) == 2
    them, me = conv.messages
    assert them.attachments[0].kind == AttachmentKind.link
    assert me.is_me and me.sender_name == "Charlie" and me.reply_to == "1"
    assert me.attachments[0].kind == AttachmentKind.image and me.reactions[0].emoji == "😂"


def test_instagram_mojibake_and_system_text(tmp_path):
    thread = tmp_path / "your_instagram_activity" / "messages" / "inbox" / "jordan_123"
    thread.mkdir(parents=True)
    laugh = "😂".encode("utf-8").decode("latin-1")
    data = {
        "participants": [{"name": "Jordan"}, {"name": "charlie"}],
        "title": "Jordan",
        "messages": [
            {"sender_name": "charlie", "timestamp_ms": 2000, "content": f"no way {laugh}"},
            {"sender_name": "Jordan", "timestamp_ms": 1000, "content": "Liked a message"},
            {"sender_name": "Jordan", "timestamp_ms": 1500, "content": "Jordan sent an attachment.", "share": {"link": "https://instagram.com/reel/1"}},
        ],
    }
    (thread / "message_1.json").write_text(json.dumps(data))
    [conv] = list(InstagramParser(tmp_path, "charlie", "Charlie").parse())
    assert [m.text for m in conv.messages] == ["", "no way 😂"]
    assert conv.messages[0].attachments[0].url == "https://instagram.com/reel/1"
    assert conv.messages[1].is_me and conv.messages[1].sender_name == "Charlie"


def test_imessage_bot_queries(tmp_path, settings):
    import asyncio
    import sys

    import pytest

    if sys.platform != "darwin":
        pytest.skip("macOS only")
    from charliegpt.bots.imessage_bot import IMessageBot
    from charliegpt.ingest import imessage as im

    make_chat_db(tmp_path / "chat.db")
    settings.imessage.db_path = tmp_path / "chat.db"
    bot = IMessageBot(settings, engine=None)
    bot.conn = im.connect(settings.imessage.db_path)
    bot._refresh_chats()
    new = bot._messages("WHERE m.ROWID > ?", (1,))
    assert [m.text for m in new[1]] == ["lol ok", ""]
    conv = bot._conv(1)
    hist = asyncio.run(bot.fetch_history(conv, new[1][-1]))
    assert [m.id for m in hist] == ["g1", "g2"]


def test_instagram_flat_layout_media_and_system_text(tmp_path):
    """Thread folders copied straight into the root; uris still carry the original export prefix."""
    thread = tmp_path / "alex_123"
    (thread / "photos").mkdir(parents=True)
    (thread / "photos" / "999").write_bytes(b"\xff\xd8\xff")  # no extension, like the real export
    data = {
        "participants": [{"name": "Alex K"}, {"name": "Sam"}, {"name": "charlie"}],
        "title": "the group",
        "messages": [
            {"sender_name": "Alex K", "timestamp_ms": 5000, "content": "Reacted ð\u009f\u0098\u0082 to your message "},
            {"sender_name": "Sam", "timestamp_ms": 4900, "content": "Sam liked a message"},
            {"sender_name": "Sam", "timestamp_ms": 4800, "content": "Alex K added Sam to the group."},
            {"sender_name": "Sam", "timestamp_ms": 4700, "content": "You missed an audio call", "call_duration": 0},
            {"sender_name": "Alex K", "timestamp_ms": 4000, "content": "Alex sent an attachment.",
             "share": {"link": "https://www.instagram.com/p/x/", "share_text": "Stanford #1 ð\u009f\u008e\u0093", "original_content_owner": "alwayscollege"}},
            {"sender_name": "charlie", "timestamp_ms": 3000, "photos": [{"uri": "your_instagram_activity/messages/inbox/alex_123/photos/999"}]},
            {"sender_name": "charlie", "timestamp_ms": 2000, "content": "I joined the group chat late lol"},
        ],
    }  # fmt: skip
    (thread / "message_1.json").write_text(json.dumps(data))
    [conv] = list(InstagramParser(tmp_path, "charlie", "Charlie").parse())
    assert conv.is_group and len(conv.messages) == 3
    real, photo, share = conv.messages
    assert real.text == "I joined the group chat late lol"  # not mistaken for a system notice
    assert photo.attachments[0].path == str(thread / "photos" / "999")
    assert share.render() == '[shared https://www.instagram.com/p/x/: "@alwayscollege: Stanford #1 🎓"]'


def test_caption_path_map(tmp_path):
    from charliegpt.ingest.caption import resolve

    m = {"/Users/me/Library/Messages/Attachments": str(tmp_path / "att")}
    assert resolve("/Users/me/Library/Messages/Attachments/ab/0/IMG_1.heic", m) == tmp_path / "att/ab/0/IMG_1.heic"
    assert resolve("data/instagram/x/photos/1", m) == Path("data/instagram/x/photos/1")
