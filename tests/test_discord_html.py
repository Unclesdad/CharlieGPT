from datetime import timezone

from charliegpt.ingest.discord import DiscordParser
from charliegpt.ingest.discord_html import snowflake_time
from charliegpt.schema import AttachmentKind

ME, SAM, BOT = "111111111111111111", "222222222222222222", "333333333333333333"


def header(uid, username, display, bot=False):
    tag = "<span class=chatlog__author-tag>APP</span>" if bot else ""
    return (f'<div class=chatlog__header><span class=chatlog__author title={username} data-user-id={uid}>{display}</span>{tag}'
            f'<span class=chatlog__timestamp title="Friday, March 1, 2024 8:00 PM"><a href=#x>3/1/2024 8:00 PM</a></span></div>')  # fmt: skip


def container(mid, inner, head=""):
    return (f"<div id=chatlog__message-container-{mid} class=chatlog__message-container data-message-id={mid}>"
            f"<div class=chatlog__message><div class=chatlog__message-aside></div><div class=chatlog__message-primary>{head}{inner}</div></div></div>")  # fmt: skip


def content(text):
    return f"<div class=chatlog__content chatlog__markdown><span class=chatlog__markdown-preserve>{text}</span></div>"


def page(groups, guild="Study Hall", channel="Social / social"):
    pre = f"<div class=preamble><div class=preamble__entries-container><div class=preamble__entry>{guild}</div><div class=preamble__entry>{channel}</div></div></div>"
    body = "".join(f"<div class=chatlog__message-group>{g}</div>" for g in groups)
    return f"<html><body>{pre}<div class=chatlog>{body}</div></body></html>"


def test_discord_html(tmp_path):
    groups = [
        container("1213700000000000001", content('yo <img class="chatlog__emoji chatlog__emoji--small" alt=":pepe:" src=x> who\'s on') , header(SAM, "sam99", "Sam"))
        + container("1213700000000000002", content("anyone?")),
        container(
            "1213700000000000003",
            '<div class=chatlog__reply><div class=chatlog__reply-content><span class=chatlog__reply-link onclick="scrollToMessage(event, \'1213700000000000002\')">anyone?</span></div></div>'
            + content('me lol <span class=chatlog__edited-timestamp title=x>(edited)</span>')
            + '<div class=chatlog__reactions><div class=chatlog__reaction title=skull><img class="chatlog__emoji chatlog__emoji--small" alt="💀" src=x><span class=chatlog__reaction-count>2</span></div></div>',
            header(ME, "charlie_x", "charlie"),
        )
        + container(
            "1213700000000000004",
            '<div class=chatlog__attachment><a href="https://cdn.discordapp.com/attachments/1/2/cat.png?ex=1"><img class=chatlog__attachment-media src=x title="Image: cat.png (1.2 MB)"></a></div>'
            + '<div class=chatlog__embed><video class=chatlog__embed-generic-gifv><source src="https://media.tenor.com/x.mp4"></video></div>',
        ),
        container("1213700000000000005", content("Reminder: practice at 4"), header(BOT, "carl", "Carl-bot", bot=True)),
        container(
            "1213700000000000006",
            content("watch https://youtu.be/abc")
            + '<div class=chatlog__embed><div class=chatlog__embed-content-container><a class=chatlog__embed-title-link href="https://youtu.be/abc"><div class=chatlog__embed-title>Funny Cats</div></a></div></div>',
            header(SAM, "sam99", "Sam"),
        ),
    ]  # fmt: skip
    (tmp_path / "Study Hall - Social - social [1100000000000000001].html").write_text(page(groups))
    quiet = [container("1213700000000000007", content("hi"), header(SAM, "sam99", "Sam"))]
    (tmp_path / "Study Hall - Archives - resources [1100000000000000002].html").write_text(page(quiet, channel="Archives / resources"))

    parser = DiscordParser(tmp_path, [ME], "Charlie")
    [conv] = list(parser.parse())
    assert parser.skipped == ["Study Hall #resources"]  # never posted there
    assert conv.thread_id == "1100000000000000001" and conv.title == "Study Hall #social" and conv.is_group
    by_id = {m.id: m for m in conv.messages}
    assert "1213700000000000005" not in by_id  # bot
    first, follow = by_id["1213700000000000001"], by_id["1213700000000000002"]
    assert first.text == "yo :pepe: who's on" and follow.sender_name == "Sam"  # follow-up inherits author
    mine = by_id["1213700000000000003"]
    assert mine.is_me and mine.sender_name == "Charlie" and mine.text == "me lol"
    assert mine.reply_to == "1213700000000000002" and [r.emoji for r in mine.reactions] == ["💀", "💀"]
    cat, gif = by_id["1213700000000000004"].attachments
    assert by_id["1213700000000000004"].is_me and cat.kind == AttachmentKind.image and cat.name == "cat.png"
    assert gif.kind == AttachmentKind.gif
    assert by_id["1213700000000000006"].render() == 'watch https://youtu.be/abc [shared https://youtu.be/abc: "Funny Cats"]'


def test_username_identifies_me_and_snowflake(tmp_path):
    g = [container("1213700000000000003", content("sup"), header(ME, "charlie_x", "c"))]
    (tmp_path / "Direct Messages - Private - sam [99].html").write_text(page(g, guild="Direct Messages", channel="Private / sam"))
    [conv] = list(DiscordParser(tmp_path, ["charlie_x"], "Charlie").parse())
    assert conv.messages[0].is_me and not conv.is_group and conv.title == "sam"
    t = snowflake_time("175928847299117063")  # example id from Discord's docs
    assert t.tzinfo == timezone.utc and (t.year, t.month, t.day) == (2016, 4, 30)
