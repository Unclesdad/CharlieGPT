from charliegpt.ingest.imessage_html import IMessageHTMLParser
from charliegpt.ingest.vcard import normalize_handle, parse_vcards
from charliegpt.schema import AttachmentKind


def _msg(guid, ts, sender, inner, sent=False, extra=""):
    cls = "sent iMessage" if sent else "received"
    return f"""<div class="message">
    <div class="{cls}">
        <p>
            <span class="timestamp"><a title="Reveal in Messages app" href="sms://open?message-guid={guid}">{ts}</a>
                (Read by you after 6 seconds)</span>
            <span class="sender">{sender}</span>
        </p>
        <hr>
        {inner}
        {extra}
    </div>
</div>"""


def part(html):
    return f'<div class="message_part">{html}</div>'


def test_html_export(tmp_path):
    reply = _msg("R1", "Nov 07, 2025  9:10:00 AM", "Me", part('<span class="bubble">replying in thread</span>'), sent=True)
    body = "\n".join(
        [
            _msg("A1", "Nov 07, 2025  6:02:52 AM", "Me", part('<span class="bubble">Free <u>tomorrow afternoon</u>?\nthanks</span>'), sent=True),
            _msg(
                "B1",
                "Nov 07, 2025  9:07:39 AM",
                "+12016630066",
                part('<span class="bubble">yes!</span>'),
                extra=f"""<div class="tapbacks"><hr><p>Tapbacks:</p>
                    <div class="tapback"><span class="tapback"><b>Loved</b> by Me</span></div>
                    <div class="tapback"><span class="tapback"><b>💀</b> by +12016630066</span></div></div>
                    <div class="replies"><div class="reply" id="r">{reply}</div></div>""",
            ),
            reply,  # the timeline copy of the threaded reply
            _msg("C1", "Nov 07, 2025 12:04:00 PM", "Me", part('<div class="attachment"><img src="/Users/x/Library/Messages/Attachments/a/IMG_1.HEIC" loading="lazy"></div>'), sent=True),
            _msg(
                "D1",
                "Nov 07, 2025 12:05:00 PM",
                "+12016630066",
                part('<div class="app"><a href="https://youtu.be/abc"><div class="app_header"><div class="name">YouTube</div></div>'
                     '<div class="app_footer"><div class="caption">Funny video</div></div></a></div>'),
            ),  # fmt: skip
            _msg(
                "E1",
                "Nov 07, 2025 12:06:00 PM",
                "Me",
                part('<div class="edited"><table><tbody><tr><td><span class="timestamp"></span></td><td>frist</td></tr></tbody>'
                     '<tfoot><tr><td><span class="timestamp">Edited 9 seconds later</span></td><td>first</td></tr></tfoot></table></div>'),
                sent=True,
            ),  # fmt: skip
            _msg("F1", "Nov 07, 2025 12:07:00 PM", "Me", part('<span class="unsent">You unsent a message!</span>'), sent=True),
            '<div class="announcement"><p><span class="timestamp">Nov 07, 2025 12:08:00 PM</span> You named the conversation.</p></div>',
        ]
    )
    (tmp_path / "+12016630066.html").write_text(f"<html><head><style>.x{{}}</style></head><body>{body}</body></html>")
    [conv] = list(IMessageHTMLParser(tmp_path, {"+12016630066": "Dana"}, "Charlie").parse())

    assert conv.title == "Dana" and not conv.is_group
    by_id = {m.id: m for m in conv.messages}
    assert set(by_id) == {"A1", "B1", "R1", "C1", "D1", "E1"}  # unsent + announcement dropped
    assert by_id["A1"].is_me and by_id["A1"].text == "Free tomorrow afternoon?\nthanks"
    assert by_id["A1"].timestamp.hour == 6 and by_id["A1"].timestamp.tzinfo is not None
    b = by_id["B1"]
    assert b.sender_id == "+12016630066" and b.sender_name == "Dana"
    assert [(r.emoji, r.sender_id) for r in b.reactions] == [("❤️", "me"), ("💀", "+12016630066")]
    assert by_id["R1"].reply_to == "B1" and by_id["R1"].is_me
    img = by_id["C1"].attachments[0]
    assert img.kind == AttachmentKind.image and img.path.endswith("IMG_1.HEIC")
    assert by_id["D1"].attachments[0].url == "https://youtu.be/abc"
    assert by_id["E1"].text == "first"


def test_group_file_names(tmp_path):
    inner = part('<span class="bubble">hi</span>')
    html = "<html><body>" + _msg("G1", "Jan 01, 2026  1:00:00 PM", "Sam Lee", inner) + "</body></html>"
    (tmp_path / "+15550000001, +15550000002.html").write_text(html)
    (tmp_path / "🌲 Trip Crew - 123.html").write_text(html.replace("G1", "G2"))
    convs = {c.thread_id: c for c in IMessageHTMLParser(tmp_path, {"+15550000001": "Ann"}, "Charlie").parse()}
    assert convs["+15550000001, +15550000002"].is_group
    assert convs["+15550000001, +15550000002"].title == "Ann, +15550000002"
    assert convs["🌲 Trip Crew - 123"].title == "🌲 Trip Crew" and convs["🌲 Trip Crew - 123"].is_group
    assert convs["🌲 Trip Crew - 123"].messages[0].sender_name == "Sam Lee"


def test_vcard(tmp_path):
    vcf = """BEGIN:VCARD
VERSION:3.0
PRODID:-//Apple Inc.//macOS 15.0//EN
N:Doe;Jane;;;
FN:Jane Doe
TEL;type=CELL;type=VOICE;type=pref:(201) 663-0066
item1.EMAIL;type=INTERNET;type=pref:Jane@Example.com
item1.X-ABLabel:_$!<Other>!$_
PHOTO;ENCODING=b;TYPE=JPEG:/9j/4AAQSkZJRgABAQAAAQABAAD
 /2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIx
END:VCARD
BEGIN:VCARD
VERSION:3.0
N:;;;;
FN:
ORG:Pizza\\, Inc.;
TEL;type=pref:+44 20 7946 0958
END:VCARD
BEGIN:VCARD
VERSION:3.0
N:Smith;Bob;;;
TEL;type=pref:1 (212) 555-0100
END:VCARD
"""
    (tmp_path / "c.vcf").write_text(vcf)
    contacts = parse_vcards(tmp_path / "c.vcf")
    assert contacts == {
        "+12016630066": "Jane Doe",
        "jane@example.com": "Jane Doe",
        "+442079460958": "Pizza, Inc.",
        "+12125550100": "Bob Smith",
    }
    assert normalize_handle("+1 (201) 663-0066") == normalize_handle("2016630066") == "+12016630066"


def test_dm_labeled_by_name_is_not_a_group(tmp_path):
    inner = part('<span class="bubble">hi</span>')
    html = "<html><body>" + _msg("H1", "Jan 01, 2026  1:00:00 PM", "Sam Lee", inner) + "</body></html>"
    (tmp_path / "+15550000003.html").write_text(html)
    [conv] = list(IMessageHTMLParser(tmp_path, {}, "Charlie").parse())
    assert not conv.is_group and conv.title == "Sam Lee"
    assert conv.messages[0].sender_id == "+15550000003" and conv.messages[0].sender_name == "Sam Lee"
