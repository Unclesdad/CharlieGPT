import json
from datetime import timedelta

from charliegpt.context import make_context
from charliegpt.dataset.build import build_dataset, examples_for
from charliegpt.ingest.normalize import group_turns, normalize
from charliegpt.prompt import BUBBLE, build_messages, split_bubbles
from charliegpt.schema import Attachment, AttachmentKind

from conftest import msg


def test_turns_and_sessions(dm, settings):
    turns = group_turns(dm.messages, settings.dataset.burst_gap_minutes)
    assert [(t.is_me, len(t.messages)) for t in turns] == [(False, 2), (True, 2), (False, 1), (True, 1)]
    late = msg(9, False, "morning", 60 * 10)
    sessions = normalize([dm.model_copy(update={"messages": dm.messages + [late]})], settings.dataset.session_gap_hours)
    assert [len(s.messages) for s in sessions] == [6, 1]


def test_examples_target_bursts(dm, settings):
    import random

    exs = examples_for(dm, settings, None, None, random.Random(0))
    assert len(exs) == 2
    first = exs[0]["messages"]
    assert first[0]["role"] == "system" and "DM with Alex" in first[0]["content"]
    assert first[1] == {"role": "user", "content": "Alex: yo\nAlex: you up?"}
    assert first[-1] == {"role": "assistant", "content": f"ya\n{BUBBLE}\nwhats good"}
    assert [m["role"] for m in exs[1]["messages"]] == ["system", "user", "assistant"]  # history never in assistant turns
    assert "Charlie: ya\nCharlie: whats good" in exs[1]["messages"][1]["content"]


def test_train_inference_parity(dm, settings):
    """The live engine must build exactly the prompt the model was trained on."""
    import random

    exs = examples_for(dm, settings, None, None, random.Random(0))
    turns = group_turns(dm.messages[:5], settings.dataset.burst_gap_minutes)
    live = build_messages(make_context(dm, turns, settings, now=dm.messages[5].timestamp))
    assert live == exs[1]["messages"][:-1]


def test_group_names_gap_marker_and_media(dm, settings):
    m = dm.messages
    m[4].timestamp = m[3].timestamp + timedelta(hours=3)
    m[4].attachments = [Attachment(kind=AttachmentKind.image, caption="a cat in a box")]
    conv = dm.model_copy(update={"is_group": True, "title": "the boys"})
    turns = group_turns(m[:5], settings.dataset.burst_gap_minutes)
    out = build_messages(make_context(conv, turns, settings, now=m[4].timestamp))
    assert 'group chat "the boys"' in out[0]["content"]
    assert out[1]["content"].startswith("Alex: yo")
    assert out[-1]["content"].endswith('[3 hours later]\nAlex: nm just bored [image: "a cat in a box"]')


def test_split_bubbles_drops_media_placeholders():
    raw = f'[image]{BUBBLE}lmao [sticker: Cat] look{BUBBLE}[shared https://instagram.com/p/x: "@a: hi"]{BUBBLE}(replying to "yo") bet{BUBBLE}[voice memo: "hey"]'
    assert split_bubbles(raw, "Charlie") == ["lmao look", "bet"]
    assert split_bubbles("[image]", "Charlie") == []  # the engine regenerates, then stays quiet


def test_split_bubbles():
    raw = f"<think>\n\n</think>\n\nCharlie: ya\n{BUBBLE}\n  \n{BUBBLE}\nwhats good"
    assert split_bubbles(raw, "Charlie") == ["ya", "whats good"]


def test_build_dataset_files(dm, settings, tmp_path):
    convs = [dm.model_copy(update={"thread_id": f"t{i}"}) for i in range(40)]
    st = build_dataset(convs, settings, tmp_path / "ds")
    assert st.train + st.valid == 80 and st.valid > 0
    row = json.loads((tmp_path / "ds" / "train.jsonl").read_text().splitlines()[0])
    assert row["messages"][-1]["role"] == "assistant"


async def test_profiles_concurrent_and_resumable(tmp_path):
    from datetime import datetime, timedelta, timezone

    from charliegpt.memory.profiles import ProfileStore, build_profiles
    from charliegpt.schema import Conversation, Message, Participant, Platform

    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    convs = [
        Conversation(platform=Platform.discord, thread_id=str(i), title=f"chat{i}", is_group=False,
                     participants=[Participant(id="me", name="Charlie", is_me=True), Participant(id="sam", name="Sam")],
                     messages=[Message(id=f"{i}-{j}", sender_id="sam", sender_name="Sam", is_me=False, timestamp=t0 + timedelta(minutes=j), text="yo") for j in range(3)])
        for i in range(5)
    ]  # fmt: skip
    calls = []

    async def complete(msgs):
        calls.append(msgs)
        return "- friend"

    store = ProfileStore(tmp_path / "profiles.json")
    assert await build_profiles(convs, store, complete, "Charlie", min_messages=1, concurrency=2) == 5
    again = ProfileStore(tmp_path / "profiles.json")  # reloaded from disk
    assert await build_profiles(convs, again, complete, "Charlie", min_messages=1) == 0 and len(calls) == 5


def test_targets_never_teach_media_placeholders(dm, settings):
    import random

    m = dm.messages
    m[2].text, m[2].attachments = "", [Attachment(kind=AttachmentKind.image)]  # first reply: a photo, then "whats good"
    m[5].text, m[5].attachments = "", [Attachment(kind=AttachmentKind.sticker, name="Cat")]  # second reply: sticker only
    exs = examples_for(dm, settings, None, None, random.Random(0))
    assert [e["messages"][-1]["content"] for e in exs] == ["whats good"]
    assert "[image]" not in exs[0]["messages"][-1]["content"]
