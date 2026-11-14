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
    assert first[1] == {"role": "user", "content": "yo\nyou up?"}
    assert first[-1] == {"role": "assistant", "content": f"ya\n{BUBBLE}\nwhats good"}


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
    assert out[-1]["content"] == '[3 hours later]\nAlex: nm just bored [image: "a cat in a box"]'


def test_split_bubbles():
    raw = f"<think>\n\n</think>\n\nCharlie: ya\n{BUBBLE}\n  \n{BUBBLE}\nwhats good"
    assert split_bubbles(raw, "Charlie") == ["ya", "whats good"]


def test_build_dataset_files(dm, settings, tmp_path):
    convs = [dm.model_copy(update={"thread_id": f"t{i}"}) for i in range(40)]
    st = build_dataset(convs, settings, tmp_path / "ds")
    assert st.train + st.valid == 80 and st.valid > 0
    row = json.loads((tmp_path / "ds" / "train.jsonl").read_text().splitlines()[0])
    assert row["messages"][-1]["role"] == "assistant"
