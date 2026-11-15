from datetime import datetime, timedelta, timezone

import pytest

from charliegpt.config import Settings
from charliegpt.schema import Conversation, Message, Participant, Platform

T0 = datetime(2026, 3, 1, 20, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    for name in ("processed", "dataset", "memory", "models", "state", "media"):
        setattr(s.paths, name, tmp_path / name)
    s.paths.captions = tmp_path / "memory" / "captions.json"
    s.bots.debounce_seconds = 0.01
    s.bots.min_typing_delay = 0
    s.bots.max_typing_delay = 0
    s.bots.typing_jitter = 0
    s.bots.typing_chars_per_second = 1e9
    s.rag.enabled = False
    s.vision.backend = "none"
    return s


def msg(i: int, me: bool, text: str, minutes: float, name: str = "Alex") -> Message:
    return Message(
        id=f"m{i}",
        sender_id="me" if me else "alex",
        sender_name="Charlie" if me else name,
        is_me=me,
        timestamp=T0 + timedelta(minutes=minutes),
        text=text,
    )


@pytest.fixture
def dm() -> Conversation:
    return Conversation(
        platform=Platform.imessage,
        thread_id="chat-alex",
        title="Alex",
        participants=[Participant(id="me", name="Charlie", is_me=True), Participant(id="alex", name="Alex")],
        messages=[
            msg(1, False, "yo", 0),
            msg(2, False, "you up?", 0.5),
            msg(3, True, "ya", 1),
            msg(4, True, "whats good", 1.2),
            msg(5, False, "nm just bored", 2),
            msg(6, True, "same lol", 3),
        ],
    )
