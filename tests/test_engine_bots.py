import asyncio

from charliegpt.bots.base import BaseBot
from charliegpt.infer.engine import ReplyEngine
from charliegpt.infer.guardrails import KillSwitch, might_be_commitment
from charliegpt.prompt import BUBBLE
from charliegpt.schema import Platform

from conftest import msg


class FakeClient:
    def __init__(self, persona=f"ya\n{BUBBLE}\nlol", commitment="NO", excuse="cant, my pet rock is getting married"):
        self.persona_out, self.commitment, self.excuse = persona, commitment, excuse
        self.calls = []

    async def persona(self, messages):
        self.calls.append("persona")
        return self.persona_out if isinstance(self.persona_out, str) else self.persona_out.pop(0)

    async def helper(self, messages, temperature=0.2, max_tokens=300):
        prompt = messages[-1]["content"]
        if "Answer with exactly YES or NO" in prompt:
            self.calls.append("classify")
            return self.commitment
        self.calls.append("excuse")
        return self.excuse


def engine(settings, client):
    return ReplyEngine(settings, client)


async def test_engine_normal_reply(dm, settings):
    dm.messages = dm.messages[:5]
    c = FakeClient()
    assert await engine(settings, c).reply(dm) == ["ya", "lol"]
    assert c.calls == ["persona"]  # "nm just bored" doesn't look like a commitment


async def test_engine_absurd_excuse(dm, settings):
    dm.messages = dm.messages[:4] + [msg(5, False, "wanna come over tonight?", 2)]
    c = FakeClient(commitment="YES")
    assert await engine(settings, c).reply(dm) == ["cant, my pet rock is getting married"]
    assert "excuse" in c.calls


async def test_engine_stays_in_character(dm, settings):
    dm.messages = dm.messages[:5]
    c = FakeClient(persona=["As an AI language model I can't", "nah im good"])
    assert await engine(settings, c).reply(dm) == ["nah im good"]


async def test_engine_quiet_when_last_is_me(dm, settings):
    assert await engine(settings, FakeClient()).reply(dm) == []


def test_commitment_prefilter():
    assert might_be_commitment("u free tmrw?")
    assert might_be_commitment("can you venmo me $20")
    assert not might_be_commitment("lmao that's crazy")


def test_kill_switch(tmp_path):
    k = KillSwitch(tmp_path)
    assert not k.paused
    k.pause("test")
    assert k.paused
    k.resume()
    assert not k.paused


class FakeBot(BaseBot):
    platform = Platform.imessage

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.sent = []

    async def run(self):
        pass

    async def send(self, thread_id, text):
        self.sent.append(text)


async def _settle(bot):
    for st in bot.chats.values():
        if st.pending:
            await asyncio.wait_for(st.pending, 1)


async def test_bot_replies_to_dm_and_debounces(dm, settings):
    bot = FakeBot(settings, engine(settings, FakeClient()))
    meta = dm.model_copy(update={"messages": []})
    await bot.handle(meta, msg(1, False, "yo", 0))
    await bot.handle(meta, msg(2, False, "you up?", 0.1))
    await _settle(bot)
    assert bot.sent == ["ya", "lol"]  # one reply for the whole burst


async def test_bot_group_needs_mention(dm, settings):
    bot = FakeBot(settings, engine(settings, FakeClient()))
    meta = dm.model_copy(update={"messages": [], "is_group": True, "title": "gc"})
    await bot.handle(meta, msg(1, False, "anyone up", 0))
    await _settle(bot)
    assert bot.sent == []
    await bot.handle(meta, msg(2, False, "charlie you there", 1))
    await _settle(bot)
    assert bot.sent == ["ya", "lol"]


async def test_bot_human_takeover_and_kill_switch(dm, settings):
    bot = FakeBot(settings, engine(settings, FakeClient()))
    meta = dm.model_copy(update={"messages": []})
    await bot.handle(meta, msg(1, True, "typed this myself", 0))  # not an echo of the bot
    await bot.handle(meta, msg(2, False, "oh hey", 0.1))
    await _settle(bot)
    assert bot.sent == []

    bot2 = FakeBot(settings, engine(settings, FakeClient()))
    bot2.kill.pause()
    await bot2.handle(meta, msg(3, False, "hello?", 0))
    await _settle(bot2)
    assert bot2.sent == []
