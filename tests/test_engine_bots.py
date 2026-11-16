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
    assert might_be_commitment("hop on the game tn") and might_be_commitment("u on later?")
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


async def test_excuse_drops_echoed_style_and_list_bullets():
    from charliegpt.config import GuardrailsCfg
    from charliegpt.infer.guardrails import absurd_excuse

    cfg = GuardrailsCfg(excuse_random_words=0)
    FALLBACK_EXCUSES = cfg.fallback_excuses

    class Echo:
        def __init__(self, out):
            self.out = out

        async def helper(self, msgs, **kw):
            return self.out

    style = ["I was just looking at my watch", f"Yep{BUBBLE}Tomorrow I wanna leave at 10"]
    out = f"I was just looking at my watch\n- Yep\n- cant my goldfish is running for mayor{BUBBLE}lol"
    assert await absurd_excuse(Echo(out), "Charlie", "gym tmrw?", style, cfg) == ["cant my goldfish is running for mayor", "lol"]
    assert (await absurd_excuse(Echo("Yep"), "Charlie", "gym tmrw?", style, cfg))[0] in FALLBACK_EXCUSES  # only echoes left


def test_leaks_and_casual_excuses():
    from charliegpt.infer.guardrails import AI_LEAK, _casual

    for leak in ["It’s because I’m not an evil AI", "I am more than my code. I am not just a chatbot", "im a bot lol"]:
        assert AI_LEAK.search(leak), leak
    for fine in ["the robot code is on GitHub", "ai generated slop in my feed", "the bot in that server is broken", "said hi"]:
        assert not AI_LEAK.search(fine), fine
    assert _casual("I’m sorry, but I’ve been hired to train llamas.") == "I’ve been hired to train llamas"


async def test_engine_deflects_ai_question_when_model_keeps_leaking(dm, settings):
    DEFLECTIONS = settings.guardrails.deflections
    c = FakeClient(persona="I am more than my code. I am a chatbot")
    conv = dm.model_copy(update={"messages": dm.messages + [msg(99, False, "are you ai", 60)]})
    out = await engine(settings, c).reply(conv)
    assert len(out) == 1 and out[0] in DEFLECTIONS


def test_random_words_are_uniform_nouns_and_verbs():
    from charliegpt.infer.guardrails import _words, random_words

    words = _words()
    assert len(words) > 40000 and len(set(words)) == len(words)  # deduped, so each word is equally likely
    assert {"dog", "run", "lighthouse"} <= set(words) and all(w.isalpha() and w.islower() for w in words[:1000])
    assert len(random_words(2)) == 2 and random_words(0) == []


async def test_excuse_prompt_uses_random_words_and_retries(settings):
    from charliegpt.infer import guardrails

    class Seq:
        def __init__(self, outs):
            self.outs, self.prompts = list(outs), []

        async def helper(self, msgs, **kw):
            self.prompts.append(msgs[0]["content"])
            return self.outs.pop(0)

    guardrails.random_words, orig = (lambda n: ["walrus", "embezzle"][:n]), guardrails.random_words
    try:
        c = Seq(["cant my horse needs me", "cant a walrus is embezzling my savings"])
        out = await guardrails.absurd_excuse(c, "Charlie", "gym tmrw?", [], settings.guardrails)
        assert out == ["cant a walrus is embezzling my savings"] and len(c.prompts) == 2  # first ignored the words
        assert '"walrus", "embezzle"' in c.prompts[0]
    finally:
        guardrails.random_words = orig


async def test_toggles_off(dm, settings):
    s = settings.model_copy(deep=True)
    s.guardrails.ai_leak_filter = False
    s.guardrails.commitment_excuses = False
    c = FakeClient(persona="im a chatbot lol", commitment="YES")
    conv = dm.model_copy(update={"messages": dm.messages + [msg(99, False, "wanna hang tomorrow", 60)]})
    assert await engine(s, c).reply(conv) == ["im a chatbot lol"] and "excuse" not in c.calls


async def test_y_to_are_you_ai_is_a_leak(dm, settings):
    c = FakeClient(persona="y")
    conv = dm.model_copy(update={"messages": dm.messages + [msg(99, False, "are you an ai", 60)]})
    out = await engine(settings, c).reply(conv)
    assert out[0] in settings.guardrails.deflections


async def test_no_repeat_loops(dm, settings):
    c = FakeClient(persona="y")
    conv = dm.model_copy(update={"messages": dm.messages + [msg(98, True, "y", 59), msg(99, False, "hello", 60)]})
    assert await engine(settings, c).reply(conv) == []  # "y" again three times -> stay quiet
    s = settings.model_copy(deep=True)
    s.guardrails.no_repeats = False
    assert await engine(s, c).reply(conv) == ["y"]
