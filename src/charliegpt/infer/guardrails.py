"""Kill switch, commitment dodging (absurd excuses), and in-character checks."""

from __future__ import annotations

import functools
import gzip
import random
import re
from datetime import datetime
from pathlib import Path

from charliegpt.config import GuardrailsCfg
from charliegpt.prompt import BUBBLE, split_bubbles

# Cheap prefilter so the classifier only runs when a message might be asking for something.
COMMITMENT_HINT = re.compile(
    r"\b(wanna|want to|down to|down for|dtf|can you|could you|will you|would you|are you free|r u free|"
    r"free (tonight|tomorrow|today|this)|come (over|thru|through|to|with)|hang( out)?|meet( up)?|pull up|"
    r"tonight|tomorrow|tmrw|tmr|this weekend|next week|lend|borrow|venmo|zelle|cash ?app|pay|owe|"
    r"promise|help me|pick (me )?up|give me a ride|drive me|join us|you coming|u coming|let'?s go|lets)\b|\$\s?\d+",
    re.I,
)

CLASSIFY_PROMPT = """Message(s) someone just sent to {me}:
\"\"\"
{text}
\"\"\"
Are they asking {me} to commit to something: make plans, meet up, come somewhere, do a favor, \
lend or pay money, or promise anything? Answer with exactly YES or NO."""

EXCUSE_PROMPT = """{me} just got this text:
\"\"\"
{text}
\"\"\"
Write {me}'s reply. {me} won't commit, so the reply declines with one absurd, over-the-top, \
obviously made-up excuse, totally deadpan. The vibe (invent a new one, never reuse these): {examples}.{words}
Keep it to one short casual text, under 20 words: no apology, no "I'm sorry, but", no period at the end, \
no explanation after the excuse. It should read like a quick text, e.g. "cant im ...". \
Swearing is fine if it fits how {me} texts.

For tone only (don't repeat these), here are some real texts {me} has sent:
\"\"\"
{style}
\"\"\"

Reply with only the text {me} sends, nothing else."""

WORDS_HINT = "\nThe excuse must be built around these words: {words}. Use each one (any form of the word is fine)."
WORDS_FILE = Path(__file__).with_name("words.txt.gz")  # English nouns + verbs from WordNet, offensive senses removed

# Anything that sounds like a bot talking about being a bot. Broad on purpose: a dropped reply just gets regenerated.
AI_LEAK = re.compile(
    r"\b(as an ai|an? (\w+ )?ai\b|not an ai|i['’]?m (just |only )?(an? )?(ai|bot|robot|chat ?bot|program|assistant)\b|"
    r"i am (just |only )?(an? )?(ai|bot|robot|chat ?bot|program|assistant)\b|chat ?bot|language model|llm\b|"
    r"artificial intelligence|my (code|programming|training data|creators?)|more than (my|just) code|"
    r"projection of you|openai|anthropic|qwen|chat ?gpt|\bgpt\b)",
    re.I,
)
# Them asking whether they're talking to a bot.
ASKS_IF_AI = re.compile(
    r"\b((are|r) (you|u) (an? )?(ai|bot|robot|real|human|chat ?gpt|gpt)|is this (an? )?(ai|bot|real)|"
    r"(ai|bot|chat ?gpt) (is )?(texting|replying|answering)|chat ?gpt)\b",
    re.I,
)

class KillSwitch:
    def __init__(self, state_dir: Path):
        self.flag = Path(state_dir) / "PAUSED"

    @property
    def paused(self) -> bool:
        return self.flag.exists()

    def pause(self, reason: str = "") -> None:
        self.flag.parent.mkdir(parents=True, exist_ok=True)
        self.flag.write_text(f"{datetime.now().isoformat()} {reason}".strip())

    def resume(self) -> None:
        self.flag.unlink(missing_ok=True)


def might_be_commitment(text: str) -> bool:
    return bool(COMMITMENT_HINT.search(text))


async def needs_excuse(client, me: str, text: str) -> bool:
    if not might_be_commitment(text):
        return False
    answer = await client.helper([{"role": "user", "content": CLASSIFY_PROMPT.format(me=me, text=text)}], temperature=0.0, max_tokens=5)
    return answer.strip().upper().startswith("YES")


@functools.cache
def _words() -> list[str]:
    with gzip.open(WORDS_FILE, "rt", encoding="utf-8") as f:
        return f.read().split()


def random_words(n: int) -> list[str]:
    """n words drawn uniformly from every English noun and verb."""
    return [random.choice(_words()) for _ in range(n)]


def _uses(text: str, word: str) -> bool:
    # Loose match so "rescuing" counts for "rescue" and "geese" doesn't need to be exact for "goose" (stem only).
    return word[: max(4, len(word) - 3)].lower() in text.lower()


async def absurd_excuse(client, me: str, text: str, style_examples: list[str], cfg: GuardrailsCfg) -> list[str]:
    # Flatten multi-bubble examples and keep short ones: they show tone without inviting a copy.
    samples = [b for ex in style_examples for b in split_bubbles(ex, me) if len(b) <= 80][-10:]
    seen = {_norm(b) for b in samples}
    words = random_words(cfg.excuse_random_words)
    prompt = EXCUSE_PROMPT.format(
        me=me,
        text=text,
        examples=" / ".join(f'"{e}"' for e in cfg.excuse_examples) or '"cant i have to ride my horse into the sunset"',
        words=WORDS_HINT.format(words=", ".join(f'"{w}"' for w in words)) if words else "",
        style="\n".join(samples) or "(casual, lowercase)",
    )
    bubbles: list[str] = []
    for _ in range(2):  # retry once if it ignored the words
        raw = await client.helper([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=60)
        bubbles = [b for b in (re.sub(r"^\s*[-•*]\s+", "", x).strip().strip('"') for x in split_bubbles(raw.replace("\n", BUBBLE), me)) if b]
        bubbles = [c for c in (_casual(b) for b in bubbles if _norm(b) not in seen) if c][:2]  # drop echoed examples
        if bubbles and all(_uses(" ".join(bubbles), w) for w in words):
            break
    return bubbles or [random.choice(cfg.fallback_excuses or ["cant my horse needs me"])]


def _casual(text: str) -> str:
    """The base model writes like an assistant; make it read like a text."""
    text = re.sub(r"^(i['’]?m |i am )?(so )?sorry,? (but )?", "", text, flags=re.I)
    text = re.sub(r"^(unfortunately|apologies),? ", "", text, flags=re.I)
    return text.rstrip(".").strip()


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", s.lower())


def breaks_character(bubbles: list[str]) -> bool:
    return bool(AI_LEAK.search(" ".join(bubbles)))
