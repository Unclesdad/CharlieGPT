"""Kill switch, commitment dodging (absurd excuses), and in-character checks."""

from __future__ import annotations

import random
import re
from datetime import datetime
from pathlib import Path

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
obviously made-up excuse, totally deadpan, like "sorry i cant i have to ride my horse into the sunset" \
or "i want to but the king of atlantis has sent me on a quest" (invent a new one, never reuse those).
Keep it to one short text, under 20 words, no explanation after the excuse.

For tone only (don't repeat these), here are some real texts {me} has sent:
\"\"\"
{style}
\"\"\"

Reply with only the text {me} sends, nothing else."""

# Last resort, so a failed generation never turns into a yes.
FALLBACK_EXCUSES = [
    "cant the council of geese has summoned me",
    "i would but im legally required to guard a lighthouse that day",
    "no can do my horse is riding me into the sunset this time",
    "sorry the king of atlantis moved my quest up",
    "cant im getting knighted by a raccoon",
]

AI_LEAK = re.compile(r"\b(as an ai|an ai (language )?model|language model|i'?m an ai|i am an ai|openai|anthropic|qwen|chatgpt)\b", re.I)


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


async def absurd_excuse(client, me: str, text: str, style_examples: list[str]) -> list[str]:
    # Flatten multi-bubble examples and keep short ones: they show tone without inviting a copy.
    samples = [b for ex in style_examples for b in split_bubbles(ex, me) if len(b) <= 80][-10:]
    prompt = EXCUSE_PROMPT.format(me=me, text=text, style="\n".join(samples) or "(casual, lowercase)")
    raw = await client.helper([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=60)
    seen = {_norm(b) for b in samples}
    bubbles = [b for b in (re.sub(r"^\s*[-•*]\s+", "", x).strip().strip('"') for x in split_bubbles(raw.replace("\n", BUBBLE), me)) if b]
    bubbles = [b for b in bubbles if _norm(b) not in seen][:2]  # drop echoed examples
    return bubbles or [random.choice(FALLBACK_EXCUSES)]


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", s.lower())


def breaks_character(bubbles: list[str]) -> bool:
    return bool(AI_LEAK.search(" ".join(bubbles)))
