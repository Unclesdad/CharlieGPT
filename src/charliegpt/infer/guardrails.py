"""Kill switch, commitment dodging (absurd excuses), and in-character checks."""

from __future__ import annotations

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

EXCUSE_PROMPT = """You are {me}, texting. They just said:
\"\"\"
{text}
\"\"\"
They want you to commit to something. Don't agree. Decline with the most absurd, over-the-top, \
obviously made-up excuse imaginable, totally deadpan, in the same vibe as \
"sorry i cant i have to ride my horse into the sunset" or "i want to but the king of atlantis has sent me on a quest" \
(but invent a new one, never reuse those).

This is how {me} texts, match it exactly (casing, punctuation, slang, emoji):
{style}

Reply with only the text message. For more than one text, put {bubble} between them."""

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
    style = "\n".join(f"- {s}" for s in style_examples[-12:]) or "- (casual, lowercase)"
    prompt = EXCUSE_PROMPT.format(me=me, text=text, style=style, bubble=BUBBLE)
    raw = await client.helper([{"role": "user", "content": prompt}], temperature=1.0, max_tokens=120)
    return split_bubbles(raw.strip().strip('"'), me)


def breaks_character(bubbles: list[str]) -> bool:
    return bool(AI_LEAK.search(" ".join(bubbles)))
