# SPDX-License-Identifier: Apache-2.0
"""The words of reply text that are spoken: what the speech service reads and the forced aligner aligns."""

import re

# Note (ruoyu): Qwen3-TTS reads an emoji as stray syllables or runs on for minutes and spells out a link target.
UNSPOKEN = re.compile(
    r"\[(?P<link_text>[^\]]*)\]\([^)]*\)"
    r"|(?P<asterisks>\*+)"
    "|[\U0001f000-\U0001faff☀-➿⭐⭕︎️‍⃣`]"
)


def is_multiplication(text: str, match: re.Match[str]) -> bool:
    """Whether the asterisks are one multiplication sign between two numbers, such as in `3*4`."""
    return (
        match.group("asterisks") == "*"
        and text[: match.start()].rstrip()[-1:].isdigit()
        and text[match.end() :].lstrip()[:1].isdigit()
    )


def speakable_text(text: str) -> tuple[str, list[int]]:
    """`text` without emoji, markdown emphasis, code marks and link targets, and for each kept character its offset in `text`."""
    kept: list[str] = []
    origin: list[int] = []
    cursor = 0
    for match in UNSPOKEN.finditer(text):
        kept.append(text[cursor : match.start()])
        origin.extend(range(cursor, match.start()))
        if match.group("link_text") is not None:
            kept.append(match.group("link_text"))
            origin.extend(range(match.start("link_text"), match.end("link_text")))
        elif match.group("asterisks") is not None and is_multiplication(text, match):
            kept.append(match.group())
            origin.extend(range(match.start(), match.end()))
        else:
            pass
        cursor = match.end()
    kept.append(text[cursor:])
    origin.extend(range(cursor, len(text)))
    return "".join(kept), origin
