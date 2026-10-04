"""Text preprocessing: turning a string into a list of tokens."""

import re

# A token is a run of letters, digits, or apostrophes ("i've" stays one token).
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase, then split into word tokens. Punctuation is dropped.

    Deliberately simple: no stopword removal, no stemming ("feel" and "feels"
    stay different tokens). Experiment 01 shows what that costs.
    """
    return _TOKEN_RE.findall(text.lower())
