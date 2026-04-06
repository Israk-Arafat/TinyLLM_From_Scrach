"""Heuristic text cleaning and filtering."""
from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def clean_sample(text: str, min_len: int = 50) -> str | None:
    """Return cleaned text or None if the sample should be dropped."""
    if not isinstance(text, str):
        return None
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if len(text) < min_len:
        return None
    return text
