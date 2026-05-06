"""Light text normalization for the VADER path.

VADER is intentionally a *minimal* preprocessing target — its rules already
handle capitalisation, punctuation emphasis, and degree adverbs, so we don't
want to lower-case or strip punctuation before handing text in. All this
module does is collapse runs of whitespace and trim, which is safe and
mostly cosmetic.

FinBERT gets the raw sentence and lets its own tokenizer deal with it; that
path doesn't go through here on purpose.
"""

from __future__ import annotations

import re
from typing import Iterable

# Pre-compiled because we'll hit it once per sentence on every run.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_vader(text: str) -> str:
    """Collapse internal whitespace and strip the ends.

    Args:
        text: Raw sentence from PhraseBank.

    Returns:
        The same sentence with runs of whitespace squashed to a single space
        and leading/trailing whitespace removed. Casing and punctuation are
        intentionally left alone — VADER uses both as features.
    """
    if not isinstance(text, str):
        # Defensive: PhraseBank shouldn't yield non-strings, but if pandas
        # ever hands us a NaN we don't want to blow up here.
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_many(texts: Iterable[str]) -> list[str]:
    """Vectorised wrapper around :func:`normalize_for_vader`.

    Args:
        texts: Iterable of raw sentences.

    Returns:
        List of normalized sentences in the same order.
    """
    return [normalize_for_vader(t) for t in texts]
