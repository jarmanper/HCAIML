"""finsent-compare — VADER vs FinBERT on Financial PhraseBank.

Importing this package configures a single project-wide logger so individual
modules don't each have to ``logging.basicConfig`` and stomp on each other.
"""

from __future__ import annotations

import logging
import os

_LEVEL = os.environ.get("FINSENT_LOG_LEVEL", "INFO").upper()

# Only touch the root logger if nobody else has — Jupyter, pytest, and the
# transformers library all set up handlers, and stacking ours on top duplicates
# every line. ``force=False`` (the default) keeps us out of their way.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

__all__: list[str] = []
