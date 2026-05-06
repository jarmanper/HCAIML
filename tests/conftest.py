"""Make the repo root importable so ``from src import ...`` works in pytest.

Without this, running ``pytest`` from inside ``tests/`` (or with
auto-discovery) means the ``src`` package isn't on ``sys.path``. Adding the
repo root once at session start is the least magical fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
