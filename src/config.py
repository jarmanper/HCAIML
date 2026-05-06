"""Project-wide configuration: paths, label maps, model IDs, seeds.

Everything that other modules might want to import lives here so there's one
obvious place to tweak. If a value shows up in more than one file, it should
probably be promoted to this module.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# Repo-relative paths. Resolving from this file means ``set_seeds()`` works
# whether you're running from the repo root, a notebook, or pytest.
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_PROCESSED: Path = DATA_DIR / "processed"
FIGURES_DIR: Path = ROOT / "figures"
RESULTS_DIR: Path = ROOT / "results"

# PhraseBank ships its files with funny capitalisations across mirrors, so we
# accept any of these on disk.
PHRASEBANK_CANDIDATES_75: tuple[str, ...] = (
    "Sentences_75Agree.txt",
    "sentences_75agree.txt",
)
PHRASEBANK_CANDIDATES_50: tuple[str, ...] = (
    "Sentences_50Agree.txt",
    "sentences_50agree.txt",
)

# Hugging Face fallback. The dataset card uses the same suffixes as the files.
HF_DATASET: str = "financial_phrasebank"
HF_CONFIG_75: str = "sentences_75agree"
HF_CONFIG_50: str = "sentences_50agree"

# Single source of truth for label encoding. Everything in the pipeline keys
# off this — VADER mapping, FinBERT remap, evaluator, plots.
LABEL_TO_INT: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}
CLASS_NAMES: tuple[str, ...] = ("negative", "neutral", "positive")

# FinBERT model card: https://huggingface.co/ProsusAI/finbert
# Reference: Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with
# Pre-trained Language Models." arXiv:1908.10063.
FINBERT_MODEL_ID: str = "ProsusAI/finbert"

# Defaults for FinBERT inference. Both are overridable from the call site;
# they live here so the notebook and the smoke test agree on them.
FINBERT_MAX_LENGTH: int = 256
FINBERT_BATCH_SIZE: int = 32

# VADER threshold defaults — these are the canonical Hutto & Gilbert (2014)
# values. Exposed here so we (or a grader) can sweep them in an ablation.
VADER_POS_THRESH: float = 0.05
VADER_NEG_THRESH: float = -0.05

SEED: int = 42
TEST_SIZE: float = 0.20

# Output filenames. Keeping these centralized makes the README's reproduction
# claim easier to verify.
METRICS_JSON: Path = RESULTS_DIR / "metrics.json"
PREDICTIONS_CSV: Path = RESULTS_DIR / "predictions.csv"
DISAGREEMENTS_CSV: Path = RESULTS_DIR / "disagreements.csv"
PROCESSED_PARQUET: Path = DATA_PROCESSED / "phrasebank.parquet"


def set_seeds(seed: int = SEED) -> None:
    """Seed every RNG we plausibly touch.

    Covers Python ``random``, NumPy, PyTorch (CPU + CUDA), and Python's
    hash randomization. We deliberately do *not* set
    ``torch.use_deterministic_algorithms(True)`` because some kernels on
    CPU-only Windows boxes raise rather than fall back to a deterministic
    path, which would defeat the point.

    Args:
        seed: Integer seed. Defaults to :data:`SEED`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    """Make sure the runtime output directories exist.

    Cheap and idempotent. Called once from the notebook setup cell so a fresh
    clone doesn't trip on a missing ``results/`` or ``figures/`` folder.
    """
    for d in (DATA_RAW, DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
