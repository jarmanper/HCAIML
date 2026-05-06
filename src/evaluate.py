"""Metrics, persistence, and edge-case extraction.

One module so VADER and FinBERT are evaluated through the exact same code
path — anything different in the numbers is the model, not the bookkeeping.

Outputs land under ``results/``:

- ``metrics.json``       — per-model accuracy, macro P/R/F1, per-class F1,
                            confusion matrix, full sklearn classification report.
- ``predictions.csv``    — sentence-level predictions for both models.
- ``disagreements.csv``  — examples where the two models disagree, plus the
                            "hardest" rows (very confident wrong, low-confidence
                            right) for the report's edge-case section.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import config

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] = config.CLASS_NAMES,
) -> dict[str, Any]:
    """Score a single model's predictions on a single test set.

    Args:
        y_true: Integer-encoded gold labels.
        y_pred: Integer-encoded model predictions.
        class_names: Class names in label-int order; defaults to the project
            canonical ``("negative", "neutral", "positive")``.

    Returns:
        A JSON-serialisable dict with:

        - ``accuracy`` (float)
        - ``macro_precision`` / ``macro_recall`` / ``macro_f1`` (float)
        - ``per_class_f1`` (``{class_name: float}``)
        - ``per_class_precision`` / ``per_class_recall`` (same shape)
        - ``confusion_matrix`` (``list[list[int]]``, rows = true, cols = pred)
        - ``classification_report`` (str, sklearn's pretty-printed version)
        - ``support`` (``{class_name: int}``)
    """
    labels = list(range(len(class_names)))
    accuracy = float(accuracy_score(y_true, y_pred))

    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_p, per_r, per_f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "per_class_f1": {n: float(v) for n, v in zip(class_names, per_f)},
        "per_class_precision": {n: float(v) for n, v in zip(class_names, per_p)},
        "per_class_recall": {n: float(v) for n, v in zip(class_names, per_r)},
        "confusion_matrix": cm,
        "support": {n: int(v) for n, v in zip(class_names, support)},
        "classification_report": report,
    }


def write_metrics_json(
    metrics_by_model: dict[str, dict[str, Any]],
    path: Path = config.METRICS_JSON,
) -> Path:
    """Persist a ``{model_name: metrics_dict}`` mapping to JSON.

    Args:
        metrics_by_model: e.g. ``{"vader": {...}, "finbert": {...}}``.
        path: Output file path. Defaults to :data:`config.METRICS_JSON`.

    Returns:
        The path that was written, for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_by_model, fh, indent=2)
    logger.info("wrote metrics to %s", path)
    return path


def build_predictions_frame(
    sentences: list[str],
    y_true: np.ndarray,
    vader_pred: np.ndarray,
    vader_compound: np.ndarray,
    finbert_pred: np.ndarray,
    finbert_confidence: np.ndarray,
) -> pd.DataFrame:
    """Stitch the per-sentence outputs of both models into one DataFrame.

    Columns match the project spec: ``sentence``, ``true_label``,
    ``vader_pred``, ``finbert_pred``, ``vader_compound``,
    ``finbert_confidence``. Plus a few derived booleans that make the
    edge-case extraction below trivial.
    """
    df = pd.DataFrame(
        {
            "sentence": sentences,
            "true_label": y_true.astype(int),
            "vader_pred": vader_pred.astype(int),
            "finbert_pred": finbert_pred.astype(int),
            "vader_compound": vader_compound.astype(float),
            "finbert_confidence": finbert_confidence.astype(float),
        }
    )
    df["true_label_str"] = df["true_label"].map(config.INT_TO_LABEL)
    df["vader_pred_str"] = df["vader_pred"].map(config.INT_TO_LABEL)
    df["finbert_pred_str"] = df["finbert_pred"].map(config.INT_TO_LABEL)
    df["vader_correct"] = df["vader_pred"] == df["true_label"]
    df["finbert_correct"] = df["finbert_pred"] == df["true_label"]
    df["models_disagree"] = df["vader_pred"] != df["finbert_pred"]
    return df


def write_predictions_csv(df: pd.DataFrame, path: Path = config.PREDICTIONS_CSV) -> Path:
    """Write the predictions frame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("wrote predictions to %s", path)
    return path


def extract_edge_cases(
    predictions: pd.DataFrame,
    n_each: int = 20,
) -> pd.DataFrame:
    """Pull out the rows that make the edge-case section interesting.

    Returns a single concatenated DataFrame with a ``category`` column so
    the report can group on it. Categories:

    - ``finbert_wins``: FinBERT right, VADER wrong.
    - ``vader_wins``:   VADER right, FinBERT wrong.
    - ``both_wrong``:   Neither model right.
    - ``finbert_overconfident_wrong``: FinBERT wrong with very high
      softmax probability — these are the embarrassing failures.
    - ``finbert_underconfident_right``: FinBERT right but only barely —
      examples where it almost flipped.

    Args:
        predictions: DataFrame from :func:`build_predictions_frame`.
        n_each: Cap per category; the function returns up to ``n_each``
            rows per group. Defaults to 20.

    Returns:
        DataFrame with the same columns as the input plus a ``category``
        string column.
    """
    parts: list[pd.DataFrame] = []

    finbert_wins = predictions[
        predictions["finbert_correct"] & ~predictions["vader_correct"]
    ].copy()
    finbert_wins["category"] = "finbert_wins"
    parts.append(finbert_wins.head(n_each))

    vader_wins = predictions[
        predictions["vader_correct"] & ~predictions["finbert_correct"]
    ].copy()
    vader_wins["category"] = "vader_wins"
    parts.append(vader_wins.head(n_each))

    both_wrong = predictions[
        ~predictions["vader_correct"] & ~predictions["finbert_correct"]
    ].copy()
    both_wrong["category"] = "both_wrong"
    parts.append(both_wrong.head(n_each))

    overconfident_wrong = (
        predictions[~predictions["finbert_correct"]]
        .sort_values("finbert_confidence", ascending=False)
        .head(n_each)
        .copy()
    )
    overconfident_wrong["category"] = "finbert_overconfident_wrong"
    parts.append(overconfident_wrong)

    underconfident_right = (
        predictions[predictions["finbert_correct"]]
        .sort_values("finbert_confidence", ascending=True)
        .head(n_each)
        .copy()
    )
    underconfident_right["category"] = "finbert_underconfident_right"
    parts.append(underconfident_right)

    return pd.concat(parts, ignore_index=True)


def write_disagreements_csv(
    edge_cases: pd.DataFrame,
    path: Path = config.DISAGREEMENTS_CSV,
) -> Path:
    """Write the edge-case frame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    edge_cases.to_csv(path, index=False)
    logger.info("wrote edge cases to %s", path)
    return path
