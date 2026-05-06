"""Tests for ``src.evaluate``.

Strategy: hand-build small ``y_true`` / ``y_pred`` arrays whose metrics we've
worked out on paper, then assert :func:`compute_metrics` agrees. If sklearn
ever changes its averaging behaviour or we accidentally swap an axis, this
catches it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import config
from src.evaluate import (
    build_predictions_frame,
    compute_metrics,
    extract_edge_cases,
)


def test_compute_metrics_known_values() -> None:
    # 6 examples, mix of all three classes, four correct -> accuracy 4/6.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    out = compute_metrics(y_true, y_pred)

    # accuracy
    assert math.isclose(out["accuracy"], 4 / 6, rel_tol=0, abs_tol=1e-9)

    # per-class F1 — derived by hand:
    # negative: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    # neutral : TP=2, FP=1, FN=0 -> P=2/3, R=1.0, F1=0.8
    # positive: TP=1, FP=0, FN=1 -> P=1.0, R=0.5, F1=2/3
    assert math.isclose(out["per_class_f1"]["negative"], 0.5, abs_tol=1e-9)
    assert math.isclose(out["per_class_f1"]["neutral"], 0.8, abs_tol=1e-9)
    assert math.isclose(out["per_class_f1"]["positive"], 2 / 3, abs_tol=1e-9)

    # macro-F1 is just the unweighted mean of the three.
    expected_macro = (0.5 + 0.8 + 2 / 3) / 3
    assert math.isclose(out["macro_f1"], expected_macro, abs_tol=1e-9)

    # Confusion matrix shape and contents.
    cm = out["confusion_matrix"]
    assert cm == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_compute_metrics_perfect_predictions_score_one() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])
    out = compute_metrics(y, y)
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    for c in config.CLASS_NAMES:
        assert out["per_class_f1"][c] == 1.0
    # Confusion matrix should be diagonal.
    cm = np.asarray(out["confusion_matrix"])
    assert np.array_equal(cm, np.diag(cm.diagonal()))


def test_build_predictions_frame_has_expected_columns() -> None:
    sentences = ["a", "b", "c"]
    y_true = np.array([0, 1, 2])
    vader_pred = np.array([0, 1, 1])
    vader_compound = np.array([-0.6, 0.0, 0.4])
    finbert_pred = np.array([0, 1, 2])
    finbert_conf = np.array([0.99, 0.55, 0.88])

    df = build_predictions_frame(
        sentences, y_true, vader_pred, vader_compound, finbert_pred, finbert_conf
    )

    expected = {
        "sentence",
        "true_label",
        "vader_pred",
        "finbert_pred",
        "vader_compound",
        "finbert_confidence",
        "true_label_str",
        "vader_pred_str",
        "finbert_pred_str",
        "vader_correct",
        "finbert_correct",
        "models_disagree",
    }
    assert expected.issubset(df.columns)
    assert df["finbert_correct"].tolist() == [True, True, True]
    assert df["vader_correct"].tolist() == [True, True, False]
    assert df["models_disagree"].tolist() == [False, False, True]


def test_extract_edge_cases_categorises_rows() -> None:
    # Designed so each of the five categories shows up at least once:
    # - row 0: VADER right, FinBERT wrong  -> vader_wins
    # - row 1: VADER wrong, FinBERT right  -> finbert_wins
    # - row 2: both right (skipped by edge-case extractor)
    # - row 3: VADER right, FinBERT wrong  -> vader_wins
    # - row 4: both wrong                  -> both_wrong
    # - row 5: VADER wrong, FinBERT right  -> finbert_wins
    df = pd.DataFrame(
        {
            "sentence": [f"s{i}" for i in range(6)],
            "true_label": [0, 0, 1, 1, 2, 2],
            "vader_pred": [0, 1, 1, 1, 0, 0],
            "finbert_pred": [1, 0, 1, 2, 0, 2],
            "vader_compound": [0.0] * 6,
            "finbert_confidence": [0.95, 0.51, 0.92, 0.40, 0.88, 0.60],
        }
    )
    df["true_label_str"] = df["true_label"].map(config.INT_TO_LABEL)
    df["vader_pred_str"] = df["vader_pred"].map(config.INT_TO_LABEL)
    df["finbert_pred_str"] = df["finbert_pred"].map(config.INT_TO_LABEL)
    df["vader_correct"] = df["vader_pred"] == df["true_label"]
    df["finbert_correct"] = df["finbert_pred"] == df["true_label"]
    df["models_disagree"] = df["vader_pred"] != df["finbert_pred"]

    edges = extract_edge_cases(df, n_each=10)
    assert "category" in edges.columns
    assert set(edges["category"]) >= {
        "finbert_wins",
        "vader_wins",
        "both_wrong",
        "finbert_overconfident_wrong",
        "finbert_underconfident_right",
    }
