"""Plotting helpers — four required figures, one consistent palette.

Everything in here:

- Uses ``seaborn``'s ``colorblind`` palette so the report is readable
  for an audience that includes colour-vision-deficient graders. No
  default matplotlib teals.
- Saves to ``figures/`` at 200 dpi with ``bbox_inches="tight"``. Filenames
  are stable so the README can link to them.
- Accepts already-computed metric dicts (from :mod:`src.evaluate`) and
  prediction DataFrames so we don't re-run inference here.

The four figures, by spec:

1. Side-by-side confusion matrices (raw and normalized).
2. Per-class F1 bar chart for both models.
3. Histogram of FinBERT confidence on correct vs incorrect predictions.
4. Per-class precision / recall comparison chart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

logger = logging.getLogger(__name__)

# One palette to rule them all. ``colorblind`` is an 8-colour seaborn preset
# that's safe across the common forms of colour-vision deficiency.
PALETTE = sns.color_palette("colorblind")
COLOR_VADER = PALETTE[0]
COLOR_FINBERT = PALETTE[2]
CMAP = "Blues"  # used for the confusion matrices; sequential and CB-safe


def _apply_style() -> None:
    """Apply the project plotting style once per figure call."""
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
        font_scale=1.0,
    )
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"


def _save(fig: plt.Figure, path: Path) -> Path:
    """Persist a figure and close it so the notebook stays clean."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("saved %s", path)
    return path


def plot_confusion_matrices(
    vader_metrics: dict[str, Any],
    finbert_metrics: dict[str, Any],
    class_names: tuple[str, ...] = config.CLASS_NAMES,
    out_dir: Path = config.FIGURES_DIR,
) -> tuple[Path, Path]:
    """Save side-by-side raw and normalized confusion matrices.

    Two PNGs are written:

    - ``confusion_matrices_raw.png``       — counts.
    - ``confusion_matrices_normalized.png`` — row-normalized (each row
      sums to 1, so the diagonal reads as per-class recall).

    Args:
        vader_metrics: Output of :func:`evaluate.compute_metrics` for VADER.
        finbert_metrics: Same for FinBERT.
        class_names: Class names in label-int order.
        out_dir: Directory to write into.

    Returns:
        Tuple ``(raw_path, normalized_path)``.
    """
    _apply_style()
    raw_path = out_dir / "confusion_matrices_raw.png"
    norm_path = out_dir / "confusion_matrices_normalized.png"

    cm_v = np.asarray(vader_metrics["confusion_matrix"])
    cm_f = np.asarray(finbert_metrics["confusion_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, title in zip(axes, [cm_v, cm_f], ["VADER", "FinBERT"]):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=CMAP,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{title}: confusion matrix (counts)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Confusion matrices — raw counts", y=1.02, fontsize=12)
    _save(fig, raw_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, title in zip(axes, [cm_v, cm_f], ["VADER", "FinBERT"]):
        # Row-normalize so each row sums to 1 (NaN-safe for empty rows).
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
        sns.heatmap(
            norm,
            annot=True,
            fmt=".2f",
            cmap=CMAP,
            vmin=0.0,
            vmax=1.0,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{title}: confusion matrix (row-normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Confusion matrices — row-normalized (recall on the diagonal)",
                 y=1.02, fontsize=12)
    _save(fig, norm_path)

    return raw_path, norm_path


def plot_per_class_f1(
    vader_metrics: dict[str, Any],
    finbert_metrics: dict[str, Any],
    class_names: tuple[str, ...] = config.CLASS_NAMES,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """Grouped bar chart of per-class F1 for VADER vs FinBERT.

    Saves to ``figures/per_class_f1.png``.
    """
    _apply_style()
    path = out_dir / "per_class_f1.png"

    f1_v = [vader_metrics["per_class_f1"][c] for c in class_names]
    f1_f = [finbert_metrics["per_class_f1"][c] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars_v = ax.bar(x - width / 2, f1_v, width, label="VADER", color=COLOR_VADER)
    bars_f = ax.bar(x + width / 2, f1_f, width, label="FinBERT", color=COLOR_FINBERT)
    for bars in (bars_v, bars_f):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    ax.set_xticks(x, class_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Class")
    ax.set_title("Per-class F1 — VADER vs FinBERT")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_finbert_confidence_hist(
    predictions: pd.DataFrame,
    out_dir: Path = config.FIGURES_DIR,
    bins: int = 20,
) -> Path:
    """Histogram of FinBERT confidence on correct vs incorrect predictions.

    Saves to ``figures/finbert_confidence_hist.png``. We use the prediction
    DataFrame directly so the buckets line up exactly with the predictions
    that landed in ``results/predictions.csv``.
    """
    _apply_style()
    path = out_dir / "finbert_confidence_hist.png"

    correct = predictions.loc[predictions["finbert_correct"], "finbert_confidence"]
    wrong = predictions.loc[~predictions["finbert_correct"], "finbert_confidence"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(
        correct,
        bins=bins,
        range=(0.0, 1.0),
        alpha=0.7,
        label=f"Correct (n={len(correct)})",
        color=PALETTE[2],
        edgecolor="white",
    )
    ax.hist(
        wrong,
        bins=bins,
        range=(0.0, 1.0),
        alpha=0.7,
        label=f"Incorrect (n={len(wrong)})",
        color=PALETTE[3],
        edgecolor="white",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("FinBERT predicted-class probability")
    ax.set_ylabel("Count")
    ax.set_title("FinBERT confidence — correct vs incorrect")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_per_class_precision_recall(
    vader_metrics: dict[str, Any],
    finbert_metrics: dict[str, Any],
    class_names: tuple[str, ...] = config.CLASS_NAMES,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """Two-panel grouped bar chart: precision and recall per class per model.

    Saves to ``figures/per_class_precision_recall.png``.
    """
    _apply_style()
    path = out_dir / "per_class_precision_recall.png"

    x = np.arange(len(class_names))
    width = 0.38

    p_v = [vader_metrics["per_class_precision"][c] for c in class_names]
    p_f = [finbert_metrics["per_class_precision"][c] for c in class_names]
    r_v = [vader_metrics["per_class_recall"][c] for c in class_names]
    r_f = [finbert_metrics["per_class_recall"][c] for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    ax_p, ax_r = axes
    bars_pv = ax_p.bar(x - width / 2, p_v, width, label="VADER", color=COLOR_VADER)
    bars_pf = ax_p.bar(x + width / 2, p_f, width, label="FinBERT", color=COLOR_FINBERT)
    ax_p.set_xticks(x, class_names)
    ax_p.set_ylim(0, 1.0)
    ax_p.set_ylabel("Score")
    ax_p.set_title("Precision by class")
    ax_p.legend(loc="lower right")
    for bars in (bars_pv, bars_pf):
        ax_p.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    bars_rv = ax_r.bar(x - width / 2, r_v, width, label="VADER", color=COLOR_VADER)
    bars_rf = ax_r.bar(x + width / 2, r_f, width, label="FinBERT", color=COLOR_FINBERT)
    ax_r.set_xticks(x, class_names)
    ax_r.set_title("Recall by class")
    ax_r.legend(loc="lower right")
    for bars in (bars_rv, bars_rf):
        ax_r.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    fig.suptitle("Per-class precision and recall — VADER vs FinBERT",
                 y=1.02, fontsize=12)
    return _save(fig, path)


def plot_label_distribution(
    df: pd.DataFrame,
    out_dir: Path = config.FIGURES_DIR,
    class_names: tuple[str, ...] = config.CLASS_NAMES,
) -> Path:
    """Quick label-distribution bar chart for the data section of the notebook.

    Not in the four "required" figures but cheap to draw and useful in the
    data exploration cell.
    """
    _apply_style()
    path = out_dir / "label_distribution.png"

    counts = df["label_str"].value_counts().reindex(class_names).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[0], PALETTE[7], PALETTE[2]])
    ax.bar_label(bars, fmt="%d", padding=2)
    ax.set_ylabel("Number of sentences")
    ax.set_xlabel("Class")
    ax.set_title("Financial PhraseBank — class distribution")
    return _save(fig, path)


def make_all_figures(
    vader_metrics: dict[str, Any],
    finbert_metrics: dict[str, Any],
    predictions: pd.DataFrame,
    out_dir: Path = config.FIGURES_DIR,
) -> dict[str, Path]:
    """Convenience wrapper that produces every required figure in one call.

    Returns:
        ``{name: path}`` mapping for use in the notebook / README.
    """
    raw_cm, norm_cm = plot_confusion_matrices(
        vader_metrics, finbert_metrics, out_dir=out_dir
    )
    f1_path = plot_per_class_f1(vader_metrics, finbert_metrics, out_dir=out_dir)
    conf_path = plot_finbert_confidence_hist(predictions, out_dir=out_dir)
    pr_path = plot_per_class_precision_recall(
        vader_metrics, finbert_metrics, out_dir=out_dir
    )
    return {
        "confusion_matrices_raw": raw_cm,
        "confusion_matrices_normalized": norm_cm,
        "per_class_f1": f1_path,
        "finbert_confidence_hist": conf_path,
        "per_class_precision_recall": pr_path,
    }
