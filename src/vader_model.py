"""VADER baseline: lexicon + heuristic rules.

This is the "before" half of the comparison. VADER is a hand-built model: a
sentiment lexicon (word -> valence float) plus a few heuristic rules for
boosters ("very"), negation ("not good"), capitalisation, punctuation, and
contrastive conjunctions ("but"). It produces a single ``compound`` score in
``[-1, 1]`` for any input sentence.

The interesting limitation for this project: VADER has no notion of
*context*. Every word's valence is fixed at look-up time, regardless of the
surrounding sentence. That's why financial language — where the same verb
("rose", "fell", "missed") flips polarity depending on what it's modifying —
is hard for it. Compare with ``finbert_model.py`` for the contextual route.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import config
from .preprocess import normalize_for_vader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VaderPrediction:
    """One VADER prediction.

    Attributes:
        label: Integer label in :data:`config.INT_TO_LABEL`.
        compound: Raw VADER compound score in ``[-1, 1]``.
    """

    label: int
    compound: float


def compound_to_label(
    compound: float,
    pos_thresh: float = config.VADER_POS_THRESH,
    neg_thresh: float = config.VADER_NEG_THRESH,
) -> int:
    """Map a VADER compound score to one of our integer labels.

    Uses the canonical Hutto & Gilbert (2014) thresholds: anything at or
    above ``+0.05`` is positive, anything at or below ``-0.05`` is
    negative, the band in between is neutral. The thresholds are exposed
    as parameters so an ablation can sweep them without touching the
    rest of the pipeline.

    Args:
        compound: VADER compound score.
        pos_thresh: Lower bound (inclusive) for the positive class.
        neg_thresh: Upper bound (inclusive) for the negative class.

    Returns:
        Integer in ``{0, 1, 2}`` per :data:`config.LABEL_TO_INT`.
    """
    if compound >= pos_thresh:
        return config.LABEL_TO_INT["positive"]
    if compound <= neg_thresh:
        return config.LABEL_TO_INT["negative"]
    return config.LABEL_TO_INT["neutral"]


class VaderClassifier:
    """Thin convenience wrapper around :class:`SentimentIntensityAnalyzer`.

    The underlying analyzer is stateless and cheap to construct, but holding
    one on the class means we don't pay the lexicon-load cost once per
    sentence in a tight loop.
    """

    def __init__(
        self,
        pos_thresh: float = config.VADER_POS_THRESH,
        neg_thresh: float = config.VADER_NEG_THRESH,
    ) -> None:
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

    def predict_one(self, text: str) -> VaderPrediction:
        """Score a single sentence."""
        normalized = normalize_for_vader(text)
        compound = float(self.analyzer.polarity_scores(normalized)["compound"])
        return VaderPrediction(
            label=compound_to_label(compound, self.pos_thresh, self.neg_thresh),
            compound=compound,
        )

    def predict(
        self,
        texts: Iterable[str],
        progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score a batch of sentences.

        Args:
            texts: Iterable of raw sentences. They will be normalized for
                VADER inside this method — pass the originals.
            progress: If True, show a tqdm bar. Disable for tests / smoke runs.

        Returns:
            Tuple ``(labels, compounds)`` of numpy arrays, both length
            ``len(texts)``. ``labels`` is ``int64``, ``compounds`` is
            ``float32``.
        """
        text_list = list(texts)
        labels = np.empty(len(text_list), dtype=np.int64)
        compounds = np.empty(len(text_list), dtype=np.float32)
        iterator = tqdm(text_list, desc="VADER", disable=not progress)
        for i, text in enumerate(iterator):
            pred = self.predict_one(text)
            labels[i] = pred.label
            compounds[i] = pred.compound
        return labels, compounds


def run_vader(
    texts: Iterable[str],
    pos_thresh: float = config.VADER_POS_THRESH,
    neg_thresh: float = config.VADER_NEG_THRESH,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One-call entry point for the notebook / pipeline.

    Args:
        texts: Sentences to classify.
        pos_thresh: Positive-class threshold on the compound score.
        neg_thresh: Negative-class threshold on the compound score.
        progress: Toggle the tqdm bar.

    Returns:
        ``(labels, compounds)`` — same as :meth:`VaderClassifier.predict`.
    """
    clf = VaderClassifier(pos_thresh=pos_thresh, neg_thresh=neg_thresh)
    return clf.predict(texts, progress=progress)
