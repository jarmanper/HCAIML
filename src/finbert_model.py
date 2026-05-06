"""FinBERT inference: batched, contextual, financial-domain-tuned.

This is the "after" half of the comparison and the actual honors extension —
moving from a fixed lexicon look-up (VADER) to a Transformer encoder whose
self-attention layers produce *contextual* embeddings for every token. Same
word, different surrounding sentence, different vector. That contextuality
is what lets the model treat "missed by a wide margin" as negative without
having to put "missed" in any lexicon by hand.

We're using ``ProsusAI/finbert`` (Araci, 2019), a BERT-base checkpoint
fine-tuned on the Reuters TRC2-financial corpus and on PhraseBank itself.
The model card lives at https://huggingface.co/ProsusAI/finbert.

There is one well-known foot-gun: the model card's ``id2label`` is
``{0: 'positive', 1: 'negative', 2: 'neutral'}`` — *not* the alphabetical
ordering you might expect. We never index into the logits directly; we
always go through the string label and re-map to our canonical
``{negative: 0, neutral: 1, positive: 2}`` scheme via :mod:`src.config`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config

logger = logging.getLogger(__name__)


@dataclass
class FinBertClassifier:
    """Holds a tokenizer + model pair ready for batched inference.

    Construct once, call :meth:`predict` as many times as you like. The
    constructor takes the slow path (loads weights, moves to device); the
    inference path is hot.
    """

    model_id: str = config.FINBERT_MODEL_ID
    device: torch.device | None = None
    max_length: int = config.FINBERT_MAX_LENGTH

    def __post_init__(self) -> None:
        self.device = self.device or _pick_device()
        logger.info("loading %s on %s", self.model_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.eval()
        self.model.to(self.device)

        # Build the index-to-our-int map up front so the hot loop just does
        # a tensor lookup. ``id2label`` keys are ints; values are 'positive'
        # / 'negative' / 'neutral'. Anything missing is a model regression.
        id2label: dict[int, str] = self.model.config.id2label
        try:
            self._idx_to_our_int = np.array(
                [config.LABEL_TO_INT[id2label[i].lower()] for i in range(len(id2label))],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"unexpected FinBERT id2label mapping: {id2label!r}"
            ) from exc
        logger.debug("FinBERT id2label = %s, mapped to ints %s",
                     id2label, self._idx_to_our_int.tolist())

    @torch.no_grad()
    def predict(
        self,
        texts: Iterable[str],
        batch_size: int = config.FINBERT_BATCH_SIZE,
        progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run batched inference on a list of sentences.

        Args:
            texts: Sentences to classify. Passed through unchanged — the
                FinBERT tokenizer handles its own normalisation, casing,
                and truncation.
            batch_size: Number of sentences per forward pass. 32 is a
                comfortable default on CPU; bump it if you're on GPU.
            progress: If True, show a tqdm bar.

        Returns:
            Tuple ``(labels, confidences)`` of numpy arrays, both length
            ``len(texts)``. ``labels`` is ``int64`` in our canonical
            scheme; ``confidences`` is the softmax probability of the
            predicted class as ``float32``.
        """
        text_list = list(texts)
        n = len(text_list)
        labels = np.empty(n, dtype=np.int64)
        confidences = np.empty(n, dtype=np.float32)

        n_batches = (n + batch_size - 1) // batch_size
        bar = tqdm(
            range(0, n, batch_size),
            total=n_batches,
            desc="FinBERT",
            disable=not progress,
        )
        for start in bar:
            end = min(start + batch_size, n)
            batch = text_list[start:end]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
            top_idx_np = top_idx.detach().cpu().numpy()
            top_prob_np = top_prob.detach().cpu().numpy().astype(np.float32)
            # Re-map model-index -> our canonical int via the lookup we
            # built in __post_init__. This is the *only* place we touch
            # FinBERT's native label ordering; do not bypass it.
            labels[start:end] = self._idx_to_our_int[top_idx_np]
            confidences[start:end] = top_prob_np

        return labels, confidences


def _pick_device() -> torch.device:
    """Choose CUDA if available, otherwise CPU.

    Apple Silicon ``mps`` is intentionally not auto-selected: at the time
    of writing, FinBERT under MPS occasionally returns NaN logits and the
    speed-up vs CPU is marginal on a model this size. CPU is the safer
    default; CUDA users get the fast path.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_finbert(
    texts: Iterable[str],
    batch_size: int = config.FINBERT_BATCH_SIZE,
    max_length: int = config.FINBERT_MAX_LENGTH,
    device: torch.device | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One-call entry point for the notebook / pipeline.

    Args:
        texts: Sentences to classify.
        batch_size: Sentences per forward pass.
        max_length: Token cap; PhraseBank sentences are short, 256 is plenty.
        device: Override the auto-detected device. Mostly for tests.
        progress: Toggle the tqdm bar.

    Returns:
        ``(labels, confidences)`` — see :meth:`FinBertClassifier.predict`.
    """
    clf = FinBertClassifier(device=device, max_length=max_length)
    return clf.predict(texts, batch_size=batch_size, progress=progress)
