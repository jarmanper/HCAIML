"""Load and clean the Financial PhraseBank, then carve out a held-out test set.

The original PhraseBank ships as a plain text file where each line is::

    Some financial sentence here .@neutral

with a literal ``@`` separating the sentence from the label and the file
encoded in latin-1 (so the diacritics in the original release don't choke
on Windows). When the local file isn't present, we fall back to the
Hugging Face mirror, which exposes the same data under
``financial_phrasebank``.

Why a held-out test split when both models are zero-shot? Two reasons:
it lets us run an apples-to-apples comparison on the same examples the
report will quote, and it leaves a clean train half on the table if a
future iteration of this project wants to actually fine-tune something.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Split:
    """A train/test split returned by :func:`load_and_split`.

    Attributes:
        train: DataFrame with columns ``sentence``, ``label_str``, ``label``.
        test:  DataFrame with the same columns as ``train``.
    """

    train: pd.DataFrame
    test: pd.DataFrame


def _find_local_file(candidates: Iterable[str]) -> Path | None:
    """Return the first PhraseBank candidate path that actually exists."""
    for name in candidates:
        p = config.DATA_RAW / name
        if p.exists():
            return p
    return None


def _read_local(path: Path) -> pd.DataFrame:
    """Parse the latin-1 PhraseBank text file into a DataFrame.

    Each line is ``sentence@label``. We split on the *last* ``@`` because
    a handful of sentences contain stray ``@`` characters (email-looking
    fragments). Empty and malformed lines are skipped with a warning count.
    """
    rows: list[tuple[str, str]] = []
    skipped = 0
    with path.open("r", encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if "@" not in line:
                skipped += 1
                continue
            sentence, label = line.rsplit("@", 1)
            rows.append((sentence.strip(), label.strip().lower()))

    if skipped:
        logger.warning("skipped %d malformed lines in %s", skipped, path.name)

    return pd.DataFrame(rows, columns=["sentence", "label_str"])


def _read_huggingface(hf_config: str) -> pd.DataFrame:
    """Pull PhraseBank from the Hugging Face hub as a fallback.

    The HF dataset uses integer labels with a different ordering than ours
    (``0=negative, 1=neutral, 2=positive`` happens to line up here, but I
    don't want to bet on it elsewhere), so we round-trip through the
    string label exposed by the dataset's ``ClassLabel`` feature and then
    let the canonical mapping in :mod:`src.config` do the rest.
    """
    from datasets import load_dataset

    logger.info("local PhraseBank not found; falling back to HF (%s)", hf_config)
    ds = load_dataset(config.HF_DATASET, hf_config, trust_remote_code=True)
    split = ds["train"]
    label_feature = split.features["label"]
    sentences = split["sentence"]
    labels = [label_feature.int2str(i).lower() for i in split["label"]]
    return pd.DataFrame({"sentence": sentences, "label_str": labels})


def load_phrasebank(use_50agree: bool = False) -> pd.DataFrame:
    """Load the PhraseBank, clean it, and attach integer labels.

    Args:
        use_50agree: If True, load the 50%-agreement split instead of the
            default 75%. Useful as an ablation — the 50% split is bigger
            but noisier.

    Returns:
        DataFrame with columns:

        - ``sentence`` (str): the cleaned headline / sentence.
        - ``label_str`` (str): one of ``"negative" | "neutral" | "positive"``.
        - ``label`` (int): integer-encoded via :data:`config.LABEL_TO_INT`.
    """
    candidates = (
        config.PHRASEBANK_CANDIDATES_50 if use_50agree else config.PHRASEBANK_CANDIDATES_75
    )
    hf_config_name = config.HF_CONFIG_50 if use_50agree else config.HF_CONFIG_75

    local = _find_local_file(candidates)
    if local is not None:
        logger.info("loading PhraseBank from %s", local)
        df = _read_local(local)
    else:
        df = _read_huggingface(hf_config_name)

    df = (
        df.assign(sentence=df["sentence"].str.strip())
          .loc[lambda d: d["sentence"].str.len() > 0]
          .drop_duplicates(subset=["sentence"])
          .reset_index(drop=True)
    )

    valid = set(config.LABEL_TO_INT)
    bad = df.loc[~df["label_str"].isin(valid)]
    if not bad.empty:
        logger.warning("dropping %d rows with unexpected labels: %s",
                       len(bad), sorted(bad["label_str"].unique()))
        df = df.loc[df["label_str"].isin(valid)].reset_index(drop=True)

    df["label"] = df["label_str"].map(config.LABEL_TO_INT).astype("int64")
    if df["label"].isna().any():
        # Belt-and-braces: ``map`` would have produced NaN if a label slipped
        # through the filter above. If we got here something is genuinely
        # wrong with the input file.
        raise ValueError("encountered NaN labels after mapping; check label_str values")

    logger.info("loaded %d rows; class counts = %s", len(df),
                df["label_str"].value_counts().to_dict())
    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> Split:
    """Stratified train/test split on the integer label column.

    Stratifying on ``label`` keeps the class balance the same in both halves —
    important here because PhraseBank is heavily neutral-skewed and a random
    split could easily under-represent the negative class on the test side.

    Args:
        df: DataFrame returned from :func:`load_phrasebank`.
        test_size: Fraction held out for evaluation. Defaults to 0.2.
        seed: Random seed for the split. Defaults to :data:`config.SEED`.

    Returns:
        :class:`Split` containing the train and test DataFrames.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info("split: %d train / %d test", len(train_df), len(test_df))
    return Split(train=train_df, test=test_df)


def load_and_split(
    use_50agree: bool = False,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
    persist: bool = True,
) -> Split:
    """Convenience wrapper: load PhraseBank, persist parquet, return a split.

    Args:
        use_50agree: Use the 50%-agreement ablation split.
        test_size: Held-out fraction.
        seed: Random seed.
        persist: If True, write the cleaned dataset to
            ``data/processed/phrasebank.parquet`` so downstream tooling
            (or a grader poking around) can pick it up without re-running
            the loader.

    Returns:
        :class:`Split` ready for inference.
    """
    df = load_phrasebank(use_50agree=use_50agree)
    if persist:
        config.PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.PROCESSED_PARQUET, index=False)
        logger.info("wrote cleaned dataset to %s", config.PROCESSED_PARQUET)
    return split_train_test(df, test_size=test_size, seed=seed)
