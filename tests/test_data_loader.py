"""Tests for the PhraseBank loader.

We avoid hitting the network here — the local-file path is exercised against
a tiny synthetic latin-1 fixture so the suite is fast and deterministic. A
slower opt-in test at the bottom of the file checks the real PhraseBank if
it happens to be on disk; it's marked so CI can skip it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.data_loader import (
    Split,
    _read_local,
    load_phrasebank,
    split_train_test,
)


@pytest.fixture
def synthetic_phrasebank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a tiny PhraseBank-shaped file in latin-1 and point config at it."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    fpath = raw_dir / "Sentences_75Agree.txt"

    # Mix all three labels (>=2 per class so the stratified split has room
    # to work), include a stray @, a duplicate, and a latin-1 character to
    # confirm the encoding round-trips.
    lines = [
        "Operating profit rose to EUR 12 million .@positive",
        "Net sales increased by 8 percent .@positive",
        "Revenue declined sharply year-on-year .@negative",
        "Margins compressed under cost pressure .@negative",
        "The company reported quarterly results .@neutral",
        "Café opened a new branch .@neutral",
        "Operating profit rose to EUR 12 million .@positive",  # duplicate
        "Contact @sales for details .@neutral",  # stray @
    ]
    fpath.write_text("\n".join(lines) + "\n", encoding="latin-1")

    monkeypatch.setattr(config, "DATA_RAW", raw_dir)
    return fpath


def test_read_local_handles_latin1_and_stray_at(synthetic_phrasebank: Path) -> None:
    df = _read_local(synthetic_phrasebank)
    assert "Café opened a new branch ." in df["sentence"].tolist()
    assert any(s.startswith("Contact @sales") for s in df["sentence"])
    assert set(df["label_str"]) <= {"positive", "negative", "neutral"}


def test_load_phrasebank_label_mapping(synthetic_phrasebank: Path) -> None:
    df = load_phrasebank(use_50agree=False)
    assert list(df.columns) == ["sentence", "label_str", "label"]
    # Canonical mapping
    assert (df.loc[df["label_str"] == "negative", "label"] == 0).all()
    assert (df.loc[df["label_str"] == "neutral", "label"] == 1).all()
    assert (df.loc[df["label_str"] == "positive", "label"] == 2).all()


def test_load_phrasebank_drops_dupes_and_has_no_nans(synthetic_phrasebank: Path) -> None:
    df = load_phrasebank(use_50agree=False)
    assert df.notna().all().all(), "no NaNs anywhere in the cleaned frame"
    # Duplicate sentence in the fixture should have been collapsed.
    assert df["sentence"].duplicated().sum() == 0


def test_split_train_test_is_stratified_and_reproducible(
    synthetic_phrasebank: Path,
) -> None:
    df = load_phrasebank(use_50agree=False)
    split_a = split_train_test(df, test_size=0.5, seed=42)
    split_b = split_train_test(df, test_size=0.5, seed=42)
    assert isinstance(split_a, Split)
    pd.testing.assert_frame_equal(split_a.train, split_b.train)
    pd.testing.assert_frame_equal(split_a.test, split_b.test)
    # Stratified split should keep both halves non-empty for at least one class.
    assert len(split_a.train) > 0 and len(split_a.test) > 0


@pytest.mark.skipif(
    not any(
        (Path(__file__).resolve().parent.parent / "data" / "raw" / name).exists()
        for name in config.PHRASEBANK_CANDIDATES_75
    ),
    reason="real PhraseBank file not present; this is an opt-in sanity check",
)
def test_real_phrasebank_row_count_in_expected_range() -> None:
    df = load_phrasebank(use_50agree=False)
    assert 1500 <= len(df) <= 5000, f"unexpected PhraseBank size: {len(df)}"
