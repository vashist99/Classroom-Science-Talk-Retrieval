"""Unit tests for src/data_loader_1.py.

Covers the Step 1 DoD:
    * row count + label distribution match xlsx (assertions in validate_corpus)
    * no duplicate utterances; whitespace + unicode normalized; cues are list-typed
    * a small unit test loads the parquet and asserts the schema
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader_1 import (
    DEFAULT_XLSX,
    SETTING_VOCAB,
    add_was_sci_coded,
    clean_corpus,
    extract_transcript_ref,
    load_and_clean,
    normalize_setting,
    normalize_text,
    parse_cue_list,
    run,
    validate_corpus,
)


def test_normalize_text_strips_and_collapses_whitespace():
    assert normalize_text("  hello  world  ") == "hello world"


def test_normalize_text_handles_nfkc_unicode():
    s = "smart \u201cquotes\u201d"
    out = normalize_text(s)
    assert out is not None and "smart" in out


def test_normalize_text_passes_nan_through():
    assert pd.isna(normalize_text(np.nan))


def test_parse_cue_list_splits_on_pipe():
    assert parse_cue_list("observe|notice|see") == ["observe", "notice", "see"]


def test_parse_cue_list_splits_on_comma_and_semicolon():
    assert parse_cue_list("a, b; c") == ["a", "b", "c"]


def test_parse_cue_list_returns_empty_for_nan_or_empty():
    assert parse_cue_list(np.nan) == []
    assert parse_cue_list("") == []
    assert parse_cue_list("   ") == []


def test_normalize_setting_strips_topic_suffix():
    setting, topic = normalize_setting("Large Group / Animal Observation")
    assert setting == "Large Group"
    assert topic == "Animal Observation"


def test_normalize_setting_preserves_blocks_engineering_slash():
    """Blocks/Engineering has no spaces around the slash and must not be split."""
    setting, topic = normalize_setting("Blocks/Engineering")
    assert setting == "Blocks/Engineering"
    assert topic is None


def test_normalize_setting_maps_nan_to_unknown():
    setting, topic = normalize_setting(np.nan)
    assert setting == "Unknown"
    assert topic is None


def test_normalize_setting_snaps_unknown_value_to_unknown():
    setting, topic = normalize_setting("Some Made Up Setting")
    assert setting == "Unknown"


def test_extract_transcript_ref_finds_id():
    notes = "Transcript match (01-2_19LG #61, SCI-coded teacher utterance): seed"
    assert extract_transcript_ref(notes) == "01-2_19LG #61"


def test_extract_transcript_ref_returns_none_when_absent():
    assert extract_transcript_ref("no reference here") is None
    assert extract_transcript_ref(np.nan) is None


def test_add_was_sci_coded_is_case_insensitive():
    df = pd.DataFrame({"Notes": [
        "blah SCI-coded blah",
        "blah sci-CODED blah",
        "no marker here",
        None,
    ]})
    out = add_was_sci_coded(df)
    assert list(out["was_sci_coded"]) == [1, 1, 0, 0]


def _xlsx_available():
    return DEFAULT_XLSX.exists()


@pytest.mark.skipif(not _xlsx_available(), reason="source xlsx not present")
def test_load_and_clean_end_to_end_produces_valid_corpus():
    corpus, seed, cat = load_and_clean(verbose=False)

    assert len(corpus) >= 195
    assert corpus["utt_id"].is_unique
    assert corpus["utterance"].notna().all()

    label_counts = Counter(corpus["label"].fillna("__NULL__"))
    assert label_counts.get("SCIENCE_TALK", 0) >= 188
    assert label_counts.get("NOT_SCIENCE_TALK", 0) >= 5

    assert set(corpus["setting"].dropna()).issubset(SETTING_VOCAB)
    assert isinstance(corpus["tier2_cues"].iloc[0], list)
    assert isinstance(corpus["tier3_cues"].iloc[0], list)
    assert corpus["was_sci_coded"].dtype.kind in ("i", "u", "b")

    assert {"term", "tier", "category", "variants"}.issubset(seed.columns)
    assert isinstance(seed["variants"].iloc[0], list)
    assert {"label", "type", "definition"}.issubset(cat.columns)


@pytest.mark.skipif(not _xlsx_available(), reason="source xlsx not present")
def test_run_writes_three_parquets_and_reread_passes_schema(tmp_path):
    paths = run(out_dir=tmp_path, verbose=False)
    assert set(paths) == {"corpus", "seed_words", "category_defs"}
    for p in paths.values():
        assert p.exists()
        assert p.stat().st_size > 0

    df = pd.read_parquet(paths["corpus"])
    required = {
        "utt_id", "utterance", "label", "setting", "source",
        "tier2_cues", "tier3_cues", "was_sci_coded",
    }
    assert required.issubset(df.columns)

    assert df["utt_id"].notna().all() and df["utt_id"].is_unique
    assert df["utterance"].notna().all()
    assert set(df["label"].dropna()).issubset({"SCIENCE_TALK", "NOT_SCIENCE_TALK"})
    assert set(df["setting"].dropna()).issubset(SETTING_VOCAB)

    sample_cue = df["tier2_cues"].iloc[0]
    assert isinstance(sample_cue, (list, np.ndarray))


@pytest.mark.skipif(not _xlsx_available(), reason="source xlsx not present")
def test_dedup_drops_duplicates_when_present(tmp_path):
    """Inject a duplicate row, run clean_corpus, verify it drops the dupe."""
    raw = pd.read_excel(DEFAULT_XLSX, sheet_name="Example utterances")
    dupe = raw.iloc[[0]].copy()
    raw_with_dup = pd.concat([raw, dupe], ignore_index=True)

    cleaned = clean_corpus(raw_with_dup, verbose=False)
    assert len(cleaned) == len(raw.dropna(subset=["Utterance"]).drop_duplicates(subset=["Utterance"]))


def test_validate_corpus_rejects_undersized_corpus():
    """validate_corpus should refuse to greenlight a too-small corpus."""
    tiny = pd.DataFrame({
        "utt_id": ["utt_0000"],
        "utterance": ["hi"],
        "label": ["SCIENCE_TALK"],
        "setting": ["Unknown"],
        "source": ["test"],
        "tier2_cues": [[]],
        "tier3_cues": [[]],
        "was_sci_coded": [0],
        "transcript_ref": [None],
        "topic": [None],
        "citation": [None],
    })
    with pytest.raises(AssertionError):
        validate_corpus(tiny)
