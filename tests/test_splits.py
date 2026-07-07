"""Unit tests for src/splits.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.splits import (
    DEFAULT_RATIOS,
    INFORMAL_SETTINGS,
    SPLIT_NAMES,
    _primary_subtype,
    _stratify_one,
    make_hard_informal_slice,
    make_splits,
    run,
    summarize,
)


@pytest.fixture
def fake_corpus():
    return pd.DataFrame([
        {"utt_id": f"utt_{i:04d}",
         "label": "SCIENCE_TALK",
         "subtype": [["observation"], ["prediction"], ["content"], ["evidence"],
                     ["causal_reasoning"]][i % 5]}
        for i in range(50)
    ])


@pytest.fixture
def fake_negatives():
    src_types = ["transcript_clean", "llm_hard_negative", "seed_word_nonscience"]
    return pd.DataFrame([
        {"neg_id": f"neg_{i:05d}",
         "text": f"negative number {i}",
         "source_type": src_types[i % 3],
         "subtype": [["not_science_shape"], ["observation"]][i % 2]}
        for i in range(60)
    ])


@pytest.fixture
def fake_pairs():
    return pd.DataFrame([
        {"pair_id": f"pair_{i:05d}",
         "anchor_id": f"utt_{i % 10:04d}",  # 10 anchors, 3 variants each
         "register": "INFORMAL"}
        for i in range(30)
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestPrimarySubtype:
    def test_picks_lex_smallest_for_multi_label(self):
        assert _primary_subtype(["observation", "evidence"]) == "evidence"
        assert _primary_subtype(["content"]) == "content"

    def test_returns_unknown_on_empty(self):
        assert _primary_subtype([]) == "unknown"
        assert _primary_subtype(None) == "unknown"

    def test_handles_numpy_array(self):
        assert _primary_subtype(np.array(["prediction", "observation"])) == "observation"


class TestStratifyOne:
    def test_each_group_split_proportionally(self):
        keys = pd.Series(["A"] * 100 + ["B"] * 50)
        splits = _stratify_one(keys, ratios=(0.6, 0.2, 0.2), seed=0)
        a_split = splits[keys == "A"].value_counts()
        b_split = splits[keys == "B"].value_counts()
        # A: 60/20/20, B: 30/10/10
        assert a_split.get("train", 0) == 60
        assert a_split.get("val", 0) == 20
        assert a_split.get("test", 0) == 20
        assert b_split.get("train", 0) == 30
        assert b_split.get("val", 0) == 10
        assert b_split.get("test", 0) == 10

    def test_deterministic_under_same_seed(self):
        keys = pd.Series(list("AABBCC") * 10)
        s1 = _stratify_one(keys, ratios=(0.7, 0.15, 0.15), seed=42)
        s2 = _stratify_one(keys, ratios=(0.7, 0.15, 0.15), seed=42)
        assert (s1 == s2).all()


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------

class TestMakeSplits:
    def test_includes_all_kinds(self, fake_corpus, fake_negatives, fake_pairs):
        df = make_splits(fake_corpus, fake_negatives, fake_pairs)
        assert set(df["kind"]) == {"positive", "negative", "pair"}
        assert len(df) == len(fake_corpus) + len(fake_negatives) + len(fake_pairs)

    def test_split_values_are_valid(self, fake_corpus, fake_negatives, fake_pairs):
        df = make_splits(fake_corpus, fake_negatives, fake_pairs)
        assert set(df["split"]).issubset(set(SPLIT_NAMES))

    def test_pairs_co_locate_with_anchor(self, fake_corpus, fake_negatives, fake_pairs):
        df = make_splits(fake_corpus, fake_negatives, fake_pairs)
        # Every pair's split must equal its anchor's split. This prevents
        # leakage where (anchor, variant_A) is in train and (anchor, variant_B)
        # is in val -- the bi-encoder would score itself.
        anchor_split = dict(zip(
            df.loc[df["kind"] == "positive", "id"],
            df.loc[df["kind"] == "positive", "split"],
        ))
        pair_rows = df[df["kind"] == "pair"]
        for _, row in pair_rows.iterrows():
            assert row["split"] == anchor_split[row["anchor_id"]], \
                f"Pair {row['id']} in {row['split']} but anchor in {anchor_split[row['anchor_id']]}"

    def test_ratios_must_sum_to_one(self, fake_corpus, fake_negatives):
        with pytest.raises(ValueError, match="ratios must sum"):
            make_splits(fake_corpus, fake_negatives, ratios=(0.5, 0.3, 0.3))

    def test_negatives_stratified_jointly_on_source_and_subtype(self, fake_negatives):
        # Build a tiny corpus so stratification has rows to work with
        corpus = pd.DataFrame([
            {"utt_id": "utt_0001", "label": "SCIENCE_TALK", "subtype": ["content"]}
        ])
        df = make_splits(corpus, fake_negatives, ratios=(0.6, 0.2, 0.2))
        neg_rows = df[df["kind"] == "negative"]
        # Every (source_type x subtype) bucket should have at least 1 row in train
        # (60% of 10 rows per bucket = 6 rows in train)
        keys = pd.Series([k for k in neg_rows["stratify_key"]])
        for k in keys.unique():
            sub_splits = neg_rows.loc[neg_rows["stratify_key"] == k, "split"].value_counts()
            assert sub_splits.get("train", 0) > 0, f"bucket {k} has no train rows"

    def test_empty_pairs_optional(self, fake_corpus, fake_negatives):
        df = make_splits(fake_corpus, fake_negatives, df_pairs=None)
        assert "pair" not in set(df["kind"])
        assert len(df) == len(fake_corpus) + len(fake_negatives)


class TestSummarize:
    def test_summarize_returns_string_with_split_names(self, fake_corpus, fake_negatives):
        df = make_splits(fake_corpus, fake_negatives)
        out = summarize(df)
        for name in SPLIT_NAMES:
            assert name in out
        assert "By kind" in out


# ---------------------------------------------------------------------------
# Hard-informal slice
# ---------------------------------------------------------------------------

@pytest.fixture
def slice_corpus():
    # 6 informal-setting positives, 14 other positives (mix of known/unknown).
    rows = []
    for i in range(6):
        rows.append({
            "utt_id": f"inf_{i:02d}",
            "utterance": f"informal utterance {i}",
            "setting": INFORMAL_SETTINGS[i % len(INFORMAL_SETTINGS)],
            "label": "SCIENCE_TALK",
            "subtype": ["observation"],
        })
    for i in range(14):
        rows.append({
            "utt_id": f"oth_{i:02d}",
            "utterance": f"other utterance {i}",
            "setting": "Unknown" if i % 2 else "Large Group",
            "label": "SCIENCE_TALK",
            "subtype": ["content"],
        })
    return pd.DataFrame(rows)


@pytest.fixture
def slice_pairs():
    # informal positives generate LARGE_GROUP/SMALL_GROUP variants;
    # 'oth_' anchors generate INFORMAL variants (the variant-family candidates).
    rows = []
    for i in range(6):
        rows.append({
            "pair_id": f"p_inf_{i:02d}", "anchor_id": f"inf_{i:02d}",
            "variant_text": f"formalized {i}", "register": "LARGE_GROUP",
            "anchor_setting": INFORMAL_SETTINGS[i % len(INFORMAL_SETTINGS)],
            "subtype": ["observation"],
        })
    for i in range(14):
        rows.append({
            "pair_id": f"p_oth_{i:02d}", "anchor_id": f"oth_{i:02d}",
            "variant_text": f"casualized {i}", "register": "INFORMAL",
            "anchor_setting": "Unknown",
            "subtype": ["content"],
        })
    return pd.DataFrame(rows)


class TestHardInformalSlice:
    def test_all_informal_positives_held_out(self, slice_corpus, slice_pairs):
        slice_df, held = make_hard_informal_slice(
            slice_corpus, slice_pairs, n_variant_families=0, seed=1)
        # All 6 informal-setting positives must be in the held-out set.
        for i in range(6):
            assert f"inf_{i:02d}" in held
        reasons = set(slice_df.loc[slice_df["kind"] == "positive", "slice_reason"])
        assert "informal_positive" in reasons

    def test_variant_families_sampled_and_whole_family_held(self, slice_corpus, slice_pairs):
        slice_df, held = make_hard_informal_slice(
            slice_corpus, slice_pairs, n_variant_families=5, seed=1)
        # 6 informal + 5 sampled variant families
        assert len(held) == 11
        sampled = [a for a in held if a.startswith("oth_")]
        assert len(sampled) == 5
        # Whole family held: every pair of a held anchor is in the slice.
        slice_pair_anchors = set(slice_df.loc[slice_df["kind"] == "pair", "anchor_id"])
        for a in held:
            anchor_pairs = set(slice_pairs.loc[slice_pairs["anchor_id"] == a, "pair_id"])
            in_slice = set(slice_df.loc[slice_df["id"].isin(anchor_pairs), "id"])
            assert anchor_pairs == in_slice

    def test_deterministic_under_seed(self, slice_corpus, slice_pairs):
        h1 = make_hard_informal_slice(slice_corpus, slice_pairs, n_variant_families=5, seed=7)[1]
        h2 = make_hard_informal_slice(slice_corpus, slice_pairs, n_variant_families=5, seed=7)[1]
        assert h1 == h2

    def test_no_leakage_between_slice_and_pool(self, slice_corpus, slice_pairs):
        slice_df, held = make_hard_informal_slice(
            slice_corpus, slice_pairs, n_variant_families=5, seed=3)
        pool_anchors = set(slice_corpus.loc[~slice_corpus["utt_id"].isin(held), "utt_id"])
        assert pool_anchors.isdisjoint(held)
        slice_anchors = set(slice_df["anchor_id"])
        assert slice_anchors.isdisjoint(pool_anchors)


class TestRunIntegration:
    def test_run_filters_rejects_and_writes_slice(
        self, slice_corpus, slice_pairs, fake_negatives, tmp_path
    ):
        # Mark one negative rejected so filter_verified drops it.
        neg = fake_negatives.copy()
        neg["decision"] = None
        neg["routing"] = "auto"
        neg.loc[0, "decision"] = "reject"
        neg.to_parquet(tmp_path / "negatives.parquet", index=False)
        slice_corpus.to_parquet(tmp_path / "corpus.parquet", index=False)
        slice_pairs.to_parquet(tmp_path / "pairs.parquet", index=False)

        out = run(tmp_path, n_variant_families=3, seed=1, verbose=False)
        assert out.exists()
        slice_path = tmp_path / "hard_informal_slice.parquet"
        assert slice_path.exists()

        splits = pd.read_parquet(out)
        slice_df = pd.read_parquet(slice_path)

        # Rejected negative is gone from splits.
        assert "neg_00000" not in set(splits["id"])
        # Slice ids never appear in train/val/test splits (no leakage).
        assert set(slice_df["id"]).isdisjoint(set(splits["id"]))
        # Splits only contain train/val/test.
        assert set(splits["split"]).issubset(set(SPLIT_NAMES))
