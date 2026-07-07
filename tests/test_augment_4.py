"""Unit tests for src/augment_4.py.

Covers each of Step 4's four DoD conditions explicitly:
    1. Each anchor has variants in >=2 registers different from source
    2. Tier2/Tier3 cues preserved in >=80% of variants
    3. No anchor leaks through unchanged
    4. Every row records model_id + prompt_version
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.augment_4 import (
    DEFAULT_LLM_MODEL,
    PRESERVATION_THRESHOLD,
    PROMPT_VERSIONS,
    REGISTERS,
    SETTING_TO_REGISTER,
    build_register_variant_prompt,
    build_variant_pairs,
    check_cue_preservation,
    differs_from_anchor,
    generate_variants_for_anchor,
    parse_variants_json,
    sample_pairs_for_review,
    setting_to_register,
    stub_llm_callable,
    target_registers_for_anchor,
    validate_variant_pairs,
    _as_list,
)
from src.data_loader_1 import DEFAULT_PROCESSED_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_corpus():
    """Mini corpus with one anchor per known setting + one Unknown."""
    return pd.DataFrame([
        {"utt_id": "u_lg",   "utterance": "I wonder what is inside this seed.",
         "label": "SCIENCE_TALK", "setting": "Large Group",
         "subtype": ["prediction"], "tier2_cues": ["wonder"], "tier3_cues": ["seed"]},
        {"utt_id": "u_sg",   "utterance": "Look how the wheel turns when we push it.",
         "label": "SCIENCE_TALK", "setting": "Small Group",
         "subtype": ["observation"], "tier2_cues": ["look"], "tier3_cues": ["wheel"]},
        {"utt_id": "u_ctr",  "utterance": "These two stick together because they are magnets.",
         "label": "SCIENCE_TALK", "setting": "Centers",
         "subtype": ["causal_reasoning"], "tier2_cues": ["because"], "tier3_cues": ["magnet"]},
        {"utt_id": "u_unk",  "utterance": "Plants need water and sunlight to grow.",
         "label": "SCIENCE_TALK", "setting": "Unknown",
         "subtype": ["content"], "tier2_cues": [], "tier3_cues": ["plant"]},
        {"utt_id": "u_blk",  "utterance": "The tower fell because the base was weak.",
         "label": "SCIENCE_TALK", "setting": "Blocks/Engineering",
         "subtype": ["causal_reasoning"], "tier2_cues": ["because"], "tier3_cues": []},
        {"utt_id": "u_neg",  "utterance": "Sit down please.",
         "label": "NOT_SCIENCE_TALK", "setting": "Large Group",
         "subtype": ["content"], "tier2_cues": [], "tier3_cues": []},
    ])


# ---------------------------------------------------------------------------
# _as_list helper (parquet-roundtrip safety)
# ---------------------------------------------------------------------------

def test_as_list_passes_through_lists():
    assert _as_list(["a", "b"]) == ["a", "b"]


def test_as_list_handles_none():
    assert _as_list(None) == []


def test_as_list_handles_nan():
    assert _as_list(float("nan")) == []


def test_as_list_handles_numpy_array():
    assert _as_list(np.array(["x", "y"])) == ["x", "y"]


# ---------------------------------------------------------------------------
# Setting / register helpers
# ---------------------------------------------------------------------------

def test_setting_to_register_known_values():
    assert setting_to_register("Large Group") == "LARGE_GROUP"
    assert setting_to_register("Small Group") == "SMALL_GROUP"
    assert setting_to_register("Centers") == "INFORMAL"
    assert setting_to_register("Blocks/Engineering") == "INFORMAL"


def test_setting_to_register_unknown_returns_none():
    assert setting_to_register("Unknown") is None
    assert setting_to_register(None) is None


def test_setting_to_register_strips_whitespace():
    assert setting_to_register("  Large Group  ") == "LARGE_GROUP"


def test_target_registers_for_known_source_excludes_source():
    assert set(target_registers_for_anchor("LARGE_GROUP")) == {"SMALL_GROUP", "INFORMAL"}
    assert set(target_registers_for_anchor("INFORMAL")) == {"LARGE_GROUP", "SMALL_GROUP"}


def test_target_registers_for_unknown_returns_all_three():
    assert set(target_registers_for_anchor(None)) == set(REGISTERS)


def test_known_source_yields_at_least_two_targets():
    """DoD #1 requires >=2 *different* target registers per anchor."""
    for r in REGISTERS:
        assert len(target_registers_for_anchor(r)) >= 2


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def test_prompt_includes_anchor_and_target():
    p = build_register_variant_prompt(
        "the seed sprouts", "INFORMAL",
        source_register="LARGE_GROUP",
        subtypes=["observation"],
        tier2_cues=["look"], tier3_cues=["seed"],
        n=2,
    )
    assert "the seed sprouts" in p
    assert "INFORMAL" in p
    assert "LARGE_GROUP" in p
    assert "Generate 2" in p
    assert "JSON" in p
    assert "'look'" in p and "'seed'" in p


def test_prompt_handles_no_cues():
    p = build_register_variant_prompt(
        "x", "LARGE_GROUP",
        source_register=None,
        subtypes=[], tier2_cues=[], tier3_cues=[],
        n=1,
    )
    assert "no required cues" in p


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def test_parse_variants_basic():
    raw = '{"variants": [{"text": "a", "self_score": 0.8}, {"text": "b", "self_score": 0.5}]}'
    out = parse_variants_json(raw)
    assert len(out) == 2
    assert out[0]["text"] == "a" and out[0]["self_score"] == 0.8


def test_parse_variants_handles_str_only_items():
    raw = '{"variants": ["plain string variant"]}'
    out = parse_variants_json(raw)
    assert out == [{"text": "plain string variant", "self_score": None}]


def test_parse_variants_clamps_score():
    raw = '{"variants": [{"text": "a", "self_score": 1.7}]}'
    assert parse_variants_json(raw)[0]["self_score"] == 1.0


def test_parse_variants_handles_garbage_score():
    raw = '{"variants": [{"text": "a", "self_score": "not-a-number"}]}'
    assert parse_variants_json(raw)[0]["self_score"] is None


def test_parse_variants_handles_surrounding_prose():
    raw = 'Here you go: {"variants": [{"text": "a", "self_score": 0.6}]} done.'
    out = parse_variants_json(raw)
    assert len(out) == 1


def test_parse_variants_skips_empty_text():
    raw = '{"variants": [{"text": "", "self_score": 0.5}, {"text": "ok", "self_score": 0.5}]}'
    out = parse_variants_json(raw)
    assert len(out) == 1 and out[0]["text"] == "ok"


def test_parse_variants_handles_garbage():
    assert parse_variants_json("nope") == []
    assert parse_variants_json("") == []


# ---------------------------------------------------------------------------
# Stub LLM callable
# ---------------------------------------------------------------------------

def test_stub_returns_valid_json_for_each_register():
    for reg in REGISTERS:
        prompt = build_register_variant_prompt(
            "look at this seed", reg,
            source_register=None, subtypes=["observation"],
            tier2_cues=["look"], tier3_cues=["seed"], n=1,
        )
        out = stub_llm_callable(prompt, PROMPT_VERSIONS["register_variant"])
        parsed = parse_variants_json(out)
        assert len(parsed) == 1
        assert parsed[0]["self_score"] is not None


def test_stub_is_deterministic():
    p = build_register_variant_prompt("x y z", "INFORMAL", source_register=None,
                                      subtypes=[], tier2_cues=[], tier3_cues=[], n=1)
    a = stub_llm_callable(p, PROMPT_VERSIONS["register_variant"])
    b = stub_llm_callable(p, PROMPT_VERSIONS["register_variant"])
    assert a == b


def test_stub_differs_from_anchor():
    """DoD #3 sanity: stub variants are not anchor-identical."""
    anchor = "look at this seed sprout"
    p = build_register_variant_prompt(anchor, "LARGE_GROUP", source_register=None,
                                      subtypes=[], tier2_cues=["look"], tier3_cues=["seed"], n=1)
    out = stub_llm_callable(p, PROMPT_VERSIONS["register_variant"])
    parsed = parse_variants_json(out)
    assert differs_from_anchor(parsed[0]["text"], anchor)


def test_stub_preserves_cues_when_present():
    """DoD #2 sanity: stub appends cues so preservation passes."""
    anchor = "tell me what you think"
    p = build_register_variant_prompt(anchor, "INFORMAL", source_register=None,
                                      subtypes=[], tier2_cues=["wonder"], tier3_cues=["force"], n=1)
    out = stub_llm_callable(p, PROMPT_VERSIONS["register_variant"])
    text = parse_variants_json(out)[0]["text"]
    _, _, passed = check_cue_preservation(text, ["wonder"], ["force"])
    assert passed


# ---------------------------------------------------------------------------
# Cue preservation
# ---------------------------------------------------------------------------

def test_cue_preservation_full():
    preserved, pct, passed = check_cue_preservation("the seed sprouts in soil",
                                                    ["look"], ["seed", "soil"])
    assert "seed" in preserved and "soil" in preserved
    assert pct == pytest.approx(2 / 3)
    assert not passed  # 0.667 < 0.80


def test_cue_preservation_passes_at_threshold():
    preserved, pct, passed = check_cue_preservation("look at the seed in the soil",
                                                    ["look"], ["seed", "soil"])
    assert pct == 1.0
    assert passed


def test_cue_preservation_vacuous_when_no_cues():
    preserved, pct, passed = check_cue_preservation("anything", [], [])
    assert preserved == [] and pct == 1.0 and passed


def test_cue_preservation_case_insensitive():
    _, pct, passed = check_cue_preservation("WONDER what THIS is", ["wonder"], [])
    assert pct == 1.0 and passed


def test_cue_preservation_handles_numpy_inputs():
    """parquet-roundtripped lists become numpy arrays; must still work."""
    preserved, pct, passed = check_cue_preservation(
        "look at the seed",
        np.array(["look"]), np.array(["seed"]),
    )
    assert pct == 1.0 and passed


# ---------------------------------------------------------------------------
# differs_from_anchor
# ---------------------------------------------------------------------------

def test_differs_basic():
    assert differs_from_anchor("hello world", "hi there")
    assert not differs_from_anchor("hello world", "hello world")


def test_differs_normalizes_whitespace_and_case():
    assert not differs_from_anchor("Hello   World", "hello world")
    assert not differs_from_anchor("  hello world  ", "hello world")


def test_differs_returns_false_for_empty():
    assert not differs_from_anchor("", "anything")
    assert not differs_from_anchor("   ", "anything")


# ---------------------------------------------------------------------------
# Per-anchor generator
# ---------------------------------------------------------------------------

def test_generate_for_known_setting_anchor_excludes_source(fake_corpus):
    anchor = fake_corpus[fake_corpus["utt_id"] == "u_lg"].iloc[0]
    rows = generate_variants_for_anchor(anchor, n_per_register=1, llm_callable=stub_llm_callable)
    assert {r["register"] for r in rows} == {"SMALL_GROUP", "INFORMAL"}
    assert all(r["source_register"] == "LARGE_GROUP" for r in rows)


def test_generate_for_unknown_setting_anchor_targets_all(fake_corpus):
    anchor = fake_corpus[fake_corpus["utt_id"] == "u_unk"].iloc[0]
    rows = generate_variants_for_anchor(anchor, n_per_register=1, llm_callable=stub_llm_callable)
    assert {r["register"] for r in rows} == set(REGISTERS)
    assert all(r["source_register"] is None for r in rows)


def test_generate_carries_anchor_metadata(fake_corpus):
    anchor = fake_corpus[fake_corpus["utt_id"] == "u_sg"].iloc[0]
    rows = generate_variants_for_anchor(anchor, n_per_register=1, llm_callable=stub_llm_callable)
    for r in rows:
        assert r["anchor_id"] == "u_sg"
        assert r["anchor_text"] == "Look how the wheel turns when we push it."
        assert r["subtype"] == ["observation"]
        assert r["prompt_version"] == PROMPT_VERSIONS["register_variant"]


# ---------------------------------------------------------------------------
# Orchestrator + DoD validation
# ---------------------------------------------------------------------------

def test_build_variant_pairs_skips_negatives(fake_corpus):
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    assert "u_neg" not in set(df["anchor_id"])


def test_build_variant_pairs_schema(fake_corpus):
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    expected = {
        "pair_id", "anchor_id", "anchor_text", "anchor_setting", "source_register",
        "variant_text", "register", "subtype",
        "preserved_cues", "preservation_pct", "preservation_check_passed",
        "differs_from_anchor",
        "llm_self_score", "prompt_version", "model_id", "created_at",
    }
    assert expected.issubset(df.columns)
    assert df["pair_id"].is_unique


def test_build_variant_pairs_dod_register_diversity(fake_corpus):
    """DoD #1: each anchor has variants in >=2 registers different from source."""
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    for anchor_id, grp in df.groupby("anchor_id"):
        src = grp["source_register"].iloc[0]
        targets_other_than_source = set(grp["register"]) - {src}
        assert len(targets_other_than_source) >= 2, f"anchor {anchor_id} fails DoD #1"


def test_build_variant_pairs_dod_preservation(fake_corpus):
    """DoD #2: >=80% of variants preserve their anchor's cues."""
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    assert df["preservation_check_passed"].mean() >= 0.80


def test_build_variant_pairs_dod_no_leak(fake_corpus):
    """DoD #3: no variant equals its anchor verbatim."""
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    assert df["differs_from_anchor"].all()


def test_build_variant_pairs_dod_provenance(fake_corpus):
    """DoD #4: every row records model_id and prompt_version."""
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    assert df["model_id"].notna().all()
    assert df["prompt_version"].notna().all()
    assert (df["model_id"] == DEFAULT_LLM_MODEL).all()


def test_build_variant_pairs_full_anchor_coverage(fake_corpus):
    """Every positive anchor should produce variants (parser must not silently fail)."""
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    n_positives = (fake_corpus["label"] == "SCIENCE_TALK").sum()
    assert df["anchor_id"].nunique() == n_positives


def test_build_variant_pairs_drops_leak_when_flag_set():
    """If a (degenerate) llm_callable returns the anchor verbatim, those rows
    are dropped before persistence."""
    def echo_llm(prompt, version):
        anchor = re.search(r'ANCHOR:\s+"([^"]+)"', prompt).group(1)
        return json.dumps({"variants": [{"text": anchor, "self_score": 1.0}]})

    import re
    df_corpus = pd.DataFrame([{
        "utt_id": "u_a", "utterance": "look at this", "label": "SCIENCE_TALK",
        "setting": "Large Group", "subtype": ["observation"],
        "tier2_cues": ["look"], "tier3_cues": [],
    }])
    df = build_variant_pairs(df_corpus, n_per_register=1, llm_callable=echo_llm,
                             drop_failed_leak_check=True, verbose=False)
    assert len(df) == 0 or df["differs_from_anchor"].all()


# ---------------------------------------------------------------------------
# Validator behaviour
# ---------------------------------------------------------------------------

def test_validate_passes_on_clean_pool(fake_corpus):
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    info = validate_variant_pairs(df, fake_corpus, verbose=False)
    assert info["leak_count"] == 0
    assert info["preservation_rate"] >= 0.80
    assert info["anchors_failing_register_diversity"] == 0
    assert info["anchor_coverage"] == 1.0


def test_validate_warns_when_register_dominates():
    """Synthetic 80/10/10 split should fire the dominance warning."""
    rows = []
    for i in range(80):
        rows.append({"pair_id": f"p{i}", "anchor_id": "a", "anchor_text": "x",
                     "anchor_setting": "Large Group", "source_register": None,
                     "variant_text": f"v{i}", "register": "INFORMAL",
                     "subtype": [], "preserved_cues": [], "preservation_pct": 1.0,
                     "preservation_check_passed": True, "differs_from_anchor": True,
                     "llm_self_score": 0.7, "prompt_version": "v", "model_id": "m"})
    for i in range(10):
        rows.append({"pair_id": f"q{i}", "anchor_id": "b", "anchor_text": "y",
                     "anchor_setting": "Large Group", "source_register": None,
                     "variant_text": f"v{i}", "register": "LARGE_GROUP",
                     "subtype": [], "preserved_cues": [], "preservation_pct": 1.0,
                     "preservation_check_passed": True, "differs_from_anchor": True,
                     "llm_self_score": 0.7, "prompt_version": "v", "model_id": "m"})
    for i in range(10):
        rows.append({"pair_id": f"r{i}", "anchor_id": "c", "anchor_text": "z",
                     "anchor_setting": "Large Group", "source_register": None,
                     "variant_text": f"v{i}", "register": "SMALL_GROUP",
                     "subtype": [], "preserved_cues": [], "preservation_pct": 1.0,
                     "preservation_check_passed": True, "differs_from_anchor": True,
                     "llm_self_score": 0.7, "prompt_version": "v", "model_id": "m"})
    df_pairs = pd.DataFrame(rows)
    df_corpus = pd.DataFrame([
        {"utt_id": "a", "label": "SCIENCE_TALK"},
        {"utt_id": "b", "label": "SCIENCE_TALK"},
        {"utt_id": "c", "label": "SCIENCE_TALK"},
    ])
    info = validate_variant_pairs(df_pairs, df_corpus, require_two_other_registers=False, verbose=False)
    assert any("dominat" in w for w in info["warnings"])


def test_validate_raises_on_surface_leak():
    df_pairs = pd.DataFrame([{
        "pair_id": "p0", "anchor_id": "a", "anchor_text": "hello", "anchor_setting": None,
        "source_register": None, "variant_text": "v", "register": "INFORMAL",
        "subtype": [], "preserved_cues": [], "preservation_pct": 1.0,
        "preservation_check_passed": True, "differs_from_anchor": False,
        "llm_self_score": 0.7, "prompt_version": "v", "model_id": "m",
    }])
    df_corpus = pd.DataFrame([{"utt_id": "a", "label": "SCIENCE_TALK"}])
    with pytest.raises(AssertionError, match="identical to their anchor"):
        validate_variant_pairs(df_pairs, df_corpus, require_two_other_registers=False, verbose=False)


def test_validate_raises_on_diversity_failure_when_required():
    """Anchor with only one variant and only one register should fail DoD #1."""
    df_pairs = pd.DataFrame([{
        "pair_id": "p0", "anchor_id": "a", "anchor_text": "x", "anchor_setting": "Large Group",
        "source_register": "LARGE_GROUP", "variant_text": "v", "register": "INFORMAL",
        "subtype": [], "preserved_cues": [], "preservation_pct": 1.0,
        "preservation_check_passed": True, "differs_from_anchor": True,
        "llm_self_score": 0.7, "prompt_version": "v", "model_id": "m",
    }])
    df_corpus = pd.DataFrame([{"utt_id": "a", "label": "SCIENCE_TALK"}])
    with pytest.raises(AssertionError, match="<2 variants in registers"):
        validate_variant_pairs(df_pairs, df_corpus,
                               require_two_other_registers=True, verbose=False)


# ---------------------------------------------------------------------------
# sample_pairs_for_review
# ---------------------------------------------------------------------------

def test_sample_pairs_returns_n_rows(fake_corpus):
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    sample = sample_pairs_for_review(df, n=min(6, len(df)))
    assert 1 <= len(sample) <= 6


def test_sample_pairs_is_deterministic(fake_corpus):
    df = build_variant_pairs(fake_corpus, n_per_register=1,
                             llm_callable=stub_llm_callable, verbose=False)
    a = sample_pairs_for_review(df, n=3, seed=99)
    b = sample_pairs_for_review(df, n=3, seed=99)
    assert list(a["pair_id"]) == list(b["pair_id"])


# ---------------------------------------------------------------------------
# End-to-end against the real on-disk corpus
# ---------------------------------------------------------------------------

def _corpus_ready():
    return (DEFAULT_PROCESSED_DIR / "corpus.parquet").exists()


@pytest.mark.skipif(not _corpus_ready(), reason="run Steps 1+2 first")
def test_run_step4_end_to_end_with_stub(tmp_path):
    """Copy corpus.parquet, run Step 4 with the stub, verify on-disk shape."""
    import shutil
    shutil.copy(DEFAULT_PROCESSED_DIR / "corpus.parquet", tmp_path / "corpus.parquet")

    from src.augment_4 import run as run_step4
    out_path = run_step4(processed_dir=tmp_path, use_real_llm=False,
                         n_per_register=1, verbose=False)
    assert out_path.exists()

    df = pd.read_parquet(out_path)
    assert df["pair_id"].is_unique
    assert df["differs_from_anchor"].all()
    assert df["model_id"].notna().all()
    assert df["prompt_version"].notna().all()
    assert df["preservation_check_passed"].mean() >= 0.80
