"""Unit tests for src/negatives_3.py.

Covers Step 3 DoD:
    * three negative source types are representable (function exists for each;
      transcript path tested with synthetic transcripts)
    * negative pool is structurally well-formed and validates against the
      target ratio
    * sample_for_review + compute_review_pass_rate work for the 50-row
      hand-check loop
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader_1 import DEFAULT_PROCESSED_DIR
from src.negatives_3 import (
    DEFAULT_LLM_MODEL,
    NEG_REVIEW_LABELS,
    NEGATIVE_SOURCE_TYPES,
    PROMPT_VERSIONS,
    TRANSCRIPT_SUBTYPES,
    apply_negatives_review,
    assign_subtypes_to_negatives,
    build_gate_prompt,
    build_hard_negative_prompt,
    build_negative_pool,
    build_seed_nonscience_prompt,
    compute_review_pass_rate,
    generate_hard_negatives_for_positive,
    generate_seed_nonscience_for_term,
    mine_transcript_negatives,
    mine_transcript_xlsx,
    parse_gate_score,
    parse_negatives_json,
    passes_structural_checks,
    sample_for_review,
    stub_llm_callable,
    validate_negatives,
    validate_negatives_review,
)
from src.subtypes_2 import LLM_STUB_PROMPT_VERSION
from src.subtypes_2 import build_seed_index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_corpus():
    return pd.DataFrame([
        {"utt_id": "utt_0000", "utterance": "Look at the seed",
         "label": "SCIENCE_TALK", "subtype": ["content", "observation"],
         "tier2_cues": ["notice"], "tier3_cues": ["seed"]},
        {"utt_id": "utt_0001", "utterance": "I wonder why it sank",
         "label": "SCIENCE_TALK", "subtype": ["prediction"],
         "tier2_cues": ["wonder"], "tier3_cues": []},
        {"utt_id": "utt_0002", "utterance": "Sit down please",
         "label": "NOT_SCIENCE_TALK", "subtype": ["observation"],
         "tier2_cues": [], "tier3_cues": []},
    ])


@pytest.fixture
def fake_seed():
    return pd.DataFrame([
        {"term": "seed", "tier": "TIER3", "category": "LIFE_SCIENCE_PLANTS",
         "variants": []},
        {"term": "force", "tier": "TIER3", "category": "PHYSICAL_SCIENCE_MECHANISMS",
         "variants": []},
        {"term": "wonder", "tier": "TIER2", "category": "ASK_QUESTION_FRAME",
         "variants": ["I wonder"]},
    ])


@pytest.fixture
def fake_cat():
    return pd.DataFrame([
        {"label": "LIFE_SCIENCE_PLANTS", "definition": "Words about living things, plants."},
        {"label": "PHYSICAL_SCIENCE_MECHANISMS", "definition": "Words about forces and machines."},
        {"label": "ASK_QUESTION_FRAME", "definition": "Question starters that open inquiry."},
    ])


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def test_parse_negatives_json_simple():
    raw = '{"negatives": ["a", "b", "c"]}'
    assert parse_negatives_json(raw) == ["a", "b", "c"]


def test_parse_negatives_json_with_surrounding_prose():
    raw = 'Sure! Here you go: {"negatives": ["x", "y"]} done.'
    assert parse_negatives_json(raw) == ["x", "y"]


def test_parse_negatives_json_handles_invalid():
    assert parse_negatives_json("garbage no json") == []
    assert parse_negatives_json("") == []


def test_parse_negatives_json_strips_whitespace():
    raw = '{"negatives": ["  hello  ", "world"]}'
    assert parse_negatives_json(raw) == ["hello", "world"]


def test_parse_gate_score_finds_decimal():
    assert parse_gate_score("0.83") == pytest.approx(0.83)
    assert parse_gate_score("Score: 0.95 -- definitely") == pytest.approx(0.95)


def test_parse_gate_score_clamps_to_unit():
    assert parse_gate_score("1.5") == 1.0
    assert parse_gate_score("-0.2") is not None  # negative number not matched by regex


def test_parse_gate_score_returns_none_for_garbage():
    assert parse_gate_score("no number here") is None
    assert parse_gate_score("") is None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_hard_negative_prompt_includes_inputs():
    p = build_hard_negative_prompt("look at the seed", ["content", "observation"], 3)
    assert "look at the seed" in p
    assert "Generate 3" in p
    assert "JSON" in p


def test_seed_nonscience_prompt_includes_inputs():
    p = build_seed_nonscience_prompt("force", "PHYSICAL_SCIENCE_MECHANISMS", "definition x", 2)
    assert "force" in p
    assert "PHYSICAL_SCIENCE_MECHANISMS" in p
    assert "definition x" in p


def test_gate_prompt_quotes_text():
    p = build_gate_prompt("hello world")
    assert "hello world" in p


# ---------------------------------------------------------------------------
# Stub LLM behaviour
# ---------------------------------------------------------------------------

def test_stub_llm_returns_valid_json_for_neg_prompt():
    out = stub_llm_callable("Generate 2 hard negatives for: 'foo'", PROMPT_VERSIONS["hard_negative"])
    parsed = parse_negatives_json(out)
    assert isinstance(parsed, list) and len(parsed) >= 1


def test_stub_llm_returns_decimal_for_gate_prompt():
    out = stub_llm_callable("Is this science?", PROMPT_VERSIONS["gate_score"])
    score = parse_gate_score(out)
    assert score is not None and 0.0 <= score <= 1.0


def test_stub_llm_is_deterministic():
    a = stub_llm_callable("same prompt", PROMPT_VERSIONS["hard_negative"])
    b = stub_llm_callable("same prompt", PROMPT_VERSIONS["hard_negative"])
    assert a == b


def test_stub_llm_seed_nonscience_includes_term():
    """For seed_nonscience prompts the stub should weave the term in."""
    prompt = build_seed_nonscience_prompt("force", "PHYSICAL_SCIENCE_MECHANISMS", "def", 2)
    out = stub_llm_callable(prompt, PROMPT_VERSIONS["seed_nonscience"])
    parsed = parse_negatives_json(out)
    assert any("force" in n.lower() for n in parsed)


# ---------------------------------------------------------------------------
# Per-source generators
# ---------------------------------------------------------------------------

def test_generate_hard_negatives_attaches_anchor(fake_corpus):
    pos_row = fake_corpus.iloc[0]
    out = generate_hard_negatives_for_positive(pos_row, n=2, llm_callable=stub_llm_callable)
    assert len(out) >= 1
    assert all(r["source_type"] == "llm_hard_negative" for r in out)
    assert all(r["anchor_utt_id"] == "utt_0000" for r in out)
    assert all(r["anchor_seed_term"] is None for r in out)
    assert all(r["prompt_version"] == PROMPT_VERSIONS["hard_negative"] for r in out)


def test_generate_seed_nonscience_attaches_anchor(fake_seed):
    seed_row = fake_seed.iloc[1]  # 'force'
    out = generate_seed_nonscience_for_term(
        seed_row, {"PHYSICAL_SCIENCE_MECHANISMS": "def"}, n=2, llm_callable=stub_llm_callable,
    )
    assert len(out) >= 1
    assert all(r["source_type"] == "seed_word_nonscience" for r in out)
    assert all(r["anchor_seed_term"] == "force" for r in out)
    assert all(r["anchor_utt_id"] is None for r in out)


def test_mine_transcript_negatives_filters_seed_lines(tmp_path, fake_seed):
    transcript = tmp_path / "demo.txt"
    transcript.write_text(
        "All fruits have seeds, okay\n"   # contains 'seed' -- should be filtered
        "Please put your shoes on\n"      # clean
        "Look at this beautiful flower\n" # clean (no seed words)
        "I wonder what time it is\n"      # contains 'wonder' -- filtered
        "\n"
        "Walk in a line please\n",        # clean
        encoding="utf-8",
    )
    seed_index = build_seed_index(fake_seed)
    out = mine_transcript_negatives([transcript], seed_index)
    texts = [r["text"] for r in out]
    assert "Please put your shoes on" in texts
    assert "Look at this beautiful flower" in texts
    assert "Walk in a line please" in texts
    assert all("seed" not in t.lower() for t in texts)
    assert all("wonder" not in t.lower() for t in texts)


def test_mine_transcript_negatives_carries_provenance(tmp_path, fake_seed):
    transcript = tmp_path / "demo.txt"
    transcript.write_text("Walking feet inside\nUse kind words please\n", encoding="utf-8")
    out = mine_transcript_negatives([transcript], build_seed_index(fake_seed))
    assert all(r["source_type"] == "transcript_clean" for r in out)
    assert all("transcript_ref" in r for r in out)


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def test_passes_structural_checks_basic():
    assert passes_structural_checks("hello world friends")
    assert not passes_structural_checks("")
    assert not passes_structural_checks("hi")  # < 2 words after split? "hi" is 1 word
    assert not passes_structural_checks(" " * 5)


def test_passes_structural_checks_seed_required():
    """seed_word_nonscience must contain the seed term."""
    assert passes_structural_checks("please use force gently", anchor_seed_term="force")
    assert not passes_structural_checks("please be gentle", anchor_seed_term="force")


def test_passes_structural_checks_rejects_too_long():
    too_long = "word " * 50
    assert not passes_structural_checks(too_long)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_build_negative_pool_runs_with_stub(fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(
        fake_corpus, fake_seed, fake_cat,
        n_hard_per_positive=2, n_per_seed_term=2,
        llm_callable=stub_llm_callable, verbose=False,
    )
    assert len(df) > 0
    expected_cols = {
        "neg_id", "text", "source_type", "subtype",
        "anchor_utt_id", "anchor_seed_term",
        "structural_check_passed", "llm_gate_score",
        "model_id", "prompt_version",
        "human_verified", "human_label",
    }
    assert expected_cols.issubset(df.columns)


def test_build_negative_pool_neg_ids_are_unique(fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(fake_corpus, fake_seed, fake_cat,
                             llm_callable=stub_llm_callable, verbose=False)
    assert df["neg_id"].is_unique


def test_build_negative_pool_assigns_subtypes(fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(fake_corpus, fake_seed, fake_cat,
                             llm_callable=stub_llm_callable, verbose=False)
    # Every row should have at least one subtype assigned.
    assert all(len(s) >= 1 for s in df["subtype"])


def test_build_negative_pool_includes_transcript_clean_when_supplied(
    tmp_path, fake_corpus, fake_seed, fake_cat,
):
    transcript = tmp_path / "demo.txt"
    transcript.write_text(
        "Use kind words please\nWalking feet inside\nClean up your toys\n",
        encoding="utf-8",
    )
    df = build_negative_pool(
        fake_corpus, fake_seed, fake_cat,
        transcript_paths=[transcript],
        llm_callable=stub_llm_callable, verbose=False,
    )
    assert "transcript_clean" in set(df["source_type"])


def test_build_negative_pool_dedups(fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(fake_corpus, fake_seed, fake_cat,
                             llm_callable=stub_llm_callable, verbose=False)
    assert df["text"].is_unique


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_warns_when_under_target(fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(fake_corpus, fake_seed, fake_cat,
                             n_hard_per_positive=1, n_per_seed_term=1,
                             llm_callable=stub_llm_callable, verbose=False)
    info = validate_negatives(df, n_positives=2, target_ratio=10.0, verbose=False)
    assert any("target" in w.lower() for w in info["warnings"])


def test_validate_passes_when_two_sources_present(fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(fake_corpus, fake_seed, fake_cat,
                             llm_callable=stub_llm_callable, verbose=False)
    info = validate_negatives(df, n_positives=2, target_ratio=0.5, verbose=False,
                              require_three_sources=False)
    assert info["n_negatives"] >= 1


def test_validate_can_require_three_sources(fake_corpus, fake_seed, fake_cat):
    """Without transcripts, default mode should hard-fail when require_three_sources=True."""
    df = build_negative_pool(fake_corpus, fake_seed, fake_cat,
                             llm_callable=stub_llm_callable, verbose=False)
    with pytest.raises(AssertionError):
        validate_negatives(df, n_positives=2, require_three_sources=True, verbose=False)


# ---------------------------------------------------------------------------
# Sampling for review + pass rate
# ---------------------------------------------------------------------------

def test_sample_for_review_returns_n_rows(fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(fake_corpus, fake_seed, fake_cat,
                             llm_callable=stub_llm_callable, verbose=False)
    sample = sample_for_review(df, n=min(10, len(df)))
    assert 1 <= len(sample) <= 10


def test_sample_for_review_is_deterministic(fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(fake_corpus, fake_seed, fake_cat,
                             llm_callable=stub_llm_callable, verbose=False)
    a = sample_for_review(df, n=5, seed=123)
    b = sample_for_review(df, n=5, seed=123)
    assert list(a["neg_id"]) == list(b["neg_id"])


def test_compute_review_pass_rate_handles_no_reviews():
    df = pd.DataFrame({"human_verified": [False, False], "human_label": [pd.NA, pd.NA]})
    info = compute_review_pass_rate(df)
    assert info["n_reviewed"] == 0
    assert info["pass_rate"] is None


def test_compute_review_pass_rate_computes_correctly():
    df = pd.DataFrame({
        "human_verified": [True, True, True, True, False],
        "human_label": ["NOT_SCIENCE_TALK", "NOT_SCIENCE_TALK", "SCIENCE_TALK",
                        "NOT_SCIENCE_TALK", pd.NA],
    })
    info = compute_review_pass_rate(df)
    assert info["n_reviewed"] == 4
    assert info["n_true_negative"] == 3
    assert info["pass_rate"] == 0.75


# ---------------------------------------------------------------------------
# Applying a filled negatives-quality review sheet
# ---------------------------------------------------------------------------

def _make_neg_pool(tmp_path) -> Path:
    df = pd.DataFrame({
        "neg_id": ["neg_00000", "neg_00001", "neg_00002", "neg_00003"],
        "text": ["a", "b", "c", "d"],
        "source_type": ["transcript_clean"] * 4,
        "human_verified": [False, False, False, False],
        "human_label": [pd.NA, pd.NA, pd.NA, pd.NA],
    })
    p = tmp_path / "negatives.parquet"
    df.to_parquet(p, index=False)
    return p


def test_validate_negatives_review_rejects_bad_label():
    df = pd.DataFrame({"neg_id": ["neg_00000"], "review_human_label": ["BOGUS"]})
    with pytest.raises(ValueError):
        validate_negatives_review(df)


def test_validate_negatives_review_skips_blank():
    df = pd.DataFrame({
        "neg_id": ["neg_00000", "neg_00001"],
        "review_human_label": ["NOT_SCIENCE_TALK", pd.NA],
    })
    reviewed = validate_negatives_review(df)
    assert len(reviewed) == 1
    assert reviewed.iloc[0]["neg_id"] == "neg_00000"


def test_apply_negatives_review_drops_leak_keeps_rest(tmp_path):
    _make_neg_pool(tmp_path)
    tmpl = pd.DataFrame({
        "neg_id": ["neg_00000", "neg_00001", "neg_00002", "neg_00003"],
        "review_human_label": ["NOT_SCIENCE_TALK", "POSSIBLE_LEAK",
                               "UNCLEAR", pd.NA],
    })
    tpath = tmp_path / "negatives_review.xlsx"
    tmpl.to_excel(tpath, index=False)

    stats = apply_negatives_review(tpath, tmp_path, verbose=False)
    assert stats["n_reviewed"] == 3
    assert stats["dropped"] == 1
    assert stats["neg_after"] == 3

    out = pd.read_parquet(tmp_path / "negatives.parquet")
    assert "neg_00001" not in set(out["neg_id"])  # leak removed
    assert "neg_00002" in set(out["neg_id"])       # unclear kept by default
    confirmed = out[out["neg_id"] == "neg_00000"].iloc[0]
    assert bool(confirmed["human_verified"]) is True
    assert confirmed["human_label"] == "NOT_SCIENCE_TALK"


def test_apply_negatives_review_drop_unclear(tmp_path):
    _make_neg_pool(tmp_path)
    tmpl = pd.DataFrame({
        "neg_id": ["neg_00001", "neg_00002"],
        "review_human_label": ["POSSIBLE_LEAK", "UNCLEAR"],
    })
    tpath = tmp_path / "negatives_review.parquet"
    tmpl.to_parquet(tpath, index=False)

    stats = apply_negatives_review(tpath, tmp_path, drop_unclear=True, verbose=False)
    assert stats["dropped"] == 2
    out = pd.read_parquet(tmp_path / "negatives.parquet")
    assert "neg_00002" not in set(out["neg_id"])


def test_apply_negatives_review_backs_up(tmp_path):
    _make_neg_pool(tmp_path)
    tmpl = pd.DataFrame({
        "neg_id": ["neg_00001"], "review_human_label": ["POSSIBLE_LEAK"],
    })
    tpath = tmp_path / "negatives_review.xlsx"
    tmpl.to_excel(tpath, index=False)
    apply_negatives_review(tpath, tmp_path, verbose=False)
    backups = list(tmp_path.glob("negatives.pre_review.*.parquet"))
    assert len(backups) == 1
    assert len(pd.read_parquet(backups[0])) == 4  # original preserved


def test_apply_negatives_review_reports_pass_rate(tmp_path):
    _make_neg_pool(tmp_path)
    tmpl = pd.DataFrame({
        "neg_id": ["neg_00000", "neg_00001"],
        "review_human_label": ["NOT_SCIENCE_TALK", "NOT_SCIENCE_TALK"],
    })
    tpath = tmp_path / "negatives_review.xlsx"
    tmpl.to_excel(tpath, index=False)
    stats = apply_negatives_review(tpath, tmp_path, verbose=False)
    assert stats["pass_rate"] == 1.0
    assert stats["dropped"] == 0


def test_apply_negatives_review_flags_missing_ids(tmp_path):
    _make_neg_pool(tmp_path)
    tmpl = pd.DataFrame({
        "neg_id": ["neg_99999"], "review_human_label": ["NOT_SCIENCE_TALK"],
    })
    tpath = tmp_path / "negatives_review.xlsx"
    tmpl.to_excel(tpath, index=False)
    stats = apply_negatives_review(tpath, tmp_path, verbose=False)
    assert "neg_99999" in stats["missing_ids"]


# ---------------------------------------------------------------------------
# End-to-end against the real on-disk corpus (skip if missing)
# ---------------------------------------------------------------------------

def _processed_ready():
    return all((DEFAULT_PROCESSED_DIR / fn).exists()
               for fn in ("corpus.parquet", "seed_words.parquet", "category_defs.parquet"))


@pytest.mark.skipif(not _processed_ready(),
                    reason="run Steps 1+2 first")
def test_run_step3_end_to_end_with_stub(tmp_path):
    """Copy the real parquets, run Step 3 with the stub, verify on-disk shape."""
    import shutil
    for fn in ("corpus.parquet", "seed_words.parquet", "category_defs.parquet"):
        shutil.copy(DEFAULT_PROCESSED_DIR / fn, tmp_path / fn)

    from src.negatives_3 import run as run_step3
    out_path = run_step3(processed_dir=tmp_path, use_real_llm=False,
                         n_hard_per_positive=1, n_per_seed_term=1,
                         seed_term_sample=10,
                         transcript_xlsx=None,
                         max_transcript_negatives=50,
                         verbose=False)
    assert out_path.exists()

    df = pd.read_parquet(out_path)
    assert {"neg_id", "text", "source_type", "subtype",
            "transcript_subtype", "transcript_code", "transcript_ref",
            "structural_check_passed"}.issubset(df.columns)
    assert df["neg_id"].is_unique
    assert set(df["source_type"]).issubset(NEGATIVE_SOURCE_TYPES)
    assert all(len(s) >= 1 for s in df["subtype"])


# ---------------------------------------------------------------------------
# xlsx transcript mining
# ---------------------------------------------------------------------------

def _write_synthetic_workbook(path: Path, sheets: dict[str, list[list]]) -> None:
    """sheets: {sheet_name: [[row0_values], [row1_values], ...]}"""
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for row in rows:
            ws.append(row)
    wb.save(path)


@pytest.fixture
def synthetic_transcript_xlsx(tmp_path):
    """A 3-sheet workbook mimicking 'Coding Transcripts.xlsx' columns A..G."""
    p = tmp_path / "synthetic_transcripts.xlsx"
    header = ["ID", "Number", "Speaker", "Utterance",
              "Behavior-disapproving?", "Language-building", "Content type (CT ONLY)"]
    sheets = {
        "TOC": [["Worksheet", "Date", "Contents"]],
        "01-99C": [
            header,
            ["01-99C", 1, "T1:", "Plants need water to grow.",      None, "CT", "SCI"],
            ["01-99C", 2, "T1:", "Sit down on the carpet please.",  "Y",  None, None],
            ["01-99C", 3, "T1:", "Let's count to three together.",  None, None, None],
            ["01-99C", 4, "C3:", "I want a turn next.",             None, None, None],
            ["01-99C", 5, "T2:", "Quiet voices in the hallway.",    "Y",  None, None],
            ["01-99C", 6, "T:",  "Open your books to page ten.",    None, None, None],
            ["01-99C", 7, "Persephone:", "I drew a butterfly!",     None, None, None],
            ["01-99C", 8, "T1",  "Today we will talk about forces.", None, "CT", "SCI"],
            ["01-99C", 9, " T7:", "Pencils in your boxes please.",  None, None, None],
        ],
        "02-99LG": [
            header,
            ["02-99LG", 1, "T9:", "Look at how the seed sprouts.",  None, "CT", "SCI"],
            ["02-99LG", 2, "T9:", "Walk in a line to the bathroom.", "Y", None, None],
        ],
    }
    _write_synthetic_workbook(p, sheets)
    return p


def test_xlsx_filters_out_sci_rows(synthetic_transcript_xlsx, fake_seed):
    out = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                               verbose=False)
    texts = [r["text"] for r in out]
    assert "Plants need water to grow." not in texts
    assert "Today we will talk about forces." not in texts
    assert "Look at how the seed sprouts." not in texts


def test_xlsx_keeps_only_teacher_speakers(synthetic_transcript_xlsx, fake_seed):
    out = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                               verbose=False)
    texts = [r["text"] for r in out]
    assert "I want a turn next." not in texts          # C3 child
    assert "I drew a butterfly!" not in texts          # named child


def test_xlsx_speaker_pattern_handles_variants(synthetic_transcript_xlsx, fake_seed):
    """T1, T1:, T:, ' T7:' all count as teachers."""
    out = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                               verbose=False)
    texts = [r["text"] for r in out]
    assert "Open your books to page ten." in texts          # T:
    assert "Pencils in your boxes please." in texts         # ' T7:' with leading space


def test_xlsx_tags_behavior_disapproving(synthetic_transcript_xlsx, fake_seed):
    out = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                               verbose=False)
    by_text = {r["text"]: r for r in out}
    assert by_text["Sit down on the carpet please."]["transcript_subtype"] == "behavior_disapproving"
    assert by_text["Quiet voices in the hallway."]["transcript_subtype"] == "behavior_disapproving"
    assert by_text["Walk in a line to the bathroom."]["transcript_subtype"] == "behavior_disapproving"
    assert by_text["Let's count to three together."]["transcript_subtype"] == "other_teacher_talk"
    assert by_text["Open your books to page ten."]["transcript_subtype"] == "other_teacher_talk"


def test_xlsx_carries_provenance(synthetic_transcript_xlsx, fake_seed):
    out = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                               verbose=False)
    refs = [r["transcript_ref"] for r in out]
    assert all(ref and "!" in ref for ref in refs)
    assert any(r.startswith("01-99C!R") for r in refs)
    assert any(r.startswith("02-99LG!R") for r in refs)


def test_xlsx_skips_meta_sheets(synthetic_transcript_xlsx, fake_seed):
    out = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                               verbose=False)
    refs = [r["transcript_ref"] for r in out]
    assert not any(ref.startswith("TOC") for ref in refs)


def test_xlsx_max_total_caps_output(synthetic_transcript_xlsx, fake_seed):
    out = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                               max_total=2, verbose=False)
    assert len(out) == 2


def test_xlsx_seed_filter_optional(synthetic_transcript_xlsx, fake_seed):
    """When apply_seed_filter=True, lines containing seed terms get dropped."""
    base = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                                apply_seed_filter=False, verbose=False)
    filt = mine_transcript_xlsx(synthetic_transcript_xlsx, build_seed_index(fake_seed),
                                apply_seed_filter=True, verbose=False)
    assert len(filt) <= len(base)


def test_build_negative_pool_with_xlsx(synthetic_transcript_xlsx, fake_corpus, fake_seed, fake_cat):
    df = build_negative_pool(
        fake_corpus, fake_seed, fake_cat,
        transcript_xlsx=synthetic_transcript_xlsx,
        n_hard_per_positive=1, n_per_seed_term=1,
        llm_callable=stub_llm_callable, verbose=False,
    )
    assert "transcript_clean" in set(df["source_type"])
    tx_rows = df[df["source_type"] == "transcript_clean"]
    assert (tx_rows["transcript_subtype"].notna()).all()
    assert set(tx_rows["transcript_subtype"]).issubset(TRANSCRIPT_SUBTYPES)
    llm_rows = df[df["source_type"] != "transcript_clean"]
    assert llm_rows["transcript_subtype"].isna().all()


def test_build_negative_pool_with_xlsx_clears_three_source_gate(
    synthetic_transcript_xlsx, fake_corpus, fake_seed, fake_cat,
):
    df = build_negative_pool(
        fake_corpus, fake_seed, fake_cat,
        transcript_xlsx=synthetic_transcript_xlsx,
        llm_callable=stub_llm_callable, verbose=False,
    )
    info = validate_negatives(df, n_positives=2, target_ratio=0.5,
                              require_three_sources=True, verbose=False)
    assert info["n_negatives"] >= 3


# ---------------------------------------------------------------------------
# Piece 5: subtype_classifier injection
# ---------------------------------------------------------------------------

class TestSubtypeClassifierInjection:
    """The default classifier path stays free; the injection path is exercised
    so a future refactor can't silently sever the wiring."""

    def test_default_classifier_is_stub_and_emits_provenance_columns(self, fake_seed):
        df_neg = pd.DataFrame([
            {"text": "qqq zzz", "source_type": "llm_hard_negative",
             "anchor_utt_id": "utt_0000", "anchor_seed_term": None,
             "prompt_version": "stub"},
        ])
        out = assign_subtypes_to_negatives(df_neg, fake_seed)
        assert "subtype" in out.columns
        assert "subtype_source" in out.columns
        assert "subtype_confidence" in out.columns
        assert "subtype_prompt_version" in out.columns
        # Cue-less + no rule/keyword hit -> falls into the LLM branch
        assert out["subtype_source"].iloc[0] == "llm"
        assert out["subtype_prompt_version"].iloc[0] == LLM_STUB_PROMPT_VERSION
        assert out["subtype_confidence"].iloc[0] == 0.0
        assert out["subtype"].iloc[0] == ["observation"]

    def test_custom_classifier_is_invoked(self, fake_seed):
        called = []

        def fake_classifier(utterance):
            called.append(utterance)
            return (["evidence", "causal_reasoning"], 0.82, "fake_v1")

        df_neg = pd.DataFrame([
            {"text": "qqq zzz mmm", "source_type": "llm_hard_negative",
             "anchor_utt_id": "utt_0000", "anchor_seed_term": None,
             "prompt_version": "stub"},
            {"text": "blah blah blah", "source_type": "transcript_clean",
             "anchor_utt_id": None, "anchor_seed_term": None,
             "prompt_version": "stub"},
        ])
        out = assign_subtypes_to_negatives(df_neg, fake_seed,
                                           llm_classifier=fake_classifier)
        assert len(called) == 2
        assert set(called) == {"qqq zzz mmm", "blah blah blah"}
        assert all(out["subtype_source"] == "llm")
        assert all(out["subtype_prompt_version"] == "fake_v1")
        assert all(out["subtype_confidence"] == 0.82)
        assert sorted(out["subtype"].iloc[0]) == ["causal_reasoning", "evidence"]

    def test_build_negative_pool_threads_subtype_classifier_through(
        self, synthetic_transcript_xlsx, fake_corpus, fake_seed, fake_cat,
    ):
        called_count = {"n": 0}

        def counting_classifier(utterance):
            called_count["n"] += 1
            return (["evidence"], 0.7, "counting_v1")

        df = build_negative_pool(
            fake_corpus, fake_seed, fake_cat,
            transcript_xlsx=synthetic_transcript_xlsx,
            n_hard_per_positive=1, n_per_seed_term=1,
            llm_callable=stub_llm_callable,
            subtype_classifier=counting_classifier,
            verbose=False,
        )
        # Provenance columns must be present in the persisted-shape DataFrame
        assert "subtype_source" in df.columns
        assert "subtype_confidence" in df.columns
        assert "subtype_prompt_version" in df.columns
        # Classifier was actually invoked for at least one row (negatives have
        # no cues so the LLM branch fires for them)
        assert called_count["n"] > 0
        llm_rows = df[df["subtype_prompt_version"] == "counting_v1"]
        assert len(llm_rows) > 0
        assert (llm_rows["subtype_confidence"] == 0.7).all()


class TestSkipKeywordScanForNegatives:
    """Default: skip the keyword path so all negatives reach the LLM and can
    self-tag `not_science_shape`. Backwards-compat: keep_keyword=True restores v1."""

    def test_default_routes_keyword_match_through_llm(self, fake_seed):
        """A negative whose TEXT contains a seed word would historically be
        keyword-tagged. By default, it should now go through the LLM."""
        called = []

        def fake_classifier(utterance):
            called.append(utterance)
            return (["not_science_shape"], 1.0, "fake_v2")

        # 'wonder' is in fake_seed (TIER2, ASK_QUESTION_FRAME -> ['prediction']).
        # Historically this would keyword-tag the negative as ['prediction']
        # without the LLM ever seeing it. With skip_keyword_scan=True (default),
        # it should now hit the LLM and get not_science_shape.
        df_neg = pd.DataFrame([
            {"text": "I wonder how many minutes till lunch.",
             "source_type": "transcript_clean",
             "anchor_utt_id": None, "anchor_seed_term": None,
             "prompt_version": "stub"},
        ])
        out = assign_subtypes_to_negatives(df_neg, fake_seed,
                                           llm_classifier=fake_classifier)
        assert called == ["I wonder how many minutes till lunch."]
        assert out["subtype_source"].iloc[0] == "llm"
        assert out["subtype"].iloc[0] == ["not_science_shape"]

    def test_keep_keyword_restores_v1_behavior(self, fake_seed):
        """skip_keyword_scan=False -> seed-word matches in text bypass the LLM."""
        called = []

        def fake_classifier(utterance):
            called.append(utterance)
            return (["not_science_shape"], 1.0, "fake_v2")

        df_neg = pd.DataFrame([
            {"text": "I wonder how many minutes till lunch.",
             "source_type": "transcript_clean",
             "anchor_utt_id": None, "anchor_seed_term": None,
             "prompt_version": "stub"},
        ])
        out = assign_subtypes_to_negatives(
            df_neg, fake_seed,
            llm_classifier=fake_classifier,
            skip_keyword_scan=False,
        )
        # The keyword path matched 'wonder' -> classifier never ran
        assert called == []
        assert out["subtype_source"].iloc[0] == "keyword"


class TestPosNegLeakCheck:
    """build_negative_pool must drop negatives whose text matches a positive."""

    def test_negative_matching_positive_text_is_dropped(
        self, synthetic_transcript_xlsx, fake_corpus, fake_seed, fake_cat,
    ):
        # fake_corpus has a positive "Look at the seed". If we synthesize a
        # transcript negative with the SAME text, it must be dropped.
        # We use the existing transcript_xlsx fixture which won't have that
        # text, then we'll inject a leak by adding one that does.
        # Simpler: make the LLM stub generate a negative that exactly matches.

        def leaky_llm(prompt, prompt_version):
            # Make hard-negative output include the positive utterance verbatim
            import json
            return json.dumps({"negatives": ["Look at the seed", "Something else"]})

        df = build_negative_pool(
            fake_corpus, fake_seed, fake_cat,
            transcript_xlsx=synthetic_transcript_xlsx,
            n_hard_per_positive=1, n_per_seed_term=1,
            llm_callable=leaky_llm, verbose=False,
        )
        # No negative text should equal a positive text (case-insensitive)
        pos_lower = {u.strip().lower() for u in fake_corpus["utterance"]}
        neg_lower = set(df["text"].str.strip().str.lower())
        assert pos_lower.isdisjoint(neg_lower)


class TestGateScoreOnlyLlmGenerated:
    def test_default_skips_transcript_clean(self):
        from src.negatives_3 import gate_score_negatives, LLM_GENERATED_SOURCES
        df = pd.DataFrame([
            {"text": "a", "source_type": "transcript_clean"},
            {"text": "b", "source_type": "llm_hard_negative"},
            {"text": "c", "source_type": "seed_word_nonscience"},
            {"text": "d", "source_type": "transcript_clean"},
        ])
        called = []

        def stub(prompt, prompt_version):
            called.append(prompt)
            import json
            return json.dumps({"score": 0.9})

        out = gate_score_negatives(df, llm_callable=stub, verbose=False)
        # Only the 2 LLM-generated rows were scored
        assert len(called) == 2
        assert pd.isna(out.loc[0, "llm_gate_score"])
        assert out.loc[1, "llm_gate_score"] == pytest.approx(0.9)
        assert out.loc[2, "llm_gate_score"] == pytest.approx(0.9)
        assert pd.isna(out.loc[3, "llm_gate_score"])

    def test_score_all_rows_when_flag_off(self):
        from src.negatives_3 import gate_score_negatives
        df = pd.DataFrame([
            {"text": "a", "source_type": "transcript_clean"},
            {"text": "b", "source_type": "llm_hard_negative"},
        ])
        called = []

        def stub(prompt, prompt_version):
            called.append(prompt)
            import json
            return json.dumps({"score": 0.5})

        out = gate_score_negatives(
            df, llm_callable=stub, only_llm_generated=False, verbose=False,
        )
        assert len(called) == 2
        assert out["llm_gate_score"].notna().all()

    def test_llm_callable_exception_yields_none_score(self):
        from src.negatives_3 import gate_score_negatives

        df = pd.DataFrame([
            {"text": "a", "source_type": "llm_hard_negative"},
        ])

        def boom(prompt, prompt_version):
            raise RuntimeError("endpoint down")

        out = gate_score_negatives(df, llm_callable=boom, verbose=False)
        assert out["llm_gate_score"].iloc[0] is None


class TestMakeRealLlmCallableResilience:
    """Endpoint blips during negative generation must NOT crash the loop --
    one bad response degrades to the parser fallback."""

    def test_swallows_endpoint_exceptions(self, monkeypatch):
        from src.negatives_3 import make_real_llm_callable

        monkeypatch.setenv("LLM_API_KEY", "fake")
        monkeypatch.setenv("COMPLETION_URL", "http://fake")

        from src import llm_client_0

        def boom(**kw):
            import requests
            raise requests.exceptions.JSONDecodeError("Expecting value", "", 0)

        monkeypatch.setattr(llm_client_0, "cached_request", boom)

        clf = make_real_llm_callable()
        result = clf("any prompt", "any_version")
        # Empty string -> downstream parsers handle gracefully
        assert result == ""
