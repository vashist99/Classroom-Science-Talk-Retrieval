"""Unit tests for src/subtypes_2.py.

Covers Step 2 DoD:
    * every utterance gets >=1 sub-type (assertion in check_subtype_distribution)
    * Tier2-cued rows labeled by deterministic mapping
    * empty-cue rows fall back to LLM (verified with a fake llm_classifier)
    * LLM-assigned subtypes carry confidence + prompt_version
    * distribution sanity (logged warnings, hard-fails on missing >=1)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader_1 import DEFAULT_PROCESSED_DIR
from src.subtypes_2 import (
    LLM_REAL_PROMPT_VERSION,
    LLM_STUB_PROMPT_VERSION,
    PRACTICE_LABELS,
    SUBTYPES_ALL,
    assign_subtypes,
    build_seed_index,
    build_subtype_prompt,
    check_subtype_distribution,
    llm_subtype_stub,
    make_real_llm_subtype_classifier,
    map_cues_to_subtypes,
    parse_subtype_response,
    run,
    scan_text_for_subtypes,
)


@pytest.fixture
def fake_seed_df():
    """Tiny seed-words table covering one term per category we care about."""
    return pd.DataFrame([
        {"term": "observe",  "tier": "TIER2", "category": "INQUIRY_VERB",
         "variants": ["observing", "notice"]},
        {"term": "predict",  "tier": "TIER2", "category": "INQUIRY_VERB",
         "variants": ["prediction", "guess"]},
        {"term": "wonder",   "tier": "TIER2", "category": "ASK_QUESTION_FRAME",
         "variants": ["I wonder"]},
        {"term": "measure",  "tier": "TIER2", "category": "MEASUREMENT_FRAME",
         "variants": ["how many"]},
        {"term": "because",  "tier": "TIER2", "category": "REASONING_FRAME",
         "variants": ["how do you know"]},
        {"term": "what happens",  "tier": "TIER2", "category": "CAUSE_EFFECT_FRAME",
         "variants": ["what changed"]},
        {"term": "seed",     "tier": "TIER3", "category": "LIFE_SCIENCE_PLANTS",
         "variants": []},
        {"term": "pulley",   "tier": "TIER3", "category": "PHYSICAL_SCIENCE_MECHANISMS",
         "variants": []},
    ])


@pytest.fixture
def seed_index(fake_seed_df):
    return build_seed_index(fake_seed_df)


def test_build_seed_index_includes_terms_and_variants(fake_seed_df, seed_index):
    assert "observe" in seed_index
    assert "observing" in seed_index
    assert "notice" in seed_index
    assert "i wonder" in seed_index
    assert "seed" in seed_index


def test_map_cues_inquiry_verb_default_is_observation(seed_index):
    assert map_cues_to_subtypes(["notice"], [], seed_index) == ["observation"]


def test_map_cues_inquiry_verb_predict_override_is_prediction(seed_index):
    assert map_cues_to_subtypes(["predict"], [], seed_index) == ["prediction"]
    assert map_cues_to_subtypes(["guess"], [], seed_index) == ["prediction"]


def test_map_cues_ask_question_frame_is_prediction(seed_index):
    assert map_cues_to_subtypes(["wonder"], [], seed_index) == ["prediction"]


def test_map_cues_reasoning_frame_yields_both_causal_and_evidence(seed_index):
    assert sorted(map_cues_to_subtypes(["because"], [], seed_index)) == \
        ["causal_reasoning", "evidence"]


def test_map_cues_tier3_cue_adds_content(seed_index):
    assert map_cues_to_subtypes([], ["seed"], seed_index) == ["content"]


def test_map_cues_combined_tier2_and_tier3(seed_index):
    out = map_cues_to_subtypes(["notice"], ["seed"], seed_index)
    assert "observation" in out
    assert "content" in out


def test_map_cues_unknown_token_defaults_to_observation(seed_index):
    """Unknown Tier2 token shouldn't silently drop -- assigns observation."""
    assert map_cues_to_subtypes(["zzz_unknown"], [], seed_index) == ["observation"]


def test_scan_text_finds_seed_with_word_boundary(seed_index):
    out = scan_text_for_subtypes("Did you notice the seed?", seed_index)
    assert "observation" in out
    assert "content" in out


def test_scan_text_does_not_match_substrings(seed_index):
    """'observe' should not match 'observed' as a separate cue (no-op example -
    here we ensure 'seed' doesn't accidentally trigger on 'seeds' boundary."""
    # 'seeds' has 'seed' as a sub-string but not at a word boundary on the right
    out = scan_text_for_subtypes("counting seeds", seed_index)
    assert "content" not in out


def test_scan_text_handles_nan(seed_index):
    assert scan_text_for_subtypes(np.nan, seed_index) == []


def test_assign_subtypes_assigns_one_per_row(fake_seed_df):
    df = pd.DataFrame([
        {"utt_id": "u0", "utterance": "Look at the seed",
         "tier2_cues": ["notice"], "tier3_cues": ["seed"]},
        {"utt_id": "u1", "utterance": "Predict what will happen",
         "tier2_cues": ["predict"], "tier3_cues": []},
        {"utt_id": "u2", "utterance": "Sit down please",
         "tier2_cues": [], "tier3_cues": []},
    ])
    out = assign_subtypes(df, fake_seed_df)
    assert len(out) == 3
    assert all(len(s) >= 1 for s in out["subtype"])


def test_assign_subtypes_uses_rule_when_cues_present(fake_seed_df):
    df = pd.DataFrame([
        {"utt_id": "u0", "utterance": "totally unrelated text here",
         "tier2_cues": ["notice"], "tier3_cues": []},
    ])
    out = assign_subtypes(df, fake_seed_df)
    assert out.iloc[0]["subtype_source"] == "rule"
    assert out.iloc[0]["subtype"] == ["observation"]
    assert out.iloc[0]["subtype_confidence"] == 1.0
    assert out.iloc[0]["subtype_prompt_version"] is None


def test_assign_subtypes_marks_rule_plus_keyword_when_text_adds(fake_seed_df):
    """Rule produces 'content' from tier3, text scan adds 'observation' from 'notice'."""
    df = pd.DataFrame([
        {"utt_id": "u0", "utterance": "Did you notice the seed?",
         "tier2_cues": [], "tier3_cues": ["seed"]},
    ])
    out = assign_subtypes(df, fake_seed_df)
    assert out.iloc[0]["subtype_source"] == "rule+keyword"
    assert "content" in out.iloc[0]["subtype"]
    assert "observation" in out.iloc[0]["subtype"]


def test_assign_subtypes_keyword_only_path(fake_seed_df):
    """No cues at all but text contains a known seed -> keyword source."""
    df = pd.DataFrame([
        {"utt_id": "u0", "utterance": "I wonder what will happen",
         "tier2_cues": [], "tier3_cues": []},
    ])
    out = assign_subtypes(df, fake_seed_df)
    assert out.iloc[0]["subtype_source"] == "keyword"
    assert "prediction" in out.iloc[0]["subtype"]
    assert out.iloc[0]["subtype_confidence"] == 1.0


def test_assign_subtypes_skip_keyword_scan_forces_llm(fake_seed_df):
    """`skip_keyword_scan=True` keeps surface tokens from bypassing the LLM."""
    df = pd.DataFrame([
        {"utt_id": "u0", "utterance": "I predict you will spill the paint",
         "tier2_cues": [], "tier3_cues": []},
    ])
    out_kw = assign_subtypes(
        df, fake_seed_df, llm_classifier=llm_subtype_stub, skip_keyword_scan=False,
    )
    assert out_kw.iloc[0]["subtype_source"] == "keyword"

    def fake_llm(u):
        return (["not_science_shape"], 0.88, "test_v")

    out_skip = assign_subtypes(
        df, fake_seed_df, llm_classifier=fake_llm, skip_keyword_scan=True,
    )
    assert out_skip.iloc[0]["subtype_source"] == "llm"
    assert out_skip.iloc[0]["subtype"] == ["not_science_shape"]
    assert out_skip.iloc[0]["subtype_confidence"] == pytest.approx(0.88)


def test_assign_subtypes_calls_llm_when_no_cues_and_no_keywords(fake_seed_df):
    """Only when both rule and keyword scan return empty does the LLM fire."""
    fake_calls: list[str] = []

    def fake_llm(utterance: str):
        fake_calls.append(utterance)
        return (["evidence"], 0.7, "fake_prompt_v1")

    df = pd.DataFrame([
        {"utt_id": "u0", "utterance": "totally unrelated to anything",
         "tier2_cues": [], "tier3_cues": []},
    ])
    out = assign_subtypes(df, fake_seed_df, llm_classifier=fake_llm)
    assert fake_calls == ["totally unrelated to anything"]
    assert out.iloc[0]["subtype_source"] == "llm"
    assert out.iloc[0]["subtype"] == ["evidence"]
    assert out.iloc[0]["subtype_confidence"] == 0.7
    assert out.iloc[0]["subtype_prompt_version"] == "fake_prompt_v1"


def test_assign_subtypes_does_not_call_llm_when_rule_or_keyword_fired(fake_seed_df):
    """The fake classifier must not be called for rows that rule/keyword resolved."""
    def boom(_):
        raise AssertionError("LLM should not be called for rule/keyword rows")

    df = pd.DataFrame([
        {"utt_id": "u0", "utterance": "Look at the seed",
         "tier2_cues": ["notice"], "tier3_cues": ["seed"]},
        {"utt_id": "u1", "utterance": "I wonder why",
         "tier2_cues": [], "tier3_cues": []},
    ])
    out = assign_subtypes(df, fake_seed_df, llm_classifier=boom)
    assert "llm" not in set(out["subtype_source"])


def test_llm_subtype_stub_returns_documented_shape():
    sts, conf, ver = llm_subtype_stub("anything")
    assert sts == ["observation"]
    assert conf == 0.0
    assert ver == LLM_STUB_PROMPT_VERSION


def test_check_distribution_raises_when_a_row_has_no_subtype():
    df = pd.DataFrame([
        {"subtype": ["observation"], "subtype_source": "rule"},
        {"subtype": [], "subtype_source": "rule"},
    ])
    with pytest.raises(AssertionError):
        check_subtype_distribution(df, verbose=False)


def test_check_distribution_returns_warnings_for_skew():
    """All rows have only 'content' -> warning for over-dominance + missing practices."""
    df = pd.DataFrame([
        {"subtype": ["content"], "subtype_source": "rule"} for _ in range(10)
    ])
    info = check_subtype_distribution(df, verbose=False)
    assert "content" in str(info["warnings"])
    assert any("0 examples" in w for w in info["warnings"])


def _processed_corpus_available():
    return (DEFAULT_PROCESSED_DIR / "corpus.parquet").exists() and \
           (DEFAULT_PROCESSED_DIR / "seed_words.parquet").exists()


@pytest.mark.skipif(not _processed_corpus_available(),
                    reason="run Step 1 first to produce the processed parquets")
def test_run_step2_end_to_end_persists_subtype_columns(tmp_path):
    """End-to-end: copy real parquets to tmp, run Step 2, verify schema on disk."""
    import shutil
    for fn in ("corpus.parquet", "seed_words.parquet", "category_defs.parquet"):
        src = DEFAULT_PROCESSED_DIR / fn
        if src.exists():
            shutil.copy(src, tmp_path / fn)

    out_path = run(processed_dir=tmp_path, verbose=False)
    assert out_path.exists()

    df = pd.read_parquet(out_path)
    for col in ("subtype", "subtype_source", "subtype_confidence", "subtype_prompt_version"):
        assert col in df.columns

    assert df["subtype"].apply(lambda s: len(s) > 0).all()

    seen = {st for sts in df["subtype"] for st in sts}
    assert seen.issubset(set(SUBTYPES_ALL))

    valid_sources = {"rule", "rule+keyword", "keyword", "llm"}
    assert set(df["subtype_source"].unique()).issubset(valid_sources)

    assert df["subtype_confidence"].between(0.0, 1.0).all()

    llm_rows = df[df["subtype_source"] == "llm"]
    if len(llm_rows) > 0:
        assert llm_rows["subtype_prompt_version"].notna().all()


# ---------------------------------------------------------------------------
# Real LLM-backed classifier: prompt + parser + factory
# ---------------------------------------------------------------------------

class TestBuildSubtypePrompt:
    def test_includes_utterance_verbatim(self):
        prompt = build_subtype_prompt("Look at the bug!")
        assert "Look at the bug!" in prompt

    def test_includes_full_closed_vocabulary(self):
        prompt = build_subtype_prompt("anything")
        for st in SUBTYPES_ALL:
            assert st in prompt

    def test_requests_strict_json(self):
        prompt = build_subtype_prompt("anything")
        assert "JSON" in prompt
        assert "subtypes" in prompt
        assert "confidence" in prompt


class TestParseSubtypeResponse:
    def test_clean_json_round_trips(self):
        sts, conf, ver = parse_subtype_response(
            '{"subtypes": ["observation", "evidence"], "confidence": 0.82}'
        )
        assert sorted(sts) == ["evidence", "observation"]
        assert conf == pytest.approx(0.82)
        assert ver == LLM_REAL_PROMPT_VERSION

    def test_filters_out_hallucinated_labels(self):
        sts, conf, _ = parse_subtype_response(
            '{"subtypes": ["observation", "vibes", "thinking"], "confidence": 0.5}'
        )
        assert sts == ["observation"]
        assert conf == pytest.approx(0.5)

    def test_clamps_out_of_range_confidence(self):
        _, conf_hi, _ = parse_subtype_response(
            '{"subtypes": ["observation"], "confidence": 7.0}'
        )
        _, conf_lo, _ = parse_subtype_response(
            '{"subtypes": ["observation"], "confidence": -0.5}'
        )
        assert conf_hi == 1.0
        assert conf_lo == 0.0

    def test_extracts_json_from_prose_wrapper(self):
        wrapped = (
            "Sure, here is my classification:\n"
            '{"subtypes": ["prediction"], "confidence": 0.9}\n'
            "Hope that helps!"
        )
        sts, conf, _ = parse_subtype_response(wrapped)
        assert sts == ["prediction"]
        assert conf == pytest.approx(0.9)

    def test_falls_back_when_unparseable(self):
        sts, conf, ver = parse_subtype_response("totally not json")
        # v2: parser-failure fallback honestly tags `not_science_shape`
        # rather than silently forcing `observation`.
        assert sts == ["not_science_shape"]
        assert conf == 0.0
        assert ver == LLM_REAL_PROMPT_VERSION

    def test_falls_back_when_empty_subtype_list(self):
        sts, _, _ = parse_subtype_response('{"subtypes": [], "confidence": 0.4}')
        assert sts == ["not_science_shape"]

    def test_falls_back_when_all_subtypes_invalid(self):
        sts, _, _ = parse_subtype_response('{"subtypes": ["xyz", "abc"], "confidence": 0.4}')
        assert sts == ["not_science_shape"]

    def test_handles_empty_string(self):
        sts, conf, ver = parse_subtype_response("")
        assert sts == ["not_science_shape"]
        assert conf == 0.0
        assert ver == LLM_REAL_PROMPT_VERSION

    def test_dedupes_repeated_subtypes(self):
        sts, _, _ = parse_subtype_response(
            '{"subtypes": ["observation", "observation", "evidence"], "confidence": 0.5}'
        )
        assert sorted(sts) == ["evidence", "observation"]

    def test_non_numeric_confidence_falls_to_zero(self):
        _, conf, _ = parse_subtype_response(
            '{"subtypes": ["observation"], "confidence": "high"}'
        )
        assert conf == 0.0

    def test_accepts_not_science_shape_as_valid_label(self):
        sts, conf, _ = parse_subtype_response(
            '{"subtypes": ["not_science_shape"], "confidence": 0.9}'
        )
        assert sts == ["not_science_shape"]
        assert conf == pytest.approx(0.9)

    def test_not_science_shape_is_exclusive(self):
        """When the LLM returns not_science_shape alongside practice labels,
        the sentinel wins and the practice labels are dropped (the prompt
        says not_science_shape must stand alone)."""
        sts, _, _ = parse_subtype_response(
            '{"subtypes": ["observation", "not_science_shape", "content"], '
            '"confidence": 0.7}'
        )
        assert sts == ["not_science_shape"]


class TestNotScienceShapeVocab:
    """Vocab-level guarantees that downstream code can rely on."""

    def test_not_science_shape_in_subtypes_all(self):
        from src.subtypes_2 import SUBTYPES_ALL
        assert "not_science_shape" in SUBTYPES_ALL

    def test_not_science_shape_is_not_a_practice_label(self):
        """Practice metrics (e.g. practice_coverage) must NOT count negatives
        as having a science practice."""
        from src.subtypes_2 import PRACTICE_LABELS
        assert "not_science_shape" not in PRACTICE_LABELS


class TestPromptIncludesNotScienceShape:
    def test_prompt_mentions_not_science_shape(self):
        prompt = build_subtype_prompt("anything")
        assert "not_science_shape" in prompt

    def test_prompt_says_sentinel_must_stand_alone(self):
        """Regression: the LLM must be instructed not to combine
        not_science_shape with practice labels."""
        prompt = build_subtype_prompt("anything")
        assert "ALONE" in prompt or "alone" in prompt

    def test_prompt_warns_against_forced_content_labels(self):
        """Place names / kid names / classroom objects shouldn't get `content`."""
        prompt = build_subtype_prompt("anything")
        assert "place names" in prompt.lower() or "names of children" in prompt.lower()


class TestDistributionCheckTreatsNotScienceShapeAsOptional:
    def test_no_warning_when_not_science_shape_is_zero(self):
        """Positive-only corpus should never have a not_science_shape row.
        The distribution checker must NOT warn about that absence."""
        df = pd.DataFrame([
            {"subtype": ["observation"], "subtype_source": "rule"},
            {"subtype": ["prediction"], "subtype_source": "rule"},
            {"subtype": ["causal_reasoning"], "subtype_source": "rule"},
            {"subtype": ["evidence"], "subtype_source": "rule"},
            {"subtype": ["content"], "subtype_source": "rule"},
        ])
        info = check_subtype_distribution(df, verbose=False)
        assert not any("not_science_shape" in w for w in info["warnings"])

    def test_still_warns_when_practice_label_is_zero(self):
        """Other missing labels still warn -- only not_science_shape is exempt."""
        df = pd.DataFrame([
            {"subtype": ["content"], "subtype_source": "rule"} for _ in range(5)
        ])
        info = check_subtype_distribution(df, verbose=False)
        assert any("observation: 0 examples" in w for w in info["warnings"])
        assert any("prediction: 0 examples" in w for w in info["warnings"])


class TestMakeRealLlmSubtypeClassifier:
    def test_raises_when_env_vars_missing(self, monkeypatch):
        # Empty string rather than delenv: llm_client_0 calls load_dotenv() on
        # import which would re-populate from .env if we just unset.
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("COMPLETION_URL", "")
        with pytest.raises(RuntimeError, match="LLM_API_KEY"):
            make_real_llm_subtype_classifier()

    def test_invokes_cached_request_with_versioned_prompt(self, monkeypatch):
        """Build a real classifier, monkey-patch cached_request, confirm wiring."""
        monkeypatch.setenv("LLM_API_KEY", "fake-key")
        monkeypatch.setenv("COMPLETION_URL", "http://fake")

        captured = {}

        def fake_cached_request(*, api_key, url, endpoint, model, params, prompt_version):
            captured["api_key"] = api_key
            captured["url"] = url
            captured["endpoint"] = endpoint
            captured["model"] = model
            captured["params"] = params
            captured["prompt_version"] = prompt_version
            return {"choices": [{"message": {"content":
                '{"subtypes": ["causal_reasoning"], "confidence": 0.7}'}}]}

        from src import llm_client_0
        monkeypatch.setattr(llm_client_0, "cached_request", fake_cached_request)

        clf = make_real_llm_subtype_classifier(model="custom-model-id")
        sts, conf, ver = clf("Why does the ball roll?")

        assert sts == ["causal_reasoning"]
        assert conf == pytest.approx(0.7)
        assert ver == LLM_REAL_PROMPT_VERSION
        assert captured["api_key"] == "fake-key"
        assert captured["url"] == "http://fake"
        assert captured["endpoint"] == "completion"
        assert captured["model"] == "custom-model-id"
        assert captured["prompt_version"] == LLM_REAL_PROMPT_VERSION
        assert captured["params"]["temperature"] == 0.0
        msgs = captured["params"]["messages"]
        assert msgs[0]["role"] == "user"
        assert "Why does the ball roll?" in msgs[0]["content"]

    def test_handles_malformed_endpoint_response(self, monkeypatch):
        """If the endpoint returns garbage, the classifier still returns a valid SubtypeResult."""
        monkeypatch.setenv("LLM_API_KEY", "fake-key")
        monkeypatch.setenv("COMPLETION_URL", "http://fake")

        from src import llm_client_0
        monkeypatch.setattr(
            llm_client_0,
            "cached_request",
            lambda **kw: {"unexpected": "shape"},
        )

        clf = make_real_llm_subtype_classifier()
        sts, conf, ver = clf("anything")
        assert sts == ["not_science_shape"]
        assert conf == 0.0
        assert ver == LLM_REAL_PROMPT_VERSION

    def test_swallows_endpoint_exceptions(self, monkeypatch):
        """Endpoint blips (JSONDecodeError, ConnectionError, etc.) must NOT
        abort the loop -- they degrade to the fallback so the run continues."""
        monkeypatch.setenv("LLM_API_KEY", "fake-key")
        monkeypatch.setenv("COMPLETION_URL", "http://fake")

        from src import llm_client_0

        def boom(**kw):
            import requests
            raise requests.exceptions.JSONDecodeError("Expecting value", "", 0)

        monkeypatch.setattr(llm_client_0, "cached_request", boom)

        clf = make_real_llm_subtype_classifier()
        sts, conf, ver = clf("anything")
        assert sts == ["not_science_shape"]
        assert conf == 0.0
        assert ver == LLM_REAL_PROMPT_VERSION

    def test_drop_in_replacement_for_stub_in_assign_subtypes(self, monkeypatch, fake_seed_df):
        """Real classifier should plug into assign_subtypes() exactly like the stub."""
        monkeypatch.setenv("LLM_API_KEY", "fake-key")
        monkeypatch.setenv("COMPLETION_URL", "http://fake")

        from src import llm_client_0
        monkeypatch.setattr(
            llm_client_0,
            "cached_request",
            lambda **kw: {"choices": [{"message": {"content":
                '{"subtypes": ["evidence"], "confidence": 0.66}'}}]},
        )

        df = pd.DataFrame([
            {"utt_id": "u1", "utterance": "qqq zzz mmm",
             "label": "SCIENCE_TALK",
             "tier2_cues": [], "tier3_cues": []},
        ])
        clf = make_real_llm_subtype_classifier()
        out = assign_subtypes(df, fake_seed_df, llm_classifier=clf)

        row = out.iloc[0]
        assert row["subtype_source"] == "llm"
        assert row["subtype"] == ["evidence"]
        assert row["subtype_confidence"] == pytest.approx(0.66)
        assert row["subtype_prompt_version"] == LLM_REAL_PROMPT_VERSION
