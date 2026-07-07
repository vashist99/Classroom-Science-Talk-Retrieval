"""Unit tests for src/reranker_9.py (Step 9 LLM pair re-ranker).

All tests run offline with a deterministic stub caller or pure-math helpers; the
real gpt-oss-120b path is exercised separately via `python src/reranker_9.py
--smoke`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import reranker_9 as r9


# ---------------------------------------------------------------------------
# Parsing (fails loudly)
# ---------------------------------------------------------------------------

def test_parse_score_json():
    s, rat = r9.parse_score('{"score": 0.8, "rationale": "ok"}')
    assert s == 0.8 and rat == "ok"


def test_parse_score_json_in_prose():
    s, _ = r9.parse_score('Here you go: {"score": 0.42, "rationale": "x"} thanks')
    assert s == 0.42


def test_parse_score_tag_fallback():
    s, _ = r9.parse_score("blah <score>0.6</score> blah")
    assert s == 0.6


def test_parse_score_clamps():
    assert r9.parse_score('{"score": 1.7}')[0] == 1.0
    assert r9.parse_score('{"score": -0.3}')[0] == 0.0


def test_parse_score_garbage_raises():
    with pytest.raises(r9.RerankerParseError):
        r9.parse_score("I cannot answer that")
    with pytest.raises(r9.RerankerParseError):
        r9.parse_score("")


# ---------------------------------------------------------------------------
# score_pair_full: retry + loud fail
# ---------------------------------------------------------------------------

def test_score_pair_full_success():
    caller = lambda s, u, pv: '{"score": 0.7, "rationale": "r"}'
    s, _ = r9.score_pair_full("a", "b", caller=caller, system_prompt="sys",
                              prompt_version="v", verbose=False)
    assert s == 0.7


def test_score_pair_full_retry_then_success():
    calls = {"n": 0}

    def caller(s, u, pv):
        calls["n"] += 1
        return "garbage" if calls["n"] == 1 else '{"score": 0.9}'

    s, _ = r9.score_pair_full("a", "b", caller=caller, system_prompt="sys",
                              prompt_version="v", max_retries=1, verbose=False)
    assert s == 0.9 and calls["n"] == 2


def test_score_pair_full_loud_fail_after_retries():
    caller = lambda s, u, pv: "never valid"
    with pytest.raises(r9.RerankerParseError):
        r9.score_pair_full("a", "b", caller=caller, system_prompt="sys",
                           prompt_version="v", max_retries=1, row_id="z",
                           verbose=False)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_auroc_perfect():
    assert r9.auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0


def test_auroc_inverted():
    assert r9.auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_auroc_half_with_ties():
    # all equal scores -> 0.5
    assert r9.auroc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auroc_single_class_none():
    assert r9.auroc([1, 1, 1], [0.1, 0.2, 0.3]) is None


def test_spearman_monotonic():
    assert r9.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_anti():
    assert r9.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_constant_none():
    assert r9.spearman([1, 2, 3], [5, 5, 5]) is None


def test_ndcg_and_rank_metrics():
    m = r9.rank_metrics(["d1", "d0", "d2"], {"d1"}, k=3)
    assert m["precision@1"] == 1.0
    assert m["mrr@3"] == 1.0
    assert m["ndcg@3"] == pytest.approx(1.0)
    m2 = r9.rank_metrics(["d0", "d1"], {"d1"}, k=3)
    assert m2["precision@1"] == 0.0
    assert m2["mrr@3"] == 0.5


# ---------------------------------------------------------------------------
# Prompt assembly + stub
# ---------------------------------------------------------------------------

def test_build_operational_definitions():
    df = pd.DataFrame({
        "label": ["INQUIRY_VERB", "TIER2"],
        "type": ["Category (Tier 2)", "Tier tag"],
        "definition": ["initiate inquiry actions", "general words"],
    })
    out = r9.build_operational_definitions(df)
    assert "INQUIRY_VERB" in out
    assert "initiate inquiry actions" in out


def test_build_system_prompt_fills_placeholder(tmp_path):
    pd.DataFrame({
        "label": ["INQUIRY_VERB"], "type": ["Category (Tier 2)"],
        "definition": ["initiate inquiry"],
    }).to_parquet(tmp_path / "category_defs.parquet", index=False)
    tmpl = tmp_path / "tmpl.txt"
    tmpl.write_text("DEFS:\n{operational_definitions}\nEND", encoding="utf-8")
    sysmsg = r9.build_system_prompt(tmp_path, template_path=tmpl)
    assert "{operational_definitions}" not in sysmsg
    assert "INQUIRY_VERB" in sysmsg


def test_stub_caller_roundtrips_through_parser():
    user = r9.build_user_message("the ice is melting fast", "the ice melts fast")
    out = r9.stub_caller("sys", user, "v")
    s, _ = r9.parse_score(out)
    assert 0.0 <= s <= 1.0


def test_call_ledger_counts_miss_for_novel_params():
    led = r9.CallLedger()
    led.classify(model="no-such-model-xyz",
                 params={"messages": [{"role": "user", "content": "zzz-unique"}]},
                 prompt_version="vtest")
    assert led.misses == 1
    assert led.total == 1


# ---------------------------------------------------------------------------
# Calibration data + scoring with stub
# ---------------------------------------------------------------------------

@pytest.fixture
def frames():
    corpus = pd.DataFrame({
        "utt_id": ["utt_a", "utt_b"],
        "utterance": ["why did the ice melt", "how did the plant grow"],
        "label": ["SCIENCE_TALK", "SCIENCE_TALK"],
    })
    pairs = pd.DataFrame({
        "pair_id": ["pair_v1", "pair_v2"],
        "anchor_id": ["utt_a", "utt_b"],
        "variant_text": ["why did the ice melt", "how did the plant grow"],
    })
    negatives = pd.DataFrame({
        "neg_id": ["neg_v1", "neg_v2"],
        "text": ["line up at the door", "put your shoes on"],
        "anchor_utt_id": [None, None],
        "source_type": ["transcript_clean", "transcript_clean"],
    })
    splits = pd.DataFrame({
        "id": ["pair_v1", "pair_v2", "neg_v1", "neg_v2"],
        "kind": ["pair", "pair", "negative", "negative"],
        "split": ["val", "val", "val", "val"],
    })
    return {"corpus": corpus, "pairs": pairs, "negatives": negatives, "splits": splits}


def test_build_pair_audit_labels(frames):
    audit = r9.build_pair_audit(frames, n=4, seed=1)
    labels = sorted(a["label"] for a in audit)
    assert set(labels) == {0, 1}
    pos = [a for a in audit if a["label"] == 1]
    # positive candidate equals a variant of a science anchor
    assert all(a["candidate"] for a in pos)


def test_score_audit_auroc_with_stub(frames):
    # stub scores by lexical overlap: positives (anchor==variant) overlap fully,
    # negatives (anchor vs management text) overlap ~0 -> perfect separation.
    audit = r9.build_pair_audit(frames, n=4, seed=1)
    res = r9.score_audit(audit, caller=r9.stub_caller, system_prompt="sys",
                         prompt_version="v", verbose=False)
    assert res["auroc"] == 1.0


def test_run_stub_end_to_end(tmp_path):
    """Offline run() smoke: zero network calls, report written, keys present."""
    # materialize a tiny processed dir
    pd.DataFrame({
        "utt_id": ["utt_a", "utt_b"],
        "utterance": ["why did the ice melt", "how did the plant grow"],
        "label": ["SCIENCE_TALK", "SCIENCE_TALK"],
    }).to_parquet(tmp_path / "corpus.parquet", index=False)
    pd.DataFrame({
        "pair_id": ["pair_v1", "pair_v2"],
        "anchor_id": ["utt_a", "utt_b"],
        "variant_text": ["why did the ice melt", "how did the plant grow"],
    }).to_parquet(tmp_path / "pairs.parquet", index=False)
    pd.DataFrame({
        "neg_id": ["neg_v1", "neg_v2"],
        "text": ["line up at the door", "put your shoes on"],
        "anchor_utt_id": [None, None],
        "source_type": ["transcript_clean", "transcript_clean"],
    }).to_parquet(tmp_path / "negatives.parquet", index=False)
    pd.DataFrame({
        "id": ["pair_v1", "pair_v2", "neg_v1", "neg_v2"],
        "kind": ["pair", "pair", "negative", "negative"],
        "label": ["SCIENCE_TALK", "SCIENCE_TALK", "NOT_SCIENCE_TALK", "NOT_SCIENCE_TALK"],
        "stratify_key": ["a", "b", "c", "d"],
        "anchor_id": [None, None, None, None],
        "split": ["val", "val", "val", "val"],
    }).to_parquet(tmp_path / "splits.parquet", index=False)
    pd.DataFrame({
        "label": ["INQUIRY_VERB"], "type": ["Category (Tier 2)"],
        "definition": ["initiate inquiry"],
    }).to_parquet(tmp_path / "category_defs.parquet", index=False)

    cfg = {
        "version": "reranker_test", "model_env": "LLM_MODEL_TRACKB",
        "prompt_version": "reranker_v1",
        "prompt_file": "prompts/reranker_v1.txt",
        "prompt_file_paraphrase": "prompts/reranker_v1b.txt",
        "temperature": 0.0, "max_tokens": 50, "max_retries": 1,
        "gate": {"min_auroc": 0.85, "min_spearman": 0.90},
        "diagnostics": {"hard_informal_min_lift": 0.0, "retrieval_lift_metrics": []},
        "audit": {"n_pairs": 4, "seed": 1}, "stability": {"n_pairs": 2, "seed": 1},
        "smoke": {"audit_pairs": 4, "stability_pairs": 2, "rerank_sample": 0,
                  "max_new_calls": 50},
        "output": {"report_json": str(tmp_path / "rep.json"),
                   "calibration_md": str(tmp_path / "cal.md")},
    }
    cfg_path = tmp_path / "reranker.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    report = r9.run(tmp_path, config_path=cfg_path, mode="smoke",
                    use_stub=True, verbose=False)
    assert report["cost_ledger"]["new_calls"] == 0
    assert "auroc" in report and "stability" in report
    assert (tmp_path / "rep.json").exists()
    assert (tmp_path / "cal.md").exists()


def test_config_loads():
    cfg = r9.load_config()
    assert cfg["model_env"] == "LLM_MODEL_TRACKB"
    assert cfg["gate"]["min_auroc"] == 0.85


# ---------------------------------------------------------------------------
# API-key rotation
# ---------------------------------------------------------------------------

def test_collect_api_keys(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "a, b")
    monkeypatch.setenv("LLM_API_KEY_2", "c")
    monkeypatch.delenv("LLM_API_KEY_3", raising=False)
    assert r9._collect_api_keys("LLM_API_KEY") == ["a", "b", "c"]


def test_response_kind():
    ok = {"choices": [{"message": {"content": '{"score":0.9}'}}]}
    assert r9._response_kind(ok) == "ok"
    assert r9._response_kind({"error": {"message": "monthly budget exceeded"}}) == "quota_auth"
    assert r9._response_kind({"error": {"message": "rate limited, slow down"}}) == "transient"
    assert r9._response_kind({"choices": [{"message": {"content": ""}}]}) == "transient"


def _good_body(score=0.91):
    return {"choices": [{"message": {"content": json.dumps(
        {"score": score, "rationale": "ok"})}}]}


def test_make_real_caller_rotates_on_quota(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k1")
    monkeypatch.setenv("LLM_API_KEY_2", "k2")
    monkeypatch.delenv("LLM_API_KEY_3", raising=False)
    monkeypatch.setenv("COMPLETION_URL", "http://example.test/v1")

    used = []

    def fake_cached_request(*, api_key, url, endpoint, model, params, prompt_version):
        used.append(api_key)
        if api_key == "k1":
            return {"error": {"message": "Your budget has been exceeded"}}
        return _good_body()

    monkeypatch.setattr("src.llm_client_0.cached_request", fake_cached_request)

    caller = r9.make_real_caller("gpt", ledger=None, verbose=False)
    out = caller("sys", "usr", "reranker_v1")
    assert json.loads(out)["score"] == 0.91
    assert used == ["k1", "k2"]  # rotated past the depleted key

    # second call is sticky on k2 (k1 is not retried)
    out2 = caller("sys2", "usr2", "reranker_v1")
    assert json.loads(out2)["score"] == 0.91
    assert used == ["k1", "k2", "k2"]


def test_make_real_caller_transient_does_not_rotate(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k1")
    monkeypatch.setenv("LLM_API_KEY_2", "k2")
    monkeypatch.delenv("LLM_API_KEY_3", raising=False)
    monkeypatch.setenv("COMPLETION_URL", "http://example.test/v1")

    used = []

    def fake_cached_request(*, api_key, url, endpoint, model, params, prompt_version):
        used.append(api_key)
        return {"choices": [{"message": {"content": ""}}]}  # transient empty

    monkeypatch.setattr("src.llm_client_0.cached_request", fake_cached_request)

    caller = r9.make_real_caller("gpt", ledger=None, verbose=False)
    assert caller("sys", "usr", "reranker_v1") == ""  # degraded, no rotation
    assert used == ["k1"]  # stayed on the same key
