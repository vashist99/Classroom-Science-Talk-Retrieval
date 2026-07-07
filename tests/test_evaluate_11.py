"""Unit tests for src/evaluate_11.py (Step 11 evaluation + threshold tuning).

Pure-logic tests only (metrics, tuning, query building, error tables). The heavy
ablation run is exercised via `python src/evaluate_11.py --smoke`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluate_11 as e11


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------

def test_classification_metrics_basic():
    # tp=2, fp=1, fn=1, tn=1
    m = e11.classification_metrics([1, 1, 0, 1, 0], [1, 1, 1, 0, 0])
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_classification_metrics_all_correct():
    m = e11.classification_metrics([1, 0, 1], [1, 0, 1])
    assert m["f1"] == 1.0


def test_classification_metrics_no_positives_predicted():
    m = e11.classification_metrics([1, 1], [0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


# ---------------------------------------------------------------------------
# threshold_grid + tune_threshold
# ---------------------------------------------------------------------------

def test_threshold_grid_bounds():
    g = e11.threshold_grid({"threshold_grid": {"start": 0.0, "stop": 1.0, "step": 0.25}})
    assert g[0] == 0.0 and g[-1] == 0.75 and 1.0 not in g


def test_tune_threshold_picks_max_f1_respecting_floor():
    # Perfectly separable: pos scores high, neg low.
    dev_scores = [0.9, 0.8, 0.2, 0.1]
    dev_labels = [1, 1, 0, 0]
    # slice all high -> floor satisfiable at high thresholds
    slice_scores = [0.85, 0.9, 0.95]
    grid = [0.1, 0.3, 0.5, 0.7, 0.95]
    out = e11.tune_threshold(dev_scores, dev_labels, slice_scores, grid, floor=0.8)
    assert out["meets_floor"] is True
    assert out["dev_f1"] == 1.0
    # threshold 0.7 keeps slice recall 1.0 and F1 1.0; 0.95 would drop slice recall to 0
    assert out["threshold"] <= 0.85


def test_tune_threshold_floor_blocks_high_threshold():
    dev_scores = [0.9, 0.4, 0.2]
    dev_labels = [1, 1, 0]
    slice_scores = [0.5, 0.55]   # recall collapses above 0.55
    grid = [0.3, 0.6, 0.9]
    out = e11.tune_threshold(dev_scores, dev_labels, slice_scores, grid, floor=0.9)
    # only thr 0.3 keeps slice recall (1.0) >= 0.9
    assert out["threshold"] == 0.3
    assert out["meets_floor"] is True


def test_tune_threshold_fallback_when_floor_unreachable():
    out = e11.tune_threshold([0.9, 0.1], [1, 0], [0.0, 0.0], [0.5], floor=0.99)
    assert out["meets_floor"] is False  # no threshold satisfies floor -> fallback


# ---------------------------------------------------------------------------
# mean_ranking_metrics
# ---------------------------------------------------------------------------

def test_mean_ranking_metrics_perfect():
    orders = [(["g", "x", "y"], "g"), (["g", "z"], "g")]
    m = e11.mean_ranking_metrics(orders, k=10)
    assert m["precision@1"] == 1.0
    assert m["mrr@10"] == 1.0
    assert m["n"] == 2


def test_mean_ranking_metrics_empty():
    m = e11.mean_ranking_metrics([], k=10)
    assert m["mrr@10"] is None and m["n"] == 0


# ---------------------------------------------------------------------------
# _primary_subtype
# ---------------------------------------------------------------------------

def test_primary_subtype_list_and_scalar():
    assert e11._primary_subtype(["observation"]) == "observation"
    assert e11._primary_subtype(["b", "a"]) == "a|b"
    assert e11._primary_subtype("content") == "content"
    assert e11._primary_subtype(float("nan")) == "unknown"


# ---------------------------------------------------------------------------
# query building
# ---------------------------------------------------------------------------

@pytest.fixture
def frames():
    corpus = pd.DataFrame({
        "utt_id": ["a", "b"], "utterance": ["why melt", "how grow"],
        "label": ["SCIENCE_TALK", "SCIENCE_TALK"],
        "subtype": [["causal_reasoning"], ["content"]]})
    pairs = pd.DataFrame({
        "pair_id": ["p1", "p2"], "anchor_id": ["a", "b"],
        "variant_text": ["why did it melt", "how did it grow"]})
    negatives = pd.DataFrame({"neg_id": ["n1", "n2"],
                              "text": ["line up", "shoes on"]})
    splits = pd.DataFrame({
        "id": ["p1", "p2", "n1", "n2"],
        "kind": ["pair", "pair", "negative", "negative"],
        "split": ["val", "test", "val", "test"]})
    return {"corpus": corpus, "pairs": pairs, "negatives": negatives, "splits": splits}


def test_build_split_queries(frames):
    q = e11.build_split_queries(frames, "val", n_pos=5, n_neg=5)
    pos = [x for x in q if x["label"] == 1]
    neg = [x for x in q if x["label"] == 0]
    assert len(pos) == 1 and len(neg) == 1
    assert pos[0]["gold_anchor"] == "a"
    assert pos[0]["subtype"] == "causal_reasoning"


def test_build_slice_queries(tmp_path):
    pd.DataFrame({
        "id": ["utt_1"], "kind": ["positive"], "anchor_id": ["utt_1"],
        "text": ["the magnets click together"], "setting": ["Centers"],
        "register": ["INFORMAL"], "subtype": [["content"]],
        "slice_reason": ["informal_positive"]}).to_parquet(
        tmp_path / "hard_informal_slice.parquet", index=False)
    q = e11.build_slice_queries(tmp_path, n=10)
    assert len(q) == 1 and q[0]["label"] == 1 and q[0]["gold_anchor"] == "utt_1"


def test_build_slice_queries_missing(tmp_path):
    assert e11.build_slice_queries(tmp_path, n=10) == []


# ---------------------------------------------------------------------------
# errors + predictions
# ---------------------------------------------------------------------------

def test_errors_and_predictions():
    test = [
        {"phrase": "why melt", "label": 1, "subtype": "causal_reasoning"},
        {"phrase": "line up", "label": 0, "subtype": "negative"},
    ]
    sys_c = {
        "tune": {"threshold": 0.5},
        "test": {
            "scores": np.array([0.9, 0.8]),  # neg scored high -> false positive
            "details": [
                {"top_match_utt_id": "a", "top_match_utterance": "why melt",
                 "top_match_llm_score": 0.9, "rationale": "sci", "degraded": False},
                {"top_match_utt_id": "b", "top_match_utterance": "x",
                 "top_match_llm_score": 0.8, "rationale": "??", "degraded": False},
            ],
        },
    }
    errors, preds = e11._errors_and_predictions(sys_c, test, 0.5)
    assert len(preds) == 2
    assert len(errors) == 1
    assert errors[0]["error_type"] == "false_positive"
    assert errors[0]["phrase"] == "line up"


def test_config_loads():
    cfg = e11.load_config()
    assert cfg["hard_informal_recall_floor"] == 0.80
    assert "smoke" in cfg and "full" in cfg
