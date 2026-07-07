"""Unit tests for src/query_10.py (Step 10 query-time pipeline).

All tests run offline: the heavy SentenceTransformer + real endpoint are
replaced by a tiny deterministic fake model and an injected caller. The real
path is exercised via `python src/query_10.py --smoke`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import query_10 as q10


class FakeST:
    """3-dim deterministic encoder: maps a keyword to a one-hot anchor."""

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        out = []
        for t in texts:
            tl = t.lower()
            if "ice" in tl:
                v = [1.0, 0.0, 0.0]
            elif "plant" in tl:
                v = [0.0, 1.0, 0.0]
            else:
                v = [0.0, 0.0, 1.0]
            out.append(v)
        a = np.asarray(out, dtype=np.float32)
        # _encode passes normalize_embeddings=True, but our vectors are unit;
        # normalize anyway for safety.
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return a / n


def _make_pipe(caller, *, threshold=0.5, degraded_thr=0.5, max_workers=1, top_k=3):
    import src.reranker_9 as r9
    p = q10.QueryPipeline.__new__(q10.QueryPipeline)
    p.config = {"model_env": "X"}
    p.verbose = False
    p.top_k = top_k
    p.aggregation = "max"
    p.score_threshold = threshold
    p.degraded_cosine_threshold = degraded_thr
    p.max_workers = max_workers
    p.wall_cap = 60
    p.prompt_version = "reranker_v1"
    p.query_prefix = ""
    p.model = FakeST()
    p.corpus_emb = np.eye(3, dtype=np.float32)  # 3 anchors, one-hot
    p.meta_utt_ids = ["ice_anchor", "plant_anchor", "other_anchor"]
    p.meta_utts = ["why did the ice melt", "how did the plant grow", "we counted blocks"]
    p.subtype_map = {}
    p.system_prompt = "sys"
    p.ledger = r9.CallLedger()
    p._r9 = r9
    p.caller = caller
    p.max_retries = 1
    return p


def _caller_by_keyword(system, user, prompt_version):
    """High score if candidate text shares the query's keyword, else low."""
    import re
    phrases = re.findall(r'"([^"]*)"', user)
    q, c = (phrases + ["", ""])[:2]
    ql, cl = q.lower(), c.lower()
    same = any(k in ql and k in cl for k in ("ice", "plant", "block", "count"))
    return json.dumps({"score": 0.95 if same else 0.05, "rationale": "kw"})


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

def test_retrieve_top1_matches_keyword():
    pipe = _make_pipe(_caller_by_keyword)
    cands = pipe.retrieve("what about the ice cube")
    assert cands[0]["utt_id"] == "ice_anchor"
    assert cands[0]["bi_score"] == pytest.approx(1.0)
    assert len(cands) == 3


def test_retrieve_respects_top_k():
    pipe = _make_pipe(_caller_by_keyword, top_k=2)
    assert len(pipe.retrieve("ice")) == 2


# ---------------------------------------------------------------------------
# classify — happy path
# ---------------------------------------------------------------------------

def test_classify_science_label_and_max_score():
    pipe = _make_pipe(_caller_by_keyword, threshold=0.5)
    res = pipe.classify("why did the ice disappear")
    assert res["label"] == q10.LABEL_SCIENCE
    assert res["score"] == pytest.approx(0.95)
    assert res["degraded"] is False
    # ranked by llm_score desc -> matching anchor first
    assert res["ranked_candidates"][0]["utt_id"] == "ice_anchor"
    assert res["ranked_candidates"][0]["llm_score"] == pytest.approx(0.95)


def test_classify_not_science_when_below_threshold():
    # caller returns 0.05 for everything that doesn't share a keyword
    def low(system, user, pv):
        return json.dumps({"score": 0.05, "rationale": "no"})
    pipe = _make_pipe(low, threshold=0.5)
    res = pipe.classify("zzz nonsense unrelated")
    assert res["label"] == q10.LABEL_NOT
    assert res["score"] == pytest.approx(0.05)


def test_classify_schema_fields():
    pipe = _make_pipe(_caller_by_keyword)
    res = pipe.classify("ice melting")
    for c in res["ranked_candidates"]:
        assert set(["utt_id", "utterance", "llm_score", "bi_score", "rationale"]).issubset(c)


# ---------------------------------------------------------------------------
# Determinism + parallelism
# ---------------------------------------------------------------------------

def test_classify_deterministic():
    pipe = _make_pipe(_caller_by_keyword)
    a = pipe.classify("ice melting")
    b = pipe.classify("ice melting")
    assert a["score"] == b["score"]
    assert [c["utt_id"] for c in a["ranked_candidates"]] == \
           [c["utt_id"] for c in b["ranked_candidates"]]


def test_parallel_matches_sequential():
    seq = _make_pipe(_caller_by_keyword, max_workers=1).classify("ice")
    par = _make_pipe(_caller_by_keyword, max_workers=4).classify("ice")
    assert [c["utt_id"] for c in seq["ranked_candidates"]] == \
           [c["utt_id"] for c in par["ranked_candidates"]]
    assert seq["score"] == par["score"]


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_classify_degrades_on_endpoint_failure():
    def down(system, user, pv):
        raise ConnectionError("endpoint down")
    pipe = _make_pipe(down, degraded_thr=0.5)
    res = pipe.classify("why did the ice melt")  # cosine top-1 bi_score 1.0
    assert res["degraded"] is True
    assert res["label"] == q10.LABEL_SCIENCE  # 1.0 >= 0.5
    assert all(c["llm_score"] is None for c in res["ranked_candidates"])
    # ranked by cosine
    assert res["ranked_candidates"][0]["utt_id"] == "ice_anchor"


def test_degraded_respects_cosine_threshold():
    def down(system, user, pv):
        raise ConnectionError("down")
    pipe = _make_pipe(down, degraded_thr=1.5)  # impossible cosine -> NOT
    res = pipe.classify("ice")
    assert res["degraded"] is True
    assert res["label"] == q10.LABEL_NOT


def test_partial_parse_failure_does_not_degrade():
    import re

    def flaky(system, user, pv):
        # key off the CANDIDATE (Phrase B), not the whole message
        m = re.search(r'Phrase B \(candidate\): "([^"]*)"', user)
        cand = (m.group(1) if m else "").lower()
        if "ice" in cand:  # this one candidate is always unparsable
            return "garbage"
        return json.dumps({"score": 0.3, "rationale": "ok"})

    pipe = _make_pipe(flaky, threshold=0.5)
    res = pipe.classify("plant growth")
    # at least one success -> not degraded; failed candidate has None llm_score
    assert res["degraded"] is False
    assert any(c["llm_score"] is None for c in res["ranked_candidates"])


# ---------------------------------------------------------------------------
# build_dev_queries
# ---------------------------------------------------------------------------

def test_build_dev_queries():
    frames = {
        "corpus": pd.DataFrame({"utt_id": ["a"], "utterance": ["x"], "label": ["SCIENCE_TALK"]}),
        "pairs": pd.DataFrame({
            "pair_id": ["p1", "p2"], "anchor_id": ["a", "a"],
            "variant_text": ["why melt", "how grow"]}),
        "negatives": pd.DataFrame({"neg_id": ["n1"], "text": ["line up please"]}),
        "splits": pd.DataFrame({
            "id": ["p1", "p2", "n1"], "kind": ["pair", "pair", "negative"],
            "split": ["val", "val", "val"]}),
    }
    pos, neg = q10.build_dev_queries(frames, n_pos=5, n_neg=5)
    assert len(pos) == 2 and len(neg) == 1
    assert all("gold_anchor" in p and "phrase" in p for p in pos)


def test_config_loads():
    cfg = q10.load_config()
    assert cfg["top_k"] == 20
    assert cfg["aggregation"] == "max"
