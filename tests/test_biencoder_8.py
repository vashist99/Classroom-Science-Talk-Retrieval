"""Unit tests for src/biencoder_8.py (Step 8 bi-encoder).

The heavy model-dependent path (download + CPU fine-tune) is gated behind the
RUN_BIENCODER_TRAIN env var so the default suite stays fast and offline. The
data-prep, retrieval-metric, eval and embed-writing logic are all tested with a
deterministic fake encoder.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import biencoder_8 as b8


# ---------------------------------------------------------------------------
# Fixtures: tiny in-memory frames
# ---------------------------------------------------------------------------

@pytest.fixture
def frames():
    corpus = pd.DataFrame({
        "utt_id": ["utt_a", "utt_b", "utt_c"],
        "utterance": ["the ice is melting", "the plant grew", "we measured it"],
        "label": ["SCIENCE_TALK", "SCIENCE_TALK", "SCIENCE_TALK"],
    })
    pairs = pd.DataFrame({
        "pair_id": ["pair_t1", "pair_t2", "pair_v1", "pair_v2"],
        "anchor_id": ["utt_a", "utt_b", "utt_a", "utt_missing"],
        "variant_text": ["ice melts fast", "plant got bigger",
                         "look the ice melted", "orphan variant"],
        "register": ["INFORMAL", "SMALL_GROUP", "INFORMAL", "INFORMAL"],
    })
    negatives = pd.DataFrame({
        "neg_id": ["neg_t1", "neg_t2", "neg_v1"],
        "text": ["sit down please", "line up now", "quiet voices friends"],
        "source_type": ["llm_hard_negative", "transcript_clean", "transcript_clean"],
        "anchor_utt_id": ["utt_a", None, None],
    })
    splits = pd.DataFrame({
        "id": ["utt_a", "utt_b", "utt_c",
               "pair_t1", "pair_t2", "pair_v1", "pair_v2",
               "neg_t1", "neg_t2", "neg_v1"],
        "kind": ["positive", "positive", "positive",
                 "pair", "pair", "pair", "pair",
                 "negative", "negative", "negative"],
        "split": ["train", "train", "train",
                  "train", "train", "val", "val",
                  "train", "train", "val"],
    })
    return {"corpus": corpus, "pairs": pairs, "negatives": negatives, "splits": splits}


@pytest.fixture
def config():
    return {
        "base_model": "fake",
        "query_prefix": "Q: ",
        "document_prefix": "D: ",
        "eval": {"k_values": [1, 5, 10], "primary_metric": "mrr@10"},
        "train": {"seed": 17, "max_seq_length": 64},
        "output": {},
    }


class FakeST:
    """Deterministic stand-in for SentenceTransformer."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(h[: self.dim], dtype=np.uint8).astype(np.float32)
            if normalize_embeddings:
                n = np.linalg.norm(v)
                v = v / n if n else v
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training-triple construction
# ---------------------------------------------------------------------------

def test_triples_use_only_train_pairs(frames):
    triples = b8.build_training_triples(frames, seed=17)
    anchors = {t["anchor_id"] for t in triples}
    queries = {t["query"] for t in triples}
    # val variant text must never appear as a training query
    assert "look the ice melted" not in queries
    assert len(triples) == 2  # pair_t1, pair_t2 only
    assert anchors == {"utt_a", "utt_b"}


def test_triples_attach_matching_hard_negative(frames):
    triples = b8.build_training_triples(frames, seed=17)
    by_anchor = {t["anchor_id"]: t for t in triples}
    # utt_a has a train llm_hard_negative anchored to it
    assert by_anchor["utt_a"]["hard_negative"] == "sit down please"
    # utt_b has none -> falls back to in-batch (None)
    assert by_anchor["utt_b"]["hard_negative"] is None


def test_triples_skip_missing_anchor_text(frames):
    # pair_v2 anchors to utt_missing but it's a val pair anyway; ensure no crash
    triples = b8.build_training_triples(frames, seed=17)
    assert all(t["positive"] for t in triples)


def test_triples_to_input_examples_prefixes_and_arity(frames, config):
    triples = b8.build_training_triples(frames, seed=17)
    examples = b8.triples_to_input_examples(triples, config)
    arities = sorted(len(e.texts) for e in examples)
    assert arities == [2, 3]  # one with hard neg (3), one without (2)
    with_hard = [e for e in examples if len(e.texts) == 3][0]
    assert with_hard.texts[0].startswith("Q: ")
    assert with_hard.texts[1].startswith("D: ")
    assert with_hard.texts[2].startswith("D: ")


# ---------------------------------------------------------------------------
# Eval data construction
# ---------------------------------------------------------------------------

def test_build_eval_data_val_queries_and_relevance(frames):
    ed = b8.build_eval_data(frames, split="val", seed=17)
    # pair_v1 is a valid val query; pair_v2 dropped (anchor not in bank)
    assert set(ed["queries"]) == {"pair_v1"}
    assert ed["relevant"]["pair_v1"] == {"utt_a"}
    assert set(ed["anchor_bank"]) == {"utt_a", "utt_b", "utt_c"}
    assert set(ed["distractors"]) == {"neg_v1"}


def test_build_eval_data_caps_distractors(frames):
    ed = b8.build_eval_data(frames, split="val", max_distractors=0, seed=17)
    assert ed["distractors"] == {}


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def test_retrieval_metrics_perfect_rank():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    docs = np.array([[0.0, 1.0], [1.0, 0.0], [0.7071, 0.7071]], dtype=np.float32)
    out = b8.retrieval_metrics(q, ["q0"], docs, ["d0", "d1", "d2"],
                               {"q0": {"d1"}}, [1, 5, 10])
    assert out["n_queries"] == 1
    assert out["mrr@10"] == pytest.approx(1.0)
    assert out["recall@1"] == 1.0


def test_retrieval_metrics_second_rank():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    docs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # relevant is d1 (orthogonal) -> ranked 2nd
    out = b8.retrieval_metrics(q, ["q0"], docs, ["d0", "d1"],
                               {"q0": {"d1"}}, [1, 5, 10])
    assert out["mrr@10"] == pytest.approx(0.5)
    assert out["recall@1"] == 0.0
    assert out["recall@5"] == 1.0


def test_retrieval_metrics_empty():
    out = b8.retrieval_metrics(np.zeros((0, 2)), [], np.zeros((0, 2)), [], {}, [1])
    assert out["n_queries"] == 0


# ---------------------------------------------------------------------------
# evaluate_model + embed_corpus with the fake encoder
# ---------------------------------------------------------------------------

def test_evaluate_model_structure(frames, config):
    model = FakeST()
    ed = b8.build_eval_data(frames, split="val", seed=17)
    res = b8.evaluate_model(model, ed, config)
    assert "distractor" in res and "anchor_bank" in res
    assert res["distractor"]["n_queries"] == 1
    assert "mrr@10" in res["anchor_bank"]


def test_embed_corpus_writes_normalized(frames, config, tmp_path):
    config["output"] = {
        "corpus_embeddings": str(tmp_path / "emb.npy"),
        "corpus_embeddings_meta": str(tmp_path / "meta.parquet"),
    }
    stats = b8.embed_corpus(FakeST(), frames, config, tmp_path)
    assert stats["n_embedded"] == 3
    assert stats["l2_ok"] is True
    emb = np.load(tmp_path / "emb.npy")
    assert emb.shape[0] == 3
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
    meta = pd.read_parquet(tmp_path / "meta.parquet")
    assert list(meta["utt_id"]) == ["utt_a", "utt_b", "utt_c"]
    assert list(meta["row"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Strict improvement helper
# ---------------------------------------------------------------------------

def test_strict_improvement_detects_gain():
    c = b8._strict_improvement({"mrr@10": 0.5}, {"mrr@10": 0.7}, "mrr@10")
    assert c["improved"] is True
    assert c["delta"] == pytest.approx(0.2)


def test_strict_improvement_no_gain():
    c = b8._strict_improvement({"mrr@10": 0.7}, {"mrr@10": 0.7}, "mrr@10")
    assert c["improved"] is False
    assert c["met"] is False


def test_strict_improvement_saturated_counts_as_met():
    # baseline already at ceiling 1.0 -> cannot improve, but counts as met
    c = b8._strict_improvement({"recall@10": 1.0}, {"recall@10": 1.0}, "recall@10")
    assert c["improved"] is False
    assert c["saturated"] is True
    assert c["met"] is True


def test_verified_positive_corpus_filters_label():
    corpus = pd.DataFrame({
        "utt_id": ["a", "b"],
        "utterance": ["x", "y"],
        "label": ["SCIENCE_TALK", "NOT_SCIENCE_TALK"],
    })
    out = b8.verified_positive_corpus(corpus)
    assert list(out["utt_id"]) == ["a"]


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

def test_config_loads_and_has_required_keys():
    cfg = b8.load_config()
    assert "base_model" in cfg
    assert cfg["base_model"].startswith("nomic")
    assert "train" in cfg and "eval" in cfg and "output" in cfg


# ---------------------------------------------------------------------------
# Heavy: real download + CPU fine-tune (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_BIENCODER_TRAIN") != "1",
    reason="set RUN_BIENCODER_TRAIN=1 to run the real download+train smoke test",
)
def test_real_train_smoke(tmp_path):
    from src.data_loader_1 import DEFAULT_PROCESSED_DIR
    if not (DEFAULT_PROCESSED_DIR / "splits.parquet").exists():
        pytest.skip("no splits.parquet")
    report = b8.run(epochs=1, verbose=False)
    assert "dod_strict_improvement_passed" in report
    assert report["embeddings"]["l2_ok"] is True
