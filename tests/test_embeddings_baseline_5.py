"""Unit tests for src/embeddings_baseline_5.py.

All tests use an injected `embedder` callable so we never hit the live
endpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import embeddings_baseline_5 as eb


def _deterministic_embedder(dim: int = 8):
    """Returns an embedder that maps each unique text to a stable vector."""
    cache: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)

    def _embed(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = rng.normal(size=dim).astype(np.float32)
        return cache[text]

    return _embed, cache


class TestCosineSimilarityPairs:
    def test_identical_vectors_yield_one(self):
        a = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
        b = a.copy()
        out = eb.cosine_similarity_pairs(a, b)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [1.0, 1.0], atol=1e-6)

    def test_orthogonal_vectors_yield_zero(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        out = eb.cosine_similarity_pairs(a, b)
        np.testing.assert_allclose(out, [0.0], atol=1e-6)

    def test_zero_vector_yields_zero_not_nan(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        out = eb.cosine_similarity_pairs(a, b)
        assert not np.any(np.isnan(out))
        np.testing.assert_allclose(out, [0.0], atol=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            eb.cosine_similarity_pairs(np.zeros((2, 3)), np.zeros((3, 3)))


class TestEmbedTexts:
    def test_dedupes_within_batch(self):
        embedder, call_cache = _deterministic_embedder()
        texts = ["a", "b", "a", "a", "b"]
        out = eb.embed_texts(texts, embedder=embedder, verbose=False)
        assert out.shape == (5, 8)
        # only 2 unique texts -> only 2 unique entries in the embedder's cache
        assert len(call_cache) == 2
        # duplicate rows should match
        np.testing.assert_array_equal(out[0], out[2])
        np.testing.assert_array_equal(out[1], out[4])

    def test_handles_null_text(self):
        embedder, _ = _deterministic_embedder()
        out = eb.embed_texts([None, "x", np.nan], embedder=embedder, verbose=False)
        assert out.shape == (3, 8)
        # null and NaN both normalize to "" and share an embedding row
        np.testing.assert_array_equal(out[0], out[2])

    def test_raises_without_env_vars_when_no_embedder(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "")
        monkeypatch.setenv("EMBEDDING_URL", "")
        with pytest.raises(RuntimeError):
            eb.embed_texts(["foo"], verbose=False)


class TestGetEmbeddingForText:
    def test_returns_zero_on_endpoint_error(self, monkeypatch):
        def broken(**kwargs):
            raise RuntimeError("simulated endpoint outage")

        monkeypatch.setattr(eb, "cached_request", broken)
        out = eb.get_embedding_for_text(
            "hello", model="m", api_key="k", url="u",
        )
        assert out.shape == (eb.EMBED_DIM,)
        assert np.allclose(out, 0.0)

    def test_empty_text_short_circuits(self):
        out = eb.get_embedding_for_text(
            "", model="m", api_key="k", url="u",
        )
        assert np.allclose(out, 0.0)

    def test_returns_array_from_endpoint_payload(self, monkeypatch):
        def fake(**kwargs):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        monkeypatch.setattr(eb, "cached_request", fake)
        out = eb.get_embedding_for_text(
            "hi", model="m", api_key="k", url="u",
        )
        np.testing.assert_allclose(out, [0.1, 0.2, 0.3])


class TestAddBaselineCosineToPairs:
    def test_writes_column_in_range(self):
        embedder, _ = _deterministic_embedder()
        df = pd.DataFrame({
            "pair_id": ["p1", "p2"],
            "anchor_text": ["hello world", "the cat sat"],
            "variant_text": ["hello world", "the dog ran"],
        })
        out = eb.add_baseline_cosine_to_pairs(df, embedder=embedder, verbose=False)
        assert "baseline_cosine" in out.columns
        # identical strings -> cosine 1.0
        assert pytest.approx(out["baseline_cosine"].iloc[0], abs=1e-6) == 1.0
        assert -1.0 <= out["baseline_cosine"].iloc[1] <= 1.0

    def test_empty_frame_returns_empty_column(self):
        df = pd.DataFrame({"pair_id": [], "anchor_text": [], "variant_text": []})
        out = eb.add_baseline_cosine_to_pairs(df, verbose=False)
        assert "baseline_cosine" in out.columns
        assert len(out) == 0


class TestAddBaselineCosineToNegatives:
    def test_only_anchored_rows_get_cosine(self):
        embedder, _ = _deterministic_embedder()
        corpus = pd.DataFrame({
            "utt_id": ["utt_0001", "utt_0002"],
            "utterance": ["plants need water", "rocks are heavy"],
        })
        neg = pd.DataFrame({
            "neg_id": ["n1", "n2", "n3"],
            "text": ["i ate cake", "tomatoes are red", "the sky was blue"],
            "source_type": ["llm_hard_negative", "transcript_clean", "llm_hard_negative"],
            "anchor_utt_id": ["utt_0001", None, "utt_0002"],
        })
        out = eb.add_baseline_cosine_to_negatives(
            neg, corpus, embedder=embedder, verbose=False,
        )
        assert "baseline_cosine" in out.columns
        assert pd.isna(out["baseline_cosine"].iloc[1])  # transcript_clean -> NaN
        assert not pd.isna(out["baseline_cosine"].iloc[0])
        assert not pd.isna(out["baseline_cosine"].iloc[2])

    def test_empty_anchored_rows_returns_all_nan(self):
        corpus = pd.DataFrame({"utt_id": ["utt_0001"], "utterance": ["x"]})
        neg = pd.DataFrame({
            "neg_id": ["n1"],
            "text": ["only transcript"],
            "source_type": ["transcript_clean"],
            "anchor_utt_id": [None],
        })
        out = eb.add_baseline_cosine_to_negatives(neg, corpus, verbose=False)
        assert pd.isna(out["baseline_cosine"].iloc[0])


class TestRunEndToEnd:
    def test_writes_both_parquets(self, tmp_path):
        embedder, _ = _deterministic_embedder()
        # Build minimal frames
        corpus = pd.DataFrame({
            "utt_id": ["utt_0001"],
            "utterance": ["plants need water"],
        })
        pairs = pd.DataFrame({
            "pair_id": ["p1"],
            "anchor_text": ["plants need water"],
            "variant_text": ["plants need water"],
        })
        neg = pd.DataFrame({
            "neg_id": ["n1", "n2"],
            "text": ["i like soup", "the sun was up"],
            "source_type": ["llm_hard_negative", "transcript_clean"],
            "anchor_utt_id": ["utt_0001", None],
        })
        corpus.to_parquet(tmp_path / "corpus.parquet", index=False)
        pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
        neg.to_parquet(tmp_path / "negatives.parquet", index=False)

        eb.run(processed_dir=tmp_path, verbose=False, embedder=embedder)

        out_pairs = pd.read_parquet(tmp_path / "pairs.parquet")
        out_neg = pd.read_parquet(tmp_path / "negatives.parquet")
        assert "baseline_cosine" in out_pairs.columns
        assert "baseline_cosine" in out_neg.columns
        assert not pd.isna(out_neg["baseline_cosine"].iloc[0])
        assert pd.isna(out_neg["baseline_cosine"].iloc[1])
