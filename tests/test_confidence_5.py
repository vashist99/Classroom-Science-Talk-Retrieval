"""Unit tests for src/confidence_5.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import confidence_5 as c5


def _make_config(**overrides) -> dict:
    base = {
        "version": "test_v1",
        "baseline_encoder": {"model": "nomic-embed-text-v1.5", "prompt_version": "v1", "dim": 768},
        "pairs": {"cosine_band_min": 0.55, "cosine_band_max": 0.95, "cosine_band_falloff": 0.30},
        "negatives": {"cosine_band_min": 0.30, "cosine_band_max": 0.85, "cosine_band_falloff": 0.30},
        "weights": {"llm": 1.0, "cosine": 1.0, "structural": 1.0},
        "routing": {"auto_min": 0.70, "spot_min": 0.45},
        "audit": {"n_rows": 50, "seed": 42, "agreement_target": 0.80},
    }
    base.update(overrides)
    return base


def _make_pair_row(**kw):
    row = {
        "pair_id": "p1",
        "anchor_id": "utt_0001",
        "anchor_text": "anchor",
        "variant_text": "variant",
        "register": "LARGE_GROUP",
        "llm_self_score": 0.9,
        "baseline_cosine": 0.75,
        "preservation_pct": 1.0,
        "preservation_check_passed": True,
        "differs_from_anchor": True,
    }
    row.update(kw)
    return row


def _make_neg_row(**kw):
    row = {
        "neg_id": "n1",
        "text": "candidate",
        "source_type": "llm_hard_negative",
        "anchor_utt_id": "utt_0001",
        "anchor_seed_term": None,
        "structural_check_passed": True,
        "llm_gate_score": 0.9,
        "baseline_cosine": 0.6,
    }
    row.update(kw)
    return row


class TestLoadConfig:
    def test_round_trip(self, tmp_path):
        cfg = _make_config()
        p = tmp_path / "c.json"
        with open(p, "w") as f:
            json.dump(cfg, f)
        loaded = c5.load_config(p)
        assert loaded["routing"]["auto_min"] == 0.70

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            c5.load_config(tmp_path / "nope.json")


class TestBandScore:
    def test_inside_band_is_one(self):
        assert c5.band_score(0.75, 0.5, 0.9, falloff=0.2).item() == 1.0

    def test_at_edge_is_one(self):
        assert c5.band_score(0.9, 0.5, 0.9, falloff=0.2).item() == 1.0
        assert c5.band_score(0.5, 0.5, 0.9, falloff=0.2).item() == 1.0

    def test_outside_band_ramps_down(self):
        # 0.10 above hi=0.9, falloff=0.20  -> score = 1 - 0.5 = 0.5
        assert c5.band_score(1.0, 0.5, 0.9, falloff=0.2).item() == pytest.approx(0.5)
        # 0.10 below lo=0.5, falloff=0.20 -> score = 0.5
        assert c5.band_score(0.4, 0.5, 0.9, falloff=0.2).item() == pytest.approx(0.5)

    def test_far_outside_clips_to_zero(self):
        assert c5.band_score(0.0, 0.5, 0.9, falloff=0.1).item() == 0.0
        assert c5.band_score(2.0, 0.5, 0.9, falloff=0.1).item() == 0.0

    def test_nan_yields_neutral_half(self):
        out = c5.band_score(np.nan, 0.5, 0.9, falloff=0.2)
        assert out.item() == pytest.approx(0.5)

    def test_vectorized(self):
        x = np.array([0.7, 1.0, 0.0, np.nan])
        out = c5.band_score(x, 0.5, 0.9, falloff=0.2)
        np.testing.assert_allclose(out, [1.0, 0.5, 0.0, 0.5], atol=1e-6)


class TestWeightedGeometricMean:
    def test_equal_weights_recover_geomean(self):
        v = np.array([[0.9, 0.9, 0.9]])
        out = c5.weighted_geometric_mean(v, np.array([1.0, 1.0, 1.0]))
        assert out[0] == pytest.approx(0.9, abs=1e-6)

    def test_low_signal_drags_mean_down(self):
        v = np.array([[0.9, 0.1, 0.9]])
        out = c5.weighted_geometric_mean(v, np.array([1.0, 1.0, 1.0]))
        assert out[0] < 0.5  # one bad signal nukes the score

    def test_floor_prevents_zero(self):
        v = np.array([[0.9, 0.0, 0.9]])
        out = c5.weighted_geometric_mean(v, np.array([1.0, 1.0, 1.0]))
        assert out[0] > 0.0  # floored

    def test_weights_zero_raises(self):
        with pytest.raises(ValueError):
            c5.weighted_geometric_mean(
                np.array([[0.5, 0.5]]),
                np.array([0.0, 0.0]),
            )

    def test_weighted_skew(self):
        # If llm weight is dominant, score should track llm signal
        v = np.array([[0.9, 0.1, 0.1]])
        out = c5.weighted_geometric_mean(v, np.array([10.0, 1.0, 1.0]))
        assert out[0] > 0.6


class TestScorePairs:
    def test_happy_path_auto(self):
        df = pd.DataFrame([_make_pair_row(
            llm_self_score=0.95, baseline_cosine=0.75,
        )])
        out = c5.score_pairs(df, _make_config())
        assert out["routing"].iloc[0] == "auto"
        assert 0.0 < out["confidence"].iloc[0] <= 1.0

    def test_anchor_leak_collapses_structural(self):
        df = pd.DataFrame([_make_pair_row(
            differs_from_anchor=False,  # variant == anchor
            llm_self_score=0.95,
        )])
        out = c5.score_pairs(df, _make_config())
        assert out["confidence_structural"].iloc[0] == 0.0
        assert out["routing"].iloc[0] == "review"

    def test_differs_only_structural_ignores_preservation_pct(self):
        cfg = _make_config()
        cfg["pairs"]["structural_mode"] = "differs_only"
        cfg["routing"] = {"auto_min": 0.5, "spot_min": 0.1}
        df = pd.DataFrame([_make_pair_row(
            preservation_check_passed=False,
            preservation_pct=0.0,
            differs_from_anchor=True,
            llm_self_score=0.9,
            baseline_cosine=0.88,
        )])
        out = c5.score_pairs(df, cfg)
        assert out["confidence_structural"].iloc[0] == 1.0
        assert out["routing"].iloc[0] == "auto"

    def test_refresh_audit_routing_preserves_row_order(self):
        pairs = pd.DataFrame([
            _make_pair_row(pair_id="p1", baseline_cosine=0.7),
            _make_pair_row(pair_id="p2", baseline_cosine=0.7),
        ])
        pairs = c5.score_pairs(pairs, _make_config())
        negs = pd.DataFrame([_make_neg_row(neg_id="n1", baseline_cosine=0.6)])
        negs = c5.score_negatives(negs, _make_config())
        audit = pd.DataFrame({
            "id": ["n1", "p1", "p2"],
            "kind": ["negative", "pair", "pair"],
            "human_routing": ["auto", "spot", "review"],
            "routing": ["review", "auto", "spot"],
        })
        out = c5.refresh_audit_routing(audit, pairs, negs)
        assert list(out["id"]) == ["n1", "p1", "p2"]
        assert out.iloc[0]["routing"] == negs.iloc[0]["routing"]
        assert out.iloc[1]["routing"] == pairs.iloc[0]["routing"]

    def test_band_extreme_cosine_routes_to_review(self):
        # cosine=1.0 with band [0.55, 0.95] falloff 0.30 -> band_score ~0.83.
        # Combined with already-modest LLM should still drop into spot/review.
        df = pd.DataFrame([_make_pair_row(
            llm_self_score=0.5, baseline_cosine=1.0,
            preservation_pct=0.6,
        )])
        out = c5.score_pairs(df, _make_config())
        assert out["routing"].iloc[0] in ("spot", "review")

    def test_missing_baseline_cosine_uses_neutral(self):
        df = pd.DataFrame([_make_pair_row(baseline_cosine=np.nan)])
        out = c5.score_pairs(df, _make_config())
        assert out["confidence_cosine"].iloc[0] == pytest.approx(0.5)

    def test_three_signal_columns_present(self):
        df = pd.DataFrame([_make_pair_row(), _make_pair_row()])
        out = c5.score_pairs(df, _make_config())
        for col in ("confidence_llm", "confidence_cosine", "confidence_structural",
                    "confidence", "routing"):
            assert col in out.columns


class TestScoreNegatives:
    def test_llm_hard_negative_routed(self):
        df = pd.DataFrame([_make_neg_row(
            llm_gate_score=0.9, baseline_cosine=0.6,
        )])
        out = c5.score_negatives(df, _make_config())
        assert out["routing"].iloc[0] in ("auto", "spot", "review")
        assert not pd.isna(out["confidence"].iloc[0])

    def test_transcript_clean_skipped(self):
        df = pd.DataFrame([_make_neg_row(
            source_type="transcript_clean",
            anchor_utt_id=None,
            baseline_cosine=np.nan,
            llm_gate_score=np.nan,
        )])
        out = c5.score_negatives(df, _make_config())
        assert out["routing"].iloc[0] is None
        assert pd.isna(out["confidence"].iloc[0])

    def test_seed_word_nonscience_is_routed(self):
        df = pd.DataFrame([_make_neg_row(
            source_type="seed_word_nonscience",
            anchor_utt_id=None,
            baseline_cosine=np.nan,
            llm_gate_score=0.8,
        )])
        out = c5.score_negatives(df, _make_config())
        assert out["routing"].iloc[0] is not None

    def test_structural_failure_collapses_score(self):
        df = pd.DataFrame([_make_neg_row(structural_check_passed=False)])
        out = c5.score_negatives(df, _make_config())
        assert out["confidence_structural"].iloc[0] == 0.0
        # With one signal = 0.0 (floored), geomean ~ floor^(1/3) * good^(2/3)
        assert out["routing"].iloc[0] == "review"


class TestRoutingThresholds:
    def test_auto_threshold(self):
        cfg = _make_config(routing={"auto_min": 0.7, "spot_min": 0.45})
        conf = np.array([0.95, 0.72, 0.50, 0.30])
        out = c5._route(conf, 0.7, 0.45)
        assert list(out) == ["auto", "auto", "spot", "review"]

    def test_threshold_at_boundary_inclusive(self):
        out = c5._route(np.array([0.70, 0.45]), 0.7, 0.45)
        assert list(out) == ["auto", "spot"]


class TestSummarizeRouting:
    def test_counts_only_non_null(self):
        df = pd.DataFrame({"routing": ["auto", "spot", "auto", None]})
        s = c5.summarize_routing(df)
        assert s["total"] == 3
        assert s["counts"]["auto"] == 2

    def test_empty_routing_returns_zero(self):
        df = pd.DataFrame({"routing": [None, None]})
        s = c5.summarize_routing(df)
        assert s["total"] == 0


class TestExportRoutingAuditSample:
    def _build_frames(self, n_pairs: int = 60, n_neg: int = 60):
        pa = pd.DataFrame([{
            "pair_id": f"p{i}",
            "anchor_id": f"utt_{i:04d}",
            "anchor_text": f"anchor {i}",
            "variant_text": f"variant {i}",
            "register": "LARGE_GROUP",
            "confidence": float(i) / max(n_pairs - 1, 1),
            "confidence_llm": 0.8,
            "confidence_cosine": 0.7,
            "confidence_structural": 0.9,
            "routing": ("auto" if i % 3 == 0 else ("spot" if i % 3 == 1 else "review")),
        } for i in range(n_pairs)])

        ne = pd.DataFrame([{
            "neg_id": f"n{i}",
            "text": f"neg {i}",
            "source_type": "llm_hard_negative" if i % 2 == 0 else "seed_word_nonscience",
            "anchor_utt_id": f"utt_{i:04d}" if i % 2 == 0 else None,
            "anchor_seed_term": None if i % 2 == 0 else "water",
            "confidence": float(i) / max(n_neg - 1, 1),
            "confidence_llm": 0.7,
            "confidence_cosine": 0.6,
            "confidence_structural": 1.0,
            "routing": ("auto" if i % 3 == 0 else ("spot" if i % 3 == 1 else "review")),
        } for i in range(n_neg)])
        return pa, ne

    def test_emits_50_rows_with_required_cols(self, tmp_path):
        pa, ne = self._build_frames()
        c5.export_routing_audit_sample(pa, ne, tmp_path, n=50, seed=1, verbose=False)
        out = pd.read_parquet(tmp_path / "review_samples" / "routing_audit_template.parquet")
        assert len(out) == 50
        for col in ("id", "kind", "anchor_text", "candidate_text", "routing",
                    "human_routing", "agree", "notes", "confidence"):
            assert col in out.columns

    def test_stratifies_across_kinds_and_buckets(self, tmp_path):
        pa, ne = self._build_frames(n_pairs=90, n_neg=90)
        c5.export_routing_audit_sample(pa, ne, tmp_path, n=50, seed=2, verbose=False)
        out = pd.read_parquet(tmp_path / "review_samples" / "routing_audit_template.parquet")
        kinds = out["kind"].value_counts().to_dict()
        # Should have both kinds represented (not 50-0 or 0-50)
        assert kinds.get("pair", 0) >= 5
        assert kinds.get("negative", 0) >= 5
        # And all three routing buckets
        routes = out["routing"].value_counts().to_dict()
        assert routes.get("auto", 0) > 0
        assert routes.get("spot", 0) > 0
        assert routes.get("review", 0) > 0

    def test_excludes_transcript_clean(self, tmp_path):
        pa, _ = self._build_frames(n_pairs=10, n_neg=0)
        # Build a negative frame where some rows have routing=None (transcript_clean)
        ne = pd.DataFrame([{
            "neg_id": "n_skip",
            "text": "transcript line",
            "source_type": "transcript_clean",
            "anchor_utt_id": None,
            "anchor_seed_term": None,
            "confidence": np.nan,
            "confidence_llm": np.nan,
            "confidence_cosine": np.nan,
            "confidence_structural": np.nan,
            "routing": None,
        }])
        c5.export_routing_audit_sample(pa, ne, tmp_path, n=10, seed=1, verbose=False)
        out = pd.read_parquet(tmp_path / "review_samples" / "routing_audit_template.parquet")
        assert "transcript_clean" not in out["source_type"].values


class TestComputeAuditAgreement:
    def test_perfect_agreement(self):
        df = pd.DataFrame({
            "routing": ["auto", "spot", "review"],
            "human_routing": ["auto", "spot", "review"],
        })
        r = c5.compute_audit_agreement(df)
        assert r["n_reviewed"] == 3
        assert r["agreement"] == 1.0

    def test_partial_agreement_with_buckets(self):
        df = pd.DataFrame({
            "routing": ["auto", "auto", "spot", "review"],
            "human_routing": ["auto", "spot", "spot", "review"],
        })
        r = c5.compute_audit_agreement(df)
        assert r["n_reviewed"] == 4
        assert r["agreement"] == 0.75
        assert "auto" in r["by_bucket"]

    def test_no_reviews_returns_none_agreement(self):
        df = pd.DataFrame({"routing": ["auto"], "human_routing": [None]})
        r = c5.compute_audit_agreement(df)
        assert r["n_reviewed"] == 0
        assert r["agreement"] is None


class TestRunIntegration:
    def test_run_requires_baseline_cosine(self, tmp_path):
        pairs = pd.DataFrame([_make_pair_row()]).drop(columns=["baseline_cosine"])
        neg = pd.DataFrame([_make_neg_row()])
        pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
        neg.to_parquet(tmp_path / "negatives.parquet", index=False)
        # Write a config in tmp
        cfg = _make_config()
        cfg_path = tmp_path / "c.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        with pytest.raises(RuntimeError, match="baseline_cosine"):
            c5.run(processed_dir=tmp_path, config_path=cfg_path, verbose=False)

    def test_run_writes_confidence_and_audit(self, tmp_path):
        pairs = pd.DataFrame([
            _make_pair_row(pair_id=f"p{i}", llm_self_score=0.9, baseline_cosine=0.7)
            for i in range(30)
        ])
        neg = pd.DataFrame([
            _make_neg_row(neg_id=f"n{i}", llm_gate_score=0.9, baseline_cosine=0.6)
            for i in range(30)
        ])
        pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
        neg.to_parquet(tmp_path / "negatives.parquet", index=False)
        cfg = _make_config()
        cfg_path = tmp_path / "c.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        c5.run(processed_dir=tmp_path, config_path=cfg_path,
               audit_n=15, audit_seed=1, verbose=False)
        out_pairs = pd.read_parquet(tmp_path / "pairs.parquet")
        out_neg = pd.read_parquet(tmp_path / "negatives.parquet")
        assert "confidence" in out_pairs.columns
        assert "routing" in out_pairs.columns
        assert "confidence" in out_neg.columns
        audit = pd.read_parquet(tmp_path / "review_samples" / "routing_audit_template.parquet")
        assert len(audit) == 15


class TestScoreRoutingAudit:
    def _filled_audit_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": ["p1", "p2", "n1"],
            "kind": ["pair", "pair", "negative"],
            "routing": ["auto", "spot", "review"],
            "human_routing": ["auto", "auto", "spot"],
            "anchor_text": ["a", "b", "c"],
            "candidate_text": ["a1", "b1", "c1"],
        })

    def test_score_writes_artifacts(self, tmp_path):
        cfg = _make_config()
        cfg_path = tmp_path / "c.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        audit_xlsx = tmp_path / "routing_audit_template.xlsx"
        self._filled_audit_df().to_excel(audit_xlsx, index=False)

        report = c5.score_routing_audit(
            audit_xlsx, config_path=cfg_path, out_dir=tmp_path,
            refresh_routing=False, verbose=False,
        )
        assert report["n_reviewed"] == 3
        assert report["agreement"] == pytest.approx(1 / 3)
        assert report["passed"] is False  # target 0.80
        assert (tmp_path / "routing_audit_scored.parquet").exists()
        assert (tmp_path / "routing_audit_scored.xlsx").exists()
        assert (tmp_path / "routing_audit_report.json").exists()
        assert (tmp_path / "routing_audit_disagreements.xlsx").exists()

        scored = pd.read_parquet(tmp_path / "routing_audit_scored.parquet")
        assert "agree" in scored.columns
        assert scored["agree"].notna().sum() == 3

        disagreements = pd.read_excel(tmp_path / "routing_audit_disagreements.xlsx")
        assert len(disagreements) == 2
        assert set(disagreements["id"]) == {"p2", "n1"}

    def test_load_parquet_audit(self, tmp_path):
        p = tmp_path / "audit.parquet"
        self._filled_audit_df().to_parquet(p, index=False)
        loaded = c5.load_routing_audit(p)
        assert len(loaded) == 3

    def test_normalize_routing_labels(self):
        s = pd.Series([" Auto ", "SPOT", None, ""])
        out = c5._normalize_routing_labels(s)
        assert out.iloc[0] == "auto"
        assert out.iloc[1] == "spot"
        assert pd.isna(out.iloc[2])
        assert pd.isna(out.iloc[3])

    def test_rescore_processed_routing(self, tmp_path):
        cfg = _make_config()
        cfg["pairs"]["structural_mode"] = "soft"
        cfg["routing"] = {"auto_min": 0.5, "spot_min": 0.1}
        cfg_path = tmp_path / "c.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        pairs = pd.DataFrame([
            _make_pair_row(pair_id="p1", baseline_cosine=0.88, llm_self_score=0.9),
        ])
        neg = pd.DataFrame([
            _make_neg_row(neg_id="n1", baseline_cosine=0.6, llm_gate_score=0.9),
        ])
        pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
        neg.to_parquet(tmp_path / "negatives.parquet", index=False)
        c5.rescore_processed_routing(tmp_path, config_path=cfg_path, verbose=False)
        out = pd.read_parquet(tmp_path / "pairs.parquet")
        assert "routing" in out.columns
        assert out["routing"].iloc[0] == "auto"
