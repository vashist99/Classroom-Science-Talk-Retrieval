"""Unit tests for src/review_6.py (plan Step 6: human review tooling)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import review_6 as r6


def _make_pairs(n_review=3, n_spot=10, n_auto=20) -> pd.DataFrame:
    rows = []
    i = 0
    for routing, count in (("review", n_review), ("spot", n_spot), ("auto", n_auto)):
        for _ in range(count):
            rows.append({
                "pair_id": f"pair_{i:05d}",
                "anchor_id": f"utt_{i:04d}",
                "anchor_text": f"anchor {i}",
                "variant_text": f"variant {i}",
                "register": "LARGE_GROUP",
                "confidence": 0.5,
                "routing": routing,
            })
            i += 1
    return pd.DataFrame(rows)


def _make_negs(n_review=5, n_spot=20, n_auto=30, n_transcript=15) -> pd.DataFrame:
    rows = []
    i = 0
    for routing, count in (("review", n_review), ("spot", n_spot), ("auto", n_auto)):
        for _ in range(count):
            rows.append({
                "neg_id": f"neg_{i:05d}",
                "text": f"negative {i}",
                "source_type": "seed_word_nonscience" if i % 2 else "llm_hard_negative",
                "anchor_seed_term": "force" if i % 2 else None,
                "confidence": 0.5,
                "routing": routing,
            })
            i += 1
    for _ in range(n_transcript):
        rows.append({
            "neg_id": f"neg_{i:05d}",
            "text": f"transcript line {i}",
            "source_type": "transcript_clean",
            "anchor_seed_term": None,
            "confidence": np.nan,
            "routing": None,
        })
        i += 1
    return pd.DataFrame(rows)


def _write_parquets(tmp_path, pairs=None, negs=None):
    pairs = _make_pairs() if pairs is None else pairs
    negs = _make_negs() if negs is None else negs
    pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
    negs.to_parquet(tmp_path / "negatives.parquet", index=False)
    return pairs, negs


class TestEnsureDecisionColumns:
    def test_adds_missing_columns(self):
        df = pd.DataFrame({"pair_id": ["p1"]})
        out = r6.ensure_decision_columns(df)
        for col in r6.DECISION_COLUMNS:
            assert col in out.columns
            assert pd.isna(out[col].iloc[0])

    def test_preserves_existing_values(self):
        df = pd.DataFrame({"pair_id": ["p1"], "decision": ["accept"]})
        out = r6.ensure_decision_columns(df)
        assert out["decision"].iloc[0] == "accept"


class TestBuildReviewQueue:
    def test_includes_all_review_rows(self):
        pairs, negs = _make_pairs(), _make_negs()
        q = r6.build_review_queue(pairs, negs, spot_rate=0.12, seed=1)
        review_ids_in_queue = set(q[q["routing"] == "review"]["id"])
        expected = set(pairs[pairs["routing"] == "review"]["pair_id"]) | \
                   set(negs[negs["routing"] == "review"]["neg_id"])
        assert review_ids_in_queue == expected

    def test_spot_sampled_at_rate(self):
        pairs, negs = _make_pairs(n_spot=100), _make_negs(n_spot=100)
        q = r6.build_review_queue(pairs, negs, spot_rate=0.10, seed=1)
        n_spot = (q["routing"] == "spot").sum()
        # 10% of 100 + 10% of 100 = ~20
        assert 15 <= n_spot <= 25

    def test_excludes_decided_ids(self):
        pairs, negs = _make_pairs(), _make_negs()
        all_review = set(pairs[pairs["routing"] == "review"]["pair_id"])
        decided = {sorted(all_review)[0]}
        q = r6.build_review_queue(pairs, negs, decided_ids=decided, seed=1)
        assert decided.isdisjoint(set(q["id"]))

    def test_transcript_negatives_never_queued(self):
        q = r6.build_review_queue(_make_pairs(0, 0, 0), _make_negs(0, 0, 0, n_transcript=10))
        assert len(q) == 0

    def test_pair_auto_sample(self):
        pairs, negs = _make_pairs(n_review=0, n_spot=0, n_auto=50), _make_negs(0, 0, 0, 0)
        q = r6.build_review_queue(pairs, negs, pair_auto_sample=5, seed=1)
        assert len(q) == 5
        assert (q["routing"] == "auto").all()

    def test_queue_has_decision_columns(self):
        q = r6.build_review_queue(_make_pairs(), _make_negs())
        for col in r6.DECISION_COLUMNS:
            assert col in q.columns


class TestExportReviewQueue:
    def test_writes_parquet_and_xlsx(self, tmp_path):
        _write_parquets(tmp_path)
        path = r6.export_review_queue(tmp_path, verbose=False)
        assert path is not None and path.exists()
        assert (tmp_path / "review_samples" / "review_queue.xlsx").exists()

    def test_empty_queue_returns_none(self, tmp_path):
        _write_parquets(
            tmp_path,
            pairs=_make_pairs(0, 0, 5),
            negs=_make_negs(0, 0, 5, 5),
        )
        path = r6.export_review_queue(tmp_path, verbose=False)
        assert path is None


class TestValidateDecisions:
    def _template(self, **overrides):
        row = {
            "id": "pair_00000", "kind": "pair",
            "decision": "accept", "reviewer": "DH",
            "decided_at": pd.NA, "corrected_text": pd.NA,
            "reject_reason": pd.NA, "reject_notes": pd.NA,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_accept_passes(self):
        out = r6.validate_decisions(self._template())
        assert len(out) == 1

    def test_undecided_rows_skipped(self):
        out = r6.validate_decisions(self._template(decision=pd.NA))
        assert len(out) == 0

    def test_invalid_decision_raises(self):
        with pytest.raises(ValueError, match="invalid decision"):
            r6.validate_decisions(self._template(decision="maybe"))

    def test_reject_requires_valid_reason(self):
        with pytest.raises(ValueError, match="reject_reason"):
            r6.validate_decisions(self._template(decision="reject"))
        with pytest.raises(ValueError, match="reject_reason"):
            r6.validate_decisions(self._template(decision="reject", reject_reason="bogus"))
        out = r6.validate_decisions(
            self._template(decision="reject", reject_reason="meaning_drift"))
        assert len(out) == 1

    def test_edit_requires_corrected_text(self):
        with pytest.raises(ValueError, match="corrected_text"):
            r6.validate_decisions(self._template(decision="edit"))
        out = r6.validate_decisions(
            self._template(decision="edit", corrected_text="fixed text"))
        assert len(out) == 1

    def test_case_and_whitespace_normalized(self):
        out = r6.validate_decisions(self._template(decision=" ACCEPT "))
        assert out["decision"].iloc[0] == "accept"


class TestApplyReviewDecisions:
    def _fill_and_apply(self, tmp_path, fills: dict, reviewer="DH"):
        """Export queue, fill given {id: decision_dict}, apply."""
        path = r6.export_review_queue(tmp_path, verbose=False)
        q = pd.read_parquet(path)
        for id_, vals in fills.items():
            for col, v in vals.items():
                q.loc[q["id"] == id_, col] = v
        filled = tmp_path / "filled.parquet"
        q.to_parquet(filled, index=False)
        return r6.apply_review_decisions(filled, tmp_path, reviewer=reviewer, verbose=False)

    def test_decisions_land_in_sidecar_and_parquets(self, tmp_path):
        pairs, negs = _write_parquets(tmp_path)
        review_pair = pairs[pairs["routing"] == "review"]["pair_id"].iloc[0]
        review_neg = negs[negs["routing"] == "review"]["neg_id"].iloc[0]
        result = self._fill_and_apply(tmp_path, {
            review_pair: {"decision": "accept"},
            review_neg: {"decision": "reject", "reject_reason": "not_actually_negative"},
        })
        assert result["n_applied"] == 2

        sidecar = r6.load_sidecar(tmp_path)
        assert set(sidecar["id"]) == {review_pair, review_neg}

        out_pairs = pd.read_parquet(tmp_path / "pairs.parquet")
        row = out_pairs[out_pairs["pair_id"] == review_pair].iloc[0]
        assert row["decision"] == "accept"
        assert row["reviewer"] == "DH"
        assert pd.notna(row["decided_at"])

        out_negs = pd.read_parquet(tmp_path / "negatives.parquet")
        row = out_negs[out_negs["neg_id"] == review_neg].iloc[0]
        assert row["decision"] == "reject"
        assert row["reject_reason"] == "not_actually_negative"

    def test_resume_excludes_decided_from_next_export(self, tmp_path):
        pairs, _ = _write_parquets(tmp_path)
        review_pair = pairs[pairs["routing"] == "review"]["pair_id"].iloc[0]
        self._fill_and_apply(tmp_path, {review_pair: {"decision": "accept"}})

        path2 = r6.export_review_queue(tmp_path, verbose=False)
        q2 = pd.read_parquet(path2)
        assert review_pair not in set(q2["id"])

    def test_reapply_newer_decision_wins(self, tmp_path):
        pairs, _ = _write_parquets(tmp_path)
        review_pair = pairs[pairs["routing"] == "review"]["pair_id"].iloc[0]
        self._fill_and_apply(tmp_path, {review_pair: {"decision": "accept"}})

        # A revision template written directly (the queue excludes decided ids,
        # so the human edits the original template / a manual correction file).
        revision = pd.DataFrame([{
            "id": review_pair, "kind": "pair",
            "decision": "reject", "reject_reason": "meaning_drift",
            "reviewer": "DH", "decided_at": pd.NA,
            "corrected_text": pd.NA, "reject_notes": pd.NA,
        }])
        rev_path = tmp_path / "revision.parquet"
        revision.to_parquet(rev_path, index=False)
        r6.apply_review_decisions(rev_path, tmp_path, reviewer="DH", verbose=False)

        sidecar = r6.load_sidecar(tmp_path)
        assert len(sidecar[sidecar["id"] == review_pair]) == 1
        assert sidecar[sidecar["id"] == review_pair]["decision"].iloc[0] == "reject"
        out_pairs = pd.read_parquet(tmp_path / "pairs.parquet")
        assert out_pairs[out_pairs["pair_id"] == review_pair]["decision"].iloc[0] == "reject"

    def test_missing_reviewer_raises(self, tmp_path):
        pairs, _ = _write_parquets(tmp_path)
        review_pair = pairs[pairs["routing"] == "review"]["pair_id"].iloc[0]
        with pytest.raises(ValueError, match="reviewer"):
            self._fill_and_apply(
                tmp_path, {review_pair: {"decision": "accept"}}, reviewer=None,
            )

    def test_decisions_survive_parquet_rescore(self, tmp_path):
        """Simulates confidence_5 rescoring rewriting the parquet: sidecar
        re-application restores decisions."""
        pairs, _ = _write_parquets(tmp_path)
        review_pair = pairs[pairs["routing"] == "review"]["pair_id"].iloc[0]
        self._fill_and_apply(tmp_path, {review_pair: {"decision": "accept"}})

        # Simulate a rescore wiping decision columns
        wiped = pd.read_parquet(tmp_path / "pairs.parquet").drop(
            columns=list(r6.DECISION_COLUMNS))
        wiped.to_parquet(tmp_path / "pairs.parquet", index=False)

        # Re-project from sidecar
        sidecar = r6.load_sidecar(tmp_path)
        r6._write_decisions_into_parquet(
            tmp_path / "pairs.parquet", "pair_id",
            sidecar[sidecar["kind"] == "pair"],
        )
        restored = pd.read_parquet(tmp_path / "pairs.parquet")
        assert restored[restored["pair_id"] == review_pair]["decision"].iloc[0] == "accept"


class TestReviewStatus:
    def test_pending_before_decisions(self, tmp_path):
        _write_parquets(tmp_path)
        s = r6.review_status(tmp_path, verbose=False)
        assert s["dod_met"] is False
        assert s["kinds"]["pair"]["n_review_pending"] == 3
        assert s["kinds"]["negative"]["n_review_pending"] == 5

    def test_dod_met_after_all_review_and_spot(self, tmp_path):
        pairs, negs = _write_parquets(tmp_path)
        path = r6.export_review_queue(tmp_path, spot_rate=0.12, verbose=False)
        q = pd.read_parquet(path)
        q["decision"] = "accept"
        filled = tmp_path / "filled.parquet"
        q.to_parquet(filled, index=False)
        r6.apply_review_decisions(filled, tmp_path, reviewer="DH", verbose=False)

        s = r6.review_status(tmp_path, spot_rate=0.12, verbose=False)
        assert s["dod_met"] is True


class TestFilterVerified:
    def _df(self):
        df = _make_pairs(n_review=2, n_spot=2, n_auto=2)
        df = r6.ensure_decision_columns(df)
        ids = df["pair_id"].tolist()
        # review row 0: accepted; review row 1: rejected
        df.loc[df["pair_id"] == ids[0], "decision"] = "accept"
        df.loc[df["pair_id"] == ids[1], "decision"] = "reject"
        df.loc[df["pair_id"] == ids[1], "reject_reason"] = "meaning_drift"
        # spot row 2: edited
        df.loc[df["pair_id"] == ids[2], "decision"] = "edit"
        df.loc[df["pair_id"] == ids[2], "corrected_text"] = "edited text"
        # spot row 3: undecided (pending) -> dropped
        # auto rows 4, 5: undecided -> kept
        return df, ids

    def test_keeps_and_drops_correctly(self):
        df, ids = self._df()
        out = r6.filter_verified(df, kind="pair")
        kept = set(out["pair_id"])
        assert ids[0] in kept          # accepted
        assert ids[1] not in kept      # rejected
        assert ids[2] in kept          # edited
        assert ids[3] not in kept      # pending spot
        assert ids[4] in kept and ids[5] in kept  # auto undecided

    def test_edit_replaces_text(self):
        df, ids = self._df()
        out = r6.filter_verified(df, kind="pair")
        assert out[out["pair_id"] == ids[2]]["variant_text"].iloc[0] == "edited text"

    def test_transcript_negatives_kept(self):
        negs = _make_negs(0, 0, 0, n_transcript=5)
        out = r6.filter_verified(negs, kind="negative")
        assert len(out) == 5

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError):
            r6.filter_verified(pd.DataFrame(), kind="bogus")
