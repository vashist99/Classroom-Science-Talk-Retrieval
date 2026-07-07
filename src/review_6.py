"""Step 6 -- Human review tooling (plan's Step 6).

Gives the reviewer a low-friction accept/reject/edit surface over the rows
that Step 5 routing flagged, and merges those decisions back into the
processed parquets without ever losing prior work.

Workflow
--------
1. Export the review queue (all `routing=review` rows + a ~10-15% sample of
   `routing=spot` rows, minus anything already decided):

       python src/review_6.py --export

   -> data/processed/review_samples/review_queue.xlsx (+ .parquet mirror)

2. Human fills three columns per row in the xlsx:
       decision        accept | reject | edit          (required)
       reject_reason   one of REJECT_REASONS           (required when reject)
       corrected_text  replacement text                (required when edit)
   Optional: reviewer, reject_notes.

3. Apply the filled template:

       python src/review_6.py --apply data/processed/review_samples/review_queue.xlsx --reviewer "DH"

   Decisions land in BOTH:
       - data/processed/review_decisions.parquet   (sidecar, source of truth --
         survives any pipeline re-run that regenerates pairs/negatives)
       - pairs.parquet / negatives.parquet          (decision, reviewer,
         decided_at, corrected_text, reject_reason columns)

4. Check DoD status anytime:

       python src/review_6.py --status

Resume support (DoD #3): re-running --export excludes every id already in the
sidecar, so a half-finished session just produces a smaller queue next time.
Re-applying a template is idempotent; a newer decision for the same id wins.

DoD addressed by this module:
  1. All `routing=review` rows are decided; `routing=spot` sampled at ~10-15%
     (queue builder enforces the sample; --status verifies completion).
  2. Reject reasons are codified (REJECT_REASONS) with free-text notes
     alongside; invalid codes are rejected at --apply time.
  3. Mid-session resume without losing prior decisions (sidecar + exclusion).

`filter_verified()` is the downstream gate: Step 7 splits / Step 8 training
should call it to drop rejected rows and apply edited text.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR


DECISIONS = ("accept", "reject", "edit")

REJECT_REASONS = (
    "register_mismatch",      # pairs: variant doesn't sound like the target register
    "meaning_drift",          # pairs: paraphrase changed the meaning
    "anchor_copy",            # pairs: variant is (nearly) identical to anchor
    "not_actually_negative",  # negatives: this IS science talk
    "implausible_classroom",  # negatives: no pre-K teacher would say this
    "nonsense",               # either: garbled / incoherent text
    "duplicate",              # either: duplicate of another row
    "pii",                    # either: contains identifying information
    "other",                  # either: catch-all, explain in reject_notes
)

DECISION_COLUMNS = (
    "decision", "reviewer", "decided_at", "corrected_text",
    "reject_reason", "reject_notes",
)

SIDECAR_NAME = "review_decisions.parquet"
QUEUE_BASENAME = "review_queue"

DEFAULT_SPOT_RATE = 0.12  # plan: ~10-15%


# ---------- schema helpers ---------------------------------------------------

def ensure_decision_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the Step 6 decision columns (as NA) if they're missing."""
    out = df.copy()
    for col in DECISION_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _normalize_str(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA, "None": pd.NA})


def load_sidecar(processed_dir: Path) -> pd.DataFrame:
    """Load the decisions sidecar; empty frame with full schema if absent."""
    path = processed_dir / SIDECAR_NAME
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["id", "kind", *DECISION_COLUMNS])


def save_sidecar(df: pd.DataFrame, processed_dir: Path) -> Path:
    path = processed_dir / SIDECAR_NAME
    df.to_parquet(path, index=False)
    return path


# ---------- queue building ----------------------------------------------------

def build_review_queue(
    df_pairs: pd.DataFrame,
    df_neg: pd.DataFrame,
    *,
    decided_ids: set[str] | None = None,
    spot_rate: float = DEFAULT_SPOT_RATE,
    pair_auto_sample: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Assemble the queue: all `review` rows + spot sample + optional auto sample.

    `decided_ids` (from the sidecar) are excluded so a partially finished
    session resumes with only the remaining rows.

    `pair_auto_sample` > 0 additionally samples that many `routing=auto` PAIRS
    for an optional safety check (useful when routing sends all pairs to auto,
    as on the retuned Y1 config).
    """
    decided_ids = decided_ids or set()
    rng_seed = seed

    def _pairs_view(rows: pd.DataFrame) -> pd.DataFrame:
        v = rows[[
            "pair_id", "anchor_text", "variant_text", "register",
            "confidence", "routing",
        ]].copy()
        v = v.rename(columns={"pair_id": "id", "variant_text": "candidate_text"})
        v["kind"] = "pair"
        v["source_type"] = "variant"
        return v

    def _neg_view(rows: pd.DataFrame) -> pd.DataFrame:
        v = rows[[
            "neg_id", "anchor_seed_term", "text", "source_type",
            "confidence", "routing",
        ]].copy()
        v = v.rename(columns={"neg_id": "id", "text": "candidate_text"})
        # Negatives anchored to a seed term show that term; hard negatives show
        # nothing here (anchor_text resolution is informational, not required).
        v["anchor_text"] = v.pop("anchor_seed_term")
        v["register"] = pd.NA
        v["kind"] = "negative"
        return v

    parts: list[pd.DataFrame] = []

    for df, view in ((df_pairs, _pairs_view), (df_neg, _neg_view)):
        routed = df[df["routing"].notna()] if "routing" in df.columns else df.iloc[0:0]
        if len(routed) == 0:
            continue
        review_rows = routed[routed["routing"] == "review"]
        if len(review_rows):
            parts.append(view(review_rows))
        spot_rows = routed[routed["routing"] == "spot"]
        if len(spot_rows):
            n_spot = max(1, int(round(len(spot_rows) * spot_rate)))
            parts.append(view(spot_rows.sample(min(n_spot, len(spot_rows)), random_state=rng_seed)))
        rng_seed += 1

    if pair_auto_sample > 0 and "routing" in df_pairs.columns:
        auto_pairs = df_pairs[df_pairs["routing"] == "auto"]
        if len(auto_pairs):
            take = min(pair_auto_sample, len(auto_pairs))
            parts.append(_pairs_view(auto_pairs.sample(take, random_state=seed + 100)))

    if not parts:
        queue = pd.DataFrame(columns=[
            "id", "kind", "source_type", "anchor_text", "candidate_text",
            "register", "confidence", "routing",
        ])
    else:
        cols = ["id", "kind", "source_type", "anchor_text", "candidate_text",
                "register", "confidence", "routing"]
        queue = pd.concat([p[cols] for p in parts], ignore_index=True)

    queue = queue[~queue["id"].isin(decided_ids)].reset_index(drop=True)
    queue = queue.drop_duplicates(subset="id").reset_index(drop=True)

    for col in DECISION_COLUMNS:
        queue[col] = pd.NA
    return queue


def export_review_queue(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    spot_rate: float = DEFAULT_SPOT_RATE,
    pair_auto_sample: int = 0,
    seed: int = 42,
    verbose: bool = True,
) -> Path | None:
    """Build + write the review queue template. Returns None if queue is empty."""
    df_pairs = pd.read_parquet(processed_dir / "pairs.parquet")
    df_neg = pd.read_parquet(processed_dir / "negatives.parquet")
    sidecar = load_sidecar(processed_dir)
    decided = set(sidecar["id"].astype(str))

    queue = build_review_queue(
        df_pairs, df_neg,
        decided_ids=decided,
        spot_rate=spot_rate,
        pair_auto_sample=pair_auto_sample,
        seed=seed,
    )

    rd = processed_dir / "review_samples"
    rd.mkdir(parents=True, exist_ok=True)

    if len(queue) == 0:
        if verbose:
            print("  Review queue is empty -- everything flagged has been decided.")
        return None

    pq = rd / f"{QUEUE_BASENAME}.parquet"
    xl = rd / f"{QUEUE_BASENAME}.xlsx"
    queue.to_parquet(pq, index=False)
    try:
        queue.to_excel(xl, index=False)
    except Exception as e:
        if verbose:
            print(f"  (could not write xlsx mirror: {e})")
    if verbose:
        by = queue.groupby(["kind", "routing"]).size()
        print(f"  Review queue -> {xl} ({len(queue)} rows)")
        print(f"  Breakdown:\n{by.to_string()}")
        print(f"  Valid decisions : {', '.join(DECISIONS)}")
        print(f"  Reject reasons  : {', '.join(REJECT_REASONS)}")
    return pq


# ---------- applying decisions -------------------------------------------------

def validate_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + normalize a filled template. Returns decided rows only.

    Raises ValueError listing every problem (bad decision value, reject
    without a valid reason code, edit without corrected_text).
    """
    df = df.copy()
    df["decision"] = _normalize_str(df["decision"]).str.lower()
    df["reject_reason"] = _normalize_str(df["reject_reason"]).str.lower()
    if "corrected_text" in df.columns:
        df["corrected_text"] = _normalize_str(df["corrected_text"])
    if "reviewer" in df.columns:
        df["reviewer"] = _normalize_str(df["reviewer"])

    decided = df[df["decision"].notna()].copy()
    problems: list[str] = []

    bad_decision = decided[~decided["decision"].isin(DECISIONS)]
    for _, r in bad_decision.iterrows():
        problems.append(f"{r['id']}: invalid decision {r['decision']!r} "
                        f"(must be one of {DECISIONS})")

    rejects = decided[decided["decision"] == "reject"]
    bad_reason = rejects[~rejects["reject_reason"].isin(REJECT_REASONS)]
    for _, r in bad_reason.iterrows():
        problems.append(f"{r['id']}: decision=reject needs a reject_reason from "
                        f"{REJECT_REASONS}, got {r['reject_reason']!r}")

    edits = decided[decided["decision"] == "edit"]
    bad_edit = edits[edits["corrected_text"].isna()]
    for _, r in bad_edit.iterrows():
        problems.append(f"{r['id']}: decision=edit requires corrected_text")

    if problems:
        raise ValueError("Review template has problems:\n  " + "\n  ".join(problems))
    return decided


def apply_review_decisions(
    template_path: Path,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    reviewer: str | None = None,
    verbose: bool = True,
) -> dict:
    """Merge a filled review template into the sidecar + both parquets.

    Rows without a decision are skipped (still pending -- resume later).
    A new decision for an id already in the sidecar replaces the old one.
    """
    template_path = Path(template_path)
    if template_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(template_path)
    elif template_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(template_path)
    else:
        raise ValueError(f"Unsupported template type: {template_path.suffix}")

    decided = validate_decisions(df)
    if len(decided) == 0:
        if verbose:
            print("  No decided rows in template; nothing to apply.")
        return {"n_applied": 0}

    now = datetime.now(timezone.utc).isoformat()
    if "reviewer" not in decided.columns:
        decided["reviewer"] = pd.NA
    if reviewer is not None:
        decided["reviewer"] = decided["reviewer"].fillna(reviewer)
    if decided["reviewer"].isna().any():
        raise ValueError(
            "Some decided rows have no reviewer. Fill the reviewer column or "
            "pass --reviewer."
        )
    if "decided_at" not in decided.columns:
        decided["decided_at"] = now
    decided["decided_at"] = decided["decided_at"].fillna(now)

    keep = ["id", "kind", *DECISION_COLUMNS]
    for col in keep:
        if col not in decided.columns:
            decided[col] = pd.NA
    decided = decided[keep]

    # Sidecar: newest decision per id wins.
    sidecar = load_sidecar(processed_dir)
    if len(sidecar) == 0:
        sidecar = decided.copy()
    else:
        sidecar = pd.concat([sidecar, decided], ignore_index=True)
    sidecar = sidecar.drop_duplicates(subset="id", keep="last").reset_index(drop=True)
    save_sidecar(sidecar, processed_dir)

    # Mirror into the parquets.
    n_pairs = _write_decisions_into_parquet(
        processed_dir / "pairs.parquet", "pair_id",
        sidecar[sidecar["kind"] == "pair"],
    )
    n_negs = _write_decisions_into_parquet(
        processed_dir / "negatives.parquet", "neg_id",
        sidecar[sidecar["kind"] == "negative"],
    )

    result = {
        "n_applied": int(len(decided)),
        "n_total_decisions": int(len(sidecar)),
        "n_pairs_updated": n_pairs,
        "n_negatives_updated": n_negs,
        "by_decision": decided["decision"].value_counts().to_dict(),
    }
    if verbose:
        print(f"  Applied {result['n_applied']} decisions "
              f"({result['by_decision']}); sidecar now has "
              f"{result['n_total_decisions']} total.")
        print(f"  pairs.parquet rows updated     : {n_pairs}")
        print(f"  negatives.parquet rows updated : {n_negs}")
    return result


def _write_decisions_into_parquet(
    parquet_path: Path,
    id_col: str,
    decisions: pd.DataFrame,
) -> int:
    """Project sidecar decisions onto one parquet, preserving all other rows."""
    df = pd.read_parquet(parquet_path)
    df = ensure_decision_columns(df)
    if len(decisions) == 0:
        df.to_parquet(parquet_path, index=False)
        return 0
    lookup = decisions.set_index("id")
    mask = df[id_col].isin(lookup.index)
    for col in DECISION_COLUMNS:
        df.loc[mask, col] = df.loc[mask, id_col].map(lookup[col])
    df.to_parquet(parquet_path, index=False)
    return int(mask.sum())


# ---------- status + downstream filter ----------------------------------------

def review_status(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    spot_rate: float = DEFAULT_SPOT_RATE,
    verbose: bool = True,
) -> dict:
    """DoD check: are all review rows decided, and is the spot sample covered?"""
    df_pairs = pd.read_parquet(processed_dir / "pairs.parquet")
    df_neg = pd.read_parquet(processed_dir / "negatives.parquet")
    sidecar = load_sidecar(processed_dir)
    decided = set(sidecar["id"].astype(str))

    out: dict = {"kinds": {}}
    for kind, df, id_col in (("pair", df_pairs, "pair_id"),
                             ("negative", df_neg, "neg_id")):
        if "routing" not in df.columns:
            continue
        review_ids = set(df.loc[df["routing"] == "review", id_col].astype(str))
        spot_ids = set(df.loc[df["routing"] == "spot", id_col].astype(str))
        n_review_pending = len(review_ids - decided)
        n_spot_decided = len(spot_ids & decided)
        spot_target = max(1, int(round(len(spot_ids) * spot_rate))) if spot_ids else 0
        out["kinds"][kind] = {
            "n_review": len(review_ids),
            "n_review_decided": len(review_ids & decided),
            "n_review_pending": n_review_pending,
            "n_spot": len(spot_ids),
            "n_spot_decided": n_spot_decided,
            "spot_target": spot_target,
            "review_complete": n_review_pending == 0,
            "spot_complete": n_spot_decided >= spot_target,
        }

    out["n_decisions"] = int(len(sidecar))
    out["dod_met"] = all(
        k["review_complete"] and k["spot_complete"] for k in out["kinds"].values()
    ) if out["kinds"] else False

    if verbose:
        print("\n=== Step 6 review status ===")
        for kind, s in out["kinds"].items():
            print(f"  {kind}:")
            print(f"    review : {s['n_review_decided']}/{s['n_review']} decided"
                  f"{'  <-- PENDING' if not s['review_complete'] else ''}")
            print(f"    spot   : {s['n_spot_decided']}/{s['spot_target']} "
                  f"(target {spot_rate:.0%} of {s['n_spot']})"
                  f"{'  <-- PENDING' if not s['spot_complete'] else ''}")
        print(f"  Total decisions recorded: {out['n_decisions']}")
        print(f"  Step 6 DoD met: {'YES' if out['dod_met'] else 'NO'}")
    return out


def filter_verified(
    df: pd.DataFrame,
    *,
    kind: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """Downstream gate for splits / training (use after review).

    Keeps:
      - rows with decision == 'accept'
      - rows with decision == 'edit' (text replaced by corrected_text)
      - undecided rows whose routing is 'auto' or None (trusted by routing /
        outside routing scope, e.g. transcript negatives)
    Drops:
      - rows with decision == 'reject'
      - undecided rows still routed 'review' or 'spot' (not yet verified)
    """
    if kind not in ("pair", "negative"):
        raise ValueError(f"kind must be 'pair' or 'negative', got {kind!r}")
    text_col = "variant_text" if kind == "pair" else "text"

    df = ensure_decision_columns(df)
    decision = _normalize_str(df["decision"]).str.lower()
    routing = (_normalize_str(df["routing"]).str.lower()
               if "routing" in df.columns
               else pd.Series(pd.NA, index=df.index, dtype="string"))

    keep_decided = decision.isin(["accept", "edit"])
    keep_undecided = decision.isna() & (routing.isna() | (routing == "auto"))
    keep = keep_decided | keep_undecided

    out = df[keep].copy()
    edits = _normalize_str(out["decision"]).str.lower() == "edit"
    if edits.any():
        out.loc[edits, text_col] = out.loc[edits, "corrected_text"]

    if verbose:
        n_rej = int((decision == "reject").sum())
        n_pending = int((decision.isna() & routing.isin(["review", "spot"])).sum())
        print(f"  filter_verified({kind}): kept {len(out)}/{len(df)} "
              f"(dropped {n_rej} rejected, {n_pending} pending review/spot)")
    return out


# ---------- CLI -----------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--export", action="store_true",
                   help="Export the review queue (excludes already-decided rows).")
    p.add_argument("--apply", type=Path, default=None, metavar="TEMPLATE",
                   help="Apply a filled review template (xlsx or parquet).")
    p.add_argument("--status", action="store_true",
                   help="Show Step 6 DoD status.")
    p.add_argument("--reviewer", type=str, default=None,
                   help="Reviewer name for --apply (used where the template's "
                        "reviewer column is blank).")
    p.add_argument("--spot-rate", type=float, default=DEFAULT_SPOT_RATE,
                   help=f"Fraction of spot rows to sample (default {DEFAULT_SPOT_RATE}).")
    p.add_argument("--pair-auto-sample", type=int, default=0,
                   help="Also sample N routing=auto pairs into the queue as a "
                        "safety check (default 0).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verbose = not args.quiet
    did_something = False
    if args.export:
        export_review_queue(
            args.processed_dir,
            spot_rate=args.spot_rate,
            pair_auto_sample=args.pair_auto_sample,
            seed=args.seed,
            verbose=verbose,
        )
        did_something = True
    if args.apply is not None:
        apply_review_decisions(
            args.apply, args.processed_dir,
            reviewer=args.reviewer, verbose=verbose,
        )
        did_something = True
    if args.status or not did_something:
        review_status(args.processed_dir, spot_rate=args.spot_rate, verbose=verbose)
