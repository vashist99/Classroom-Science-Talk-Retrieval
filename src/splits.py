"""Stratified train / val / test splits for the bi-encoder + re-ranker.

Produces a single `splits.parquet` keyed by (utt_id | neg_id | pair_id) with a
`split` column in {train, val, test}. Stratification guarantees each split
contains:
    * positives across all subtypes (observation, prediction, ...)
    * not_science_shape negatives (the bulk after Option A) and any practice-
      shape negatives (LLM- or keyword-tagged)
    * variant pairs grouped by anchor_id (so an anchor and its variants land
      in the SAME split -- never train an anchor in train and evaluate it on
      one of its own variants in val)

Before splitting, human-review verdicts are applied via `filter_verified`
(rejected rows dropped, edits applied) so only verified / auto-trusted data ever
reaches training. A held-out `hard_informal_slice.parquet` is also carved out
(whole anchor families, never used in training) for the informal-register cases
the model is most likely to miss.

This module is intentionally self-contained and stage-agnostic so that Step 5
(bi-encoder), Step 6 (re-ranker), and Step 7 (evaluation) can all import it.

Usage:
    python src/splits.py                    # writes data/processed/splits.parquet
    python src/splits.py --train 0.8 --val 0.1 --test 0.1
    python src/splits.py --informal-variant-families 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR
from src.review_6 import filter_verified

DEFAULT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "val", "test")

# Corpus `setting` values that count as casual / informal classroom talk.
# These are the genuinely-informal positives that anchor the hard-informal
# held-out slice (centers, free play, transitions, side conversations).
INFORMAL_SETTINGS: tuple[str, ...] = (
    "Centers",
    "Blocks/Engineering",
    "Transition",
    "Conversation",
)
INFORMAL_REGISTER = "INFORMAL"
# Default number of *extra* anchor families (beyond the informal-setting
# positives) to pull into the slice purely to give it genuine
# informal-register variant coverage. Whole families are held out so no
# anchor ever appears in both the slice and a train/val/test split.
DEFAULT_INFORMAL_VARIANT_FAMILIES = 30


def _stratify_one(
    keys: pd.Series,
    *,
    ratios: tuple[float, float, float],
    seed: int,
) -> pd.Series:
    """Assign each row in `keys` to a split, stratifying so each unique
    `keys` value is split according to ratios. Used for both subtype-stratified
    positives and source-type-stratified negatives.

    Returns a Series of split names indexed identically to `keys`.
    """
    rng = np.random.default_rng(seed)
    splits = pd.Series(index=keys.index, dtype="object")
    for k, group_idx in keys.groupby(keys).groups.items():
        idx = list(group_idx)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # Make sure remainder lands in test (sums to n)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train += n_test  # shrink train if rounding overshot
            n_test = 0
        splits.loc[idx[:n_train]] = "train"
        splits.loc[idx[n_train : n_train + n_val]] = "val"
        splits.loc[idx[n_train + n_val :]] = "test"
    return splits


def _primary_subtype(subtypes) -> str:
    """A row's subtype list -> single string for stratification.
    Multi-label rows pick the lexicographically smallest practice label
    (so they bucket consistently across runs)."""
    if subtypes is None:
        return "unknown"
    sts = list(subtypes) if not isinstance(subtypes, list) else subtypes
    if len(sts) == 0:
        return "unknown"
    return sorted(sts)[0]


def make_splits(
    df_corpus: pd.DataFrame,
    df_negatives: pd.DataFrame,
    df_pairs: pd.DataFrame | None = None,
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 17,
) -> pd.DataFrame:
    """Build a `splits.parquet` row for every utt_id (positives + nulls) in
    corpus, every neg_id in negatives, and every pair_id in pairs.

    Stratification keys:
      * positives:  primary subtype label
      * negatives:  source_type x primary subtype label (joint)
      * pairs:      grouped by anchor_id (whole anchor goes to one split)

    Pairs co-locate with their anchor. If anchor_id is in `train`, all
    of that anchor's variants are also in `train`. This prevents leakage
    where the bi-encoder trains on (anchor, variant_A) and is evaluated
    on (anchor, variant_B) which would wildly inflate metrics.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")

    rows: list[pd.DataFrame] = []

    # --- Positives -----------------------------------------------------------
    pos = df_corpus.copy()
    pos["primary_subtype"] = pos["subtype"].apply(_primary_subtype)
    pos["split"] = _stratify_one(
        pos["primary_subtype"], ratios=ratios, seed=seed,
    )
    rows.append(pd.DataFrame({
        "id": pos["utt_id"],
        "kind": "positive" if "label" in pos.columns else "corpus",
        "label": pos.get("label", "SCIENCE_TALK"),
        "stratify_key": pos["primary_subtype"],
        "anchor_id": None,
        "split": pos["split"],
    }))

    # --- Negatives -----------------------------------------------------------
    neg = df_negatives.copy()
    neg["primary_subtype"] = neg["subtype"].apply(_primary_subtype)
    neg["stratify_key"] = neg["source_type"].astype(str) + "::" + neg["primary_subtype"]
    neg["split"] = _stratify_one(
        neg["stratify_key"], ratios=ratios, seed=seed + 1,
    )
    rows.append(pd.DataFrame({
        "id": neg["neg_id"],
        "kind": "negative",
        "label": "NOT_SCIENCE_TALK",
        "stratify_key": neg["stratify_key"],
        "anchor_id": None,
        "split": neg["split"],
    }))

    # --- Pairs (co-located with their anchor) --------------------------------
    if df_pairs is not None and len(df_pairs) > 0:
        # Build anchor_id -> split lookup from positives' split assignment
        utt_to_split = dict(zip(pos["utt_id"], pos["split"]))
        pairs = df_pairs.copy()
        pairs["split"] = pairs["anchor_id"].map(utt_to_split).fillna("train")
        rows.append(pd.DataFrame({
            "id": pairs["pair_id"],
            "kind": "pair",
            "label": "SCIENCE_TALK",
            "stratify_key": "pair::" + pairs["register"].astype(str),
            "anchor_id": pairs["anchor_id"],
            "split": pairs["split"],
        }))

    return pd.concat(rows, axis=0, ignore_index=True)


def make_hard_informal_slice(
    df_corpus: pd.DataFrame,
    df_pairs: pd.DataFrame | None,
    *,
    settings: tuple[str, ...] = INFORMAL_SETTINGS,
    n_variant_families: int = DEFAULT_INFORMAL_VARIANT_FAMILIES,
    seed: int = 17,
) -> tuple[pd.DataFrame, set]:
    """Build the held-out hard-informal evaluation slice.

    The slice is the set of *casual / informal-register* science-talk cases the
    model is most likely to miss. It is **never** used in training: every anchor
    that contributes to the slice is held out as a *whole family* (the anchor
    positive plus all of its variant pairs) so an anchor can never appear in both
    the slice and a train/val/test split (no leakage).

    Two kinds of anchor families enter the slice:
      1. ``informal_positive`` -- every corpus positive whose ``setting`` is one
         of ``settings`` (centers, blocks/engineering, transition, conversation).
         These utterances are genuinely informal classroom talk.
      2. ``informal_variant_family`` -- a seeded random sample of up to
         ``n_variant_families`` *additional* anchors that have at least one
         INFORMAL-register LLM variant. These give the slice real
         informal-register variant text (the bulk of anchors are
         ``setting=Unknown`` and only acquire an informal register via
         augmentation). Holding out the whole family keeps the eval honest.

    Returns ``(slice_df, held_out_anchor_ids)`` where ``slice_df`` has columns
    ``[id, kind, anchor_id, text, setting, register, subtype, slice_reason]`` and
    ``held_out_anchor_ids`` is the set of corpus ``utt_id`` values to exclude from
    the train/val/test pools.
    """
    informal_pos_ids: set = set(
        df_corpus.loc[df_corpus["setting"].isin(settings), "utt_id"]
    )

    sampled_ids: set = set()
    if (
        df_pairs is not None
        and len(df_pairs) > 0
        and "register" in df_pairs.columns
        and n_variant_families > 0
    ):
        has_inf_variant = set(
            df_pairs.loc[df_pairs["register"] == INFORMAL_REGISTER, "anchor_id"]
        )
        candidates = sorted(has_inf_variant - informal_pos_ids)
        if candidates:
            rng = np.random.default_rng(seed)
            k = min(n_variant_families, len(candidates))
            picks = rng.choice(len(candidates), size=k, replace=False)
            sampled_ids = {candidates[i] for i in picks}

    held_out = informal_pos_ids | sampled_ids

    def _reason(utt_id) -> str:
        return "informal_positive" if utt_id in informal_pos_ids else "informal_variant_family"

    rows: list[pd.DataFrame] = []

    pos = df_corpus[df_corpus["utt_id"].isin(held_out)].copy()
    if len(pos) > 0:
        rows.append(pd.DataFrame({
            "id": pos["utt_id"],
            "kind": "positive",
            "anchor_id": pos["utt_id"],
            "text": pos["utterance"],
            "setting": pos["setting"],
            "register": pos["setting"].map(
                lambda s: INFORMAL_REGISTER if s in settings else None
            ),
            "subtype": pos["subtype"].apply(_primary_subtype),
            "slice_reason": pos["utt_id"].map(_reason),
        }))

    if df_pairs is not None and len(df_pairs) > 0:
        slc_pairs = df_pairs[df_pairs["anchor_id"].isin(held_out)].copy()
        if len(slc_pairs) > 0:
            rows.append(pd.DataFrame({
                "id": slc_pairs["pair_id"],
                "kind": "pair",
                "anchor_id": slc_pairs["anchor_id"],
                "text": slc_pairs["variant_text"],
                "setting": slc_pairs.get("anchor_setting"),
                "register": slc_pairs["register"],
                "subtype": slc_pairs["subtype"].apply(_primary_subtype),
                "slice_reason": slc_pairs["anchor_id"].map(_reason),
            }))

    if rows:
        slice_df = pd.concat(rows, axis=0, ignore_index=True)
    else:
        slice_df = pd.DataFrame(
            columns=["id", "kind", "anchor_id", "text", "setting",
                     "register", "subtype", "slice_reason"]
        )
    return slice_df, held_out


def summarize_slice(slice_df: pd.DataFrame, held_out: set) -> str:
    lines = [
        f"Hard-informal slice: {len(slice_df):,} rows from "
        f"{len(held_out):,} held-out anchor families",
    ]
    if len(slice_df) > 0:
        by = slice_df.groupby(["slice_reason", "kind"]).size().unstack(fill_value=0)
        lines.append(by.to_string())
        if "register" in slice_df.columns:
            reg = slice_df["register"].fillna("(none)").value_counts().to_dict()
            lines.append(f"  by register: {reg}")
    return "\n".join(lines)


def summarize(df_splits: pd.DataFrame) -> str:
    """Pretty-print the split distribution."""
    lines = ["Split distribution:"]
    overall = df_splits["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0)
    total = len(df_splits)
    for name in SPLIT_NAMES:
        n = int(overall.get(name, 0))
        lines.append(f"  {name:6s}: {n:5,} ({n / total:.1%} of {total:,})")
    by_kind = df_splits.groupby(["kind", "split"]).size().unstack(fill_value=0)
    lines.append("\nBy kind:")
    lines.append(by_kind.to_string())
    return "\n".join(lines)


def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = 17,
    n_variant_families: int = DEFAULT_INFORMAL_VARIANT_FAMILIES,
    verbose: bool = True,
) -> Path:
    df_corpus = pd.read_parquet(processed_dir / "corpus.parquet")
    df_neg = pd.read_parquet(processed_dir / "negatives.parquet")
    pairs_path = processed_dir / "pairs.parquet"
    df_pairs = pd.read_parquet(pairs_path) if pairs_path.exists() else None

    # --- Apply human-review verdicts before anything else --------------------
    # Drop rejected rows and apply edits so the splits and the held-out slice
    # only ever contain verified (or auto-trusted / out-of-scope) data.
    if verbose:
        print("Applying human-review gate (filter_verified):")
    df_neg = filter_verified(df_neg, kind="negative", verbose=verbose)
    if df_pairs is not None and len(df_pairs) > 0:
        df_pairs = filter_verified(df_pairs, kind="pair", verbose=verbose)

    # --- Carve out the held-out hard-informal slice -------------------------
    slice_df, held_out = make_hard_informal_slice(
        df_corpus, df_pairs,
        n_variant_families=n_variant_families, seed=seed,
    )
    if verbose:
        print("\n" + summarize_slice(slice_df, held_out))

    # Remove the held-out anchor families from the train/val/test pools.
    pool_corpus = df_corpus[~df_corpus["utt_id"].isin(held_out)].copy()
    if df_pairs is not None and len(df_pairs) > 0:
        pool_pairs = df_pairs[~df_pairs["anchor_id"].isin(held_out)].copy()
    else:
        pool_pairs = df_pairs

    df_splits = make_splits(
        pool_corpus, df_neg, pool_pairs, ratios=ratios, seed=seed,
    )
    if verbose:
        print("\n" + summarize(df_splits))

    out_path = processed_dir / "splits.parquet"
    df_splits.to_parquet(out_path, index=False)
    slice_path = processed_dir / "hard_informal_slice.parquet"
    slice_df.to_parquet(slice_path, index=False)
    if verbose:
        print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")
        print(f"Wrote {slice_path} ({slice_path.stat().st_size:,} bytes)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified train/val/test splits.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--train", type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument("--val", type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument("--test", type=float, default=DEFAULT_RATIOS[2])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--informal-variant-families", type=int,
        default=DEFAULT_INFORMAL_VARIANT_FAMILIES,
        help="Extra anchor families (with INFORMAL variants) to hold out in the "
             "hard-informal slice, beyond the informal-setting positives.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(
        args.processed_dir,
        ratios=(args.train, args.val, args.test),
        seed=args.seed,
        n_variant_families=args.informal_variant_families,
        verbose=not args.quiet,
    )
