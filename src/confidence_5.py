"""Step 5b -- Confidence scoring and human-review routing.

Combines three signals into a single `confidence` score in [0, 1] for every
generated pair (variant in pairs.parquet) and every LLM-generated negative
(llm_hard_negative + seed_word_nonscience in negatives.parquet):

  (a) LLM self-report
        - pairs:     `llm_self_score`  (from Step 4 augment LLM)
        - negatives: `llm_gate_score`  (from Step 3 negatives gate)
  (b) Baseline cosine band score
        - cosine(anchor, candidate) is good only in a band: too high =
          paraphrase didn't change anything / negative looks like a positive;
          too low = meaning drift / negative is too easy.
        - Band edges are externalized in `config/confidence.json`.
  (c) Deterministic structural checks
        - pairs:     `preservation_pct * preservation_check_passed * differs_from_anchor`
        - negatives: `structural_check_passed` (bool -> {0, 1})

Composition is a weighted geometric mean (penalizes any signal being low --
the spirit of "all three should agree before we auto-accept").

Routing decision uses two thresholds from the config:
    confidence >= auto_min  -> "auto"     (auto-accept into the training set)
    confidence >= spot_min  -> "spot"     (random 10% spot-check)
    else                    -> "review"   (full manual review before use)

Transcript-mined negatives (`source_type == 'transcript_clean'`) are
hand-coded -- they bypass routing and stay marked routing=None.

DoD addressed by this module:
  1. Each pair has a confidence column in [0, 1] derived from the three signals.
  2. Thresholds and band edges are loaded from `config/confidence.json` --
     never hard-coded.
  3. A stratified 50-row audit template is written to
     `data/processed/review_samples/routing_audit_template.parquet` for the
     50-pair human audit (see notebook cell + this module's docstring).
  4. Re-runnable: drops & recomputes confidence/routing on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR


DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "confidence.json"


# ---------- config ----------------------------------------------------------

def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load externalized thresholds + band edges. Raises if missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Confidence config not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- numeric primitives ---------------------------------------------

def band_score(
    x,
    lo: float,
    hi: float,
    falloff: float = 0.30,
) -> np.ndarray:
    """Score in [0, 1] favoring values inside [lo, hi].

    Inside the band -> 1.0.
    Outside the band -> linear ramp down across a `falloff`-wide window.
    NaN -> 0.5 (neutral; we don't know either way).

    >>> float(band_score(0.7, 0.5, 0.9))
    1.0
    >>> float(band_score(1.0, 0.5, 0.9, falloff=0.20))  # 0.5 above hi
    0.5
    """
    arr = np.asarray(x, dtype=np.float64)
    nan_mask = np.isnan(arr)
    in_band = (arr >= lo) & (arr <= hi)
    below = arr < lo
    above = arr > hi
    safe_falloff = max(float(falloff), 1e-9)

    score = np.ones_like(arr)
    score = np.where(in_band, 1.0, score)
    score = np.where(below, np.clip(1.0 - (lo - arr) / safe_falloff, 0.0, 1.0), score)
    score = np.where(above, np.clip(1.0 - (arr - hi) / safe_falloff, 0.0, 1.0), score)
    score = np.where(nan_mask, 0.5, score)
    return score


def weighted_geometric_mean(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    floor: float = 1e-3,
) -> np.ndarray:
    """Per-row weighted geometric mean over the rightmost axis.

    values: shape (..., k); weights: shape (k,). Returns shape (...,).
    Floors values to `floor` before logging so a single zero doesn't nuke the
    score (it strongly punishes, but doesn't return exactly 0).
    """
    v = np.clip(np.asarray(values, dtype=np.float64), floor, 1.0)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("Weights must sum to > 0")
    w = w / w.sum()
    return np.exp(np.sum(np.log(v) * w, axis=-1))


# ---------- scoring ---------------------------------------------------------

def _route(conf: np.ndarray, auto_min: float, spot_min: float) -> np.ndarray:
    """Apply the routing thresholds to a confidence array."""
    return np.where(
        conf >= auto_min, "auto",
        np.where(conf >= spot_min, "spot", "review"),
    ).astype(object)


def score_pairs(df_pairs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add confidence_* + confidence + routing columns to pairs.parquet."""
    cfg_p = config["pairs"]
    cfg_w = config["weights"]
    cfg_r = config["routing"]

    llm = df_pairs["llm_self_score"].astype(float).fillna(0.5).to_numpy()
    cos = (df_pairs["baseline_cosine"].astype(float).to_numpy()
           if "baseline_cosine" in df_pairs.columns
           else np.full(len(df_pairs), np.nan))
    cos_score = band_score(
        cos,
        lo=cfg_p["cosine_band_min"],
        hi=cfg_p["cosine_band_max"],
        falloff=cfg_p["cosine_band_falloff"],
    )

    pres_pct = df_pairs["preservation_pct"].astype(float).fillna(0.0).to_numpy()
    pres_ok = df_pairs["preservation_check_passed"].astype(bool).astype(float).to_numpy()
    differs = df_pairs["differs_from_anchor"].astype(bool).astype(float).to_numpy()
    structural_mode = cfg_p.get("structural_mode", "strict")
    if structural_mode == "differs_only":
        # Variant must differ from anchor; cue preservation handled separately.
        structural = differs
    elif structural_mode == "soft":
        # Human audit: reviewers accepted variants that missed the 80% cue
        # substring bar but still preserved meaning. Use preservation_pct as a
        # graded signal instead of a hard pass/fail gate.
        structural = np.clip(pres_pct * differs, 0.0, 1.0)
    else:
        structural = np.clip(pres_pct * pres_ok * differs, 0.0, 1.0)

    values = np.stack([llm, cos_score, structural], axis=1)
    weights = np.array([cfg_w["llm"], cfg_w["cosine"], cfg_w["structural"]], dtype=float)
    confidence = weighted_geometric_mean(values, weights)
    routing = _route(confidence, cfg_r["auto_min"], cfg_r["spot_min"])

    out = df_pairs.copy()
    out["confidence_llm"] = llm
    out["confidence_cosine"] = cos_score
    out["confidence_structural"] = structural
    out["confidence"] = confidence
    out["routing"] = routing
    return out


def score_negatives(df_neg: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add confidence + routing to negatives.parquet.

    Only LLM-generated rows (`llm_hard_negative`, `seed_word_nonscience`)
    receive a routing label. Transcript-mined rows stay routing=None,
    confidence=NaN.
    """
    cfg_n = config["negatives"]
    cfg_w = config["weights"]
    cfg_r = config["routing"]

    llm_raw = df_neg["llm_gate_score"].astype(float).to_numpy()
    # Missing LLM signal = neutral (0.5) so confidence doesn't crater on
    # rows that legitimately didn't have a gate score (e.g. early-cached rows)
    llm = np.where(np.isnan(llm_raw), 0.5, llm_raw)

    cos = (df_neg["baseline_cosine"].astype(float).to_numpy()
           if "baseline_cosine" in df_neg.columns
           else np.full(len(df_neg), np.nan))
    cos_score = band_score(
        cos,
        lo=cfg_n["cosine_band_min"],
        hi=cfg_n["cosine_band_max"],
        falloff=cfg_n["cosine_band_falloff"],
    )

    structural = df_neg["structural_check_passed"].astype(bool).astype(float).to_numpy()

    values = np.stack([llm, cos_score, structural], axis=1)
    weights = np.array([cfg_w["llm"], cfg_w["cosine"], cfg_w["structural"]], dtype=float)
    confidence = weighted_geometric_mean(values, weights)
    routing = _route(confidence, cfg_r["auto_min"], cfg_r["spot_min"])

    is_llm_gen = df_neg["source_type"].isin(
        ["llm_hard_negative", "seed_word_nonscience"]
    ).to_numpy()
    routing_masked = np.where(is_llm_gen, routing, None)
    confidence_masked = np.where(is_llm_gen, confidence, np.nan)

    out = df_neg.copy()
    out["confidence_llm"] = np.where(is_llm_gen, llm, np.nan)
    out["confidence_cosine"] = np.where(is_llm_gen, cos_score, np.nan)
    out["confidence_structural"] = np.where(is_llm_gen, structural, np.nan)
    out["confidence"] = confidence_masked
    out["routing"] = routing_masked
    return out


# ---------- reporting + audit ----------------------------------------------

def summarize_routing(df: pd.DataFrame) -> dict:
    """Routing bucket counts + percentages over rows with non-null routing."""
    if "routing" not in df.columns:
        return {"counts": {}, "pct": {}, "total": 0}
    valid = df["routing"][df["routing"].notna()]
    total = int(len(valid))
    if total == 0:
        return {"counts": {}, "pct": {}, "total": 0}
    counts = valid.value_counts().to_dict()
    pct = {k: v / total for k, v in counts.items()}
    return {"counts": counts, "pct": pct, "total": total}


def export_routing_audit_sample(
    df_pairs: pd.DataFrame,
    df_neg: pd.DataFrame,
    processed_dir: Path,
    *,
    n: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Write a stratified routing audit template.

    Stratified across (kind, routing) so each bucket (auto, spot, review)
    in each kind (pair, negative) gets roughly equal representation. With
    6 strata and n=50, that's ~8 rows per stratum; we top up from
    under-represented strata to hit exactly n.

    Output schema (Excel-friendly, one row per audited candidate):

        id                str
        kind              "pair" | "negative"
        source_type       "variant" | "llm_hard_negative" | "seed_word_nonscience"
        anchor_text       str    (utt_id-resolved positive for negatives)
        candidate_text    str    (variant_text for pairs; text for negatives)
        register          str    (pairs only)
        confidence        float
        confidence_llm    float
        confidence_cosine float
        confidence_structural float
        routing           "auto" | "spot" | "review"   (model's decision)
        human_routing     "auto" | "spot" | "review"   (you fill in)
        agree             bool                          (auto-computed)
        notes             str                           (you fill in)
    """
    rd = processed_dir / "review_samples"
    rd.mkdir(parents=True, exist_ok=True)

    pa = df_pairs[[
        "pair_id", "anchor_text", "variant_text", "register",
        "confidence", "confidence_llm", "confidence_cosine",
        "confidence_structural", "routing",
    ]].copy()
    pa["kind"] = "pair"
    pa["source_type"] = "variant"
    pa = pa.rename(columns={"pair_id": "id", "variant_text": "candidate_text"})

    nrows = df_neg[df_neg["routing"].notna()].copy()
    utt_to_text_lookup: dict[str, str] = {}
    # Resolve negative anchor_utt_id -> anchor_text by joining with pairs (cheap
    # because pairs.parquet carries anchor_id + anchor_text).
    if "anchor_id" in df_pairs.columns and "anchor_text" in df_pairs.columns:
        utt_to_text_lookup = dict(zip(df_pairs["anchor_id"], df_pairs["anchor_text"]))
    nrows["anchor_text"] = nrows["anchor_utt_id"].map(utt_to_text_lookup).fillna(
        nrows["anchor_seed_term"].fillna("<no anchor>")
    )
    na = nrows[[
        "neg_id", "anchor_text", "text", "source_type",
        "confidence", "confidence_llm", "confidence_cosine",
        "confidence_structural", "routing",
    ]].copy()
    na["kind"] = "negative"
    na["register"] = pd.NA
    na = na.rename(columns={"neg_id": "id", "text": "candidate_text"})

    cols = [
        "id", "kind", "source_type", "anchor_text", "candidate_text", "register",
        "confidence", "confidence_llm", "confidence_cosine",
        "confidence_structural", "routing",
    ]
    unified = pd.concat([pa[cols], na[cols]], ignore_index=True, sort=False)

    strata = list(unified.groupby(["kind", "routing"], dropna=False))
    n_strata = max(len(strata), 1)
    per = max(1, n // n_strata)
    parts = []
    state = seed
    for _, grp in strata:
        take = min(per, len(grp))
        parts.append(grp.sample(take, random_state=state))
        state += 1
    sampled = pd.concat(parts, ignore_index=True)

    if len(sampled) < n:
        remainder = unified.drop(sampled.index, errors="ignore")
        extra = remainder.sample(
            n=min(n - len(sampled), len(remainder)), random_state=seed,
        )
        sampled = pd.concat([sampled, extra], ignore_index=True)
    sampled = sampled.head(n).reset_index(drop=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    sampled["human_routing"] = pd.NA
    sampled["agree"] = pd.NA
    sampled["notes"] = pd.NA

    path = rd / "routing_audit_template.parquet"
    sampled.to_parquet(path, index=False)
    # Also dump an Excel mirror so the user can open it directly.
    try:
        sampled.to_excel(rd / "routing_audit_template.xlsx", index=False)
    except Exception as e:
        if verbose:
            print(f"  (could not also write xlsx mirror: {e})")
    if verbose:
        print(f"  Audit template -> {path} ({len(sampled)} rows)")
    return path


def _normalize_routing_labels(series: pd.Series) -> pd.Series:
    """Lowercase, strip, and coerce blank/nan strings to NA."""
    out = series.astype("string").str.strip().str.lower()
    return out.replace({"": pd.NA, "nan": pd.NA, "<na>": pd.NA})


def compute_audit_agreement(df_audit: pd.DataFrame) -> dict:
    """Given a filled-in audit template, compute model-vs-human agreement.

    Use after the human has filled in `human_routing` for every row.
    Returns a dict with `n_reviewed`, `agreement` (proportion), per-bucket
    breakdowns, and a confusion matrix (model routing vs human routing).
    """
    df = df_audit.copy()
    if "routing" not in df.columns or "human_routing" not in df.columns:
        raise ValueError("Audit frame must contain `routing` and `human_routing` columns.")

    df["routing"] = _normalize_routing_labels(df["routing"])
    df["human_routing"] = _normalize_routing_labels(df["human_routing"])
    reviewed = df[df["human_routing"].notna()].copy()
    n = len(reviewed)
    if n == 0:
        return {
            "n_reviewed": 0,
            "agreement": None,
            "by_bucket": {},
            "by_kind": {},
            "confusion": {},
            "n_disagreements": 0,
        }

    agree = reviewed["human_routing"] == reviewed["routing"]
    reviewed["agree"] = agree
    overall = float(agree.mean())
    by_bucket = (
        reviewed.groupby("routing")["agree"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n", "mean": "agreement"})
        .to_dict(orient="index")
    )
    by_kind: dict = {}
    if "kind" in reviewed.columns:
        by_kind = (
            reviewed.groupby("kind")["agree"]
            .agg(["count", "mean"])
            .rename(columns={"count": "n", "mean": "agreement"})
            .to_dict(orient="index")
        )
    # routing (model) -> human_routing counts
    confusion = (
        reviewed.groupby(["routing", "human_routing"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .astype(int)
        .to_dict(orient="index")
    )
    confusion = {k: {hk: int(hv) for hk, hv in v.items()} for k, v in confusion.items()}
    return {
        "n_reviewed": int(n),
        "agreement": overall,
        "by_bucket": by_bucket,
        "by_kind": by_kind,
        "confusion": confusion,
        "n_disagreements": int((~reviewed["agree"]).sum()),
    }


def refresh_audit_routing(
    df_audit: pd.DataFrame,
    df_pairs: pd.DataFrame,
    df_neg: pd.DataFrame,
) -> pd.DataFrame:
    """Replace audit `routing` and confidence columns with current parquet values.

    Joins on `id` (pair_id / neg_id) so a filled human audit can be re-scored
    after config retuning without regenerating the sample. Row order is
    preserved so human_routing labels stay aligned.
    """
    pair_cols = [
        "pair_id", "confidence", "confidence_llm", "confidence_cosine",
        "confidence_structural", "routing",
    ]
    neg_cols = [
        "neg_id", "confidence", "confidence_llm", "confidence_cosine",
        "confidence_structural", "routing",
    ]
    pair_lookup = df_pairs[pair_cols].rename(columns={"pair_id": "id"})
    neg_lookup = df_neg[neg_cols].rename(columns={"neg_id": "id"})
    lookup = pd.concat([pair_lookup, neg_lookup], ignore_index=True)

    refresh_cols = [
        "confidence", "confidence_llm", "confidence_cosine",
        "confidence_structural", "routing",
    ]
    out = df_audit.drop(columns=[c for c in refresh_cols if c in df_audit.columns])
    return out.merge(lookup, on="id", how="left")


def rescore_processed_routing(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    verbose: bool = True,
) -> tuple[Path, Path]:
    """Recompute confidence + routing on pairs/negatives without regenerating audit."""
    config = load_config(config_path)
    pairs_path = processed_dir / "pairs.parquet"
    negs_path = processed_dir / "negatives.parquet"
    df_pairs = pd.read_parquet(pairs_path)
    df_neg = pd.read_parquet(negs_path)

    if "baseline_cosine" not in df_pairs.columns or "baseline_cosine" not in df_neg.columns:
        raise RuntimeError(
            "Missing `baseline_cosine`. Run Step 5 (embeddings_baseline_5) first."
        )

    df_pairs = score_pairs(df_pairs, config)
    df_neg = score_negatives(df_neg, config)
    df_pairs.to_parquet(pairs_path, index=False)
    df_neg.to_parquet(negs_path, index=False)

    if verbose:
        ps = summarize_routing(df_pairs)
        ns = summarize_routing(df_neg)
        print("  Pairs routing (rescored):")
        for k in ("auto", "spot", "review"):
            c = ps["counts"].get(k, 0)
            p = ps["pct"].get(k, 0.0)
            print(f"    {k:6s}: {c:5,} ({p:.1%})")
        print("  Negatives (LLM-generated) routing (rescored):")
        for k in ("auto", "spot", "review"):
            c = ns["counts"].get(k, 0)
            p = ns["pct"].get(k, 0.0)
            print(f"    {k:6s}: {c:5,} ({p:.1%})")
    return pairs_path, negs_path


def load_routing_audit(path: Path) -> pd.DataFrame:
    """Load a routing audit template from .xlsx or .parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Routing audit file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported audit file type: {path.suffix} (use .xlsx or .parquet)")


def score_routing_audit(
    audit_path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    out_dir: Path | None = None,
    refresh_routing: bool = True,
    verbose: bool = True,
) -> dict:
    """Score a filled routing audit and write scored artifacts.

    Reads `audit_path` (xlsx or parquet), fills the `agree` column, writes:
      - routing_audit_scored.xlsx / .parquet  (full frame with agree)
      - routing_audit_report.json             (summary metrics)
      - routing_audit_disagreements.xlsx        (rows where model != human)

    Returns the report dict from `compute_audit_agreement`, plus paths written.
    """
    audit_path = Path(audit_path)
    if out_dir is None:
        out_dir = audit_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    agreement_target = config.get("audit", {}).get("agreement_target", 0.80)

    df = load_routing_audit(audit_path)
    human_routing = _normalize_routing_labels(df["human_routing"])
    notes = df["notes"].copy() if "notes" in df.columns else pd.Series([pd.NA] * len(df))

    if refresh_routing:
        df_pairs = pd.read_parquet(processed_dir / "pairs.parquet")
        df_neg = pd.read_parquet(processed_dir / "negatives.parquet")
        df = refresh_audit_routing(df, df_pairs, df_neg)

    df["human_routing"] = human_routing
    if "notes" in df.columns:
        df["notes"] = notes
    df["routing"] = _normalize_routing_labels(df["routing"])
    df["human_routing"] = _normalize_routing_labels(df["human_routing"])
    reviewed_mask = df["human_routing"].notna()
    if "agree" in df.columns:
        df = df.drop(columns=["agree"])
    df["agree"] = False
    df.loc[reviewed_mask, "agree"] = (
        df.loc[reviewed_mask, "human_routing"] == df.loc[reviewed_mask, "routing"]
    )

    report = compute_audit_agreement(df)
    report["agreement_target"] = agreement_target
    report["passed"] = (
        report["agreement"] is not None and report["agreement"] >= agreement_target
    )
    report["source_audit"] = str(audit_path.resolve())
    report["config_version"] = config.get("version")
    report["refresh_routing"] = refresh_routing

    scored_parquet = out_dir / "routing_audit_scored.parquet"
    scored_xlsx = out_dir / "routing_audit_scored.xlsx"
    report_json = out_dir / "routing_audit_report.json"
    disagreements_xlsx = out_dir / "routing_audit_disagreements.xlsx"

    df.to_parquet(scored_parquet, index=False)
    df.to_excel(scored_xlsx, index=False)

    disagreements = df[reviewed_mask & ~df["agree"].astype(bool)].copy()
    disagreements.to_excel(disagreements_xlsx, index=False)

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    report["outputs"] = {
        "scored_parquet": str(scored_parquet.resolve()),
        "scored_xlsx": str(scored_xlsx.resolve()),
        "report_json": str(report_json.resolve()),
        "disagreements_xlsx": str(disagreements_xlsx.resolve()),
    }

    if verbose:
        _print_audit_report(report)
    return report


def _print_audit_report(report: dict) -> None:
    """Human-readable console summary of an audit scoring run."""
    n = report.get("n_reviewed", 0)
    agreement = report.get("agreement")
    target = report.get("agreement_target", 0.80)
    passed = report.get("passed", False)

    print(f"\n=== Routing audit score ===")
    print(f"  Rows reviewed     : {n}")
    if agreement is None:
        print("  Overall agreement : (no human_routing values found)")
        return
    print(f"  Overall agreement : {agreement:.1%}")
    print(f"  Target            : {target:.0%}")
    print(f"  Result            : {'PASS' if passed else 'FAIL'}")
    print(f"  Disagreements     : {report.get('n_disagreements', 0)}")

    print("\n  By model routing bucket:")
    for bucket, stats in sorted(report.get("by_bucket", {}).items()):
        print(f"    {bucket:6s}: n={stats['n']:3d}  agreement={stats['agreement']:.1%}")

    if report.get("by_kind"):
        print("\n  By kind:")
        for kind, stats in sorted(report.get("by_kind", {}).items()):
            print(f"    {kind:8s}: n={stats['n']:3d}  agreement={stats['agreement']:.1%}")

    if report.get("confusion"):
        print("\n  Confusion (model routing -> human routing counts):")
        for model_route, human_counts in sorted(report["confusion"].items()):
            parts = ", ".join(f"{h}={c}" for h, c in sorted(human_counts.items()))
            print(f"    model={model_route:6s}: {parts}")

    outputs = report.get("outputs", {})
    if outputs:
        print("\n  Wrote:")
        for label, path in outputs.items():
            print(f"    {label}: {path}")


# ---------- orchestration ---------------------------------------------------

def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    audit_n: int | None = None,
    audit_seed: int | None = None,
    verbose: bool = True,
) -> tuple[Path, Path, Path]:
    """End-to-end Step 5b: score, route, emit audit template."""
    config = load_config(config_path)
    if audit_n is None:
        audit_n = config.get("audit", {}).get("n_rows", 50)
    if audit_seed is None:
        audit_seed = config.get("audit", {}).get("seed", 42)

    pairs_path = processed_dir / "pairs.parquet"
    negs_path = processed_dir / "negatives.parquet"
    df_pairs = pd.read_parquet(pairs_path)
    df_neg = pd.read_parquet(negs_path)

    if "baseline_cosine" not in df_pairs.columns:
        raise RuntimeError(
            "pairs.parquet is missing `baseline_cosine`. Run Step 5a "
            "(embeddings_baseline_5.run / `--steps 5`) before Step 5b."
        )
    if "baseline_cosine" not in df_neg.columns:
        raise RuntimeError(
            "negatives.parquet is missing `baseline_cosine`. Run Step 5a "
            "(embeddings_baseline_5.run / `--steps 5`) before Step 5b."
        )

    df_pairs = score_pairs(df_pairs, config)
    df_neg = score_negatives(df_neg, config)

    if verbose:
        ps = summarize_routing(df_pairs)
        ns = summarize_routing(df_neg)
        print("  Pairs routing:")
        for k in ("auto", "spot", "review"):
            c = ps["counts"].get(k, 0)
            p = ps["pct"].get(k, 0.0)
            print(f"    {k:6s}: {c:5,} ({p:.1%})")
        print("  Negatives (LLM-generated) routing:")
        for k in ("auto", "spot", "review"):
            c = ns["counts"].get(k, 0)
            p = ns["pct"].get(k, 0.0)
            print(f"    {k:6s}: {c:5,} ({p:.1%})")

    df_pairs.to_parquet(pairs_path, index=False)
    df_neg.to_parquet(negs_path, index=False)
    audit_path = export_routing_audit_sample(
        df_pairs, df_neg, processed_dir,
        n=audit_n, seed=audit_seed, verbose=verbose,
    )
    return pairs_path, negs_path, audit_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--audit-n", type=int, default=None,
                   help="Audit template size (default: config.audit.n_rows = 50)")
    p.add_argument("--audit-seed", type=int, default=None,
                   help="Audit template RNG seed (default: config.audit.seed = 42)")
    p.add_argument(
        "--score-audit", type=Path, default=None, metavar="PATH",
        help="Score a filled routing audit (xlsx or parquet) and write "
             "routing_audit_scored.* + routing_audit_report.json. "
             "Default input if omitted when using --score-audit-only: "
             "review_samples/routing_audit_template.xlsx",
    )
    p.add_argument(
        "--score-audit-only", action="store_true",
        help="Only score the filled audit template; skip confidence routing run.",
    )
    p.add_argument(
        "--rescore-only", action="store_true",
        help="Recompute confidence + routing on pairs/negatives parquets only; "
             "does not regenerate the audit template.",
    )
    p.add_argument(
        "--no-refresh-routing", action="store_true",
        help="When scoring an audit, use the routing column baked into the audit "
             "file instead of refreshing from current parquets.",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.rescore_only:
        rescore_processed_routing(
            processed_dir=args.processed_dir,
            config_path=args.config,
            verbose=not args.quiet,
        )
    elif args.score_audit_only or args.score_audit is not None:
        audit_path = args.score_audit
        if audit_path is None:
            audit_path = args.processed_dir / "review_samples" / "routing_audit_template.xlsx"
        score_routing_audit(
            audit_path,
            config_path=args.config,
            processed_dir=args.processed_dir,
            out_dir=audit_path.parent,
            refresh_routing=not args.no_refresh_routing,
            verbose=not args.quiet,
        )
    else:
        run(
            processed_dir=args.processed_dir,
            config_path=args.config,
            audit_n=args.audit_n,
            audit_seed=args.audit_seed,
            verbose=not args.quiet,
        )
