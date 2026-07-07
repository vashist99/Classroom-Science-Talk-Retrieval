"""Step 4 -- LLM register-variant augmentation.

Multiplies the positive corpus by producing meaning-preserving paraphrases of
each anchor in different teacher *registers* (large group, small group,
informal/centers). The bi-encoder (Step 8) then learns invariance to register
rather than overfitting to surface form.

Produces `data/processed/pairs.parquet` with one row per (anchor, variant):

    pair_id, anchor_id, anchor_text, anchor_setting, source_register,
    variant_text, register, subtype,
    preserved_cues, preservation_pct, preservation_check_passed,
    differs_from_anchor,
    llm_self_score, prompt_version, model_id

DoD addressed by this module:
    1. Each anchor has variants in >=2 registers different from its source.
       Anchors with `setting=Unknown` fan out to all 3 registers.
    2. Tier2 cue verbs and Tier3 content nouns are preserved in >=80% of
       variants (substring check, vacuous-truth pass for cue-less anchors).
    3. No anchor leaks through unchanged -- variant text must differ from anchor.
    4. Every row records `model_id` and `prompt_version`.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR
from src.negatives_3 import (
    DEFAULT_LLM_MODEL,
    LLMCallable,
    make_real_llm_callable,
)


REGISTERS: tuple[str, ...] = ("LARGE_GROUP", "SMALL_GROUP", "INFORMAL")

SETTING_TO_REGISTER: dict[str, str | None] = {
    "Large Group": "LARGE_GROUP",
    "Small Group": "SMALL_GROUP",
    "Centers": "INFORMAL",
    "Blocks/Engineering": "INFORMAL",
    "Unknown": None,
}

REGISTER_DESCRIPTIONS = {
    "LARGE_GROUP": "addressing the whole class at circle/carpet time; "
                   "slightly more formal, often longer turns, narrating for everyone",
    "SMALL_GROUP": "addressing 2-5 children at a small-group activity; "
                   "conversational, child-directed, mid-length turns",
    "INFORMAL":    "talking with one or two children at a center or during free play; "
                   "casual, side-by-side, often short and observational",
}

PROMPT_VERSIONS = {
    "register_variant": "aug_var_v1",
}

PRESERVATION_THRESHOLD = 0.80  # cue-preservation rate per variant (DoD #2)


def _as_list(x) -> list:
    """Coerce parquet-roundtripped list-likes (numpy arrays, NaN, None) to list."""
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "tolist"):
        return list(x.tolist())
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    return [] if x is None else [x]


# ---------------------------------------------------------------------------
# Setting / register helpers
# ---------------------------------------------------------------------------

def setting_to_register(setting: str | None) -> str | None:
    """Map a corpus `setting` value to a register code, or None if unknown."""
    if setting is None or (isinstance(setting, float) and math.isnan(setting)):
        return None
    return SETTING_TO_REGISTER.get(str(setting).strip(), None)


def target_registers_for_anchor(source_register: str | None) -> list[str]:
    """For an anchor with the given source register, which registers should we
    generate variants in?

      - Known source register -> the *other* two registers (DoD #1).
      - Unknown source        -> all three registers (DoD #1 still holds: 3 >= 2).
    """
    if source_register is None:
        return list(REGISTERS)
    return [r for r in REGISTERS if r != source_register]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_register_variant_prompt(
    anchor_text: str,
    target_register: str,
    *,
    source_register: str | None,
    subtypes: list[str],
    tier2_cues: list[str],
    tier3_cues: list[str],
    n: int,
) -> str:
    cues = sorted({*(tier2_cues or []), *(tier3_cues or [])})
    cue_str = ", ".join(repr(c) for c in cues) if cues else "(no required cues)"
    return (
        "You are helping create training data for a classifier that recognizes "
        "pre-K classroom SCIENCE TALK across teacher registers.\n\n"
        f"Generate {n} paraphrase variants of the ANCHOR utterance that:\n"
        "  1. Preserve the original's scientific meaning and intent.\n"
        "  2. Preserve any cue words/verbs and content nouns that are present "
        "in the anchor (listed below).\n"
        "  3. Match the TARGET_REGISTER's style.\n"
        "  4. Differ in surface form from the anchor (don't echo verbatim).\n\n"
        f"ANCHOR: \"{anchor_text}\"\n"
        f"SOURCE_REGISTER: {source_register or 'UNKNOWN'}\n"
        f"TARGET_REGISTER: {target_register} ({REGISTER_DESCRIPTIONS[target_register]})\n"
        f"PRACTICE TYPES: {subtypes or ['content']}\n"
        f"PRESERVE THESE CUES: {cue_str}\n\n"
        "Return STRICTLY as JSON, no commentary, no markdown:\n"
        '{"variants": [{"text": "...", "self_score": 0.0_to_1.0}, ...]}'
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def parse_variants_json(raw_text: str) -> list[dict]:
    """Pull `variants: [{text, self_score}]` out of an LLM response."""
    if not raw_text:
        return []

    def _pull(payload: dict) -> list[dict]:
        items = payload.get("variants")
        if not isinstance(items, list):
            return []
        out: list[dict] = []
        for item in items:
            if isinstance(item, dict) and "text" in item:
                text = str(item["text"]).strip()
                if not text:
                    continue
                score = item.get("self_score")
                try:
                    score = max(0.0, min(1.0, float(score))) if score is not None else None
                except (TypeError, ValueError):
                    score = None
                out.append({"text": text, "self_score": score})
            elif isinstance(item, str) and item.strip():
                out.append({"text": item.strip(), "self_score": None})
        return out

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            parsed = _pull(data)
            if parsed:
                return parsed
    except json.JSONDecodeError:
        pass

    m = _JSON_OBJ_RE.search(raw_text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return _pull(data)
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# Stub LLM callable (deterministic, no API needed)
# ---------------------------------------------------------------------------

_STUB_REGISTER_PREFIX = {
    "LARGE_GROUP": "Friends, everyone listen --",
    "SMALL_GROUP": "Hmm, look you two,",
    "INFORMAL": "Hey, look here --",
}


def stub_llm_callable(prompt: str, prompt_version: str) -> str:
    """Deterministic register-aware variant generator. Used for tests/CI.

    Builds variants by prefixing the anchor with a register-flavored opener and
    appending the required cues. This trivially satisfies DoD #2 (cue
    preservation) and DoD #3 (differs from anchor)."""
    anchor_match = re.search(r'ANCHOR:\s+"([^"]+)"', prompt)
    target_match = re.search(r"TARGET_REGISTER:\s+(\w+)", prompt)
    n_match = re.search(r"Generate (\d+)", prompt)
    cues_match = re.search(r"PRESERVE THESE CUES:\s+(.+)$", prompt, re.MULTILINE)

    anchor = anchor_match.group(1) if anchor_match else "the thing"
    target = target_match.group(1) if target_match else "INFORMAL"
    n = int(n_match.group(1)) if n_match else 1

    cue_terms: list[str] = []
    if cues_match:
        for m in re.finditer(r"'([^']+)'", cues_match.group(1)):
            cue_terms.append(m.group(1))

    prefix = _STUB_REGISTER_PREFIX.get(target, "")
    variants: list[dict] = []
    for i in range(n):
        suffix_cues = ""
        if cue_terms:
            missing = [c for c in cue_terms if c.lower() not in anchor.lower()]
            if missing:
                suffix_cues = " (" + ", ".join(missing) + ")"
        text = f"{prefix} {anchor}{suffix_cues}".strip()
        if i > 0:
            text = f"{text} -- variant {i + 1}"
        variants.append({"text": text, "self_score": 0.7})

    return json.dumps({"variants": variants})


# ---------------------------------------------------------------------------
# Cue-preservation + leak checks
# ---------------------------------------------------------------------------

def check_cue_preservation(
    variant_text: str,
    tier2_cues: list[str],
    tier3_cues: list[str],
    *,
    threshold: float = PRESERVATION_THRESHOLD,
) -> tuple[list[str], float, bool]:
    """Return (preserved_cues, preservation_pct, passed).

    A cue is "preserved" if it appears as a case-insensitive substring of the
    variant. Anchors with no cues vacuously pass (preservation_pct = 1.0).
    """
    cues = list({*_as_list(tier2_cues), *_as_list(tier3_cues)})
    if not cues:
        return [], 1.0, True
    text_lc = (variant_text or "").lower()
    preserved = [c for c in cues if str(c).lower() in text_lc]
    pct = len(preserved) / len(cues)
    return preserved, pct, pct >= threshold


def differs_from_anchor(variant_text: str, anchor_text: str) -> bool:
    """DoD #3: variant must not be the anchor verbatim (after normalization)."""
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())
    return _norm(variant_text) != _norm(anchor_text) and bool(_norm(variant_text))


# ---------------------------------------------------------------------------
# Per-anchor generator
# ---------------------------------------------------------------------------

def generate_variants_for_anchor(
    anchor_row: pd.Series,
    *,
    n_per_register: int,
    llm_callable: LLMCallable,
    target_registers: list[str] | None = None,
) -> list[dict]:
    """Generate variants of one anchor in each target register. Returns a list
    of dicts ready to drop into `pairs.parquet`."""
    source_reg = setting_to_register(anchor_row.get("setting"))
    targets = target_registers if target_registers is not None else target_registers_for_anchor(source_reg)
    tier2 = _as_list(anchor_row.get("tier2_cues"))
    tier3 = _as_list(anchor_row.get("tier3_cues"))
    subtypes = _as_list(anchor_row.get("subtype"))

    out: list[dict] = []
    for target_reg in targets:
        prompt = build_register_variant_prompt(
            anchor_text=anchor_row["utterance"],
            target_register=target_reg,
            source_register=source_reg,
            subtypes=subtypes,
            tier2_cues=tier2,
            tier3_cues=tier3,
            n=n_per_register,
        )
        raw = llm_callable(prompt, PROMPT_VERSIONS["register_variant"])
        for parsed in parse_variants_json(raw):
            preserved, pct, passed = check_cue_preservation(parsed["text"], tier2, tier3)
            out.append({
                "anchor_id": anchor_row["utt_id"],
                "anchor_text": anchor_row["utterance"],
                "anchor_setting": anchor_row.get("setting"),
                "source_register": source_reg,
                "variant_text": parsed["text"],
                "register": target_reg,
                "subtype": subtypes,
                "preserved_cues": preserved,
                "preservation_pct": pct,
                "preservation_check_passed": passed,
                "differs_from_anchor": differs_from_anchor(parsed["text"], anchor_row["utterance"]),
                "llm_self_score": parsed["self_score"],
                "prompt_version": PROMPT_VERSIONS["register_variant"],
            })
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_variant_pairs(
    df_corpus: pd.DataFrame,
    *,
    n_per_register: int = 1,
    llm_callable: LLMCallable = stub_llm_callable,
    drop_failed_leak_check: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """End-to-end build of the register-variant pair pool. Returns a DataFrame
    matching `pairs.parquet`'s schema."""
    positives = df_corpus[df_corpus["label"] == "SCIENCE_TALK"].reset_index(drop=True)
    if verbose:
        print(f"Generating register variants for {len(positives)} positive anchors "
              f"({n_per_register} per target register)...")

    all_rows: list[dict] = []
    for _, row in positives.iterrows():
        all_rows.extend(generate_variants_for_anchor(
            row, n_per_register=n_per_register, llm_callable=llm_callable,
        ))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No variants generated -- check llm_callable.")

    if verbose:
        print(f"  -> {len(df)} raw variants generated")

    if drop_failed_leak_check:
        before = len(df)
        df = df[df["differs_from_anchor"]].reset_index(drop=True)
        if verbose and before != len(df):
            print(f"  Dropped {before - len(df)} variants identical to anchor")

    df["model_id"] = DEFAULT_LLM_MODEL
    df["created_at"] = pd.Timestamp.utcnow().isoformat()
    df["pair_id"] = [f"pair_{i:05d}" for i in range(len(df))]

    df = df[[
        "pair_id", "anchor_id", "anchor_text", "anchor_setting", "source_register",
        "variant_text", "register", "subtype",
        "preserved_cues", "preservation_pct", "preservation_check_passed",
        "differs_from_anchor",
        "llm_self_score", "prompt_version", "model_id", "created_at",
    ]]

    if verbose:
        print(f"\nFinal pair pool: {len(df)} variants")
        print(f"  by register: {dict(Counter(df['register']))}")
        print(f"  preserved (>={PRESERVATION_THRESHOLD:.0%}): "
              f"{df['preservation_check_passed'].sum()}/{len(df)} "
              f"({df['preservation_check_passed'].mean():.1%})")
        print(f"  unique anchors: {df['anchor_id'].nunique()}/{len(positives)}")

    return df


# ---------------------------------------------------------------------------
# Validation -- the four DoD checks
# ---------------------------------------------------------------------------

def validate_variant_pairs(
    df_pairs: pd.DataFrame,
    df_corpus: pd.DataFrame,
    *,
    preservation_min: float = PRESERVATION_THRESHOLD,
    require_two_other_registers: bool = True,
    verbose: bool = True,
) -> dict:
    """Hard + soft assertions for Step 4's DoD."""
    info: dict = {
        "n_pairs": len(df_pairs),
        "n_anchors": df_pairs["anchor_id"].nunique(),
        "register_distribution": dict(Counter(df_pairs["register"])),
        "preservation_rate": float(df_pairs["preservation_check_passed"].mean()),
        "leak_count": int((~df_pairs["differs_from_anchor"]).sum()),
        "warnings": [],
    }

    # DoD #1 -- each anchor has variants in >=2 registers != source
    bad_anchors: list[str] = []
    for anchor_id, grp in df_pairs.groupby("anchor_id"):
        src = grp["source_register"].iloc[0]
        target_regs = set(grp["register"]) - {src}
        if len(target_regs) < 2:
            bad_anchors.append(anchor_id)
    if bad_anchors:
        msg = (f"{len(bad_anchors)} anchors have <2 variants in registers different "
               f"from their source (e.g. {bad_anchors[:3]}).")
        if require_two_other_registers:
            assert not bad_anchors, msg
        else:
            info["warnings"].append(msg)
    info["anchors_failing_register_diversity"] = len(bad_anchors)

    # Setting balance: register_distribution shouldn't be dominated >70% by one
    reg_counts = info["register_distribution"]
    total = sum(reg_counts.values())
    if total > 0:
        max_share = max(reg_counts.values()) / total
        if max_share > 0.70:
            info["warnings"].append(
                f"One register dominates ({max_share:.1%} of variants). "
                "Consider re-balancing source corpus or lowering n_per_register for it."
            )

    # DoD #2 -- preservation >=80% of variants
    pres = info["preservation_rate"]
    if pres < preservation_min:
        info["warnings"].append(
            f"Only {pres:.1%} of variants preserve cues at the {preservation_min:.0%} "
            "threshold (DoD #2 wants this fraction high). Tighten the prompt or "
            "switch llm_callable."
        )

    # DoD #3 -- no surface leak
    assert info["leak_count"] == 0, (
        f"{info['leak_count']} variants are identical to their anchor; drop them or "
        "re-prompt."
    )

    # DoD #4 -- model + prompt provenance
    assert df_pairs["model_id"].notna().all(), "Every row needs model_id"
    assert df_pairs["prompt_version"].notna().all(), "Every row needs prompt_version"

    # Coverage: how many positive anchors got at least one variant?
    n_positives = int((df_corpus["label"] == "SCIENCE_TALK").sum())
    info["anchor_coverage"] = info["n_anchors"] / max(n_positives, 1)
    if info["anchor_coverage"] < 1.0:
        info["warnings"].append(
            f"Only {info['n_anchors']}/{n_positives} positives have at least one "
            f"variant ({info['anchor_coverage']:.1%}). Investigate parser failures."
        )

    if verbose:
        print(f"\nValidation:")
        print(f"  pairs: {info['n_pairs']}, unique anchors: {info['n_anchors']} "
              f"({info['anchor_coverage']:.1%} of {n_positives})")
        print(f"  register distribution: {info['register_distribution']}")
        print(f"  cue preservation rate: {info['preservation_rate']:.1%}")
        print(f"  surface-leak count:    {info['leak_count']}")
        print(f"  anchors failing >=2-register-diversity: "
              f"{info['anchors_failing_register_diversity']}")
        if info["warnings"]:
            print("  warnings:")
            for w in info["warnings"]:
                print(f"    - {w}")
        else:
            print("  no warnings")

    return info


# ---------------------------------------------------------------------------
# Sample for human review
# ---------------------------------------------------------------------------

def sample_pairs_for_review(
    df_pairs: pd.DataFrame,
    n: int = 50,
    *,
    seed: int = 42,
    stratify_by_register: bool = True,
) -> pd.DataFrame:
    """Pull n variants for hand-checking, stratified across registers."""
    if stratify_by_register and df_pairs["register"].nunique() > 1:
        per_reg = max(1, n // df_pairs["register"].nunique())
        parts = []
        for _, grp in df_pairs.groupby("register"):
            parts.append(grp.sample(min(per_reg, len(grp)), random_state=seed))
        sampled = pd.concat(parts, ignore_index=True)
        if len(sampled) < n:
            extra = df_pairs.drop(sampled.index, errors="ignore")
            extra = extra.sample(min(n - len(sampled), len(extra)), random_state=seed)
            sampled = pd.concat([sampled, extra], ignore_index=True)
        return sampled.head(n).reset_index(drop=True)
    return df_pairs.sample(min(n, len(df_pairs)), random_state=seed).reset_index(drop=True)


def export_pairs_review_template(
    df_pairs: pd.DataFrame,
    processed_dir: Path,
    *,
    n: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Path | None:
    """Stratified register-variant sample for human QC (cue preservation, register)."""
    if len(df_pairs) == 0:
        return None
    rd = processed_dir / "review_samples"
    rd.mkdir(parents=True, exist_ok=True)
    sample = sample_pairs_for_review(df_pairs, n=min(n, len(df_pairs)), seed=seed)
    out = sample.copy()
    out["review_human_ok"] = pd.NA
    out["review_notes"] = pd.NA
    path = rd / "pairs_review_template.parquet"
    out.to_parquet(path, index=False)
    if verbose:
        print(f"  Review template: {path} ({len(out)} rows)")
    return path


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    use_real_llm: bool = False,
    n_per_register: int = 1,
    verbose: bool = True,
) -> Path:
    df_corpus = pd.read_parquet(processed_dir / "corpus.parquet")

    llm_callable = (
        make_real_llm_callable() if use_real_llm else stub_llm_callable
    )
    if verbose:
        print(f"LLM mode: {'REAL (Llama 3.3-70b)' if use_real_llm else 'STUB (deterministic)'}")

    df_pairs = build_variant_pairs(
        df_corpus,
        n_per_register=n_per_register,
        llm_callable=llm_callable,
        verbose=verbose,
    )

    validate_variant_pairs(df_pairs, df_corpus, verbose=verbose)

    export_pairs_review_template(df_pairs, processed_dir, verbose=verbose)

    out_path = processed_dir / "pairs.parquet"
    df_pairs.to_parquet(out_path, index=False)
    if verbose:
        print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes, {len(df_pairs)} rows)")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 4: register-variant augmentation.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--use-llm", action="store_true",
                        help="Use the real Llama 3.3-70b endpoint (default: stub).")
    parser.add_argument("--n-per-register", type=int, default=1,
                        help="Number of variants per target register (default: 1).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run(
        args.processed_dir,
        use_real_llm=args.use_llm,
        n_per_register=args.n_per_register,
        verbose=not args.quiet,
    )
