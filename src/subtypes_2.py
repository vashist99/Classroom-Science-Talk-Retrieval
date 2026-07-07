"""Step 2 — Sub-type ("soft practice") labeling.

Reads `corpus.parquet` and `seed_words.parquet` (produced by Step 1), assigns
each utterance one or more practice labels via a three-stage pipeline:

    1. Rule-based mapping from explicit Tier2/Tier3 cues
    2. Additive keyword scan of the utterance text against seed-word variants
    3. LLM zero-shot fallback for rows with no rule and no keyword hits

Persists the augmented corpus back to `corpus.parquet` with four new columns:
    subtype, subtype_source, subtype_confidence, subtype_prompt_version

DoD addressed by this module:
    * Every utterance gets >=1 sub-type
    * Tier2-cued rows labeled by deterministic mapping; LLM fallback for empties
    * LLM-assigned subtypes carry a confidence and prompt version
    * Distribution sanity checks (no class with 0; logs over-dominance >70%)
"""

from __future__ import annotations

import json
import os
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

SUBTYPES_ALL = [
    "observation", "prediction", "causal_reasoning", "evidence", "content",
    # Sentinel for utterances that don't fit any science-practice surface form.
    # Should NEVER appear on SCIENCE_TALK rows (positives are gated by Tier3 cues
    # so they never reach the LLM); appears on negatives that the LLM judges
    # not to resemble any practice subtype on its surface.
    "not_science_shape",
]
PRACTICE_LABELS = {"observation", "prediction", "causal_reasoning", "evidence"}
# Subtypes the LLM is allowed to assign. `not_science_shape` is reachable
# ONLY via the LLM (not via rules or keyword scan) -- by design, since the
# rule + keyword paths look for science-shape evidence by construction.
EXPECTED_PRACTICE_AND_CONTENT = PRACTICE_LABELS | {"content"}

CATEGORY_TO_SUBTYPES = {
    "INQUIRY_VERB": ["observation"],
    "COMMUNICATE_INFO": ["observation"],
    "ORGANIZE_THINKING": ["observation"],
    "ASK_QUESTION_FRAME": ["prediction"],
    "MEASUREMENT_FRAME": ["evidence"],
    "CAUSE_EFFECT_FRAME": ["causal_reasoning"],
    "REASONING_FRAME": ["causal_reasoning", "evidence"],
    "LIFE_SCIENCE_PLANTS": ["content"],
    "PHYSICAL_SCIENCE_MECHANISMS": ["content"],
    "PROPERTIES_MATERIALS": ["content"],
    "STRUCTURE_SHAPE_SPACE": ["content"],
    "MEASUREMENT_SHAPE": ["content"],
    "HYPOTHESIS_NOTE": ["prediction"],
    "FROGSTREET_CONTENT_BUCKET": [],
    "FROGSTREET_VOCAB_ROUTINE": [],
}

TERM_OVERRIDES = {
    "predict": ["prediction"],
    "prediction": ["prediction"],
    "guess": ["prediction"],
    "what if": ["prediction"],
    "hypothesis": ["prediction"],
    "hypothesise": ["prediction"],
    "hypothesize": ["prediction"],
}

LLM_STUB_PROMPT_VERSION = "subtype_v0_stub"
# Short slug for cache keys / logs. Content matches the not_science_shape prompt;
# bump this string any time the classification prompt or model changes.
LLM_REAL_PROMPT_VERSION = "subtype_v3"
DEFAULT_REAL_LLM_MODEL = "llama-3.3-70b-instruct"


SubtypeResult = tuple[list[str], float, str]


def llm_subtype_stub(utterance: str) -> SubtypeResult:
    """Default stub for the LLM zero-shot fallback.

    Returns ['observation'] with confidence=0.0 so stubbed rows are easy to
    re-find later (filter on `subtype_confidence==0.0` and
    `subtype_prompt_version=='subtype_v0_stub'`). Use
    `make_real_llm_subtype_classifier()` for production.
    """
    return (["observation"], 0.0, LLM_STUB_PROMPT_VERSION)


# ---------------------------------------------------------------------------
# Real LLM-backed classifier (Llama 3.3-70b via cached_request)
# ---------------------------------------------------------------------------

def build_subtype_prompt(utterance: str) -> str:
    """Versioned classification prompt over the closed SUBTYPES_ALL vocabulary.

    The prompt explicitly allows `not_science_shape` so the LLM has an
    honest escape hatch instead of being forced to pick a practice label
    on a non-science utterance. Tied to `LLM_REAL_PROMPT_VERSION` for caching.
    """
    return (
        "You are tagging pre-K classroom utterances with practice sub-types for "
        "a science-talk classifier. Many utterances are NOT science-related "
        "(behavioral cues, transitions, songs, classroom logistics) -- those "
        "should be tagged `not_science_shape`, NOT forced into a practice label.\n\n"
        f"UTTERANCE: \"{utterance}\"\n\n"
        "Pick one or more sub-types from this CLOSED vocabulary (do NOT invent new ones):\n"
        "  - observation        (notice, look, see, describe what's happening in the world)\n"
        "  - prediction         (wonder, guess, what-if, hypothesis about a phenomenon)\n"
        "  - causal_reasoning   (because/so/that's-why explanations of physical events)\n"
        "  - evidence           (point at, measure, count, show data about a phenomenon)\n"
        "  - content            (names a science topic: plants, force, materials, structure)\n"
        "  - not_science_shape  (the utterance does NOT resemble any of the above on its\n"
        "                        surface -- e.g. classroom management, songs, naming kids,\n"
        "                        non-science place names, social pleasantries, transitions)\n\n"
        "Rules:\n"
        "  * Use `not_science_shape` ALONE -- never combine it with other labels.\n"
        "  * Only assign a practice label if the utterance plausibly fits that practice's\n"
        "    surface form. Don't stretch.\n"
        "  * `content` requires a science topic. Place names, food, names of children,\n"
        "    and classroom objects do NOT count as content.\n\n"
        "Also report your confidence as a number between 0.0 and 1.0.\n"
        "Return STRICTLY as JSON, no commentary, no markdown:\n"
        '{"subtypes": ["..."], "confidence": 0.0_to_1.0}'
    )


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def parse_subtype_response(raw_text: str) -> SubtypeResult:
    """Parse an LLM response into (subtypes, confidence, prompt_version).

    Robust to:
      * prose-wrapped JSON (extracts the embedded {...} block)
      * hallucinated labels (filtered against `SUBTYPES_ALL`)
      * out-of-range confidence (clamped to [0,1])
      * `not_science_shape` combined with practice labels (sentinel wins)
      * outright failure (falls back to (['not_science_shape'], 0.0) so the
        DoD assertion that every row has >=1 subtype never breaks, AND the
        failure is honestly tagged as not-shape-matched rather than silently
        forced into `observation`)
    """
    data = None
    if raw_text:
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            m = _JSON_OBJ_RE.search(raw_text)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = None

    if not isinstance(data, dict):
        # Endpoint returned non-JSON / unparseable text. Mark honestly as
        # not-shape-matched rather than silently tagging `observation`.
        return (["not_science_shape"], 0.0, LLM_REAL_PROMPT_VERSION)

    raw_subs = data.get("subtypes")
    if not isinstance(raw_subs, list):
        raw_subs = []
    valid_set = {str(st).strip() for st in raw_subs if str(st).strip() in SUBTYPES_ALL}

    # `not_science_shape` is a sentinel -- when present it must stand alone.
    # Drop any practice / content labels that came alongside it.
    if "not_science_shape" in valid_set:
        valid_set = {"not_science_shape"}

    valid = sorted(valid_set)
    if not valid:
        # Empty / fully-hallucinated response: fall back to not_science_shape
        # rather than the old default of `observation`. Forced practice labels
        # on parse failures were the original Option-A motivation.
        valid = ["not_science_shape"]

    try:
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0

    return (valid, conf, LLM_REAL_PROMPT_VERSION)


def make_real_llm_subtype_classifier(
    model: str = DEFAULT_REAL_LLM_MODEL,
    *,
    api_key_env: str = "LLM_API_KEY",
    completion_url_env: str = "COMPLETION_URL",
    temperature: float = 0.0,
    max_tokens: int = 120,
) -> Callable[[str], SubtypeResult]:
    """Build a Step-2 sub-type classifier that hits the UF LLM endpoint via
    `llm_client_0.cached_request`.

    The returned callable matches the same `(utterance) -> SubtypeResult`
    interface as `llm_subtype_stub`, so it drops into `assign_subtypes(...,
    llm_classifier=...)` without further changes.

    Determinism: temperature=0.0 + cached_request keying on the params dict
    means re-runs are byte-identical and free.
    """
    from src.llm_client_0 import cached_request

    api_key = os.getenv(api_key_env)
    url = os.getenv(completion_url_env)
    if not api_key or not url:
        raise RuntimeError(
            f"Real LLM mode requires env vars {api_key_env} and {completion_url_env}."
        )

    def _call(utterance: str) -> SubtypeResult:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": build_subtype_prompt(utterance)}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Endpoint can occasionally return non-JSON (empty body, HTML error page,
        # rate-limit blip, transient connection drop). One bad response must
        # NOT abort a 1000-row loop; degrade gracefully to the parser fallback
        # so the row still gets a valid SubtypeResult. The failure isn't cached,
        # so a re-run will retry the offending row.
        try:
            raw = cached_request(
                api_key=api_key,
                url=url,
                endpoint="completion",
                model=model,
                params=params,
                prompt_version=LLM_REAL_PROMPT_VERSION,
            )
            text = raw["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001 - intentional broad catch with fallback
            text = ""
        return parse_subtype_response(text)

    return _call


def build_seed_index(df_seed: pd.DataFrame) -> dict[str, pd.Series]:
    """Map every lowercase term + variant -> the seed-words row it came from."""
    index: dict[str, pd.Series] = {}
    for _, row in df_seed.iterrows():
        index[row["term"].lower()] = row
        for v in row["variants"]:
            index[str(v).lower()] = row
    return index


def map_cues_to_subtypes(
    tier2_cues,
    tier3_cues,
    seed_index: dict[str, pd.Series],
) -> list[str]:
    """Deterministic rule: turn explicit cue lists into subtype labels."""
    subtypes: set[str] = set()
    for cue in tier2_cues:
        cue_lc = str(cue).lower().strip()
        if cue_lc in TERM_OVERRIDES:
            subtypes.update(TERM_OVERRIDES[cue_lc])
            continue
        row = seed_index.get(cue_lc)
        if row is not None:
            subtypes.update(CATEGORY_TO_SUBTYPES.get(row["category"], []))
        else:
            subtypes.add("observation")
    if len(tier3_cues) > 0:
        subtypes.add("content")
    return sorted(subtypes)


def scan_text_for_subtypes(text, seed_index: dict[str, pd.Series]) -> list[str]:
    """Word-boundary search of utterance text for known seed terms / variants."""
    if pd.isna(text):
        return []
    text_lc = str(text).lower()
    subtypes: set[str] = set()
    for token in sorted(seed_index.keys(), key=len, reverse=True):
        if not re.search(r"\b" + re.escape(token) + r"\b", text_lc):
            continue
        row = seed_index[token]
        if str(row["tier"]) == "TIER3":
            subtypes.add("content")
        elif token in TERM_OVERRIDES:
            subtypes.update(TERM_OVERRIDES[token])
        else:
            subtypes.update(CATEGORY_TO_SUBTYPES.get(row["category"], []))
    return sorted(subtypes)


def assign_subtypes(
    df_corpus: pd.DataFrame,
    df_seed: pd.DataFrame,
    *,
    llm_classifier: Callable[[str], SubtypeResult] = llm_subtype_stub,
    skip_keyword_scan: bool = False,
) -> pd.DataFrame:
    """Add subtype, subtype_source, subtype_confidence, subtype_prompt_version columns.

    Logic per row:
      - if rule produced anything AND keyword adds extra:  source = 'rule+keyword'
      - elif rule produced anything:                       source = 'rule'
      - elif keyword scan produced anything:               source = 'keyword'
      - else:                                              source = 'llm' (calls `llm_classifier`)

    All non-llm rows get confidence=1.0 and prompt_version=None.

    `skip_keyword_scan=True` skips the seed-word surface scan (for NOT_SCIENCE
    rows where we want the LLM's not_science_shape escape hatch instead of
    keyword shortcuts). Does not affect rule-based cue mapping.
    """
    seed_index = build_seed_index(df_seed)
    df = df_corpus.copy()

    rule_subtypes: list[list[str]] = []
    kw_subtypes: list[list[str]] = []
    for _, row in df.iterrows():
        rule_subtypes.append(map_cues_to_subtypes(row["tier2_cues"], row["tier3_cues"], seed_index))
        if skip_keyword_scan:
            kw_subtypes.append([])
        else:
            kw_subtypes.append(scan_text_for_subtypes(row["utterance"], seed_index))

    final_subtypes: list[list[str]] = []
    final_sources: list[str] = []
    final_confs: list[float] = []
    final_versions: list[str | None] = []

    for i in range(len(df)):
        rule_st = rule_subtypes[i]
        kw_st = kw_subtypes[i]
        kw_new = sorted(set(kw_st) - set(rule_st))

        if len(rule_st) > 0 and len(kw_new) > 0:
            final_subtypes.append(sorted(set(rule_st) | set(kw_new)))
            final_sources.append("rule+keyword")
            final_confs.append(1.0)
            final_versions.append(None)
        elif len(rule_st) > 0:
            final_subtypes.append(sorted(set(rule_st)))
            final_sources.append("rule")
            final_confs.append(1.0)
            final_versions.append(None)
        elif len(kw_st) > 0:
            final_subtypes.append(sorted(set(kw_st)))
            final_sources.append("keyword")
            final_confs.append(1.0)
            final_versions.append(None)
        else:
            sts, conf, ver = llm_classifier(df["utterance"].iloc[i])
            final_subtypes.append(list(sts))
            final_sources.append("llm")
            final_confs.append(conf)
            final_versions.append(ver)

    df["subtype"] = final_subtypes
    df["subtype_source"] = final_sources
    df["subtype_confidence"] = final_confs
    df["subtype_prompt_version"] = final_versions
    return df


def check_subtype_distribution(df_corpus: pd.DataFrame, *, verbose: bool = True) -> dict:
    """Hard-assert >=1 sub-type per row; log soft warnings about skew."""
    assert df_corpus["subtype"].apply(lambda s: len(s) > 0).all(), \
        "DoD violation: every utterance must have >=1 sub-type"

    flat = [st for sts in df_corpus["subtype"] for st in sts]
    subtype_counts = Counter(flat)
    n_rows = len(df_corpus)

    warnings = []
    # `not_science_shape` is a sentinel -- on a positive-only corpus it's
    # SUPPOSED to be 0 (positives are gated out of the LLM path by Tier3 cues).
    # Only warn on 0 for the practice + content labels.
    for st in SUBTYPES_ALL:
        n = subtype_counts.get(st, 0)
        if n == 0 and st in EXPECTED_PRACTICE_AND_CONTENT:
            warnings.append(f"{st}: 0 examples")
        elif n / n_rows > 0.70:
            warnings.append(f"{st}: over-dominant ({n / n_rows:.1%})")

    has_practice = df_corpus["subtype"].apply(
        lambda s: any(st in PRACTICE_LABELS for st in s)
    )
    content_only = df_corpus["subtype"].apply(lambda s: set(s) == {"content"})
    sources = Counter(df_corpus["subtype_source"])

    if verbose:
        print(f"Sub-type distribution ({n_rows} rows):")
        for st in SUBTYPES_ALL:
            n = subtype_counts.get(st, 0)
            print(f"  {st:18s}: {n:3d} ({n / n_rows:.1%})")
        print(f"\nPractice coverage: {has_practice.sum()} ({has_practice.mean():.1%})")
        print(f"Content-only:      {content_only.sum()} ({content_only.mean():.1%})")
        print(f"Source breakdown:  {dict(sources)}")
        if warnings:
            print(f"\nDistribution warnings (soft, inspect manually):")
            for w in warnings:
                print(f"  - {w}")

    return {
        "subtype_counts": dict(subtype_counts),
        "practice_coverage": int(has_practice.sum()),
        "content_only": int(content_only.sum()),
        "source_distribution": dict(sources),
        "warnings": warnings,
    }


def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    llm_classifier: Callable[[str], SubtypeResult] = llm_subtype_stub,
    verbose: bool = True,
) -> Path:
    """End-to-end Step 2: load corpus + seeds, assign, validate, persist."""
    corpus_path = processed_dir / "corpus.parquet"
    seed_path = processed_dir / "seed_words.parquet"

    df_corpus = pd.read_parquet(corpus_path)
    df_seed = pd.read_parquet(seed_path)

    df_corpus = assign_subtypes(df_corpus, df_seed, llm_classifier=llm_classifier)
    check_subtype_distribution(df_corpus, verbose=verbose)
    df_corpus.to_parquet(corpus_path, index=False)

    if verbose:
        print(f"  wrote {corpus_path} ({corpus_path.stat().st_size:,} bytes)")
    return corpus_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: assign sub-type labels.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                        help=f"Directory containing Step 1 parquets (default: {DEFAULT_PROCESSED_DIR})")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use real Llama 3.3-70b for the LLM fallback (default: stub). "
                             "On the Y1 corpus every row has Tier3 cues so this fires zero "
                             "API calls; meaningful for cue-less corpora (e.g. Y2 transcripts).")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    args = parser.parse_args()

    classifier = make_real_llm_subtype_classifier() if args.use_llm else llm_subtype_stub
    run(args.processed_dir, llm_classifier=classifier, verbose=not args.quiet)
