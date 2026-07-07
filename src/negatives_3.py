"""Step 3 — Negative mining (gating step).

Builds a NOT_SCIENCE_TALK pool from three source types and persists to
`data/processed/negatives.parquet`:

    1. transcript_clean       -- transcript utterances with NO seed-word match
    2. llm_hard_negative      -- LLM-generated, anchored to a positive utterance
    3. seed_word_nonscience   -- LLM-generated, anchored to a seed term used in
                                 a non-scientific sense

DoD addressed by this module:
    * At least three negative source types are represented (function exists
      for transcript_clean even if no transcripts are supplied; warn if absent)
    * Final negative pool is >=3x positive count; logs balance across sub-types
      so each positive sub-type has a credible negative counterpart
    * `sample_for_review(n=50)` and `compute_review_pass_rate` give you the
      hand-check loop required for the >=90% true-negative gate

Each output row carries an LLM gate score (0.0 = clearly science .. 1.0 =
clearly non-science), a structural-check flag, and full provenance
(`source_type`, `anchor_*`, `model_id`, `prompt_version`).
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR
from src.subtypes_2 import (
    PRACTICE_LABELS,
    SUBTYPES_ALL,
    SubtypeResult,
    assign_subtypes,
    build_seed_index,
    llm_subtype_stub,
    make_real_llm_subtype_classifier,
)


NEGATIVE_SOURCE_TYPES = {"transcript_clean", "llm_hard_negative", "seed_word_nonscience"}

# Sub-categories of transcript_clean for downstream auditing / review weighting.
# behavior_disapproving = column E == 'Y'  (high-confidence negatives)
# other_teacher_talk    = any other non-SCI teacher utterance (broader pool)
TRANSCRIPT_SUBTYPES = {"behavior_disapproving", "other_teacher_talk"}

PROMPT_VERSIONS = {
    "hard_negative": "neg_hard_v1",
    "seed_nonscience": "neg_seed_v1",
    "gate_score": "neg_gate_v1",
}

DEFAULT_LLM_MODEL = "llama-3.3-70b-instruct"

DEFAULT_TRANSCRIPT_XLSX = (
    _PROJECT_ROOT / "data" / "transcripts" / "Coding Transcripts.xlsx"
)

# Coded-transcript schema (col indices); see data/transcripts/Coding Transcripts.xlsx
XLSX_SPEAKER_COL = 2          # C
XLSX_UTTERANCE_COL = 3        # D
XLSX_BEHAVIOR_CODE_COL = 4    # E -- 'Y' means behavior-disapproving
XLSX_CONTENT_CODE_COL = 6     # G -- 'SCI' / 'MATH' / 'VOCAB' / NaN
XLSX_TEACHER_SPEAKER_RE = re.compile(r"^T\d*:?$")  # T1, T1:, T:, T17:, etc.
XLSX_META_SHEETS = {"TOC", "Sheet1", "Sheet5", "Coding Scheme"}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_hard_negative_prompt(positive_utterance: str, subtypes: list[str], n: int) -> str:
    return (
        "You are helping build training data for a classifier that distinguishes pre-K "
        "classroom SCIENCE TALK from non-science talk.\n\n"
        "I will give you a science-talk utterance. Generate "
        f"{n} HARD NEGATIVE utterances that:\n"
        "  1. Mirror the original's syntactic structure and approximate length.\n"
        "  2. May reuse common function words (\"I wonder\", \"let's see\", \"what if\").\n"
        "  3. Are clearly NOT scientific -- they're about social interactions, classroom "
        "management, SEL, literacy, transitions, or other non-science topics.\n"
        "  4. Sound like things a real pre-K teacher would actually say.\n\n"
        f"ORIGINAL (science talk): \"{positive_utterance}\"\n"
        f"PRACTICE TYPES: {subtypes}\n\n"
        "Return STRICTLY as JSON, no commentary, no markdown:\n"
        "{\"negatives\": [\"utterance1\", \"utterance2\", ...]}"
    )


def build_seed_nonscience_prompt(term: str, category: str, definition: str, n: int) -> str:
    return (
        "You are helping build hard negatives for a pre-K classroom science-talk classifier.\n\n"
        f"The word \"{term}\" (category {category}) can appear in classroom talk in either "
        "a scientific sense or a clearly non-scientific sense.\n"
        f"Operational definition of the scientific sense: {definition}\n\n"
        f"Generate {n} short utterances a pre-K teacher might say that USE THE WORD "
        f"\"{term}\" but in a clearly NON-SCIENTIFIC context (social, classroom management, "
        "SEL, literacy, transitions, etc.).\n\n"
        "Return STRICTLY as JSON, no commentary:\n"
        "{\"negatives\": [\"utterance1\", \"utterance2\", ...]}"
    )


def build_gate_prompt(text: str) -> str:
    return (
        "Is the following pre-K classroom utterance SCIENCE TALK?\n\n"
        "SCIENCE TALK = inquiry about the world (notice/try/test/predict/explain) "
        "and/or science content terms (plants, forces, materials, structure).\n"
        "NOT SCIENCE TALK = routines, classroom management, SEL, literacy, social.\n\n"
        f"UTTERANCE: \"{text}\"\n\n"
        "Respond with ONLY a single number between 0.0 (definitely science) "
        "and 1.0 (definitely NOT science). No other text."
    )


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_negatives_json(raw_text: str) -> list[str]:
    """Best-effort extraction of `negatives` list from an LLM response."""
    if not raw_text:
        return []
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and isinstance(data.get("negatives"), list):
            return [str(x).strip() for x in data["negatives"] if str(x).strip()]
    except json.JSONDecodeError:
        pass
    for m in _JSON_OBJ_RE.finditer(raw_text):
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and isinstance(data.get("negatives"), list):
                return [str(x).strip() for x in data["negatives"] if str(x).strip()]
        except json.JSONDecodeError:
            continue
    return []


def parse_gate_score(raw_text: str) -> float | None:
    """Pull a 0..1 float out of the gate-prompt response."""
    if not raw_text:
        return None
    m = re.search(r"\b([01](?:\.\d+)?|0?\.\d+)\b", raw_text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# LLM callables
# ---------------------------------------------------------------------------

LLMCallable = Callable[[str, str], str]
"""(prompt: str, prompt_version: str) -> raw_text_response: str"""


_STUB_HARD_NEGATIVES_POOL = [
    "Please put your shoes back on the rug.",
    "Remember we use kind words with our friends.",
    "Let's all sit criss-cross applesauce.",
    "Time to clean up and head to centers.",
    "Use a quiet voice when we're listening.",
    "Hands to yourself please.",
    "Take turns -- it's your friend's turn next.",
    "Walking feet inside, please.",
    "Let's read this book together at the carpet.",
    "I need you to pack up your backpack now.",
    "Can you write your name on the paper?",
    "Wait your turn at the water fountain.",
    "Show me you're ready to listen.",
    "Let's count to three: one, two, three!",
    "Eyes on me when I'm talking.",
]


def stub_llm_callable(prompt: str, prompt_version: str) -> str:
    """Deterministic LLM stub. Returns plausible non-science classroom talk
    in the JSON shape the parsers expect. Used by default so the pipeline
    runs without API calls.

    For seeded determinism the pool is deterministic; the same prompt always
    returns the same negatives.
    """
    if prompt_version.startswith("neg_gate"):
        return "0.85"

    seed = sum(ord(c) for c in prompt) % len(_STUB_HARD_NEGATIVES_POOL)
    pool = _STUB_HARD_NEGATIVES_POOL
    n_match = re.search(r"Generate (\d+)", prompt)
    n = int(n_match.group(1)) if n_match else 2

    chosen = [pool[(seed + i) % len(pool)] for i in range(n)]

    if prompt_version.startswith("neg_seed"):
        term_match = re.search(r'the word "([^"]+)"', prompt, re.IGNORECASE)
        if term_match:
            term = term_match.group(1)
            chosen = [
                c if re.search(r"\b" + re.escape(term) + r"\b", c.lower())
                else f"{c.rstrip('. ')}, {term}."
                for c in chosen
            ]

    return json.dumps({"negatives": chosen})


def make_real_llm_callable(
    model: str = DEFAULT_LLM_MODEL,
    *,
    api_key_env: str = "LLM_API_KEY",
    completion_url_env: str = "COMPLETION_URL",
    temperature: float = 0.3,
    max_tokens: int = 400,
) -> LLMCallable:
    """Build a callable that hits the real UF LLM endpoint via cached_request."""
    from src.llm_client_0 import cached_request

    api_key = os.getenv(api_key_env)
    url = os.getenv(completion_url_env)
    if not api_key or not url:
        raise RuntimeError(
            f"Real LLM mode requires env vars {api_key_env} and {completion_url_env}"
        )

    def _call(prompt: str, prompt_version: str) -> str:
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Endpoint can occasionally return non-JSON (empty body, HTML error,
        # rate-limit blip, transient connection drop). One bad response must
        # NOT abort a multi-thousand-row loop -- return "" so the parser falls
        # back gracefully. The failure isn't cached, so a re-run retries it.
        try:
            raw = cached_request(
                api_key=api_key,
                url=url,
                endpoint="completion",
                model=model,
                params=params,
                prompt_version=prompt_version,
            )
            return raw["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001 - intentional broad catch with fallback
            return ""

    return _call


# ---------------------------------------------------------------------------
# Per-source generators
# ---------------------------------------------------------------------------

def generate_hard_negatives_for_positive(
    positive_row: pd.Series,
    *,
    n: int,
    llm_callable: LLMCallable,
) -> list[dict]:
    """Generate N LLM-produced hard negatives anchored to one positive."""
    prompt = build_hard_negative_prompt(
        positive_utterance=positive_row["utterance"],
        subtypes=list(positive_row.get("subtype", [])),
        n=n,
    )
    raw = llm_callable(prompt, PROMPT_VERSIONS["hard_negative"])
    texts = parse_negatives_json(raw)
    return [
        {
            "text": t,
            "source_type": "llm_hard_negative",
            "anchor_utt_id": positive_row["utt_id"],
            "anchor_seed_term": None,
            "prompt_version": PROMPT_VERSIONS["hard_negative"],
        }
        for t in texts
    ]


def generate_seed_nonscience_for_term(
    seed_row: pd.Series,
    cat_definitions: dict[str, str],
    *,
    n: int,
    llm_callable: LLMCallable,
) -> list[dict]:
    """Generate N LLM-produced non-science uses of one seed term."""
    definition = cat_definitions.get(seed_row["category"], "(no definition available)")
    prompt = build_seed_nonscience_prompt(
        term=seed_row["term"],
        category=seed_row["category"],
        definition=definition,
        n=n,
    )
    raw = llm_callable(prompt, PROMPT_VERSIONS["seed_nonscience"])
    texts = parse_negatives_json(raw)
    return [
        {
            "text": t,
            "source_type": "seed_word_nonscience",
            "anchor_utt_id": None,
            "anchor_seed_term": seed_row["term"],
            "prompt_version": PROMPT_VERSIONS["seed_nonscience"],
        }
        for t in texts
    ]


def mine_transcript_negatives(
    transcript_paths: list[Path],
    seed_index: dict[str, pd.Series],
    *,
    max_per_file: int | None = None,
) -> list[dict]:
    """Read transcripts (one utterance per line) and keep lines that contain
    NO seed token or variant. Each line is a candidate clean negative.

    Plain-text path: one utterance per line. Caller is responsible for any
    pre-processing / speaker filtering. For richer hand-coded xlsx files use
    `mine_transcript_xlsx` instead -- that one trusts the human codes and is
    much more selective.
    """
    tokens = sorted(seed_index.keys(), key=len, reverse=True)
    out: list[dict] = []
    for path in transcript_paths:
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        kept = 0
        for ln, line in enumerate(lines):
            text = line.strip()
            if not text:
                continue
            text_lc = text.lower()
            # Conservative substring match: drops 'seed' AND 'seeds', 'force' AND
            # 'forces', etc. False positives cost us a negative; false negatives
            # would pollute the pool, which is worse.
            if any(t in text_lc for t in tokens):
                continue
            out.append({
                "text": text,
                "source_type": "transcript_clean",
                "transcript_subtype": "other_teacher_talk",
                "transcript_code": None,
                "anchor_utt_id": None,
                "anchor_seed_term": None,
                "prompt_version": None,
                "transcript_ref": f"{Path(path).name}#L{ln + 1}",
            })
            kept += 1
            if max_per_file is not None and kept >= max_per_file:
                break
    return out


def mine_transcript_xlsx(
    xlsx_path: Path,
    seed_index: dict[str, pd.Series],
    *,
    speaker_col: int = XLSX_SPEAKER_COL,
    utterance_col: int = XLSX_UTTERANCE_COL,
    behavior_code_col: int = XLSX_BEHAVIOR_CODE_COL,
    content_code_col: int = XLSX_CONTENT_CODE_COL,
    speaker_pattern: re.Pattern[str] = XLSX_TEACHER_SPEAKER_RE,
    science_code_token: str = "SCI",
    behavior_code_value: str = "Y",
    skip_sheets: set[str] = XLSX_META_SHEETS,
    apply_seed_filter: bool = False,
    max_total: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Mine clean negatives from a hand-coded transcript workbook.

    Per-sheet logic (one row per utterance, header on row 0):
        * Drop rows whose Speaker (after .strip()) doesn't match `speaker_pattern`
          (defaults to T1, T1:, T:, T17: ... -- teachers only).
        * Drop rows whose Content code (col G) contains 'SCI' (case-insensitive).
          These are the rows the original positives were drawn from -- including
          them would create label leakage.
        * Tag the survivors:
            transcript_subtype = 'behavior_disapproving' if col E == 'Y',
                                 else 'other_teacher_talk'
        * Optionally also drop any row whose text contains a seed-word substring
          (apply_seed_filter=True). Default False because the human SCI code is
          a stronger signal than substring matching, and most curriculum seed
          words (light, force, plant, animal, ...) appear in ordinary non-science
          talk too.

    Provenance: `transcript_ref = "<sheet>!R<excel_row_1_indexed>"`.
    """
    xl = pd.ExcelFile(xlsx_path)
    data_sheets = [s for s in xl.sheet_names if s not in skip_sheets]
    out: list[dict] = []
    sheet_stats: list[tuple[str, int, int, int]] = []
    seed_tokens = sorted(seed_index.keys(), key=len, reverse=True) if apply_seed_filter else []

    for sheet in data_sheets:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, header=0, dtype=str)
        except Exception:
            continue
        n_raw = len(df)
        if df.shape[1] <= max(speaker_col, utterance_col, content_code_col):
            continue

        speakers = df.iloc[:, speaker_col].fillna("").str.strip()
        utterances = df.iloc[:, utterance_col].fillna("").str.strip()
        content_codes = df.iloc[:, content_code_col].fillna("").str.strip()
        behavior_codes = (
            df.iloc[:, behavior_code_col].fillna("").str.strip()
            if df.shape[1] > behavior_code_col
            else pd.Series([""] * len(df))
        )

        is_teacher = speakers.apply(lambda s: bool(speaker_pattern.match(s)))
        is_sci = content_codes.str.contains(science_code_token, case=False, na=False)
        keep = is_teacher & ~is_sci & (utterances != "")
        kept_idx = df.index[keep]

        n_kept = 0
        for i in kept_idx:
            text = utterances.iloc[i]
            if apply_seed_filter:
                tlc = text.lower()
                if any(t in tlc for t in seed_tokens):
                    continue
            tx_subtype = (
                "behavior_disapproving"
                if behavior_codes.iloc[i].upper() == behavior_code_value.upper()
                else "other_teacher_talk"
            )
            out.append({
                "text": text,
                "source_type": "transcript_clean",
                "transcript_subtype": tx_subtype,
                "transcript_code": content_codes.iloc[i] or None,
                "anchor_utt_id": None,
                "anchor_seed_term": None,
                "prompt_version": None,
                "transcript_ref": f"{sheet}!R{i + 2}",  # +2: header row + 1-indexed
            })
            n_kept += 1
            if max_total is not None and len(out) >= max_total:
                break
        sheet_stats.append((sheet, n_raw, int(is_teacher.sum()), n_kept))
        if max_total is not None and len(out) >= max_total:
            break

    if verbose:
        total_raw = sum(s[1] for s in sheet_stats)
        total_teacher = sum(s[2] for s in sheet_stats)
        total_kept = sum(s[3] for s in sheet_stats)
        print(f"  xlsx mining: {len(sheet_stats)} sheets, {total_raw} rows total, "
              f"{total_teacher} teacher rows, {total_kept} kept after dropping SCI"
              f"{' (capped at max_total)' if max_total and len(out) >= max_total else ''}")

    return out


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def passes_structural_checks(text: str, *, anchor_seed_term: str | None = None) -> bool:
    """Cheap rules to drop obviously-bad LLM output before LLM gating."""
    if not text or not text.strip():
        return False
    n_words = len(text.split())
    if n_words < 2 or n_words > 40:
        return False
    if anchor_seed_term is not None:
        # seed_word_nonscience must actually contain the seed term
        if not re.search(r"\b" + re.escape(anchor_seed_term.lower()) + r"\b", text.lower()):
            return False
    return True


# ---------------------------------------------------------------------------
# Subtype assignment for negatives
# ---------------------------------------------------------------------------

def assign_subtypes_to_negatives(
    df_neg: pd.DataFrame,
    df_seed: pd.DataFrame,
    *,
    llm_classifier: Callable[[str], SubtypeResult] = llm_subtype_stub,
    skip_keyword_scan: bool = True,
) -> pd.DataFrame:
    """Reuse Step 2's logic so negatives carry the same sub-type vocabulary
    as positives -- letting Step 7 stratify and Step 4 build sub-type-balanced pairs.

    Negatives have no hand-coded Tier2/Tier3 cues, so the rule-based branch in
    Step 2 always misses for them. The keyword-scan branch, however, would
    surface-match seed words in the *text* of a negative (e.g. "I predict
    you're going to spill") and assign a practice label that bypasses the
    LLM's `not_science_shape` escape hatch entirely.

    `skip_keyword_scan=True` (default) routes ALL negatives through
    `llm_classifier` so they get the proper escape hatch. Pass False only if
    you genuinely want surface-form keyword tagging on negatives (e.g. for
    backwards-compat on an older artifact).
    """
    # Subtype assignment expects `utterance`, `tier2_cues`, `tier3_cues`.
    df = df_neg.copy()
    df["utterance"] = df["text"]
    df["tier2_cues"] = [[] for _ in range(len(df))]
    df["tier3_cues"] = [[] for _ in range(len(df))]
    enriched = assign_subtypes(
        df, df_seed, llm_classifier=llm_classifier,
        skip_keyword_scan=skip_keyword_scan,
    )
    df_neg = df_neg.copy()
    df_neg["subtype"] = enriched["subtype"].tolist()
    df_neg["subtype_source"] = enriched["subtype_source"].tolist()
    df_neg["subtype_confidence"] = enriched["subtype_confidence"].tolist()
    df_neg["subtype_prompt_version"] = enriched["subtype_prompt_version"].tolist()
    return df_neg


# ---------------------------------------------------------------------------
# Optional LLM gate scoring
# ---------------------------------------------------------------------------

LLM_GENERATED_SOURCES = {"llm_hard_negative", "seed_word_nonscience"}


def gate_score_negatives(
    df_neg: pd.DataFrame,
    *,
    llm_callable: LLMCallable,
    only_llm_generated: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add an llm_gate_score in [0,1] (1.0 = clearly non-science) per row.

    `only_llm_generated=True` (default) scores ONLY rows where source_type is
    `llm_hard_negative` or `seed_word_nonscience`. This is the cheap default
    because:
      - transcript_clean rows are already human-coded as not-SCI by experts;
        a second-pass LLM rating adds little signal there.
      - the LLM-generated rows (~671 on Y1) are the bucket where we have NO
        upstream quality signal, so this is where the LLM gate adds value.

    Pass `only_llm_generated=False` to score every row (~7,800 calls, ~100min).
    Rows that are skipped get `llm_gate_score = NaN`.
    """
    df_neg = df_neg.copy()
    scores: list[float | None] = []
    failures = 0
    n_to_score = 0
    for _, row in df_neg.iterrows():
        if only_llm_generated and row["source_type"] not in LLM_GENERATED_SOURCES:
            scores.append(None)
            continue
        n_to_score += 1
        prompt = build_gate_prompt(row["text"])
        try:
            raw = llm_callable(prompt, PROMPT_VERSIONS["gate_score"])
        except Exception:  # noqa: BLE001
            raw = ""
        score = parse_gate_score(raw)
        if score is None:
            failures += 1
        scores.append(score)
    df_neg["llm_gate_score"] = scores
    if verbose:
        n_ok = sum(1 for s in scores if s is not None)
        if only_llm_generated:
            print(f"Gate scoring (LLM-generated only): scored {n_ok}/{n_to_score}, "
                  f"{failures} failures, {len(df_neg) - n_to_score} skipped")
        else:
            print(f"Gate scoring (all rows): parsed {n_ok}/{len(df_neg)} ({failures} failures)")
    return df_neg


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_negative_pool(
    df_corpus: pd.DataFrame,
    df_seed: pd.DataFrame,
    df_cat: pd.DataFrame | None = None,
    *,
    transcript_paths: list[Path] | None = None,
    transcript_xlsx: Path | None = None,
    max_transcript_negatives: int | None = None,
    n_hard_per_positive: int = 2,
    n_per_seed_term: int = 2,
    seed_term_sample: int | None = None,
    llm_callable: LLMCallable = stub_llm_callable,
    subtype_classifier: Callable[[str], SubtypeResult] = llm_subtype_stub,
    skip_keyword_scan_for_negatives: bool = True,
    target_min_size: int = 600,
    verbose: bool = True,
) -> pd.DataFrame:
    """End-to-end build of the negative pool. Returns a tidy negatives DataFrame."""
    cat_definitions = (
        dict(zip(df_cat["label"], df_cat["definition"])) if df_cat is not None else {}
    )

    positives = df_corpus[df_corpus["label"] == "SCIENCE_TALK"].reset_index(drop=True)
    if verbose:
        print(f"Positives in corpus: {len(positives)}")

    rows: list[dict] = []

    if verbose:
        print(f"Generating hard negatives ({n_hard_per_positive} per positive)...")
    for _, pos in positives.iterrows():
        rows.extend(generate_hard_negatives_for_positive(
            pos, n=n_hard_per_positive, llm_callable=llm_callable,
        ))
    if verbose:
        print(f"  -> {len(rows)} hard-negative candidates")

    n_before_seed = len(rows)
    seed_terms = df_seed
    if seed_term_sample is not None and seed_term_sample < len(df_seed):
        seed_terms = df_seed.sample(seed_term_sample, random_state=42).reset_index(drop=True)
    if verbose:
        print(f"Generating seed-word nonscience ({n_per_seed_term} per term, "
              f"{len(seed_terms)} terms)...")
    for _, srow in seed_terms.iterrows():
        rows.extend(generate_seed_nonscience_for_term(
            srow, cat_definitions, n=n_per_seed_term, llm_callable=llm_callable,
        ))
    if verbose:
        print(f"  -> {len(rows) - n_before_seed} seed-nonscience candidates")

    if transcript_xlsx is not None or transcript_paths:
        seed_index = build_seed_index(df_seed)
        n_before_tx = len(rows)
        if transcript_xlsx is not None:
            if verbose:
                print(f"Mining transcripts (xlsx): {transcript_xlsx.name}...")
            rows.extend(mine_transcript_xlsx(
                Path(transcript_xlsx), seed_index,
                max_total=max_transcript_negatives,
                verbose=verbose,
            ))
        if transcript_paths:
            rows.extend(mine_transcript_negatives(transcript_paths, seed_index))
        if verbose:
            print(f"  -> {len(rows) - n_before_tx} transcript_clean candidates")
    else:
        if verbose:
            print("  no transcripts supplied -> source_type='transcript_clean' is empty")

    df_neg = pd.DataFrame(rows)
    if df_neg.empty:
        raise RuntimeError("No negative candidates were generated; check LLM callable.")

    df_neg["text"] = df_neg["text"].apply(lambda s: s.strip())
    df_neg["structural_check_passed"] = df_neg.apply(
        lambda r: passes_structural_checks(r["text"], anchor_seed_term=r["anchor_seed_term"]),
        axis=1,
    )
    n_struct = (~df_neg["structural_check_passed"]).sum()
    failed = df_neg[~df_neg["structural_check_passed"]]
    if verbose and len(failed) > 0:
        n_show = min(12, len(failed))
        print(f"  Structural-check failures (sample {n_show} of {len(failed)}):")
        for _, r in failed.head(n_show).iterrows():
            t = str(r["text"])[:140]
            ast = r.get("anchor_seed_term")
            extra = f" anchor_seed_term={ast!r}" if pd.notna(ast) and ast else ""
            print(f"    - {t!r}{extra}")
    if verbose:
        print(f"Structural checks: dropped {n_struct} rows that failed; kept "
              f"{df_neg['structural_check_passed'].sum()}")
    df_neg = df_neg[df_neg["structural_check_passed"]].reset_index(drop=True)

    before = len(df_neg)
    df_neg = df_neg.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if verbose:
        print(f"De-duplication: {before} -> {len(df_neg)} rows")

    # Cross-leak check: drop any negative whose text matches a positive
    # utterance (case-insensitive, whitespace-stripped). Without this, the
    # bi-encoder would see the same text with both labels -- a contradictory
    # training signal. Most likely source: a transcript-mined teacher
    # utterance that happens to coincide with a hand-curated positive.
    pos_texts = {
        str(u).strip().lower()
        for u in df_corpus.loc[df_corpus["label"] == "SCIENCE_TALK", "utterance"]
    }
    neg_text_lower = df_neg["text"].str.strip().str.lower()
    leak_mask = neg_text_lower.isin(pos_texts)
    n_leaks = int(leak_mask.sum())
    if verbose:
        print(f"Cross-leak check: dropped {n_leaks} negatives matching a positive")
    if n_leaks > 0:
        df_neg = df_neg[~leak_mask].reset_index(drop=True)

    if verbose:
        print(f"Assigning sub-types to {len(df_neg)} negatives...")
    df_neg = assign_subtypes_to_negatives(
        df_neg, df_seed, llm_classifier=subtype_classifier,
        skip_keyword_scan=skip_keyword_scan_for_negatives,
    )
    if verbose:
        sub_src_counts = Counter(df_neg["subtype_source"])
        print(f"  subtype source breakdown: {dict(sub_src_counts)}")

    if "transcript_subtype" not in df_neg.columns:
        df_neg["transcript_subtype"] = None
    if "transcript_code" not in df_neg.columns:
        df_neg["transcript_code"] = None
    if "transcript_ref" not in df_neg.columns:
        df_neg["transcript_ref"] = None

    df_neg["llm_gate_score"] = pd.NA
    df_neg["model_id"] = DEFAULT_LLM_MODEL
    df_neg["human_verified"] = False
    df_neg["human_label"] = pd.NA
    df_neg["neg_id"] = [f"neg_{i:05d}" for i in range(len(df_neg))]

    df_neg = df_neg[[
        "neg_id", "text", "source_type", "subtype",
        "subtype_source", "subtype_confidence", "subtype_prompt_version",
        "transcript_subtype", "transcript_code", "transcript_ref",
        "anchor_utt_id", "anchor_seed_term",
        "structural_check_passed", "llm_gate_score",
        "model_id", "prompt_version",
        "human_verified", "human_label",
    ]]

    if verbose:
        print(f"\nFinal negative pool: {len(df_neg)} rows")
        print(f"  by source_type: {dict(Counter(df_neg['source_type']))}")
        tx_subs = df_neg.loc[df_neg["source_type"] == "transcript_clean", "transcript_subtype"]
        if len(tx_subs):
            print(f"  transcript split: {dict(Counter(tx_subs))}")
        flat_st = [s for sts in df_neg["subtype"] for s in sts]
        print(f"  by subtype:     {dict(Counter(flat_st))}")

    return df_neg


# ---------------------------------------------------------------------------
# Validation, sampling for review
# ---------------------------------------------------------------------------

def validate_negatives(
    df_neg: pd.DataFrame,
    n_positives: int,
    *,
    target_ratio: float = 3.0,
    require_three_sources: bool = False,
    verbose: bool = True,
) -> dict:
    """Hard + soft assertions for the Step 3 DoD."""
    src_counts = Counter(df_neg["source_type"])
    flat_st = [s for sts in df_neg["subtype"] for s in sts]
    st_counts = Counter(flat_st)

    info = {
        "n_negatives": len(df_neg),
        "n_positives": n_positives,
        "ratio": len(df_neg) / max(n_positives, 1),
        "source_distribution": dict(src_counts),
        "subtype_distribution": dict(st_counts),
        "warnings": [],
    }

    target = target_ratio * n_positives
    if len(df_neg) < target:
        info["warnings"].append(
            f"Negative pool is {len(df_neg)} rows (<{target:.0f} target = "
            f"{target_ratio}x positives). Increase n_hard_per_positive or n_per_seed_term."
        )

    represented = set(src_counts) & NEGATIVE_SOURCE_TYPES
    missing = NEGATIVE_SOURCE_TYPES - represented
    if missing:
        msg = f"Source types missing from pool: {sorted(missing)}"
        if require_three_sources:
            assert not missing, msg
        else:
            info["warnings"].append(msg)

    for st in PRACTICE_LABELS | {"content"}:
        if st_counts.get(st, 0) == 0:
            info["warnings"].append(f"Sub-type '{st}' has zero negatives (limits stratification).")

    bad_ids = df_neg[df_neg["neg_id"].duplicated()]["neg_id"].tolist()
    assert not bad_ids, f"neg_id must be unique, found dupes: {bad_ids[:5]}"
    assert df_neg["text"].notna().all(), "Every negative must have non-null text"

    if verbose:
        print(f"\nValidation:")
        print(f"  total negatives: {len(df_neg)} (positives={n_positives}, "
              f"ratio={info['ratio']:.2f}x, target>={target_ratio}x)")
        print(f"  source types:    {dict(src_counts)}")
        print(f"  subtypes:        {dict(st_counts)}")
        if info["warnings"]:
            print(f"  warnings:")
            for w in info["warnings"]:
                print(f"    - {w}")
        else:
            print(f"  no warnings")

    return info


def sample_for_review(
    df_neg: pd.DataFrame,
    n: int = 50,
    *,
    seed: int = 42,
    stratify_by_source: bool = True,
) -> pd.DataFrame:
    """Return n rows for hand-checking, stratified across source_type by default.

    DoD requires a 50-row hand-check with >=90% true-negative rate.
    """
    if stratify_by_source and df_neg["source_type"].nunique() > 1:
        per_source = max(1, n // df_neg["source_type"].nunique())
        parts = []
        for _, group in df_neg.groupby("source_type"):
            take = min(per_source, len(group))
            parts.append(group.sample(take, random_state=seed))
        sampled = pd.concat(parts, ignore_index=True)
        if len(sampled) < n:
            remaining = df_neg.drop(sampled.index, errors="ignore")
            extra = remaining.sample(min(n - len(sampled), len(remaining)), random_state=seed)
            sampled = pd.concat([sampled, extra], ignore_index=True)
        return sampled.head(n).reset_index(drop=True)

    return df_neg.sample(min(n, len(df_neg)), random_state=seed).reset_index(drop=True)


def export_negative_review_template(
    df_neg: pd.DataFrame,
    processed_dir: Path,
    *,
    n: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> Path | None:
    """Stratified hand-review template under ``review_samples/``.

    Fill ``review_human_label`` with NOT_SCIENCE_TALK / POSSIBLE_LEAK / UNCLEAR;
    use ``review_notes`` for free text. Merging back into the main parquet is
    a manual step.
    """
    if len(df_neg) == 0:
        return None
    rd = processed_dir / "review_samples"
    rd.mkdir(parents=True, exist_ok=True)
    sample = sample_for_review(df_neg, n=min(n, len(df_neg)), seed=seed)
    out = sample.copy()
    out["review_human_label"] = pd.NA
    out["review_notes"] = pd.NA
    path = rd / "negatives_review_template.parquet"
    out.to_parquet(path, index=False)
    if verbose:
        print(f"  Review template: {path} ({len(out)} rows)")
    return path


def compute_review_pass_rate(df_neg: pd.DataFrame) -> dict:
    """Of the rows the human marked, what % were truly NOT_SCIENCE_TALK?"""
    reviewed = df_neg[df_neg["human_verified"] == True]
    if len(reviewed) == 0:
        return {"n_reviewed": 0, "n_true_negative": 0, "pass_rate": None}
    n_true_neg = (reviewed["human_label"] == "NOT_SCIENCE_TALK").sum()
    return {
        "n_reviewed": int(len(reviewed)),
        "n_true_negative": int(n_true_neg),
        "pass_rate": float(n_true_neg) / len(reviewed),
    }


# ---------------------------------------------------------------------------
# Applying a filled negatives-quality review sheet
# ---------------------------------------------------------------------------

NEG_REVIEW_LABELS = ("NOT_SCIENCE_TALK", "POSSIBLE_LEAK", "UNCLEAR")
NEG_REVIEW_FILENAME = "negatives_review.xlsx"


def validate_negatives_review(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a filled negatives review sheet; return reviewed rows only.

    A blank ``review_human_label`` means "not reviewed" (skipped). Raises
    ValueError on unknown labels or a missing id / label column.
    """
    if "neg_id" not in df.columns:
        raise ValueError("Template missing `neg_id` column.")
    if "review_human_label" not in df.columns:
        raise ValueError("Template missing `review_human_label` column.")
    df = df.copy()
    lab = df["review_human_label"].astype("string").str.strip().str.upper()
    df["review_human_label"] = lab
    bad = sorted(set(lab.dropna()) - set(NEG_REVIEW_LABELS))
    if bad:
        raise ValueError(
            f"Invalid review_human_label values {bad}; allowed: "
            f"{NEG_REVIEW_LABELS} (or blank=not reviewed)."
        )
    return df[lab.notna()].copy()


def apply_negatives_review(
    template_path: Path,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    drop_unclear: bool = False,
    verbose: bool = True,
) -> dict:
    """Merge a filled negatives review sheet into negatives.parquet (with backup).

    For each reviewed ``neg_id``: set ``human_verified=True`` and ``human_label``.
    Rows labeled ``POSSIBLE_LEAK`` are removed from the pool (they look like real
    science talk and would poison contrastive training). ``UNCLEAR`` rows are
    kept by default (pass ``drop_unclear=True`` to remove them too).
    ``NOT_SCIENCE_TALK`` rows are confirmed and kept.
    """
    template_path = Path(template_path)
    if template_path.suffix.lower() in {".xlsx", ".xls"}:
        tmpl = pd.read_excel(template_path)
    elif template_path.suffix.lower() == ".parquet":
        tmpl = pd.read_parquet(template_path)
    else:
        raise ValueError(f"Unsupported template type: {template_path.suffix}")

    reviewed = validate_negatives_review(tmpl)
    if len(reviewed) == 0:
        if verbose:
            print("  No reviewed rows in template; nothing to apply.")
        return {"n_reviewed": 0}

    label_map = dict(zip(reviewed["neg_id"].astype(str),
                         reviewed["review_human_label"].astype(str)))

    neg_path = processed_dir / "negatives.parquet"
    df_neg = pd.read_parquet(neg_path)
    nid = df_neg["neg_id"].astype(str)
    pool_ids = set(nid)
    missing_ids = sorted(set(label_map) - pool_ids)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df_neg.to_parquet(processed_dir / f"negatives.pre_review.{ts}.parquet", index=False)

    mask = nid.isin(label_map)
    df_neg.loc[mask, "human_verified"] = True
    df_neg.loc[mask, "human_label"] = nid[mask].map(label_map)

    drop_labels = {"POSSIBLE_LEAK"}
    if drop_unclear:
        drop_labels.add("UNCLEAR")
    drop_ids = {k for k, v in label_map.items() if v in drop_labels}
    keep = ~nid.isin(drop_ids)
    n_before = len(df_neg)
    df_neg = df_neg[keep].reset_index(drop=True)
    df_neg.to_parquet(neg_path, index=False)

    n_true_neg = sum(1 for v in label_map.values() if v == "NOT_SCIENCE_TALK")
    pass_rate = n_true_neg / len(label_map) if label_map else None

    stats = {
        "n_reviewed": len(label_map),
        "by_label": dict(Counter(label_map.values())),
        "dropped": int(n_before - len(df_neg)),
        "neg_before": n_before,
        "neg_after": len(df_neg),
        "missing_ids": missing_ids,
        "pass_rate": pass_rate,
    }
    if verbose:
        print(f"Applied negatives review: {stats}")
        if pass_rate is not None:
            tgt = "PASS" if pass_rate >= 0.90 else "BELOW 90% TARGET"
            print(f"  true-negative rate: {pass_rate:.1%} ({tgt})")
        if missing_ids:
            print(f"  WARNING: {len(missing_ids)} reviewed neg_id(s) not in pool "
                  f"(stale template?): {missing_ids[:5]}")
        if stats["dropped"]:
            print("\nNOTE: negatives pool changed. Rebuild splits so training "
                  "excludes the removed rows:\n  python src/splits.py")
    return stats


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    use_real_llm: bool = False,
    use_real_llm_negatives_subtype: bool = False,
    transcript_dir: Path | None = None,
    transcript_xlsx: Path | None = None,
    max_transcript_negatives: int | None = None,
    n_hard_per_positive: int = 2,
    n_per_seed_term: int = 2,
    seed_term_sample: int | None = None,
    apply_gate_scoring: bool = True,
    gate_score_only_llm_generated: bool = True,
    skip_keyword_scan_for_negatives: bool = True,
    verbose: bool = True,
) -> Path:
    df_corpus = pd.read_parquet(processed_dir / "corpus.parquet")
    df_seed = pd.read_parquet(processed_dir / "seed_words.parquet")
    df_cat = pd.read_parquet(processed_dir / "category_defs.parquet")

    llm_callable = (
        make_real_llm_callable() if use_real_llm else stub_llm_callable
    )
    if verbose:
        print(f"LLM mode (negative generation): "
              f"{'REAL (Llama 3.3-70b)' if use_real_llm else 'STUB (deterministic)'}")

    subtype_classifier = (
        make_real_llm_subtype_classifier()
        if use_real_llm_negatives_subtype else llm_subtype_stub
    )
    if verbose:
        print(f"LLM mode (negative subtype labeling): "
              f"{'REAL (Llama 3.3-70b)' if use_real_llm_negatives_subtype else 'STUB'}")

    transcript_paths: list[Path] = []
    if transcript_dir is not None:
        transcript_paths = sorted(Path(transcript_dir).glob("*.txt"))
        if verbose:
            print(f"Found {len(transcript_paths)} transcript files in {transcript_dir}")

    if transcript_xlsx is None and DEFAULT_TRANSCRIPT_XLSX.exists():
        transcript_xlsx = DEFAULT_TRANSCRIPT_XLSX
        if verbose:
            print(f"Auto-detected transcript workbook: {transcript_xlsx}")

    df_neg = build_negative_pool(
        df_corpus, df_seed, df_cat,
        transcript_paths=transcript_paths or None,
        transcript_xlsx=transcript_xlsx,
        max_transcript_negatives=max_transcript_negatives,
        n_hard_per_positive=n_hard_per_positive,
        n_per_seed_term=n_per_seed_term,
        seed_term_sample=seed_term_sample,
        llm_callable=llm_callable,
        subtype_classifier=subtype_classifier,
        skip_keyword_scan_for_negatives=skip_keyword_scan_for_negatives,
        verbose=verbose,
    )

    if apply_gate_scoring:
        df_neg = gate_score_negatives(
            df_neg,
            llm_callable=llm_callable,
            only_llm_generated=gate_score_only_llm_generated,
            verbose=verbose,
        )
    else:
        # Make the column explicit even when gate-scoring is off, so the
        # schema is stable across runs.
        if "llm_gate_score" not in df_neg.columns:
            df_neg["llm_gate_score"] = pd.NA

    n_pos = (df_corpus["label"] == "SCIENCE_TALK").sum()
    validate_negatives(df_neg, n_positives=int(n_pos), verbose=verbose)

    export_negative_review_template(df_neg, processed_dir, verbose=verbose)

    out_path = processed_dir / "negatives.parquet"
    df_neg.to_parquet(out_path, index=False)
    if verbose:
        print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: build the NOT_SCIENCE_TALK pool.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--use-llm", action="store_true",
                        help="Use the real Llama 3.3-70b endpoint for negative generation "
                             "(default: deterministic stub).")
    parser.add_argument("--use-llm-negatives-subtype", action="store_true",
                        help="Use real Llama 3.3-70b for sub-type labeling of negatives "
                             "(~1,300 cached calls on the Y1 transcript pool). Default: stub.")
    parser.add_argument("--transcript-dir", type=Path, default=None,
                        help="Directory of plain-text transcripts (one utterance per line) to "
                             "mine for clean negatives.")
    parser.add_argument("--transcript-xlsx", type=Path, default=None,
                        help=f"Coded-transcript workbook to mine. Default auto-detect: "
                             f"{DEFAULT_TRANSCRIPT_XLSX.relative_to(_PROJECT_ROOT)}")
    parser.add_argument("--max-transcript-negatives", type=int, default=None,
                        help="Cap on transcript_clean rows so they don't drown LLM sources.")
    parser.add_argument("--n-hard-per-positive", type=int, default=2)
    parser.add_argument("--n-per-seed-term", type=int, default=2)
    parser.add_argument("--seed-term-sample", type=int, default=None,
                        help="Cap the number of seed terms used (default: all).")
    parser.add_argument("--no-gate-scoring", action="store_true",
                        help="Skip second-pass LLM gate (~700 calls on LLM-generated rows). "
                             "Default: gate ON.")
    parser.add_argument("--gate-score-all-rows", action="store_true",
                        help="When gate scoring is on: score every row, not just the "
                             "LLM-generated bucket (~7800 calls vs ~700).")
    parser.add_argument("--keep-keyword-scan-for-negatives", action="store_true",
                        help="Restore the v1 behavior where seed-word matches in the negative "
                             "TEXT bypass the LLM. By default we route all negatives through "
                             "the LLM so they get the not_science_shape escape hatch.")
    parser.add_argument("--apply-negatives-review", type=Path, default=None, metavar="TEMPLATE",
                        help="Apply a filled negatives-quality review sheet (xlsx/parquet): "
                             "mark human_verified/human_label and drop POSSIBLE_LEAK rows.")
    parser.add_argument("--drop-unclear", action="store_true",
                        help="With --apply-negatives-review, also drop UNCLEAR-labeled rows "
                             "(default: keep them).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.apply_negatives_review is not None:
        apply_negatives_review(
            args.apply_negatives_review,
            args.processed_dir,
            drop_unclear=args.drop_unclear,
            verbose=not args.quiet,
        )
        sys.exit(0)

    run(
        args.processed_dir,
        use_real_llm=args.use_llm,
        use_real_llm_negatives_subtype=args.use_llm_negatives_subtype,
        transcript_dir=args.transcript_dir,
        transcript_xlsx=args.transcript_xlsx,
        max_transcript_negatives=args.max_transcript_negatives,
        n_hard_per_positive=args.n_hard_per_positive,
        n_per_seed_term=args.n_per_seed_term,
        seed_term_sample=args.seed_term_sample,
        apply_gate_scoring=not args.no_gate_scoring,
        gate_score_only_llm_generated=not args.gate_score_all_rows,
        skip_keyword_scan_for_negatives=not args.keep_keyword_scan_for_negatives,
        verbose=not args.quiet,
    )
