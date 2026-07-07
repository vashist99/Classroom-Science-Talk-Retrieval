"""Mine real science-talk positives from the hand-coded transcript workbook.

This is the *mirror image* of `negatives_3.mine_transcript_xlsx`: that function
keeps teacher rows whose content code is NOT 'SCI' (to build negatives); here we
keep teacher rows whose content code IS 'SCI'. Those rows were human-coded as
science talk by the project's coders, so they are real, in-distribution
positives -- the opposite of the LLM-paraphrase positives in `pairs.parquet`.

Why this exists
---------------
The fine-tuned bi-encoder (Step 8) is starved of *real* positive signal: the
curated corpus is ~196 rows and the augmented pairs are synthetic. Harvesting the
existing SCI codes adds genuine positives at zero new labeling cost.

Pipeline
--------
    extract_sci_positives()   teacher & content-code contains 'SCI'
        -> filter_fragments()     drop too-short / fragmentary utterances
        -> dedup_within()         collapse exact (normalized) duplicates
        -> net_new_vs_corpus()    drop rows already present in corpus.parquet
        -> build_corpus_rows()    project onto the Step-1 corpus schema
        -> assign subtypes        rule+keyword (LLM fallback optional)
        -> merge_into_corpus()    append + back up corpus.parquet

Every mined row is tagged `source = MINED_SOURCE_TAG` and gets a distinct
`utt_id` prefix (`utt_sci_####`) so it is always distinguishable from the
curated set. `transcript_ref` records `<sheet>!R<excel_row>` provenance.

IMPORTANT: merging changes corpus.parquet, which makes the *downstream*
artifacts (pairs, baseline cosines, confidence/routing, splits) stale. This
module deliberately does NOT touch them. After merging, re-run Steps 2-7 (and
re-export any review queue) so the new positives flow through augmentation and
splitting. The new rows have no register variants until Step 4 is re-run.

Usage:
    python src/positives_mining.py --dry-run        # report only, write nothing
    python src/positives_mining.py                  # merge into corpus.parquet
    python src/positives_mining.py --min-words 6    # stricter fragment filter
    python src/positives_mining.py --use-llm-subtypes
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR, normalize_text
from src.negatives_3 import (
    DEFAULT_TRANSCRIPT_XLSX,
    XLSX_CONTENT_CODE_COL,
    XLSX_META_SHEETS,
    XLSX_SPEAKER_COL,
    XLSX_TEACHER_SPEAKER_RE,
    XLSX_UTTERANCE_COL,
)
from src.subtypes_2 import (
    assign_subtypes,
    llm_subtype_stub,
    make_real_llm_subtype_classifier,
)

SCIENCE_CODE_TOKEN = "SCI"
MINED_SOURCE_TAG = "transcript_sci_mined"
MINED_ID_PREFIX = "utt_sci_"

# Fragment filter defaults. SCI rows were coded in conversational context, so
# many are mid-turn fragments ("Plants right?", "Plants, trees,") that are weak
# as standalone retrieval targets for a non-contextual bi-encoder.
DEFAULT_MIN_WORDS = 5
DEFAULT_MIN_CHARS = 20

# Map the trailing token of a sheet name to a corpus `setting` value, matching
# augment_4.SETTING_TO_REGISTER (Centers -> INFORMAL, etc.). Sheets look like
# "01-2_19LG", "8-15C", "04-04SG".
_SHEET_SUFFIX_RE = re.compile(r"([A-Za-z]+)\d*$")
SUFFIX_TO_SETTING = {
    "LG": "Large Group",
    "SG": "Small Group",
    "C": "Centers",
}


def sheet_to_setting(sheet: str) -> str:
    """Best-effort map a transcript sheet name to a corpus `setting`."""
    m = _SHEET_SUFFIX_RE.search(str(sheet).strip())
    if not m:
        return "Unknown"
    return SUFFIX_TO_SETTING.get(m.group(1).upper(), "Unknown")


def _canon(text: object) -> str:
    """Canonical key for de-dup: NFKC + whitespace-collapse + casefold."""
    norm = normalize_text(text)
    if isinstance(norm, float):  # NaN sentinel from normalize_text
        return ""
    return norm.casefold()


# ---------------------------------------------------------------------------
# Extraction (mirror of negatives_3.mine_transcript_xlsx)
# ---------------------------------------------------------------------------

def extract_sci_positives(
    xlsx_path: Path = DEFAULT_TRANSCRIPT_XLSX,
    *,
    speaker_col: int = XLSX_SPEAKER_COL,
    utterance_col: int = XLSX_UTTERANCE_COL,
    content_code_col: int = XLSX_CONTENT_CODE_COL,
    speaker_pattern: re.Pattern[str] = XLSX_TEACHER_SPEAKER_RE,
    science_code_token: str = SCIENCE_CODE_TOKEN,
    skip_sheets: set[str] = XLSX_META_SHEETS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return all teacher rows whose content code (col G) contains 'SCI'.

    Columns: sheet, excel_row, speaker, utterance (normalized), content_code,
    setting. This is intentionally the inverse of the negative miner's
    `~is_sci` mask.
    """
    xl = pd.ExcelFile(xlsx_path)
    data_sheets = [s for s in xl.sheet_names if s not in skip_sheets]
    rows: list[dict] = []

    for sheet in data_sheets:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, header=0, dtype=str)
        except Exception:
            continue
        if df.shape[1] <= max(speaker_col, utterance_col, content_code_col):
            continue

        speakers = df.iloc[:, speaker_col].fillna("").str.strip()
        utterances = df.iloc[:, utterance_col].fillna("").str.strip()
        content_codes = df.iloc[:, content_code_col].fillna("").str.strip()

        is_teacher = speakers.apply(lambda s: bool(speaker_pattern.match(s)))
        is_sci = content_codes.str.contains(science_code_token, case=False, na=False)
        has_text = utterances != ""
        keep = is_teacher & is_sci & has_text

        setting = sheet_to_setting(sheet)
        for i in df.index[keep]:
            norm = normalize_text(utterances.iloc[i])
            if isinstance(norm, float) or not norm:
                continue
            rows.append({
                "sheet": sheet,
                "excel_row": int(i) + 2,  # header row 0 -> first data row = excel row 2
                "speaker": speakers.iloc[i],
                "utterance": norm,
                "content_code": content_codes.iloc[i],
                "setting": setting,
            })

    out = pd.DataFrame(
        rows,
        columns=["sheet", "excel_row", "speaker", "utterance", "content_code", "setting"],
    )
    if verbose:
        print(f"Extracted {len(out)} SCI-coded teacher rows from {len(data_sheets)} sheets")
    return out


# ---------------------------------------------------------------------------
# Quality / dedup filters
# ---------------------------------------------------------------------------

def is_fragment(
    text: str,
    *,
    min_words: int = DEFAULT_MIN_WORDS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> bool:
    """True if `text` is too short/fragmentary to be a useful standalone positive."""
    t = str(text).strip()
    if len(t) < min_chars:
        return True
    n_words = len(re.findall(r"\b\w+\b", t))
    return n_words < min_words


def filter_fragments(
    df: pd.DataFrame,
    *,
    min_words: int = DEFAULT_MIN_WORDS,
    min_chars: int = DEFAULT_MIN_CHARS,
    verbose: bool = True,
) -> pd.DataFrame:
    if len(df) == 0:
        return df
    mask = ~df["utterance"].apply(
        lambda t: is_fragment(t, min_words=min_words, min_chars=min_chars)
    )
    out = df[mask].reset_index(drop=True)
    if verbose:
        print(f"Fragment filter (min_words={min_words}, min_chars={min_chars}): "
              f"kept {len(out)}/{len(df)} (dropped {len(df) - len(out)})")
    return out


def dedup_within(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    if len(df) == 0:
        return df
    df = df.copy()
    df["canon"] = df["utterance"].map(_canon)
    out = df[df["canon"] != ""].drop_duplicates(subset=["canon"]).reset_index(drop=True)
    if verbose:
        print(f"Within-set dedup: kept {len(out)}/{len(df)} (dropped {len(df) - len(out)})")
    return out


def net_new_vs_corpus(
    df: pd.DataFrame,
    df_corpus: pd.DataFrame,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop rows whose (normalized, casefolded) utterance already exists in corpus."""
    if len(df) == 0:
        return df
    df = df.copy()
    if "canon" not in df.columns:
        df["canon"] = df["utterance"].map(_canon)
    corpus_keys = set(df_corpus["utterance"].map(_canon)) - {""}
    out = df[~df["canon"].isin(corpus_keys)].reset_index(drop=True)
    if verbose:
        print(f"Net-new vs corpus: kept {len(out)}/{len(df)} "
              f"(dropped {len(df) - len(out)} already in corpus)")
    return out


# ---------------------------------------------------------------------------
# Projection onto corpus schema + subtype assignment
# ---------------------------------------------------------------------------

def _next_mined_index(df_corpus: pd.DataFrame) -> int:
    """First free integer suffix for the utt_sci_#### id scheme (idempotent)."""
    existing = df_corpus["utt_id"].astype(str)
    nums = [
        int(m.group(1))
        for s in existing
        if (m := re.match(rf"^{re.escape(MINED_ID_PREFIX)}(\d+)$", s))
    ]
    return (max(nums) + 1) if nums else 0


def build_corpus_rows(
    df: pd.DataFrame,
    df_corpus: pd.DataFrame,
    *,
    source_tag: str = MINED_SOURCE_TAG,
) -> pd.DataFrame:
    """Project mined rows onto the Step-1 corpus schema (minus subtype cols)."""
    start = _next_mined_index(df_corpus)
    rows: list[dict] = []
    for offset, (_, r) in enumerate(df.iterrows()):
        rows.append({
            "utt_id": f"{MINED_ID_PREFIX}{start + offset:04d}",
            "utterance": r["utterance"],
            "label": "SCIENCE_TALK",
            "setting": r["setting"],
            "source": source_tag,
            "tier2_cues": [],
            "tier3_cues": [],
            "was_sci_coded": 1,
            "transcript_ref": f"{r['sheet']}!R{r['excel_row']}",
            "topic": None,
            "citation": None,
        })
    return pd.DataFrame(rows)


def assign_mined_subtypes(
    df_rows: pd.DataFrame,
    df_seed: pd.DataFrame,
    *,
    use_llm: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add subtype columns to mined rows via rule+keyword (LLM fallback optional).

    SCI-coded positives are real science talk, so the `not_science_shape` escape
    hatch is irrelevant here; we keep the keyword scan ON (unlike negatives).
    """
    classifier = make_real_llm_subtype_classifier() if use_llm else llm_subtype_stub
    labeled = assign_subtypes(df_rows, df_seed, llm_classifier=classifier)
    if verbose:
        from collections import Counter
        srcs = Counter(labeled["subtype_source"])
        print(f"Subtype sources for mined rows: {dict(srcs)}")
        n_stub = int((labeled["subtype_prompt_version"] == "subtype_v0_stub").sum())
        if n_stub:
            print(f"  WARNING: {n_stub} rows used the deterministic subtype STUB "
                  f"(confidence 0.0). Re-run Step 2 with --use-llm-step2 to label them "
                  f"properly, or pass --use-llm-subtypes here.")
    return labeled


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_into_corpus(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    xlsx_path: Path = DEFAULT_TRANSCRIPT_XLSX,
    seed_path: Path | None = None,
    min_words: int = DEFAULT_MIN_WORDS,
    min_chars: int = DEFAULT_MIN_CHARS,
    source_tag: str = MINED_SOURCE_TAG,
    use_llm_subtypes: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """Full pipeline: extract -> filter -> dedup -> label -> merge into corpus.

    Returns a stats dict. With `dry_run=True` nothing is written.
    """
    corpus_path = processed_dir / "corpus.parquet"
    seed_path = seed_path or (processed_dir / "seed_words.parquet")

    df_corpus = pd.read_parquet(corpus_path)
    df_seed = pd.read_parquet(seed_path)
    n_before = len(df_corpus)

    sci = extract_sci_positives(xlsx_path, verbose=verbose)
    sci = filter_fragments(sci, min_words=min_words, min_chars=min_chars, verbose=verbose)
    sci = dedup_within(sci, verbose=verbose)
    sci = net_new_vs_corpus(sci, df_corpus, verbose=verbose)

    new_rows = build_corpus_rows(sci, df_corpus, source_tag=source_tag)
    new_labeled = (
        assign_mined_subtypes(new_rows, df_seed, use_llm=use_llm_subtypes, verbose=verbose)
        if len(new_rows) > 0
        else new_rows
    )

    stats = {
        "corpus_before": n_before,
        "net_new_positives": len(new_labeled),
        "corpus_after": n_before + len(new_labeled),
        "dry_run": dry_run,
        "source_tag": source_tag,
    }

    if dry_run:
        if verbose:
            print(f"\n[DRY RUN] would add {len(new_labeled)} positives "
                  f"({n_before} -> {n_before + len(new_labeled)}). Nothing written.")
        stats["sample"] = new_labeled["utterance"].head(10).tolist()
        return stats

    if len(new_labeled) == 0:
        if verbose:
            print("\nNo net-new positives to add; corpus.parquet unchanged.")
        return stats

    # Align columns to the existing corpus (existing has subtype cols already).
    merged = pd.concat([df_corpus, new_labeled], axis=0, ignore_index=True)
    merged = merged[df_corpus.columns]
    assert merged["utt_id"].is_unique, "utt_id collision after merge"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = processed_dir / f"corpus.pre_mining.{ts}.parquet"
    df_corpus.to_parquet(backup_path, index=False)
    merged.to_parquet(corpus_path, index=False)

    stats["backup"] = str(backup_path)
    if verbose:
        print(f"\nBacked up old corpus -> {backup_path.name}")
        print(f"Merged corpus.parquet: {n_before} -> {len(merged)} rows "
              f"(+{len(new_labeled)} mined positives, source='{source_tag}')")
        print("\nNOTE: downstream artifacts (pairs, baseline_cosine, confidence/"
              "routing, splits) are now STALE. Re-run Steps 2-7 so the new "
              "positives get subtypes verified, register variants, and splits.")
    return stats


# ---------------------------------------------------------------------------
# not_science_shape review (human adjudication of LLM-flagged mined positives)
# ---------------------------------------------------------------------------

NS_DECISIONS = ("keep", "drop", "edit")
NS_REVIEW_FILENAME = "not_science_review.xlsx"
NS_SUBTYPE = "not_science_shape"


def _has_ns(subtype) -> bool:
    try:
        return NS_SUBTYPE in list(subtype)
    except TypeError:
        return False


def export_not_science_review(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> Path | None:
    """Export corpus positives tagged `not_science_shape` for human adjudication.

    These were human-coded SCI in lesson context but the LLM judged them
    non-science as standalone text. A reviewer decides keep / drop / edit.
    Writes a sheet with blank `decision`, `corrected_text`, `notes` columns.
    """
    corpus = pd.read_parquet(processed_dir / "corpus.parquet")
    flagged = corpus[corpus["subtype"].apply(_has_ns)].copy()
    if len(flagged) == 0:
        if verbose:
            print("No not_science_shape rows in corpus; nothing to review.")
        return None

    out_dir = out_dir or (processed_dir / "review_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = pd.DataFrame({
        "utt_id": flagged["utt_id"].values,
        "utterance": flagged["utterance"].values,
        "setting": flagged["setting"].values,
        "source": flagged["source"].values,
        "transcript_ref": flagged["transcript_ref"].values,
        "current_subtype": flagged["subtype"].apply(lambda s: ", ".join(list(s))).values,
        "why_flagged": "LLM tagged not_science_shape (looks non-science standalone)",
        "decision": pd.NA,        # keep | drop | edit
        "corrected_text": pd.NA,  # required only if decision == edit
        "notes": pd.NA,
    })

    xlsx_path = out_dir / NS_REVIEW_FILENAME
    sheet.to_excel(xlsx_path, index=False)
    sheet.to_parquet(xlsx_path.with_suffix(".parquet"), index=False)
    if verbose:
        print(f"Wrote {xlsx_path} ({len(sheet)} rows for review)")
        print("  Fill `decision` with keep / drop / edit "
              "(corrected_text required for edit).")
    return xlsx_path


def validate_ns_decisions(df: pd.DataFrame) -> None:
    """Validate a filled not_science review sheet. Blank decision == keep."""
    if "decision" not in df.columns:
        raise ValueError("Template missing `decision` column.")
    dec = df["decision"].astype("string").str.strip().str.lower()
    bad = sorted(set(dec.dropna()) - set(NS_DECISIONS))
    if bad:
        raise ValueError(f"Invalid decision values {bad}; allowed: {NS_DECISIONS} (or blank=keep).")
    edits = dec == "edit"
    if edits.any():
        ct = df["corrected_text"].astype("string").str.strip()
        missing = edits & (ct.isna() | (ct == ""))
        if missing.any():
            ids = df.loc[missing, "utt_id"].tolist()
            raise ValueError(f"`edit` rows missing corrected_text: {ids}")


def apply_not_science_review(
    template_path: Path,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    prune_pairs: bool = True,
    verbose: bool = True,
) -> dict:
    """Apply a filled not_science review sheet to corpus.parquet (with backup).

    drop -> remove the row; edit -> replace utterance with corrected_text;
    keep/blank -> no change. Orphaned/stale variant pairs (for dropped or edited
    anchors) are pruned so the data stays consistent even before a full rebuild.
    """
    template_path = Path(template_path)
    if template_path.suffix == ".parquet":
        tmpl = pd.read_parquet(template_path)
    else:
        tmpl = pd.read_excel(template_path)
    validate_ns_decisions(tmpl)

    dec = tmpl["decision"].astype("string").str.strip().str.lower().fillna("keep")
    drop_ids = set(tmpl.loc[dec == "drop", "utt_id"])
    edit_map = {
        r["utt_id"]: str(r["corrected_text"]).strip()
        for _, r in tmpl[dec == "edit"].iterrows()
    }

    corpus_path = processed_dir / "corpus.parquet"
    corpus = pd.read_parquet(corpus_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    corpus.to_parquet(processed_dir / f"corpus.pre_nsreview.{ts}.parquet", index=False)

    n_before = len(corpus)
    corpus = corpus[~corpus["utt_id"].isin(drop_ids)].reset_index(drop=True)
    for uid, text in edit_map.items():
        corpus.loc[corpus["utt_id"] == uid, "utterance"] = text
    corpus.to_parquet(corpus_path, index=False)

    changed_ids = drop_ids | set(edit_map)
    pruned_pairs = 0
    pairs_path = processed_dir / "pairs.parquet"
    if prune_pairs and pairs_path.exists() and changed_ids:
        pairs = pd.read_parquet(pairs_path)
        keep = ~pairs["anchor_id"].isin(changed_ids)
        pruned_pairs = int((~keep).sum())
        if pruned_pairs:
            pairs.to_parquet(processed_dir / f"pairs.pre_nsreview.{ts}.parquet", index=False)
            pairs[keep].reset_index(drop=True).to_parquet(pairs_path, index=False)

    stats = {
        "reviewed": len(tmpl),
        "dropped": len(drop_ids),
        "edited": len(edit_map),
        "kept": len(tmpl) - len(drop_ids) - len(edit_map),
        "corpus_before": n_before,
        "corpus_after": len(corpus),
        "pruned_pairs": pruned_pairs,
    }
    if verbose:
        print(f"Applied not_science review: {stats}")
        if changed_ids:
            print("\nNOTE: corpus changed. Rebuild downstream so variants/embeddings/"
                  "splits reflect it:\n"
                  "  python src/pipeline.py --steps 2 4 5 6 7 --use-llm --use-llm-step2\n"
                  "  (or rerun this script with --rebuild)")
    return stats


def rebuild_downstream(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    verbose: bool = True,
) -> None:
    """Re-run Steps 2,4,5,6,7 with the real LLM after a corpus change."""
    from src import pipeline
    pipeline.run(
        out_dir=processed_dir,
        steps=(2, 4, 5, 6, 7),
        use_real_llm=True,
        use_real_llm_step2=True,
        verbose=verbose,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--xlsx", type=Path, default=DEFAULT_TRANSCRIPT_XLSX)
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    p.add_argument("--source-tag", type=str, default=MINED_SOURCE_TAG)
    p.add_argument("--use-llm-subtypes", action="store_true",
                   help="Use real Llama for subtype LLM fallback (default: stub).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report counts only; write nothing.")
    # not_science_shape review workflow
    p.add_argument("--export-not-science", action="store_true",
                   help="Export not_science_shape positives for human review (no merge).")
    p.add_argument("--apply-not-science", type=Path, default=None, metavar="TEMPLATE",
                   help="Apply a filled not_science review sheet to corpus.parquet.")
    p.add_argument("--rebuild", action="store_true",
                   help="After --apply-not-science, rerun Steps 2,4,5,6,7 (real LLM).")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    verbose = not args.quiet
    if args.export_not_science:
        export_not_science_review(args.processed_dir, verbose=verbose)
    elif args.apply_not_science is not None:
        apply_not_science_review(args.apply_not_science, args.processed_dir, verbose=verbose)
        if args.rebuild:
            rebuild_downstream(args.processed_dir, verbose=verbose)
    else:
        merge_into_corpus(
            processed_dir=args.processed_dir,
            xlsx_path=args.xlsx,
            min_words=args.min_words,
            min_chars=args.min_chars,
            source_tag=args.source_tag,
            use_llm_subtypes=args.use_llm_subtypes,
            dry_run=args.dry_run,
            verbose=verbose,
        )
