"""Step 1 — Data ingestion and normalization.

Loads the labeled-utterance xlsx, normalizes columns to a stable analytical
schema, and writes three parquet artifacts under data/processed/:

    corpus.parquet         — analytical rows + light audit columns
    seed_words.parquet     — seed terms with variants parsed to lists
    category_defs.parquet  — category definitions, columns snake_cased

DoD addressed by this module (assert in `validate_corpus`):
    * row count and label distribution match the xlsx (logged, not silently fixed)
    * no duplicate utterances; whitespace + unicode normalized; cues are list-typed
    * a small unit test (see tests/test_data_loader_1.py) loads the parquet
      and asserts the schema
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = PROJECT_ROOT / "data" / "2026-04-17_science_talk_examples.xlsx"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SETTING_VOCAB = {
    "Large Group", "Small Group", "Centers", "Transition",
    "Conversation", "Blocks/Engineering", "Unknown",
}

VALID_LABELS = {"SCIENCE_TALK", "NOT_SCIENCE_TALK"}

TRANSCRIPT_REF_RE = re.compile(r"\(([^,)]+#\d+)")


def normalize_text(s) -> str | float:
    """NFKC-normalize, collapse internal whitespace, and strip. Returns NaN unchanged."""
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", s).strip()


def parse_cue_list(s) -> list[str]:
    """Split a cue cell on `|`, `,`, or `;`. NaN/empty becomes []."""
    if pd.isna(s) or str(s).strip() == "":
        return []
    parts = re.split(r"[|,;]", str(s))
    return [p.strip() for p in parts if p.strip()]


def normalize_setting(s) -> tuple[str, str | None]:
    """Strip topic suffix (' / Foo'), map NaN/unknown to 'Unknown'.

    Returns `(setting, topic)` where topic is None unless the source value
    encoded one (e.g. 'Large Group / Animal Observation' -> ('Large Group', 'Animal Observation')).
    """
    if pd.isna(s) or str(s).strip() == "":
        return ("Unknown", None)
    parts = [p.strip() for p in str(s).split(" / ")]
    setting = parts[0]
    topic = " / ".join(parts[1:]) if len(parts) > 1 else None
    if setting not in SETTING_VOCAB:
        setting = "Unknown"
    return (setting, topic)


def extract_transcript_ref(notes) -> str | None:
    """Pull the `01-2_19LG #61`-style id out of free-text notes."""
    if pd.isna(notes):
        return None
    m = TRANSCRIPT_REF_RE.search(str(notes))
    return m.group(1).strip() if m else None


def add_was_sci_coded(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean-int column: was the original utterance pre-coded as scientific by another project?"""
    df = df.copy()
    df["was_sci_coded"] = (
        df["Notes"].astype("string").str.contains("SCI-coded", case=False, na=False).astype(int)
    )
    return df


def clean_corpus(ex_utter_raw: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Apply Step 1 normalization to the raw `Example utterances` sheet.

    Returns a tidy dataframe with the analytical schema:
      utt_id, utterance, label, setting, source, tier2_cues, tier3_cues,
      was_sci_coded, transcript_ref, topic, citation
    """
    df = ex_utter_raw.copy()

    if verbose:
        print(f"Raw row count: {len(df)}")
        print(f"Raw label distribution: {dict(Counter(df['Label'].fillna('__NULL__')))}")
        print(f"Raw setting distribution: {dict(Counter(df['Setting'].fillna('__NULL__')))}")

    df = add_was_sci_coded(df)
    df["Utterance"] = df["Utterance"].apply(normalize_text)
    df = df.dropna(subset=["Utterance"]).reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=["Utterance"]).reset_index(drop=True)
    if verbose:
        print(f"De-dup: {before} -> {len(df)} (dropped {before - len(df)})")

    df["Tier2 cues"] = df["Tier2 cues"].apply(parse_cue_list)
    df["Tier3 cues"] = df["Tier3 cues"].apply(parse_cue_list)

    normalized = df["Setting"].apply(normalize_setting)
    df["Setting"] = normalized.apply(lambda x: x[0])
    df["topic"] = normalized.apply(lambda x: x[1])

    df["transcript_ref"] = df["Notes"].apply(extract_transcript_ref)

    df["utt_id"] = [f"utt_{i:04d}" for i in range(len(df))]

    corpus = df.rename(columns={
        "Utterance": "utterance",
        "Label": "label",
        "Setting": "setting",
        "Source": "source",
        "Tier2 cues": "tier2_cues",
        "Tier3 cues": "tier3_cues",
        "Article citation": "citation",
    })[[
        "utt_id", "utterance", "label", "setting", "source",
        "tier2_cues", "tier3_cues", "was_sci_coded",
        "transcript_ref", "topic", "citation",
    ]]

    if verbose:
        print(f"Final label distribution: {dict(Counter(corpus['label'].fillna('__NULL__')))}")
        print(f"Normalized settings: {dict(Counter(corpus['setting']))}")

    return corpus


def clean_seed_words(seed_raw: pd.DataFrame) -> pd.DataFrame:
    df = seed_raw.rename(columns={
        "Term": "term", "Tier": "tier", "Category": "category",
        "Variants (optional)": "variants",
        "Curriculum_source": "curriculum_source", "Source_URL": "source_url",
    })
    df["variants"] = df["variants"].apply(parse_cue_list)
    df["term"] = df["term"].apply(normalize_text)
    return df


def clean_category_defs(cat_raw: pd.DataFrame) -> pd.DataFrame:
    return cat_raw.rename(columns={
        "Label": "label", "Type": "type",
        "Definition (operational)": "definition",
        "Include if…": "include_if",
        "Exclude if…": "exclude_if",
        "Examples (teacher wording)": "examples",
    })


def validate_corpus(corpus: pd.DataFrame) -> None:
    """Hard assertions enforcing Step 1 DoD on the cleaned corpus."""
    assert len(corpus) >= 195, f"Row count too low after dedup: {len(corpus)}"
    assert corpus["utt_id"].is_unique, "utt_id must be unique"
    assert corpus["utterance"].notna().all(), "Every row must have an utterance"

    label_counts = Counter(corpus["label"].fillna("__NULL__"))
    assert label_counts.get("SCIENCE_TALK", 0) >= 188, \
        f"SCIENCE_TALK count unexpectedly low: {label_counts}"
    assert label_counts.get("NOT_SCIENCE_TALK", 0) >= 5, \
        f"NOT_SCIENCE_TALK count unexpectedly low: {label_counts}"

    bad_settings = set(corpus["setting"].dropna()) - SETTING_VOCAB
    assert not bad_settings, f"Unexpected setting values: {bad_settings}"


def load_and_clean(
    xlsx_path: Path = DEFAULT_XLSX,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three sheets and return cleaned (corpus, seed_words, category_defs)."""
    ex_utter = pd.read_excel(xlsx_path, sheet_name="Example utterances")
    seed_words = pd.read_excel(xlsx_path, sheet_name="Seed words")
    cat_def = pd.read_excel(xlsx_path, sheet_name="Category definitions")

    corpus = clean_corpus(ex_utter, verbose=verbose)
    validate_corpus(corpus)
    seed = clean_seed_words(seed_words)
    cat = clean_category_defs(cat_def)
    return corpus, seed, cat


def write_processed(
    corpus: pd.DataFrame,
    seed: pd.DataFrame,
    cat: pd.DataFrame,
    out_dir: Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Path]:
    """Write the three parquets to `out_dir`. Returns the file map."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "corpus": out_dir / "corpus.parquet",
        "seed_words": out_dir / "seed_words.parquet",
        "category_defs": out_dir / "category_defs.parquet",
    }
    corpus.to_parquet(paths["corpus"], index=False)
    seed.to_parquet(paths["seed_words"], index=False)
    cat.to_parquet(paths["category_defs"], index=False)
    return paths


def run(
    xlsx_path: Path = DEFAULT_XLSX,
    out_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    verbose: bool = True,
) -> dict[str, Path]:
    """End-to-end Step 1: load -> clean -> validate -> write parquets."""
    corpus, seed, cat = load_and_clean(xlsx_path, verbose=verbose)
    paths = write_processed(corpus, seed, cat, out_dir)
    if verbose:
        for name, p in paths.items():
            print(f"  wrote {name}: {p} ({p.stat().st_size:,} bytes)")
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: load and normalize the labeled corpus.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX,
                        help=f"Path to the source xlsx (default: {DEFAULT_XLSX})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                        help=f"Output directory for parquets (default: {DEFAULT_PROCESSED_DIR})")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    args = parser.parse_args()

    run(args.xlsx, args.out_dir, verbose=not args.quiet)
