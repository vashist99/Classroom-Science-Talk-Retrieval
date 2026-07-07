"""Assemble ONE consistent review package for Dr. Hadley.

Produces four standardized reviewer workbooks under data/processed/review_samples/,
each with the same layout:
  * Sheet 0 = "Review"        -> the rows to review (reviewer-fill columns up front,
                                 model/internal columns pushed to the end).
  * Sheet 1 = "INSTRUCTIONS"  -> how to fill this specific sheet.

Sheet 0 stays the data sheet so the existing apply/scoring paths keep working:
  - review_6.apply_review_decisions  (pairs)       reads sheet 0 by column name
  - confidence_5.load_routing_audit  (routing)     reads sheet 0 by column name
  - positives_mining.apply_not_science_review      reads sheet 0 by column name

Also (re)exports the negatives-quality audit sheet (the open Step 3 DoD) and
writes MANIFEST.txt listing the package.

Usage:
    python scripts/build_review_bundle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR
from src.negatives_3 import export_negative_review_template

REVIEW_DIR = DEFAULT_PROCESSED_DIR / "review_samples"

COMMON_FOOTER = [
    "",
    "GROUND RULES (all sheets):",
    "  - Work only on the 'Review' tab.",
    "  - Fill ONLY the columns named below; leave the rest as-is.",
    "  - Do NOT add, delete, reorder, or sort rows.",
    "  - Do NOT change the ID column (that is how we match your answers back).",
    "  - Leaving a row blank = skip it (counts as the default noted below).",
    "  - Save in .xlsx format and send the file back.",
]


def _order_columns(df: pd.DataFrame, id_col: str, content: list[str],
                   fill: list[str], internal: list[str]) -> pd.DataFrame:
    """Reorder: ID, content, reviewer-fill, then model/internal columns."""
    def keep(cols):
        return [c for c in cols if c in df.columns]
    ordered = [id_col] + keep(content) + keep(fill) + keep(internal)
    # append anything not explicitly listed (defensive) at the very end
    leftover = [c for c in df.columns if c not in ordered]
    return df[ordered + leftover]


def _write_workbook(df: pd.DataFrame, path: Path, instructions: list[str]) -> None:
    instr = pd.DataFrame({"INSTRUCTIONS": instructions + COMMON_FOOTER})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Review", index=False)
        instr.to_excel(xw, sheet_name="INSTRUCTIONS", index=False)
    print(f"  standardized {path.name}: {len(df)} rows, "
          f"fill cols grouped, INSTRUCTIONS tab added")


def standardize_not_science(review_dir: Path) -> int:
    path = review_dir / "not_science_review.xlsx"
    if not path.exists():
        print("  (skip) not_science_review.xlsx missing")
        return 0
    df = pd.read_excel(path)
    df = _order_columns(
        df, "utt_id",
        content=["utterance", "setting", "transcript_ref", "current_subtype", "why_flagged"],
        fill=["decision", "corrected_text", "notes"],
        internal=["source"],
    )
    _write_workbook(df, path, [
        "TASK 1 - Real utterances flagged as possibly NOT science talk",
        "",
        "These lines were coded SCI in a lesson, but look non-science on their own.",
        "The model reads each line ALONE, so context-only lines are a problem.",
        "",
        "Fill the 'decision' column with one of:",
        "  keep  - it really is science talk on its own",
        "  drop  - only makes sense in context; remove it",
        "  edit  - reword it to stand alone, and put the new wording in 'corrected_text'",
        "Use 'notes' for any comment. Blank = keep.",
    ])
    return len(df)


def standardize_pairs(review_dir: Path) -> int:
    path = review_dir / "review_queue.xlsx"
    if not path.exists():
        print("  (skip) review_queue.xlsx missing")
        return 0
    df = pd.read_excel(path)
    df = _order_columns(
        df, "id",
        content=["anchor_text", "candidate_text", "register"],
        fill=["decision", "reject_reason", "corrected_text", "reject_notes"],
        internal=["kind", "source_type", "confidence", "routing", "reviewer", "decided_at"],
    )
    _write_workbook(df, path, [
        "TASK 2 - AI-generated paraphrases of real teacher utterances",
        "",
        "'anchor_text' is the original; 'candidate_text' is the AI rewrite into the",
        "register shown. Confirm the rewrite keeps the meaning and sounds real.",
        "",
        "Fill the 'decision' column with one of:",
        "  accept - faithful and natural for that register",
        "  reject - something is wrong; put a code in 'reject_reason'",
        "  edit   - close; put your fix in 'corrected_text'",
        "",
        "reject_reason codes: register_mismatch, meaning_drift, anchor_copy,",
        "  implausible_classroom, nonsense, duplicate, pii, other",
        "Use 'reject_notes' for detail. Blank = skip (no decision).",
        "Ignore the 'confidence'/'routing' columns (model internals).",
    ])
    return len(df)


def standardize_routing(review_dir: Path) -> int:
    path = review_dir / "routing_audit_template.xlsx"
    if not path.exists():
        print("  (skip) routing_audit_template.xlsx missing")
        return 0
    df = pd.read_excel(path)
    df = _order_columns(
        df, "id",
        content=["kind", "anchor_text", "candidate_text", "register"],
        fill=["human_routing", "notes"],
        internal=["routing", "confidence", "confidence_llm", "confidence_cosine",
                  "confidence_structural", "agree"],
    )
    _write_workbook(df, path, [
        "TASK 3 - Spot-check of the auto-sorting (routing) decisions",
        "",
        "The system sorts each item into one of three piles. Tell us which pile",
        "YOU think it belongs in, based on your own judgment.",
        "",
        "Fill the 'human_routing' column with one of:",
        "  auto   - clearly good; safe to use without review",
        "  spot   - probably fine; worth a quick glance",
        "  review - questionable; a person should read it",
        "Use 'notes' for comments. Blank = skip.",
        "",
        "IMPORTANT: please decide BEFORE looking at the 'routing' column at the far",
        "right (that is the model's own answer; we compare yours to it afterwards).",
    ])
    return len(df)


def export_and_standardize_negatives(review_dir: Path) -> int:
    df_neg = pd.read_parquet(DEFAULT_PROCESSED_DIR / "negatives.parquet")
    export_negative_review_template(df_neg, DEFAULT_PROCESSED_DIR, n=50, verbose=False)
    pq = review_dir / "negatives_review_template.parquet"
    sample = pd.read_parquet(pq)

    cols = [c for c in ["neg_id", "text", "source_type", "anchor_seed_term",
                        "transcript_ref"] if c in sample.columns]
    view = sample[cols].copy()
    view["review_human_label"] = pd.NA
    view["review_notes"] = pd.NA

    path = review_dir / "negatives_review.xlsx"
    _write_workbook(view, path, [
        "TASK 4 - Quality check on the 'NOT science talk' examples",
        "",
        "These were auto-collected as NON-science examples for the model. We need to",
        "confirm they really are not science (a stray science line here would teach",
        "the model the wrong thing).",
        "",
        "Fill the 'review_human_label' column with one of:",
        "  NOT_SCIENCE_TALK - correct, this is not science",
        "  POSSIBLE_LEAK    - this actually IS (or might be) science talk",
        "  UNCLEAR          - cannot tell from the text",
        "Use 'review_notes' for comments. Blank = skip.",
    ])
    return len(view)


def write_manifest(review_dir: Path, counts: dict) -> None:
    lines = [
        "REVIEW PACKAGE FOR DR. HADLEY",
        "=" * 40,
        "",
        "Open each .xlsx, go to the 'Review' tab, and follow its 'INSTRUCTIONS' tab.",
        "See email_to_dr_hadley.txt for the overview.",
        "",
        "FILES (please return all four):",
        f"  1. not_science_review.xlsx   ({counts.get('not_science', 0)} rows) - keep/drop/edit real utterances",
        f"  2. review_queue.xlsx         ({counts.get('pairs', 0)} rows) - accept/reject/edit AI paraphrases",
        f"  3. routing_audit_template.xlsx ({counts.get('routing', 0)} rows) - spot-check auto-sorting",
        f"  4. negatives_review.xlsx     ({counts.get('negatives', 0)} rows) - confirm non-science examples",
        "",
        "Each workbook: 'Review' tab = your work; 'INSTRUCTIONS' tab = how to fill it.",
        "Do not reorder/delete rows or change ID columns.",
        "",
        "DO NOT send back: review_queue_corrected.xlsx, routing_audit_scored.xlsx,",
        "  routing_audit_disagreements.xlsx (these are from a previous round).",
    ]
    (review_dir / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote MANIFEST.txt")


def main() -> None:
    print(f"Building review bundle in {REVIEW_DIR}")
    counts = {}
    print("\n[1/5] negatives-quality audit (export + standardize)")
    counts["negatives"] = export_and_standardize_negatives(REVIEW_DIR)
    print("\n[2/5] not_science_review")
    counts["not_science"] = standardize_not_science(REVIEW_DIR)
    print("\n[3/5] review_queue (pairs)")
    counts["pairs"] = standardize_pairs(REVIEW_DIR)
    print("\n[4/5] routing_audit_template")
    counts["routing"] = standardize_routing(REVIEW_DIR)
    print("\n[5/5] manifest")
    write_manifest(REVIEW_DIR, counts)
    print(f"\nDone. Package counts: {counts}")


if __name__ == "__main__":
    main()
