"""Read-only audit: how many SCI-coded teacher utterances live in the
hand-coded transcript workbook, and how many are NET-NEW vs. the current
corpus.parquet positives?

This is the *mirror image* of `negatives_3.mine_transcript_xlsx`: that function
keeps teacher rows whose content code is NOT 'SCI' (to build negatives); here we
keep teacher rows whose content code IS 'SCI' (candidate real positives). It
writes nothing -- it only counts and prints a report so we can decide whether
mining real positives is worth doing before Step 8.

Usage:
    python scripts/count_sci_positives.py
"""

from __future__ import annotations

import sys
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

SCIENCE_CODE_TOKEN = "SCI"


def _canon(text: object) -> str:
    """Canonical key for de-dup: NFKC + whitespace-collapse + casefold."""
    norm = normalize_text(text)
    if isinstance(norm, float):  # NaN sentinel from normalize_text
        return ""
    return norm.casefold()


def extract_sci_rows(xlsx_path: Path) -> pd.DataFrame:
    """Return all teacher rows whose content code (col G) contains 'SCI'."""
    xl = pd.ExcelFile(xlsx_path)
    data_sheets = [s for s in xl.sheet_names if s not in XLSX_META_SHEETS]
    rows: list[dict] = []
    per_sheet: list[tuple[str, int, int, int]] = []

    for sheet in data_sheets:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet, header=0, dtype=str)
        except Exception:
            continue
        n_raw = len(df)
        if df.shape[1] <= max(XLSX_SPEAKER_COL, XLSX_UTTERANCE_COL, XLSX_CONTENT_CODE_COL):
            per_sheet.append((sheet, n_raw, 0, 0))
            continue

        speakers = df.iloc[:, XLSX_SPEAKER_COL].fillna("").str.strip()
        utterances = df.iloc[:, XLSX_UTTERANCE_COL].fillna("").str.strip()
        content_codes = df.iloc[:, XLSX_CONTENT_CODE_COL].fillna("").str.strip()

        is_teacher = speakers.apply(lambda s: bool(XLSX_TEACHER_SPEAKER_RE.match(s)))
        is_sci = content_codes.str.contains(SCIENCE_CODE_TOKEN, case=False, na=False)
        has_text = utterances != ""

        n_sci_all = int((is_sci & has_text).sum())
        keep = is_teacher & is_sci & has_text
        n_kept = int(keep.sum())
        per_sheet.append((sheet, n_raw, n_sci_all, n_kept))

        for i in df.index[keep]:
            rows.append({
                "sheet": sheet,
                "excel_row": int(i) + 2,  # header on row 0 -> first data row is excel row 2
                "speaker": speakers.iloc[i],
                "utterance": utterances.iloc[i],
                "content_code": content_codes.iloc[i],
            })

    out = pd.DataFrame(rows)
    out.attrs["per_sheet"] = per_sheet
    return out


def main() -> None:
    xlsx_path = DEFAULT_TRANSCRIPT_XLSX
    corpus_path = DEFAULT_PROCESSED_DIR / "corpus.parquet"
    print(f"Transcript workbook: {xlsx_path}")
    print(f"Corpus parquet:      {corpus_path}\n")

    if not xlsx_path.exists():
        print("ERROR: transcript workbook not found.")
        return
    if not corpus_path.exists():
        print("ERROR: corpus.parquet not found.")
        return

    sci = extract_sci_rows(xlsx_path)

    print("Per-sheet scan (sheet | rows | SCI-coded (any speaker) | SCI & teacher):")
    for sheet, n_raw, n_sci_all, n_kept in sci.attrs["per_sheet"]:
        print(f"  {sheet:<28} {n_raw:>5} {n_sci_all:>10} {n_kept:>10}")

    if len(sci) == 0:
        print("\nNo SCI-coded teacher rows found.")
        return

    # De-dup within the extracted SCI rows.
    sci["canon"] = sci["utterance"].map(_canon)
    sci_nonempty = sci[sci["canon"] != ""]
    unique_sci = sci_nonempty.drop_duplicates(subset=["canon"])

    # Compare against current corpus positives.
    corpus = pd.read_parquet(corpus_path)
    corpus_keys = set(corpus["utterance"].map(_canon)) - {""}

    unique_keys = set(unique_sci["canon"])
    overlap = unique_keys & corpus_keys
    net_new = unique_keys - corpus_keys
    net_new_rows = unique_sci[unique_sci["canon"].isin(net_new)]

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"SCI-coded teacher rows (raw):          {len(sci):>6}")
    print(f"  ... non-empty text:                  {len(sci_nonempty):>6}")
    print(f"  ... unique utterances (deduped):     {len(unique_sci):>6}")
    print(f"Current corpus positives:              {len(corpus):>6}")
    print(f"  ... unique corpus utterances:        {len(corpus_keys):>6}")
    print("-" * 64)
    print(f"Already in corpus (overlap):           {len(overlap):>6}")
    print(f"NET-NEW real positives available:      {len(net_new):>6}")
    print("=" * 64)

    if len(net_new_rows) > 0:
        print("\nSample net-new SCI utterances (up to 15):")
        for _, r in net_new_rows.head(15).iterrows():
            txt = r["utterance"]
            txt = txt if len(txt) <= 100 else txt[:97] + "..."
            print(f"  [{r['sheet']}!R{r['excel_row']}] ({r['content_code']}) {txt}")


if __name__ == "__main__":
    main()
