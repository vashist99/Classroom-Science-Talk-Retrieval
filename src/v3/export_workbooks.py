"""Exact 18-column `Human coded` Excel workbook exporter (fixes child_base collision bug!).

Schema guaranteed 100% identical to Human coded/*.xlsx so Dr. Hadley's review process
and Data Aggregation/aggregate_human_coded.py work out-of-the-box:
  ['line_no', 'seg_start', 'seg_end', 'speaker', 'child_id', 'utterance',
   'confidence', 'predicted_label', 'review', 'score', 'scored_by',
   'cosine_score', 'llm_score', 'top_match_in_corpus', 'top_match_score',
   'predicted_subtype', 'rationale', 'degraded']

Fixes vs legacy src/deploy_y2_12.py:
  1. Pre-assigns deterministic sheet names across ALL source files in a classroom
     folder BEFORE filtering to SCIENCE_TALK rows -> eliminates the `child_base`
     sheet-name shifting bug confirmed in Classroom46_111825.xlsx!
  2. Supports both `screening` (high-recall human-review) and `standalone`
     (calibrated max-F1 automated counts) operating thresholds cleanly.
  3. Includes exact dropdown validation on `review`, bottom agreement tally formulas,
     and an updated `INSTRUCTIONS` sheet.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

LABEL_SCIENCE = "SCIENCE_TALK"
LABEL_NOT = "NOT_SCIENCE_TALK"

HUMAN_CODED_COLS = [
    "line_no",
    "seg_start",
    "seg_end",
    "speaker",
    "child_id",
    "utterance",
    "confidence",
    "predicted_label",
    "review",
    "score",
    "scored_by",
    "cosine_score",
    "llm_score",
    "top_match_in_corpus",
    "top_match_score",
    "predicted_subtype",
    "rationale",
    "degraded",
]

_INVALID_SHEET = re.compile(r"[\[\]:*?/\\]")


def child_id_from_filename(name: str) -> str:
    stem = Path(name).name.split("_", 1)[0]
    parts = stem.split("-")
    return parts[3] if len(parts) >= 4 else "base"


def assign_deterministic_sheet_names(source_files: list[str]) -> dict[str, str]:
    """Pre-assign stable, unique <=31-char sheet names for ALL files BEFORE filtering.

    Prevents a second `child_base` file from stealing `child_base` when the first
    file has 0 predicted SCIENCE_TALK rows!
    """
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for fname in sorted(set(source_files)):
        cid = child_id_from_filename(fname)
        base = _INVALID_SHEET.sub("", f"child_{cid}")[:31] or "sheet"
        name = base
        i = 1
        while name in used:
            suffix = f"_{i}"
            name = base[: 31 - len(suffix)] + suffix
            i += 1
        used.add(name)
        mapping[fname] = name
    return mapping


def _add_review_tools(ws, *, n_rows: int) -> None:
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    if n_rows <= 0:
        return
    rev_i = HUMAN_CODED_COLS.index("review") + 1
    pred_i = HUMAN_CODED_COLS.index("predicted_label") + 1
    rev_c = get_column_letter(rev_i)
    pred_c = get_column_letter(pred_i)
    first, last = 2, n_rows + 1
    rev_rng = f"{rev_c}{first}:{rev_c}{last}"
    pred_rng = f"{pred_c}{first}:{pred_c}{last}"

    dv = DataValidation(
        type="list",
        formula1=f'"{LABEL_SCIENCE},{LABEL_NOT}"',
        allow_blank=True,
        showDropDown=False,
    )
    dv.prompt = "Enter your judgment for this utterance."
    dv.promptTitle = "Reviewer label"
    ws.add_data_validation(dv)
    dv.add(rev_rng)

    s = last + 2
    ws.cell(row=s, column=1, value="Reviewed (non-blank):")
    ws.cell(row=s, column=2, value=f'=COUNTIF({rev_rng},"<>")')
    ws.cell(row=s + 1, column=1, value="Agreements with model:")
    ws.cell(row=s + 1, column=2, value=f"=SUMPRODUCT(--({rev_rng}={pred_rng}))")
    ws.cell(row=s + 2, column=1, value="Agreement rate:")
    ws.cell(row=s + 2, column=2, value=f'=IFERROR(B{s + 1}/B{s},"")')


def _build_instructions(mode: str, threshold: float, engine: str, science_only: bool) -> list[list[str]]:
    lines = [
        ["Y2 science-talk scoring (TeLI v3 Context-Aware Pipeline) — review guide"],
        [""],
        [f"Engine: {engine} | Operating mode: {mode} (threshold={threshold:.3f})"],
        ["Each sheet is one child's point-of-view recorder for this classroom session."],
    ]
    if science_only:
        lines.append(["Sheets contain ONLY utterances predicted as SCIENCE_TALK at the configured operating mode."])
    else:
        lines.append(["Sheets contain all kept classroom utterances in chronological line_no order."])
    lines += [
        [""],
        ["Columns (exact Human coded 18-column schema):"],
        ["  predicted_label  : SCIENCE_TALK / NOT_SCIENCE_TALK from the calibrated v3 pipeline."],
        ["  review           : YOU fill this in -- pick SCIENCE_TALK or NOT_SCIENCE_TALK from the dropdown."],
        ["  score            : Calibrated probability (0-1) that this utterance is science talk in context."],
        ["  scored_by        : 'hybrid_v3' (Stage 1 + Stage 2 LLM) or 'local_v3' (Stage 1 standalone)."],
        ["  cosine_score     : Stage 1 local conversation-sequence probability (0-1)."],
        ["  llm_score        : Stage 2 context-reading LLM score (0-1, when evaluated by Stage 2)."],
        ["  predicted_subtype: Science-talk conceptual route/subtype (content/observation/prediction/causal_reasoning/evidence)."],
        ["  rationale        : Context-grounded justification under the Sept 2026 TeLI Coding Manual."],
        [""],
        ["Note: Sheet names are pre-assigned deterministically across all files in the classroom folder"],
        ["to guarantee exact (classroom + sheet + line_no) alignment with raw transcripts."],
    ]
    return lines


def write_human_coded_workbook(
    folder_name: str,
    records: list[dict[str, Any]],
    out_dir: Path,
    *,
    mode: str = "screening",
    threshold: float = 0.50,
    engine: str = "hybrid_v3",
    science_only: bool = True,
    all_source_files: list[str] | None = None,
) -> Path | None:
    """Write one classroom workbook strictly matching the 18-column `Human coded` schema."""
    all_files = all_source_files if all_source_files is not None else [str(r["source_file"]) for r in records]
    sheet_map = assign_deterministic_sheet_names(all_files)

    rows_to_write = [r for r in records if r.get("predicted_label") == LABEL_SCIENCE] if science_only else list(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{folder_name}.xlsx"

    if not rows_to_write:
        # Emit a clean INSTRUCTIONS status workbook rather than silently dropping the session!
        with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
            notes = _build_instructions(mode, threshold, engine, science_only) + [
                [""],
                [f"STATUS: 0 utterances exceeded the '{mode}' threshold ({threshold:.3f}) in {folder_name}."],
            ]
            pd.DataFrame(notes).to_excel(xl, sheet_name="INSTRUCTIONS", index=False, header=False)
        return out_path

    by_file: dict[str, list[dict[str, Any]]] = {}
    for r in rows_to_write:
        by_file.setdefault(str(r["source_file"]), []).append(r)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        for fname in sorted(by_file):
            rows = sorted(by_file[fname], key=lambda x: int(x["line_no"]))
            sn = sheet_map[fname]
            df = pd.DataFrame(rows)
            for col in HUMAN_CODED_COLS:
                if col not in df.columns:
                    df[col] = None
            df["review"] = None
            df[HUMAN_CODED_COLS].to_excel(xl, sheet_name=sn, index=False)
            _add_review_tools(xl.sheets[sn], n_rows=len(df))

        pd.DataFrame(_build_instructions(mode, threshold, engine, science_only)).to_excel(
            xl, sheet_name="INSTRUCTIONS", index=False, header=False
        )
    return out_path
