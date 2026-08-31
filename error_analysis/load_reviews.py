"""Load reviewed SCIENCE_TALK flags from Human coded/*.xlsx."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HUMAN = ROOT / "Human coded"

SKIP_SHEETS = {"INSTRUCTIONS", "SUMMARY"}


def _data_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "line_no" not in df.columns:
        return df.iloc[0:0]
    keep = pd.to_numeric(df["line_no"], errors="coerce").notna()
    return df.loc[keep].copy()


def find_reviewed_sheet(path: Path) -> tuple[str, pd.DataFrame] | None:
    xl = pd.ExcelFile(path, engine="openpyxl")
    candidates: list[tuple[str, pd.DataFrame, int]] = []
    for sheet in xl.sheet_names:
        if sheet.upper() in SKIP_SHEETS:
            continue
        df = _data_rows(pd.read_excel(path, sheet_name=sheet, engine="openpyxl"))
        if df.empty or "review" not in df.columns:
            continue
        n_rev = int(df["review"].notna().sum())
        if n_rev > 0:
            candidates.append((sheet, df, n_rev))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[2])
    return candidates[0][0], candidates[0][1]


def norm_review(val) -> str:
    if pd.isna(val):
        return "unreviewed"
    s = str(val).strip().upper().replace(" ", "_")
    if s in {"SCIENCE_TALK", "SCIENCE", "YES", "Y", "TRUE", "1"}:
        return "verified"
    if s in {"NOT_SCIENCE_TALK", "NOT_SCIENCE", "NO", "N", "FALSE", "0"}:
        return "false_positive"
    return f"other:{s}"


def load_all_flagged() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(HUMAN.glob("Classroom*_ebh.xlsx")):
        found = find_reviewed_sheet(path)
        if found is None:
            continue
        sheet, df = found
        df = df.copy()
        base = path.name.replace("_ebh.xlsx", "")
        m = re.match(r"Classroom(\w+?)_(\d{6})", base)
        df["classroom"] = m.group(1) if m else "?"
        df["obs_date"] = m.group(2) if m else "?"
        df["workbook"] = path.name
        df["sheet"] = sheet
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No reviewed workbooks in {HUMAN}")
    out = pd.concat(frames, ignore_index=True)
    out["review_norm"] = out["review"].map(norm_review)
    out["speaker_norm"] = (
        out.get("speaker", pd.Series(index=out.index))
        .astype(str).str.strip().str.lower()
    )
    out["speaker_norm"] = out["speaker_norm"].where(
        out["speaker_norm"].isin(["adult", "child"]), "other"
    )
    out["subtype_norm"] = (
        out.get("predicted_subtype", pd.Series(index=out.index))
        .astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})
    )
    out["utt"] = out["utterance"].astype(str).str.strip()
    out["n_words"] = out["utt"].str.split().str.len()
    out["score_num"] = pd.to_numeric(out.get("score"), errors="coerce")
    out["cosine_num"] = pd.to_numeric(out.get("cosine_score"), errors="coerce")
    out["llm_num"] = pd.to_numeric(out.get("llm_score"), errors="coerce")
    out["scored_by_norm"] = out.get("scored_by", pd.Series(index=out.index)).astype(str).str.lower()
    return out


def reviewed_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["review_norm"].isin(["verified", "false_positive"])].copy()
