"""Unified, leak-free dataset builder for TeLI Science Talk v3.

Produces under data/v3/:
  - y2_utterances.parquet         : All non-empty turns from the 83 focal Y2
                                    transcripts (174,979 screened + context
                                    neighbors), aligned with Human coded labels,
                                    v1/v2 replay columns, pause gaps, Manual-v0.3
                                    features, FP taxonomy, and 5-fold Stratified
                                    Classroom GroupKFold assignments.
  - y1_coding_transcripts.parquet : All 40 valid sheets of Coding Transcripts.xlsx
                                    in chronological order (teacher + child turns)
                                    with clean unbiased recall mask excluding
                                    uncoded BK TEXT / OEPQ rows from negatives.
  - y1_seed.parquet               : Cleaned 196-row seed examples (191 SCI, 5 NOT).
  - dataset_summary.json          : Machine-readable audit & fold verification.
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from datetime import time as dtime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.codebook_v3 import extract_codebook_features

HUMAN_DIR = PROJECT_ROOT / "Human coded"
HUMAN_ZIP = PROJECT_ROOT / "Human coded.zip"
RAW_Y2_ROOT = PROJECT_ROOT / "data" / "transcripts" / "Y2_Technology Observation Audio_Transcripts"
V2_SCORES_DIR = PROJECT_ROOT / "reports" / "y2_scores_codebook_v2"
MAP_FILE = PROJECT_ROOT / "reports" / "human_coded_reviewed_sheet_to_transcript_map.txt"
FP400_CSV = PROJECT_ROOT / "error_analysis" / "output" / "fp_sample_400_coded.csv"
Y1_CODING_XLSX = PROJECT_ROOT / "data" / "transcripts" / "Coding Transcripts.xlsx"
Y1_SEED_XLSX = PROJECT_ROOT / "data" / "2026-04-17_science_talk_examples.xlsx"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "v3"

TRANSCRIPT_COL = "Transcription (Confidence)"
_CONF_RE = re.compile(r"\((\d+(?:\.\d+)?)\s*%\)\s*$")
_SKIP_SHEETS = {"INSTRUCTIONS", "SUMMARY"}
_Y1_SKIP_SHEETS = {"TOC", "Sheet1", "Sheet5", "Coding Scheme", "Coding Instructions", "Codebook", "Metadata"}
_Y1_TEACHER_RE = re.compile(
    r"^(?:T\d*\s*:?|Ms\.?\s*\w*:?|Mr\.?\s*\w*:?|Mrs\.?\s*\w*:?|Teacher|Adult|LT|TA)\b",
    re.I,
)


def parse_transcription(cell: Any) -> tuple[str, float | None]:
    s = "" if cell is None or (isinstance(cell, float) and np.isnan(cell)) else str(cell).strip()
    m = _CONF_RE.search(s)
    if m:
        return s[: m.start()].strip(), float(m.group(1)) / 100.0
    return s, None


def to_seconds(val: Any) -> float:
    if val is None:
        return np.nan
    if isinstance(val, dtime):
        return float(val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val) if np.isfinite(val) else np.nan
    s = str(val).strip()
    if not s or s.lower() in ("nan", "-", "none", "nat"):
        return np.nan
    parts = s.split(":")
    try:
        return float(sum(float(p) * (60.0 ** i) for i, p in enumerate(reversed(parts))))
    except ValueError:
        return np.nan


def norm_review(val: Any) -> str:
    if pd.isna(val):
        return "unreviewed"
    s = str(val).strip().upper().replace(" ", "_")
    if s in {"SCIENCE_TALK", "SCIENCE", "YES", "Y", "TRUE", "1"}:
        return "verified"
    if s in {"NOT_SCIENCE_TALK", "NOT_SCIENCE", "NO", "N", "FALSE", "0"}:
        return "false_positive"
    return f"other:{s}"


def _open_excel_safe(path: Path, zip_fallback: Path | None = None) -> pd.ExcelFile:
    """Open an Excel workbook; fall back to zip archive if locked by Excel."""
    try:
        return pd.ExcelFile(path, engine="openpyxl")
    except PermissionError:
        if zip_fallback is not None and zip_fallback.exists():
            with zipfile.ZipFile(zip_fallback, "r") as zf:
                matches = [n for n in zf.namelist() if Path(n).name == path.name]
                if matches:
                    data = zf.read(matches[0])
                    return pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
        raise


def _data_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "line_no" not in df.columns:
        return df.iloc[0:0]
    keep = pd.to_numeric(df["line_no"], errors="coerce").notna()
    return df.loc[keep].copy()


def _find_primary_reviewed_sheet(path: Path) -> tuple[str, pd.DataFrame] | None:
    xl = _open_excel_safe(path, HUMAN_ZIP)
    cands: list[tuple[str, pd.DataFrame, int]] = []
    for sheet in xl.sheet_names:
        if sheet.upper() in _SKIP_SHEETS:
            continue
        df = _data_rows(xl.parse(sheet))
        if df.empty or "review" not in df.columns:
            continue
        n_rev = int(df["review"].notna().sum())
        if n_rev > 0:
            cands.append((sheet, df, n_rev))
    if not cands:
        return None
    cands.sort(key=lambda t: -t[2])
    return cands[0][0], cands[0][1]


def _load_sheet_to_raw_map() -> dict[tuple[str, str], str]:
    raw_map: dict[tuple[str, str], str] = {}
    for line in MAP_FILE.read_text(encoding="utf-8").splitlines():
        if "|" in line and ".xlsx" in line and not line.startswith("Format:"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                raw_map[(parts[0], parts[1])] = parts[2]
    return raw_map


def assign_stratified_classroom_folds(df_screened: pd.DataFrame, n_folds: int = 5) -> dict[str, int]:
    """Stratified GroupKFold by classroom, balancing verified science-talk rate & size."""
    stats = []
    for room, g in df_screened.groupby("classroom"):
        n_tp = int((g["label"] == 1).sum())
        n_rev = int(g["label"].isin([0, 1]).sum())
        n_all = len(g)
        tp_rate = n_tp / max(n_all, 1)
        stats.append((str(room), tp_rate, n_tp, n_rev, n_all))
    # Sort descending by tp_rate (primary) then n_tp (tie-break)
    stats.sort(key=lambda x: (-x[1], -x[2], x[0]))
    # Snake assignment across folds balances both high/low rate and counts
    fold_map: dict[str, int] = {}
    for idx, (room, *_rest) in enumerate(stats):
        cycle = idx // n_folds
        pos = idx % n_folds
        fold = pos if (cycle % 2 == 0) else (n_folds - 1 - pos)
        fold_map[room] = fold
    return fold_map


def build_y2_dataset(conf_min: float = 0.40, verbose: bool = True) -> pd.DataFrame:
    """Align all 83 human-coded focal sheets to their full raw Y2 transcripts."""
    from error_analysis.codebook import assign_fp_label

    raw_map = _load_sheet_to_raw_map()
    fp400 = pd.read_csv(FP400_CSV) if FP400_CSV.exists() else pd.DataFrame()
    fp400_lookup: dict[tuple[str, str, str], str] = {}
    if not fp400.empty:
        for _, r in fp400.iterrows():
            fp400_lookup[(str(r["workbook"]), str(r["sheet"]), str(r["utt"]).strip())] = str(r["fp_label"])

    books = sorted(HUMAN_DIR.glob("Classroom*_ebh.xlsx"))
    if verbose:
        print(f"[dataset] Loading {len(books)} human-coded workbooks + raw Y2 transcripts...")

    all_frames: list[pd.DataFrame] = []
    missing_review_lines = 0
    for tx_idx, wb_path in enumerate(books):
        found = _find_primary_reviewed_sheet(wb_path)
        if found is None:
            continue
        sheet, rev_df = found
        wb_name = wb_path.name
        folder_name = wb_name.replace("_ebh.xlsx", "")
        m = re.match(r"Classroom(\w+?)_(\d{6})", folder_name)
        classroom = m.group(1) if m else "?"
        obs_date = m.group(2) if m else "?"
        child_id = sheet.replace("child_", "", 1)

        raw_fname = raw_map.get((wb_name, sheet))
        if not raw_fname:
            raise FileNotFoundError(f"No raw transcript map entry for ({wb_name}, {sheet})")
        raw_path = RAW_Y2_ROOT / folder_name / raw_fname
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw transcript: {raw_path}")

        # Parse full raw transcript (retain ALL non-empty turns for rich context)
        raw_df = pd.read_excel(raw_path, engine="openpyxl")
        rows: list[dict[str, Any]] = []
        for pos, row in enumerate(raw_df.itertuples(index=False)):
            rowd = dict(zip(raw_df.columns, row))
            raw_cell = rowd.get(TRANSCRIPT_COL)
            if raw_cell is None or str(raw_cell).strip() in ("-", "", "nan"):
                continue
            text, conf = parse_transcription(raw_cell)
            if not text or text == "-":
                continue
            spk_raw = str(rowd.get("Speaker", "")).strip().lower()
            speaker = spk_raw if spk_raw in ("adult", "child") else "unknown"
            screened = bool(speaker in ("adult", "child") and (conf is None or conf >= conf_min))
            start_s = to_seconds(rowd.get("Segment Start Time"))
            end_s = to_seconds(rowd.get("Segment End Time"))
            rows.append({
                "transcript_id": f"{folder_name}__{sheet}",
                "folder": folder_name,
                "workbook": wb_name,
                "sheet": sheet,
                "source_file": raw_fname,
                "classroom": classroom,
                "obs_date": obs_date,
                "child_id": child_id,
                "line_no": int(pos + 2),
                "seg_start": str(rowd.get("Segment Start Time", "")),
                "seg_end": str(rowd.get("Segment End Time", "")),
                "seg_start_s": start_s,
                "seg_end_s": end_s,
                "speaker": speaker,
                "utterance": text,
                "asr_conf": np.nan if conf is None else float(conf),
                "screened": screened,
            })

        tx_df = pd.DataFrame(rows)
        if tx_df.empty:
            continue
        tx_df = tx_df.sort_values("line_no").reset_index(drop=True)
        tx_df["turn_idx"] = np.arange(len(tx_df), dtype=np.int32)

        # Compute elapsed silence gaps (in seconds) across ALL non-empty turns
        starts = tx_df["seg_start_s"].to_numpy(dtype=float)
        ends = tx_df["seg_end_s"].to_numpy(dtype=float)
        gap_prev = np.full(len(tx_df), np.nan, dtype=float)
        gap_next = np.full(len(tx_df), np.nan, dtype=float)
        if len(tx_df) > 1:
            gap_prev[1:] = np.maximum(0.0, starts[1:] - ends[:-1])
            gap_next[:-1] = np.maximum(0.0, starts[1:] - ends[:-1])
        tx_df["gap_prev_s"] = gap_prev
        tx_df["gap_next_s"] = gap_next

        # Join Human coded v1 flags & review verdicts by line_no
        rev_df = rev_df.copy()
        rev_df["line_no_int"] = pd.to_numeric(rev_df["line_no"], errors="coerce").astype(int)
        rev_df["review_norm"] = rev_df["review"].map(norm_review)
        rev_lookup = {}
        for _, rr in rev_df.iterrows():
            lno = int(rr["line_no_int"])
            rev_lookup[lno] = {
                "v1_flagged": True,
                "review_norm": rr["review_norm"],
                "v1_scored_by": str(rr.get("scored_by", "")).strip().lower(),
                "v1_score": pd.to_numeric(rr.get("score"), errors="coerce"),
                "v1_cosine": pd.to_numeric(rr.get("cosine_score"), errors="coerce"),
                "v1_llm_score": pd.to_numeric(rr.get("llm_score"), errors="coerce"),
                "v1_subtype": str(rr.get("predicted_subtype", "")).strip().lower(),
                "v1_rationale": str(rr.get("rationale", "")).strip(),
            }

        # Join v2 codebook deploy flags by line_no
        v2_lookup: dict[int, float] = {}
        v2_path = V2_SCORES_DIR / f"{folder_name}.xlsx"
        if v2_path.exists():
            try:
                xl2 = pd.ExcelFile(v2_path, engine="openpyxl")
                if sheet in xl2.sheet_names:
                    v2_df = _data_rows(xl2.parse(sheet))
                    for _, r2 in v2_df.iterrows():
                        lno2 = int(pd.to_numeric(r2["line_no"]))
                        sc2 = pd.to_numeric(r2.get("score"), errors="coerce")
                        v2_lookup[lno2] = float(sc2) if pd.notna(sc2) else 0.9
            except Exception:
                pass

        matched_lines = set()
        v1_flagged, review_norms, v1_scored_by = [], [], []
        v1_score, v1_cosine, v1_llm, v1_sub, v1_rat = [], [], [], [], []
        v2_flagged, v2_score = [], []
        labels, fp_rule, fp_hand = [], [], []

        for lno, utt in zip(tx_df["line_no"].tolist(), tx_df["utterance"].tolist()):
            info = rev_lookup.get(lno)
            if info is not None:
                matched_lines.add(lno)
                v1_flagged.append(True)
                rn = info["review_norm"]
                review_norms.append(rn)
                v1_scored_by.append(info["v1_scored_by"])
                v1_score.append(info["v1_score"])
                v1_cosine.append(info["v1_cosine"])
                v1_llm.append(info["v1_llm_score"])
                v1_sub.append(info["v1_subtype"])
                v1_rat.append(info["v1_rationale"])
                if rn == "verified":
                    labels.append(1)
                    fp_rule.append("")
                    fp_hand.append("")
                elif rn == "false_positive":
                    labels.append(0)
                    fp_rule.append(assign_fp_label(utt))
                    fp_hand.append(fp400_lookup.get((wb_name, sheet, utt.strip()), ""))
                else:
                    labels.append(-2)  # flagged but unreviewed (35 rows)
                    fp_rule.append("")
                    fp_hand.append("")
            else:
                v1_flagged.append(False)
                review_norms.append("unflagged")
                v1_scored_by.append("unflagged")
                v1_score.append(np.nan)
                v1_cosine.append(np.nan)
                v1_llm.append(np.nan)
                v1_sub.append("")
                v1_rat.append("")
                labels.append(-1)  # unflagged U_0
                fp_rule.append("")
                fp_hand.append("")

            if lno in v2_lookup:
                v2_flagged.append(True)
                v2_score.append(v2_lookup[lno])
            else:
                v2_flagged.append(False)
                v2_score.append(np.nan)

        missing_review_lines += len(set(rev_lookup) - matched_lines)

        tx_df["v1_flagged"] = v1_flagged
        tx_df["review_norm"] = review_norms
        tx_df["v1_scored_by"] = v1_scored_by
        tx_df["v1_score"] = v1_score
        tx_df["v1_cosine"] = v1_cosine
        tx_df["v1_llm_score"] = v1_llm
        tx_df["v1_subtype"] = v1_sub
        tx_df["v1_rationale"] = v1_rat
        tx_df["v2_flagged"] = v2_flagged
        tx_df["v2_score"] = v2_score
        tx_df["label"] = np.array(labels, dtype=np.int8)
        tx_df["fp_label_rule"] = fp_rule
        tx_df["fp_label_hand"] = fp_hand

        all_frames.append(tx_df)

    df = pd.concat(all_frames, ignore_index=True)

    # Extract Manual-v0.3 lexical/codebook features vectorized via list comprehension
    feat_dicts = [
        extract_codebook_features(u, spk)
        for u, spk in zip(df["utterance"].tolist(), df["speaker"].tolist())
    ]
    feat_df = pd.DataFrame(feat_dicts, index=df.index).astype(np.float32)
    for col in feat_df.columns:
        df[f"cb_{col}"] = feat_df[col]

    # Assign Stratified 5-Fold GroupKFold by Classroom
    fold_map = assign_stratified_classroom_folds(df[df["screened"]], n_folds=5)
    df["fold"] = df["classroom"].map(fold_map).astype(np.int8)

    # Assertions
    scr = df[df["screened"]]
    n_tp = int((scr["label"] == 1).sum())
    n_fp = int((scr["label"] == 0).sum())
    assert missing_review_lines == 0, f"Unaligned review lines: {missing_review_lines}"
    assert n_tp == 1984, f"Expected 1984 verified TPs, got {n_tp}"
    assert n_fp == 5985, f"Expected 5985 verified FPs, got {n_fp}"
    # Verify zero classroom leakage across folds
    for f in range(5):
        val_rooms = set(df.loc[df["fold"] == f, "classroom"].unique())
        tr_rooms = set(df.loc[df["fold"] != f, "classroom"].unique())
        assert val_rooms.isdisjoint(tr_rooms), f"Classroom leakage in fold {f}"

    if verbose:
        print(
            f"[dataset] Y2 built: {len(df):,} total non-empty rows | "
            f"{len(scr):,} screened | TP={n_tp}, FP={n_fp}, U0={(scr['label']==-1).sum():,} | "
            f"classrooms={df['classroom'].nunique()} across 5 folds"
        )
    return df


def build_y1_coding_dataset(verbose: bool = True) -> pd.DataFrame:
    """Load all 40 valid hand-coded sheets from Coding Transcripts.xlsx in turn order.

    Sets `eval_clean_mask = True` ONLY for teacher turns whose Col F is NOT an
    uncoded category (`BK TEXT`, `OEPQ`, `RESPONSE_OEPQ`), OR any SCI-coded row.
    Child turns remain in the dataframe as context neighbors (`eval_clean_mask=False`).
    """
    import openpyxl

    wb = openpyxl.load_workbook(Y1_CODING_XLSX, read_only=True, data_only=True)
    rows: list[dict[str, Any]] = []
    sheet_count = 0
    for sname in wb.sheetnames:
        if sname in _Y1_SKIP_SHEETS:
            continue
        ws = wb[sname]
        it = ws.iter_rows(values_only=True)
        hdr = next(it, None)
        if not hdr or len(hdr) < 7:
            continue
        sheet_count += 1
        turn_idx = 0
        for r_idx, row in enumerate(it, start=2):
            if len(row) < 7:
                continue
            spk_raw = str(row[2] or "").strip()
            utt = re.sub(r"\s+", " ", str(row[3] or "")).strip()
            if not utt or utt == "-":
                continue
            col_f = str(row[5] or "").strip()
            col_g = str(row[6] or "").strip()
            is_teacher = bool(_Y1_TEACHER_RE.match(spk_raw))
            speaker = "adult" if is_teacher else "child"
            is_sci = "sci" in col_g.lower()
            col_f_up = col_f.upper()
            uncoded_cat = col_f_up in {"BK TEXT", "OEPQ", "RESPONSE_OEPQ"}
            eval_clean = bool(is_sci or (is_teacher and not uncoded_cat))
            rows.append({
                "transcript_id": f"y1_{sname}",
                "sheet": sname,
                "line_no": int(r_idx),
                "turn_idx": int(turn_idx),
                "speaker_raw": spk_raw,
                "speaker": speaker,
                "utterance": utt,
                "col_f": col_f,
                "col_g": col_g,
                "label": 1 if is_sci else 0,
                "eval_clean_mask": eval_clean,
                "gap_prev_s": 2.0 if turn_idx > 0 else np.nan,
                "gap_next_s": 2.0,
                "asr_conf": 1.0,
            })
            turn_idx += 1
    wb.close()

    df = pd.DataFrame(rows)
    # Last turn of each transcript has NaN gap_next_s
    last_mask = df.groupby("transcript_id")["turn_idx"].transform("max") == df["turn_idx"]
    df.loc[last_mask, "gap_next_s"] = np.nan

    feat_dicts = [
        extract_codebook_features(u, spk)
        for u, spk in zip(df["utterance"].tolist(), df["speaker"].tolist())
    ]
    feat_df = pd.DataFrame(feat_dicts, index=df.index).astype(np.float32)
    for col in feat_df.columns:
        df[f"cb_{col}"] = feat_df[col]

    if verbose:
        clean = df[df["eval_clean_mask"]]
        print(
            f"[dataset] Y1 Coding Transcripts built: {sheet_count} sheets, "
            f"{len(df):,} total turns | clean eval={len(clean):,} "
            f"(SCI={(clean['label']==1).sum()}, clean non-SCI={(clean['label']==0).sum()})"
        )
    return df


def build_y1_seed_dataset(verbose: bool = True) -> pd.DataFrame:
    df = pd.read_excel(Y1_SEED_XLSX, sheet_name="Example utterances", engine="openpyxl")
    df["utterance"] = df["Utterance"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.dropna(subset=["utterance"]).drop_duplicates(subset=["utterance"]).reset_index(drop=True)
    df["label"] = (df["Label"].astype(str).str.strip().str.upper() == "SCIENCE_TALK").astype(np.int8)
    out = df[["utterance", "label", "Setting", "Source"]].rename(
        columns={"Setting": "setting", "Source": "source"}
    )
    if verbose:
        print(
            f"[dataset] Y1 Seed built: {len(out)} rows "
            f"(SCI={(out['label']==1).sum()}, NOT={(out['label']==0).sum()})"
        )
    return out


def build_all(out_dir: Path = DEFAULT_OUT_DIR, verbose: bool = True) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y2 = build_y2_dataset(verbose=verbose)
    y1 = build_y1_coding_dataset(verbose=verbose)
    seed = build_y1_seed_dataset(verbose=verbose)

    y2.to_parquet(out_dir / "y2_utterances.parquet", index=False)
    y1.to_parquet(out_dir / "y1_coding_transcripts.parquet", index=False)
    seed.to_parquet(out_dir / "y1_seed.parquet", index=False)

    scr = y2[y2["screened"]]
    folds_info = {}
    for f in range(5):
        fg = scr[scr["fold"] == f]
        folds_info[str(f)] = {
            "classrooms": sorted(fg["classroom"].unique().tolist()),
            "n_transcripts": int(fg["transcript_id"].nunique()),
            "n_screened": int(len(fg)),
            "n_tp": int((fg["label"] == 1).sum()),
            "n_fp": int((fg["label"] == 0).sum()),
            "n_u0": int((fg["label"] == -1).sum()),
            "tp_pct_of_screened": round(100.0 * (fg["label"] == 1).mean(), 3),
        }

    summary = {
        "y2_total_rows": int(len(y2)),
        "y2_screened_rows": int(len(scr)),
        "y2_transcripts": int(y2["transcript_id"].nunique()),
        "y2_classrooms": int(y2["classroom"].nunique()),
        "y2_tp": int((scr["label"] == 1).sum()),
        "y2_fp": int((scr["label"] == 0).sum()),
        "y2_unreviewed_flagged": int((scr["label"] == -2).sum()),
        "y2_unflagged_u0": int((scr["label"] == -1).sum()),
        "y2_v2_flagged_on_screened": int(scr["v2_flagged"].sum()),
        "y1_total_rows": int(len(y1)),
        "y1_clean_eval_rows": int(y1["eval_clean_mask"].sum()),
        "y1_clean_sci": int((y1.loc[y1["eval_clean_mask"], "label"] == 1).sum()),
        "y1_seed_rows": int(len(seed)),
        "folds": folds_info,
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print(f"[dataset] Wrote data/v3/* | Folds summary:")
        for k, v in folds_info.items():
            print(
                f"  Fold {k}: rooms={v['classrooms']} | tx={v['n_transcripts']} | "
                f"TP={v['n_tp']} FP={v['n_fp']} U0={v['n_u0']:,} (TP%={v['tp_pct_of_screened']:.2f}%)"
            )
    return summary


if __name__ == "__main__":
    build_all(verbose=True)
