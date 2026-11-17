"""Aggregate the human-checked Y2 analytic sample per Dr. Hadley's checklist.

Inputs
------
- ``Human coded/Classroom<id>_<date>_ebh.xlsx`` : one workbook per
  classroom-observation. Exactly one child sheet per workbook has the
  ``review`` column filled -> that sheet is the sampled focal transcript.
  Sheets contain ONLY utterances the LLM flagged as possible science talk.
- ``data/transcripts/Y2_Technology Observation Audio_Transcripts/<folder>/``:
  raw per-child ASR transcripts -> denominators (screened utterances) and
  audio hours.

Outputs (Data Aggregation/output/)
----------------------------------
- sampled_transcripts.xlsx     one row per sampled transcript + column dictionary
- overall_summary.docx         checklist totals (all / adult / child)
- descriptive_statistics.docx  mean/SD/min/max across transcripts
- subtype_rationale_summary.docx  subtype + rationale summaries among verified
- aggregation_log.txt          per-workbook processing log

Definitions (also in the column dictionary)
-------------------------------------------
- "Screened utterances": adult/child speech rows at or above the ASR-confidence
  floor used by the scoring pipeline (config/deploy.json, 0.40); rows with no
  parsable confidence are kept. This is the set the LLM actually evaluated and
  is the denominator for all percentages.
- "Hours (span)": last segment end minus first segment start of the child's
  recording -> observation time represented; the base for per-hour rates.
- "Hours (talk time)": summed duration of segments containing speech
  (reference only).
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
HUMAN_DIR = PROJECT_ROOT / "Human coded"
RAW_ROOT = (PROJECT_ROOT / "data" / "transcripts"
            / "Y2_Technology Observation Audio_Transcripts")
OUT_DIR = HERE / "output"

TRANSCRIPT_COL = "Transcription (Confidence)"
_CONF_RE = re.compile(r"\((\d+(?:\.\d+)?)\s*%\)\s*$")

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    LOG_LINES.append(msg)
    print(msg)


# ---------------------------------------------------------------------------
# Shared parsing helpers (mirrors src/deploy_y2_12.py ingest filters)
# ---------------------------------------------------------------------------

def parse_transcription(cell) -> tuple[str, float | None]:
    s = "" if cell is None else str(cell).strip()
    m = _CONF_RE.search(s)
    if m:
        return s[: m.start()].strip(), float(m.group(1)) / 100.0
    return s, None


def child_id_from_filename(name: str) -> str:
    stem = Path(name).name.split("_", 1)[0]
    parts = stem.split("-")
    return parts[3] if len(parts) >= 4 else "base"


def to_seconds(val) -> float | None:
    """Robust timestamp -> seconds. Handles 'H:MM:SS(.f)', 'MM:SS', datetime.time."""
    if val is None:
        return None
    if isinstance(val, dtime):
        return val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val) if np.isfinite(val) else None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "-", "none", "nat"):
        return None
    parts = s.split(":")
    try:
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
    except ValueError:
        return None


def confidence_floor() -> float:
    cfg = PROJECT_ROOT / "config" / "deploy.json"
    try:
        with open(cfg, encoding="utf-8") as f:
            return float(json.load(f).get("confidence_min", 0.40))
    except Exception:  # noqa: BLE001
        return 0.40


# ---------------------------------------------------------------------------
# Step 1: reviewed sheet extraction from a human-coded workbook
# ---------------------------------------------------------------------------

def _data_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the tally block: keep rows whose line_no is numeric."""
    if "line_no" not in df.columns:
        return df.iloc[0:0]
    keep = pd.to_numeric(df["line_no"], errors="coerce").notna()
    return df.loc[keep].copy()


def find_reviewed_sheet(path: Path) -> tuple[str, pd.DataFrame] | None:
    """Return (sheet_name, data_rows) for the single human-reviewed child sheet."""
    xl = pd.ExcelFile(path, engine="openpyxl")
    candidates: list[tuple[str, pd.DataFrame, int]] = []
    for sheet in xl.sheet_names:
        if sheet.upper() in ("INSTRUCTIONS", "SUMMARY"):
            continue
        df = _data_rows(pd.read_excel(path, sheet_name=sheet, engine="openpyxl"))
        if df.empty or "review" not in df.columns:
            continue
        n_rev = int(df["review"].notna().sum())
        if n_rev > 0:
            candidates.append((sheet, df, n_rev))
    if not candidates:
        log(f"  [WARN] {path.name}: no reviewed sheet found -> EXCLUDED")
        return None
    if len(candidates) > 1:
        names = ", ".join(f"{s}({n})" for s, _, n in candidates)
        log(f"  [WARN] {path.name}: multiple reviewed sheets [{names}]; "
            f"using the most-reviewed one")
        candidates.sort(key=lambda t: -t[2])
    sheet, df, _ = candidates[0]
    return sheet, df


def normalize_flagged(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize review/speaker/subtype on the flagged rows."""
    out = df.copy()
    rev = out["review"].astype(str).str.strip().str.upper()
    out["review_norm"] = np.where(
        rev.eq("SCIENCE_TALK"), "verified",
        np.where(rev.eq("NOT_SCIENCE_TALK"), "false_positive", "unreviewed"))
    out.loc[out["review"].isna(), "review_norm"] = "unreviewed"
    out["speaker_norm"] = (out["speaker"].astype(str).str.strip().str.lower()
                           .where(lambda s: s.isin(["adult", "child"]), "other"))
    out["subtype_norm"] = (out.get("predicted_subtype", pd.Series(index=out.index))
                           .astype(str).str.strip().str.lower()
                           .replace({"nan": "", "none": ""}))
    return out


# ---------------------------------------------------------------------------
# Step 2: denominators + hours from the raw transcript
# ---------------------------------------------------------------------------

def raw_denominators(folder: Path, child_id: str, conf_min: float) -> dict | None:
    """Screened counts (total + per speaker), raw rows, span/talk hours."""
    matches = [p for p in sorted(folder.glob("*.xlsx"))
               if child_id_from_filename(p.name) == child_id]
    if not matches:
        log(f"  [WARN] no raw file for child {child_id} in {folder.name}")
        return None
    if len(matches) > 1:
        log(f"  [WARN] {folder.name}: {len(matches)} raw files for child "
            f"{child_id}; using {matches[0].name}")
    path = matches[0]
    df = pd.read_excel(path, engine="openpyxl")
    if TRANSCRIPT_COL not in df.columns:
        log(f"  [WARN] {path.name} missing '{TRANSCRIPT_COL}'")
        return None

    n_raw = len(df)
    speaker = df.get("Speaker", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    parsed = df[TRANSCRIPT_COL].apply(parse_transcription)
    text = parsed.str[0]
    conf = parsed.str[1]
    has_text = text.notna() & ~text.isin(["", "-", "nan"])
    conf_ok = conf.isna() | (conf >= conf_min)
    screened = has_text & conf_ok & speaker.isin(["adult", "child"])

    starts = df.get("Segment Start Time", pd.Series(dtype=object)).apply(to_seconds)
    ends = df.get("Segment End Time", pd.Series(dtype=object)).apply(to_seconds)
    valid = starts.notna() & ends.notna()
    span_h = float((ends[valid].max() - starts[valid].min()) / 3600) if valid.any() else np.nan
    talk = (ends[valid & has_text] - starts[valid & has_text]).clip(lower=0)
    talk_h = float(talk.sum() / 3600) if len(talk) else np.nan

    return {
        "raw_file": path.name,
        "utt_raw_ref": int(n_raw),
        "utt_screened_total": int(screened.sum()),
        "utt_screened_adult": int((screened & speaker.eq("adult")).sum()),
        "utt_screened_child": int((screened & speaker.eq("child")).sum()),
        "hours_span": round(span_h, 3) if np.isfinite(span_h) else np.nan,
        "hours_talk_ref": round(talk_h, 3) if np.isfinite(talk_h) else np.nan,
    }


# ---------------------------------------------------------------------------
# Step 3: per-transcript rows
# ---------------------------------------------------------------------------

def _pct(num: float, den: float) -> float:
    return round(100.0 * num / den, 2) if den else np.nan


def _rate(num: float, hours: float) -> float:
    return round(num / hours, 2) if hours and np.isfinite(hours) and hours > 0 else np.nan


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-transcript df, all verified flagged rows for subtype work)."""
    conf_min = confidence_floor()
    log(f"ASR confidence floor (from config/deploy.json): {conf_min}")
    books = sorted(HUMAN_DIR.glob("Classroom*_ebh.xlsx"))
    log(f"human-coded workbooks found: {len(books)}")

    rows: list[dict] = []
    flagged_frames: list[pd.DataFrame] = []
    for path in books:
        base = path.name.replace("_ebh.xlsx", "")          # Classroom10_020526
        m = re.match(r"Classroom(\w+?)_(\d{6})(.*)$", base)
        if not m:
            log(f"  [WARN] cannot parse name {path.name} -> EXCLUDED")
            continue
        classroom, date, suffix = m.group(1), m.group(2), m.group(3)
        folder = RAW_ROOT / base
        if not folder.exists():
            log(f"  [WARN] raw folder missing for {base} -> EXCLUDED")
            continue

        found = find_reviewed_sheet(path)
        if found is None:
            continue
        sheet, df = found
        child_id = sheet.replace("child_", "", 1)
        df = normalize_flagged(df)

        den = raw_denominators(folder, child_id, conf_min)
        if den is None:
            continue

        is_adult = df["speaker_norm"].eq("adult")
        is_child = df["speaker_norm"].eq("child")
        ver = df["review_norm"].eq("verified")
        fpos = df["review_norm"].eq("false_positive")
        unrev = df["review_norm"].eq("unreviewed")

        n_flagged = len(df)
        n_fl_a, n_fl_c = int(is_adult.sum()), int(is_child.sum())
        n_ver = int(ver.sum())
        n_ver_a, n_ver_c = int((ver & is_adult).sum()), int((ver & is_child).sum())
        n_fp = int(fpos.sum())
        n_un = int(unrev.sum())
        hours = den["hours_span"]

        rows.append({
            "classroom_id": classroom,
            "observation_date": date,
            "child_id": child_id,
            "hours_span": hours,
            "hours_talk_ref": den["hours_talk_ref"],
            "utt_screened_total": den["utt_screened_total"],
            "utt_screened_adult": den["utt_screened_adult"],
            "utt_screened_child": den["utt_screened_child"],
            "utt_raw_ref": den["utt_raw_ref"],
            "n_flagged": n_flagged,
            "n_flagged_adult": n_fl_a,
            "n_flagged_child": n_fl_c,
            "pct_flagged": _pct(n_flagged, den["utt_screened_total"]),
            "pct_flagged_adult": _pct(n_fl_a, den["utt_screened_adult"]),
            "pct_flagged_child": _pct(n_fl_c, den["utt_screened_child"]),
            "n_verified": n_ver,
            "n_verified_adult": n_ver_a,
            "n_verified_child": n_ver_c,
            "pct_verified_of_flagged": _pct(n_ver, n_flagged),
            "pct_verified_of_flagged_adult": _pct(n_ver_a, n_fl_a),
            "pct_verified_of_flagged_child": _pct(n_ver_c, n_fl_c),
            "pct_verified_of_all": _pct(n_ver, den["utt_screened_total"]),
            "pct_verified_of_all_adult": _pct(n_ver_a, den["utt_screened_adult"]),
            "pct_verified_of_all_child": _pct(n_ver_c, den["utt_screened_child"]),
            "verified_per_hour": _rate(n_ver, hours),
            "verified_per_hour_adult": _rate(n_ver_a, hours),
            "verified_per_hour_child": _rate(n_ver_c, hours),
            "n_false_positive": n_fp,
            "n_flagged_unreviewed": n_un,
            "raw_file": den["raw_file"],
            "reviewed_sheet": sheet,
        })

        keep = df[["review_norm", "speaker_norm", "subtype_norm"]].copy()
        keep["rationale"] = df.get("rationale", pd.Series(index=df.index)).astype(str)
        keep["classroom_id"] = classroom
        keep["observation_date"] = date
        keep["child_id"] = child_id
        flagged_frames.append(keep)

        log(f"  ok {base}: sheet={sheet} flagged={n_flagged} verified={n_ver} "
            f"fp={n_fp} unreviewed={n_un} screened={den['utt_screened_total']} "
            f"hours={hours}")

    per_tx = pd.DataFrame(rows)
    flagged_all = (pd.concat(flagged_frames, ignore_index=True)
                   if flagged_frames else pd.DataFrame())
    return per_tx, flagged_all


# ---------------------------------------------------------------------------
# Step 4: summaries
# ---------------------------------------------------------------------------

GROUPS = {"all": "", "adult": "_adult", "child": "_child"}


def overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for gname, sfx in GROUPS.items():
        utt = df[f"utt_screened{'_total' if gname == 'all' else sfx}"].sum()
        flg = df[f"n_flagged{sfx}"].sum()
        ver = df[f"n_verified{sfx}"].sum()
        if gname == "all":
            fp = df["n_false_positive"].sum()
            unrev = df["n_flagged_unreviewed"].sum()
        else:
            # reviewed flagged in group = flagged - unreviewed (per speaker not
            # tracked separately) -> conservative: FP = reviewed flagged - verified.
            fp = np.nan
            unrev = np.nan
        col = {
            "Classrooms represented": df["classroom_id"].nunique(),
            "Observations represented": df[["classroom_id", "observation_date"]]
                .drop_duplicates().shape[0],
            "Child-level transcripts": len(df),
            "Total audio hours (span)": round(df["hours_span"].sum(), 2),
            "Total utterances (screened)": int(utt),
            "Utterances flagged by the LLM": int(flg),
            "% of utterances flagged": _pct(flg, utt),
            "Flagged utterances human-verified as science talk": int(ver),
            "% of all utterances verified as science talk": _pct(ver, utt),
            "% of flagged utterances verified": _pct(ver, flg),
            "Flagged utterances that were false positives": (int(fp) if np.isfinite(fp) else None),
            "% of flagged that were false positives": (_pct(fp, flg) if np.isfinite(fp) else None),
            "Flagged utterances not human-checked": (int(unrev) if np.isfinite(unrev) else None),
        }
        out[gname] = col
    res = pd.DataFrame(out)
    res.columns = ["All utterances", "Teacher/adult", "Child"]
    return res


def _fp_by_speaker(flagged_all: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Fill in adult/child FP + unreviewed counts from the row-level data."""
    for gname, col in (("adult", "Teacher/adult"), ("child", "Child")):
        sub = flagged_all[flagged_all["speaker_norm"].eq(gname)]
        fp = int(sub["review_norm"].eq("false_positive").sum())
        un = int(sub["review_norm"].eq("unreviewed").sum())
        flg = len(sub)
        summary.loc["Flagged utterances that were false positives", col] = fp
        summary.loc["% of flagged that were false positives", col] = _pct(fp, flg)
        summary.loc["Flagged utterances not human-checked", col] = un
    return summary


DESC_METRICS = {
    "Audio hours (span)": "hours_span",
    "Total utterances (screened)": "utt_screened{sfx}",
    "LLM-flagged utterances": "n_flagged{sfx}",
    "% utterances flagged": "pct_flagged{sfx}",
    "Human-verified science talk utterances": "n_verified{sfx}",
    "Verified science talk per hour": "verified_per_hour{sfx}",
    "% of all utterances verified": "pct_verified_of_all{sfx}",
    "% of flagged verified": "pct_verified_of_flagged{sfx}",
}


def descriptive_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    for gname, sfx in GROUPS.items():
        rows = {}
        for label, tmpl in DESC_METRICS.items():
            if label == "Audio hours (span)":
                col = "hours_span"  # identical across speaker groups
            elif label == "Total utterances (screened)":
                col = "utt_screened_total" if gname == "all" else f"utt_screened{sfx}"
            else:
                col = tmpl.format(sfx=sfx)
            s = pd.to_numeric(df[col], errors="coerce")
            rows[label] = {
                "Mean": round(s.mean(), 2), "SD": round(s.std(ddof=1), 2),
                "Min": round(s.min(), 2), "Max": round(s.max(), 2),
            }
        tables[gname] = pd.DataFrame(rows).T
    return tables


def subtype_summaries(flagged_all: pd.DataFrame, per_tx: pd.DataFrame):
    ver = flagged_all[flagged_all["review_norm"].eq("verified")].copy()
    ver["subtype_list"] = ver["subtype_norm"].str.split("|")
    exploded = ver.explode("subtype_list")
    exploded["subtype_list"] = exploded["subtype_list"].str.strip()
    exploded = exploded[exploded["subtype_list"].ne("")]

    def counts(sub: pd.DataFrame, base: int) -> pd.DataFrame:
        c = sub["subtype_list"].value_counts()
        return pd.DataFrame({
            "Count": c,
            "% of verified utterances": (100 * c / base).round(1) if base else np.nan,
        })

    n_ver = len(ver)
    tabs = {"all": counts(exploded, n_ver)}
    for g in ("adult", "child"):
        sub = exploded[exploded["speaker_norm"].eq(g)]
        base = int(ver["speaker_norm"].eq(g).sum())
        tabs[g] = counts(sub, base)

    # per-transcript subtype-count descriptive stats
    per_counts = (exploded.groupby(["classroom_id", "observation_date",
                                    "subtype_list"]).size().unstack(fill_value=0))
    idx = per_tx.set_index(["classroom_id", "observation_date"]).index
    per_counts = per_counts.reindex(idx, fill_value=0)
    desc = pd.DataFrame({
        "Mean": per_counts.mean().round(2), "SD": per_counts.std(ddof=1).round(2),
        "Min": per_counts.min(), "Max": per_counts.max(),
    })

    # rationale: model-generated free text -> top recurring strings
    rat = ver["rationale"].astype(str).str.strip().str.lower()
    rat = rat[~rat.isin(["", "nan", "none"])]
    top_rat = rat.value_counts().head(15)
    rat_df = pd.DataFrame({
        "Count": top_rat,
        "% of verified utterances": (100 * top_rat / n_ver).round(1) if n_ver else np.nan,
    })
    return tabs, desc, rat_df, n_ver


# ---------------------------------------------------------------------------
# Step 5: writers
# ---------------------------------------------------------------------------

# Dr. Hadley's exact per-transcript columns: internal name -> sheet header.
MAIN_COLS = [
    ("classroom_id", "Classroom ID"),
    ("observation_date", "Observation date"),
    ("child_id", "Child ID"),
    ("hours_span", "Total audio hours"),
    ("utt_screened_total", "Total utterances"),
    ("n_flagged", "LLM-flagged science talk utterances"),
    ("pct_flagged", "% of utterances flagged by LLM"),
    ("n_verified", "Human-verified science talk utterances"),
    ("pct_verified_of_flagged", "% of flagged verified as science talk"),
    ("pct_verified_of_all", "% of all utterances verified as science talk"),
    ("verified_per_hour", "Verified science talk per hour"),
]

ID_COLS = ["classroom_id", "observation_date", "child_id"]

MAIN_DICT = [
    ("Classroom ID", "Classroom identifier (from workbook name)"),
    ("Observation date", "Observation date MMDDYY (from workbook name)"),
    ("Child ID", "Sampled focal child whose transcript was human-reviewed"),
    ("Total audio hours", "Audio represented: last segment end - first segment "
                          "start of the child's recording"),
    ("Total utterances", "Utterances screened by the pipeline (adult+child "
                         "speech at/above the 40% ASR-confidence floor) - the "
                         "set the LLM evaluated"),
    ("LLM-flagged science talk utterances",
     "Utterances the LLM initially flagged as possible science talk"),
    ("% of utterances flagged by LLM", "Flagged / total utterances x 100"),
    ("Human-verified science talk utterances",
     "Flagged utterances verified as actual science talk by human review"),
    ("% of flagged verified as science talk", "Verified / flagged x 100"),
    ("% of all utterances verified as science talk",
     "Verified / total utterances x 100"),
    ("Verified science talk per hour", "Verified / total audio hours"),
]

SUPP_DICT = [
    ("hours_talk_ref", "Reference: summed duration of speech segments (hours)"),
    ("utt_screened_adult", "Screened utterances attributed to adults"),
    ("utt_screened_child", "Screened utterances attributed to children"),
    ("utt_raw_ref", "Reference: all raw rows in the child's transcript file"),
    ("n_flagged_adult", "LLM-flagged utterances attributed to adults"),
    ("n_flagged_child", "LLM-flagged utterances attributed to children"),
    ("pct_flagged_adult", "% of adult screened utterances flagged"),
    ("pct_flagged_child", "% of child screened utterances flagged"),
    ("n_verified_adult", "Verified science talk attributed to adults"),
    ("n_verified_child", "Verified science talk attributed to children"),
    ("pct_verified_of_flagged_adult", "% of adult flagged verified"),
    ("pct_verified_of_flagged_child", "% of child flagged verified"),
    ("pct_verified_of_all_adult", "Adult verified / adult screened x 100"),
    ("pct_verified_of_all_child", "Child verified / child screened x 100"),
    ("verified_per_hour_adult", "Adult verified per audio hour"),
    ("verified_per_hour_child", "Child verified per audio hour"),
    ("n_false_positive", "Flagged utterances human-marked NOT science talk"),
    ("n_flagged_unreviewed", "Flagged utterances left blank by the reviewer"),
    ("raw_file", "Raw transcript file used for denominators/hours"),
    ("reviewed_sheet", "Sheet in the human-coded workbook that was reviewed"),
]


def write_excel(per_tx: pd.DataFrame) -> Path:
    path = OUT_DIR / "sampled_transcripts.xlsx"
    main = per_tx[[c for c, _ in MAIN_COLS]].rename(columns=dict(MAIN_COLS))
    supp_cols = ID_COLS + [c for c, _ in SUPP_DICT]
    supp = per_tx[supp_cols]
    dictionary = pd.concat([
        pd.DataFrame([("sampled_transcripts", h, d) for h, d in MAIN_DICT],
                     columns=["sheet", "column", "definition"]),
        pd.DataFrame([("supplementary_detail", c, d) for c, d in SUPP_DICT],
                     columns=["sheet", "column", "definition"]),
    ], ignore_index=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        main.to_excel(xl, sheet_name="sampled_transcripts", index=False)
        supp.to_excel(xl, sheet_name="supplementary_detail", index=False)
        dictionary.to_excel(xl, sheet_name="column_dictionary", index=False)
    return path


def _fmt_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        if v == int(v):
            return f"{int(v):,}"
        return f"{v:,.2f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def _doc_table(doc, df: pd.DataFrame, index_label: str = ""):
    from docx.shared import Pt
    t = doc.add_table(rows=len(df) + 1, cols=len(df.columns) + 1)
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    hdr[0].text = index_label
    for j, c in enumerate(df.columns):
        hdr[j + 1].text = str(c)
    for i, (idx, row) in enumerate(df.iterrows()):
        cells = t.rows[i + 1].cells
        cells[0].text = str(idx)
        for j, v in enumerate(row):
            cells[j + 1].text = _fmt_cell(v)
    for r in t.rows:
        for c in r.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()


CAVEATS = [
    "Speaker attribution (teacher/adult vs child) comes from automatic speech "
    "recognition and diarization and is not perfectly accurate; adult/child "
    "splits should be interpreted with caution.",
    "'Total utterances' = utterances screened by the pipeline (adult/child "
    "speech at or above the 40% ASR-confidence floor) - the set the LLM "
    "actually evaluated. Raw row counts are provided as a reference column in "
    "the Excel file.",
    "Audio hours use the recording span (first to last segment); a talk-time "
    "reference column is provided in the Excel file.",
    "Flagged utterances left blank by the reviewer are reported separately as "
    "'not human-checked' and are excluded from verified counts.",
]


def write_overall_docx(summary: pd.DataFrame, per_tx: pd.DataFrame) -> Path:
    import docx
    doc = docx.Document()
    doc.add_heading("Human-Checked Y2 Analytic Sample - Overall Summary", 0)
    doc.add_paragraph(
        f"Generated {datetime.now():%B %d, %Y}. Analytic sample: one human-"
        f"reviewed focal-child transcript per classroom-observation "
        f"({len(per_tx)} transcripts, {per_tx['classroom_id'].nunique()} "
        f"classrooms). Only human-checked data are included.")
    doc.add_paragraph(
        "Key point: verified science talk is a low-frequency feature of these "
        "transcripts. The LLM deliberately generated a broad first-pass "
        "candidate set, which human review then verified.")
    doc.add_heading("Summary across the full human-checked sample", level=1)
    _doc_table(doc, summary, index_label="Measure")
    doc.add_heading("Cautions", level=1)
    for c in CAVEATS:
        doc.add_paragraph(c, style="List Bullet")
    path = OUT_DIR / "overall_summary.docx"
    doc.save(path)
    return path


def write_desc_docx(tables: dict[str, pd.DataFrame], per_tx: pd.DataFrame) -> Path:
    import docx
    doc = docx.Document()
    doc.add_heading("Descriptive Statistics Across Sampled Transcripts", 0)
    doc.add_paragraph(
        f"Unit of analysis: sampled focal-child transcript (n = {len(per_tx)}). "
        "Statistics are computed across transcripts: mean, standard deviation, "
        "minimum, and maximum.")
    titles = {"all": "All utterances", "adult": "Teacher/adult utterances",
              "child": "Child utterances"}
    for g, tab in tables.items():
        doc.add_heading(titles[g], level=1)
        if g != "all":
            doc.add_paragraph("Note: audio hours are transcript-level and "
                              "identical across speaker groups.")
        _doc_table(doc, tab, index_label="Metric")
    doc.add_heading("Cautions", level=1)
    for c in CAVEATS[:1]:
        doc.add_paragraph(c, style="List Bullet")
    path = OUT_DIR / "descriptive_statistics.docx"
    doc.save(path)
    return path


def write_subtype_docx(tabs, desc, rat_df, n_ver, per_tx) -> Path:
    import docx
    doc = docx.Document()
    doc.add_heading("Science Talk Sub-Types and Rationales (Human-Verified)", 0)
    doc.add_paragraph(
        f"Among the {n_ver} human-verified science talk utterances across "
        f"{len(per_tx)} sampled transcripts. Sub-type labels are model-"
        "predicted; an utterance can carry more than one sub-type, so "
        "percentages can sum to more than 100%.")
    titles = {"all": "All verified utterances",
              "adult": "Teacher/adult verified utterances",
              "child": "Child verified utterances"}
    for g, tab in tabs.items():
        doc.add_heading(titles[g], level=1)
        _doc_table(doc, tab, index_label="Sub-type")
    doc.add_heading("Sub-type counts per sampled transcript", level=1)
    doc.add_paragraph("Mean/SD/Min/Max of verified-utterance sub-type counts "
                      "across the sampled transcripts.")
    _doc_table(doc, desc, index_label="Sub-type")
    doc.add_heading("Most common model rationales", level=1)
    doc.add_paragraph(
        "Rationales are short free-text explanations generated by the language "
        "model, not a human-coded category system; identical strings are "
        "counted below.")
    _doc_table(doc, rat_df, index_label="Rationale")
    path = OUT_DIR / "subtype_rationale_summary.docx"
    doc.save(path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== Human-coded aggregation | {datetime.now():%Y-%m-%d %H:%M} ===")
    per_tx, flagged_all = build_dataset()
    if per_tx.empty:
        log("No transcripts aggregated; aborting.")
        return 1
    log(f"per-transcript rows: {len(per_tx)} | classrooms: "
        f"{per_tx['classroom_id'].nunique()} | observations: "
        f"{per_tx[['classroom_id', 'observation_date']].drop_duplicates().shape[0]}")

    xl_path = write_excel(per_tx)
    log(f"wrote {xl_path.name}")

    summary = overall_summary(per_tx)
    summary = _fp_by_speaker(flagged_all, summary)
    p1 = write_overall_docx(summary, per_tx)
    log(f"wrote {p1.name}")

    tables = descriptive_stats(per_tx)
    p2 = write_desc_docx(tables, per_tx)
    log(f"wrote {p2.name}")

    tabs, desc, rat_df, n_ver = subtype_summaries(flagged_all, per_tx)
    p3 = write_subtype_docx(tabs, desc, rat_df, n_ver, per_tx)
    log(f"wrote {p3.name}")

    # reconciliation checks
    assert int(summary.loc["Utterances flagged by the LLM", "All utterances"]) \
        == int(per_tx["n_flagged"].sum())
    assert int(summary.loc["Flagged utterances human-verified as science talk",
                           "All utterances"]) == int(per_tx["n_verified"].sum())
    log("reconciliation checks passed (Docx totals == Excel sums)")

    with open(OUT_DIR / "aggregation_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
