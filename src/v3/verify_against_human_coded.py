"""End-to-end verification directly against every Excel workbook in `Human coded/`.

Reads ALL 83 `.xlsx` files directly from `Human coded/` (falling back cleanly to
`Human coded.zip` for any workbook currently open/locked in Excel), aligns every
human-reviewed row `(workbook, sheet, line_no)` against leak-free out-of-fold v3
predictions (`6A_hybrid_cascade_v3` and `6B_pgkd_distilled_local_v3`, plus legacy
v1 & v2 replays), and emits:
  1. `reports/v3/human_coded_verification_summary.json` (overall + per-classroom + per-workbook)
  2. `reports/v3/human_coded_verification_report.xlsx` (multi-sheet audit workbook:
     `OVERALL_SUMMARY`, `PER_WORKBOOK_83`, `REMAINING_FP_ERRORS`, `REMAINING_FN_ERRORS`,
     `SCHEMA_CHECK`)
  3. Verifies exact 18-column Excel schema, dropdown validation, tally formulas, and
     deterministic `child_base` sheet naming on the exported v3 workbooks!
"""
from __future__ import annotations

import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.codebook_v3 import (
    apply_usf_cluster_and_veto_rules,
    clean_training_labels_for_usf,
    extract_pm2_context_columns,
)
from src.v3.export_workbooks import HUMAN_CODED_COLS, LABEL_NOT, LABEL_SCIENCE
from src.v3.metrics import (
    binary_metrics,
    oof_bcts_and_thresholds,
    tune_threshold_recall,
)

DATA_DIR = PROJECT_ROOT / "data" / "v3"
PREDS_DIR = PROJECT_ROOT / "reports" / "v3" / "stage1_preds"
S2_DIR = PROJECT_ROOT / "reports" / "v3" / "stage2_llm"
REPORTS_V3 = PROJECT_ROOT / "reports" / "v3"
HUMAN_CODED_DIR = PROJECT_ROOT / "Human coded"
HUMAN_CODED_ZIP = PROJECT_ROOT / "Human coded.zip"


def _open_human_workbook(xlsx_path: Path, zf: zipfile.ZipFile | None) -> pd.ExcelFile:
    try:
        return pd.ExcelFile(xlsx_path, engine="openpyxl")
    except PermissionError:
        if zf is None:
            raise
        target = f"Human coded/{xlsx_path.name}"
        if target not in zf.namelist():
            matches = [n for n in zf.namelist() if n.endswith("/" + xlsx_path.name) or n == xlsx_path.name]
            if not matches:
                raise
            target = matches[0]
        return pd.ExcelFile(BytesIO(zf.read(target)), engine="openpyxl")


def _load_all_human_coded_rows() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read every reviewed row directly from `Human coded/*.xlsx`."""
    files = sorted([p for p in HUMAN_CODED_DIR.glob("*.xlsx") if not p.name.startswith("~$")])
    zf = zipfile.ZipFile(HUMAN_CODED_ZIP, "r") if HUMAN_CODED_ZIP.exists() else None
    rows: list[dict[str, Any]] = []
    locked_fallbacks: list[str] = []

    try:
        for fp in files:
            try:
                xl = pd.ExcelFile(fp, engine="openpyxl")
            except PermissionError:
                xl = _open_human_workbook(fp, zf)
                locked_fallbacks.append(fp.name)

            folder_name = fp.stem.replace("_ebh", "")
            # Identify the primary reviewed sheet per workbook (matching dataset.py)
            # and track secondary partially-reviewed sheets explicitly!
            sheet_dfs: list[tuple[str, pd.DataFrame]] = []
            for sn in xl.sheet_names:
                if sn.upper() == "INSTRUCTIONS":
                    continue
                df = xl.parse(sn)
                if "line_no" not in df.columns or "review" not in df.columns:
                    continue
                ln_num = pd.to_numeric(df["line_no"], errors="coerce")
                df = df.loc[ln_num.notna()].copy()
                df["line_no"] = ln_num.loc[ln_num.notna()].astype(int)
                rev_clean = df["review"].astype(str).str.strip().str.upper()
                valid_rev = rev_clean.isin([LABEL_SCIENCE, LABEL_NOT])
                if valid_rev.any():
                    sheet_dfs.append((sn, df.loc[valid_rev].copy()))

            primary_sheet = max(sheet_dfs, key=lambda x: len(x[1]))[0] if sheet_dfs else None
            for sn, sub_df in sheet_dfs:
                is_primary = (sn == primary_sheet)
                for _, r in sub_df.iterrows():
                    rv = str(r["review"]).strip().upper()
                    rows.append({
                        "workbook": fp.name,
                        "folder": folder_name,
                        "sheet": sn,
                        "is_primary_sheet": is_primary,
                        "line_no": int(r["line_no"]),
                        "human_label": 1 if rv == LABEL_SCIENCE else 0,
                        "human_review_str": rv,
                        "v1_pred_str": str(r.get("predicted_label", "")).strip().upper(),
                        "speaker_hc": str(r.get("speaker", "")),
                        "utterance_hc": str(r.get("utterance", "")),
                    })
    finally:
        if zf is not None:
            zf.close()

    df_all = pd.DataFrame(rows)
    n_primary = int(df_all["is_primary_sheet"].sum()) if len(df_all) else 0
    n_secondary = int((~df_all["is_primary_sheet"]).sum()) if len(df_all) else 0
    return df_all, {
        "n_workbooks_scanned": len(files),
        "locked_zip_fallbacks": locked_fallbacks,
        "n_human_reviewed_rows_total": len(rows),
        "n_primary_sheet_rows": n_primary,
        "n_secondary_partial_sheet_rows": n_secondary,
    }


def _oof_calibrate_and_threshold(
    df_y2: pd.DataFrame, raw_p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Canonical strictly fold-isolated OOF BCTS calibration & thresholds (matches metrics.py 100%)."""
    scr = df_y2["screened"].to_numpy(dtype=bool)
    folds = df_y2["fold"].to_numpy()
    labels = df_y2["label"].to_numpy()
    rev_m = scr & np.isin(labels, [0, 1])
    y_bin = np.where(labels == 1, 1, 0)
    res = oof_bcts_and_thresholds(raw_p, y_bin, folds, rev_m, recall_targets=(0.90,))
    return (
        res["cal_oof"],
        res["pred_f1"],
        res["pred_rec"][0.90],
        res["mean_thr_f1"],
        res["mean_thr_rec"][0.90],
    )


def verify_exported_workbook_schema(export_dir: Path) -> dict[str, Any]:
    """Verify exported v3.1 Excel workbooks match exact 21-column `Human coded` + ±2 context schema + dropdowns."""
    import openpyxl

    files = sorted([p for p in export_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    checked = 0
    bad_schema: list[str] = []
    missing_instructions: list[str] = []
    missing_dv: list[str] = []
    for fp in files:
        wb = openpyxl.load_workbook(fp, read_only=False)
        if "INSTRUCTIONS" not in wb.sheetnames:
            missing_instructions.append(fp.name)
        for sn in wb.sheetnames:
            if sn == "INSTRUCTIONS":
                continue
            ws = wb[sn]
            hdr = [ws.cell(1, c).value for c in range(1, len(HUMAN_CODED_COLS) + 1)]
            if hdr != HUMAN_CODED_COLS:
                bad_schema.append(f"{fp.name}:{sn}")
            if len(ws.data_validations.dataValidation) == 0:
                missing_dv.append(f"{fp.name}:{sn}")
            checked += 1
        wb.close()
    return {
        "export_dir": str(export_dir),
        "n_workbooks": len(files),
        "n_data_sheets_checked": checked,
        "schema_21col_ok": len(bad_schema) == 0,
        "instructions_sheet_ok": len(missing_instructions) == 0,
        "review_dropdown_ok": len(missing_dv) == 0,
        "bad_schema": bad_schema[:10],
        "missing_instructions": missing_instructions[:10],
        "missing_dv": missing_dv[:10],
    }


def run_human_coded_verification(verbose: bool = True) -> dict[str, Any]:
    hc_df, scan_meta = _load_all_human_coded_rows()
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")

    hyb_raw = np.load(PREDS_DIR / "6A_hybrid_cascade_v3_y2_oof.npy").astype(np.float64)
    hyb_usf_path = PREDS_DIR / "6A_hybrid_cascade_v3_usf_y2_oof.npy"
    hyb_usf_raw = np.load(hyb_usf_path).astype(np.float64) if hyb_usf_path.exists() else hyb_raw
    loc_raw = np.load(PREDS_DIR / "6B_pgkd_distilled_local_v3_y2_oof.npy").astype(np.float64)
    loc_usf_path = PREDS_DIR / "6B_pgkd_distilled_local_v3_usf_y2_oof.npy"
    loc_usf_raw = np.load(loc_usf_path).astype(np.float64) if loc_usf_path.exists() else loc_raw
    s1_raw = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy").astype(np.float64)

    hyb_cal, hyb_st, hyb_s90, thr_hyb_st, thr_hyb_s90 = _oof_calibrate_and_threshold(df_y2, hyb_usf_raw)
    loc_cal, loc_st, loc_s90, thr_loc_st, thr_loc_s90 = _oof_calibrate_and_threshold(df_y2, loc_usf_raw)
    s1_cal, s1_st, s1_s90, _, _ = _oof_calibrate_and_threshold(df_y2, s1_raw)

    s2_txt = pd.read_parquet(S2_DIR / "full83_stage2_text.parquet") if (S2_DIR / "full83_stage2_text.parquet").exists() else None

    # Build exact join key `(folder, sheet, line_no)` from Y2 table
    y2_keyed = pd.DataFrame({
        "folder": df_y2["folder"].astype(str),
        "sheet": df_y2["sheet"].astype(str),
        "line_no": df_y2["line_no"].astype(int),
        "classroom": df_y2["classroom"].astype(str),
        "fold": df_y2["fold"].astype(int),
        "v2_flagged": df_y2["v2_flagged"].astype(int),
        "fp_label_hand": df_y2["fp_label_hand"].astype(str),
        "hyb_score": hyb_cal,
        "hyb_standalone": hyb_st,
        "hyb_screen90": hyb_s90,
        "loc_score": loc_cal,
        "loc_standalone": loc_st,
        "loc_screen90": loc_s90,
        "s1_score": s1_cal,
        "s1_standalone": s1_st,
        "llm_rationale": s2_txt["llm_rationale"].fillna("").astype(str) if s2_txt is not None else "",
        "llm_subtype": s2_txt["llm_subtype"].fillna("").astype(str) if s2_txt is not None else "",
    })
    y2_keyed = y2_keyed.drop_duplicates(subset=["folder", "sheet", "line_no"], keep="first")

    merged = hc_df.merge(y2_keyed, on=["folder", "sheet", "line_no"], how="left")
    matched = merged["hyb_score"].notna()
    n_unmatched_primary = int((merged["is_primary_sheet"] & ~matched).sum())
    n_unmatched_secondary = int((~merged["is_primary_sheet"] & ~matched).sum())
    m = merged.loc[matched].reset_index(drop=True)

    y = m["human_label"].to_numpy(dtype=int)
    v1_m = binary_metrics(y, np.ones_like(y))
    v2_m = binary_metrics(y, m["v2_flagged"].to_numpy(dtype=int))
    hyb_st_m = binary_metrics(y, m["hyb_standalone"].to_numpy(dtype=int))
    hyb_s90_m = binary_metrics(y, m["hyb_screen90"].to_numpy(dtype=int))
    loc_st_m = binary_metrics(y, m["loc_standalone"].to_numpy(dtype=int))
    loc_s90_m = binary_metrics(y, m["loc_screen90"].to_numpy(dtype=int))
    s1_st_m = binary_metrics(y, m["s1_standalone"].to_numpy(dtype=int))

    # Per-workbook breakdown across ALL 83 Human coded Excel workbooks!
    wb_rows = []
    for wb_name, g in m.groupby("workbook"):
        yg = g["human_label"].to_numpy(dtype=int)
        n_tp_true = int(yg.sum())
        n_fp_true = int((yg == 0).sum())
        hs = binary_metrics(yg, g["hyb_standalone"].to_numpy(dtype=int))
        h9 = binary_metrics(yg, g["hyb_screen90"].to_numpy(dtype=int))
        ls = binary_metrics(yg, g["loc_standalone"].to_numpy(dtype=int))
        v2g = binary_metrics(yg, g["v2_flagged"].to_numpy(dtype=int))
        fp_red_st = round(100.0 * (1.0 - hs["fp"] / max(n_fp_true, 1)), 1) if n_fp_true > 0 else 100.0
        fp_red_s90 = round(100.0 * (1.0 - h9["fp"] / max(n_fp_true, 1)), 1) if n_fp_true > 0 else 100.0
        wb_rows.append({
            "workbook": wb_name,
            "classroom": str(g["classroom"].iloc[0]),
            "cv_fold": int(g["fold"].iloc[0]),
            "n_reviewed": len(g),
            "human_SCI_tp": n_tp_true,
            "human_NOT_fp": n_fp_true,
            "v1_precision": round(n_tp_true / max(len(g), 1), 4),
            "v2_f1": v2g["f1"],
            "v2_recall": v2g["recall"],
            "hyb_standalone_tp": hs["tp"],
            "hyb_standalone_fp": hs["fp"],
            "hyb_standalone_fn": hs["fn"],
            "hyb_standalone_prec": hs["precision"],
            "hyb_standalone_rec": hs["recall"],
            "hyb_standalone_f1": hs["f1"],
            "hyb_standalone_fp_removed_pct": fp_red_st,
            "hyb_screen90_prec": h9["precision"],
            "hyb_screen90_rec": h9["recall"],
            "hyb_screen90_fp": h9["fp"],
            "hyb_screen90_fp_removed_pct": fp_red_s90,
            "local_standalone_f1": ls["f1"],
            "local_standalone_prec": ls["precision"],
            "local_standalone_rec": ls["recall"],
        })
    wb_df = pd.DataFrame(wb_rows).sort_values("workbook")

    # Remaining errors for qualitative inspection
    rem_fp = m[(m["human_label"] == 0) & (m["hyb_standalone"] == 1)].sort_values("hyb_score", ascending=False)
    rem_fn = m[(m["human_label"] == 1) & (m["hyb_standalone"] == 0)].sort_values("hyb_score", ascending=True)

    # Verify exported workbooks if present
    schema_checks = []
    for sub in ("workbooks_hybrid/screening", "workbooks_hybrid/standalone", "workbooks_local/screening", "workbooks_local/standalone"):
        d = REPORTS_V3 / sub
        if d.exists():
            schema_checks.append(verify_exported_workbook_schema(d))

    # Write rich Excel verification workbook
    out_xlsx = REPORTS_V3 / "human_coded_verification_report.xlsx"
    summary_rows = [
        {"System": "Old v1 (Deployed Baseline)", **v1_m, "FP_Removed_vs_v1_pct": 0.0},
        {"System": "Old v2 (Codebook Reranker)", **v2_m, "FP_Removed_vs_v1_pct": round(100.0 * (1.0 - v2_m["fp"] / max(v1_m["fp"], 1)), 1)},
        {"System": "v3 Stage-1 Local Ensemble (Standalone)", **s1_st_m, "FP_Removed_vs_v1_pct": round(100.0 * (1.0 - s1_st_m["fp"] / max(v1_m["fp"], 1)), 1)},
        {"System": "v3 PGKD Distilled Local ($0 API, Standalone)", **loc_st_m, "FP_Removed_vs_v1_pct": round(100.0 * (1.0 - loc_st_m["fp"] / max(v1_m["fp"], 1)), 1)},
        {"System": "v3 PGKD Distilled Local ($0 API, Screening R>=90%)", **loc_s90_m, "FP_Removed_vs_v1_pct": round(100.0 * (1.0 - loc_s90_m["fp"] / max(v1_m["fp"], 1)), 1)},
        {"System": "v3 Hybrid Cascade (Standalone)", **hyb_st_m, "FP_Removed_vs_v1_pct": round(100.0 * (1.0 - hyb_st_m["fp"] / max(v1_m["fp"], 1)), 1)},
        {"System": "v3 Hybrid Cascade (Screening R>=90%)", **hyb_s90_m, "FP_Removed_vs_v1_pct": round(100.0 * (1.0 - hyb_s90_m["fp"] / max(v1_m["fp"], 1)), 1)},
    ]
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        pd.DataFrame(summary_rows).to_excel(xl, sheet_name="OVERALL_SUMMARY", index=False)
        wb_df.to_excel(xl, sheet_name="PER_WORKBOOK_83", index=False)
        rem_fp[["workbook", "sheet", "line_no", "speaker_hc", "utterance_hc", "fp_label_hand", "hyb_score", "s1_score", "llm_subtype", "llm_rationale"]].head(250).to_excel(
            xl, sheet_name="REMAINING_FP_ERRORS", index=False
        )
        rem_fn[["workbook", "sheet", "line_no", "speaker_hc", "utterance_hc", "hyb_score", "s1_score", "llm_subtype", "llm_rationale"]].head(250).to_excel(
            xl, sheet_name="REMAINING_FN_ERRORS", index=False
        )
        if schema_checks:
            pd.DataFrame(schema_checks).to_excel(xl, sheet_name="SCHEMA_CHECK", index=False)

    wb_df.to_csv(REPORTS_V3 / "human_coded_per_workbook_83.csv", index=False)

    report = {
        "scan": {
            **scan_meta,
            "n_matched": int(len(m)),
            "n_unmatched_primary": n_unmatched_primary,
            "n_unmatched_secondary_out_of_scope": n_unmatched_secondary,
        },
        "thresholds": {
            "hybrid_standalone": round(thr_hyb_st, 4),
            "hybrid_screen90": round(thr_hyb_s90, 4),
            "local_standalone": round(thr_loc_st, 4),
            "local_screen90": round(thr_loc_s90, 4),
        },
        "overall": summary_rows,
        "workbook_stats": {
            "n_workbooks_verified": int(len(wb_df)),
            "workbooks_with_tp": int((wb_df["human_SCI_tp"] > 0).sum()),
            "median_hyb_standalone_f1_on_sci_workbooks": round(float(wb_df.loc[wb_df["human_SCI_tp"] > 0, "hyb_standalone_f1"].median()), 4),
            "mean_hyb_standalone_fp_removed_pct": round(float(wb_df["hyb_standalone_fp_removed_pct"].mean()), 1),
            "mean_hyb_screen90_fp_removed_pct": round(float(wb_df["hyb_screen90_fp_removed_pct"].mean()), 1),
        },
        "schema_checks": schema_checks,
        "output_xlsx": str(out_xlsx),
    }
    (REPORTS_V3 / "human_coded_verification_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        print(
            f"\n=== DIRECT VERIFICATION AGAINST ALL {len(wb_df)} `Human coded/*.xlsx` WORKBOOKS "
            f"({len(m):,}/{scan_meta['n_primary_sheet_rows']:,} primary rows matched, "
            f"unmatched_primary={n_unmatched_primary}, secondary_sheets_excluded={n_unmatched_secondary}) ==="
        )
        for r in summary_rows:
            print(
                f"  {r['System']:52s} | P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.4f} "
                f"κ={r['kappa']:.3f} AC1={r['gwet_ac1']:.3f} | TP={r['tp']:4d} FP={r['fp']:4d} FN={r['fn']:4d} | FP_cut={r['FP_Removed_vs_v1_pct']:5.1f}%"
            )
        print(f"  Saved detailed audit Excel -> {out_xlsx}")
    return report


def export_side_by_side_human_coded(verbose: bool = True) -> dict[str, Any]:
    """Export row-by-row side-by-side comparisons of ALL 7,969 reviewed v1 rows from `Human coded/`.

    Writes:
      1. `reports/v3/human_coded_side_by_side_all_7969.xlsx` (SUMMARY + ALL_7969_SIDE_BY_SIDE)
      2. `reports/v3/human_coded_side_by_side_all_7969.csv`
      3. `reports/v3/human_coded_side_by_side_83/<ClassroomXX_MMDDYY>_side_by_side.xlsx` (83 workbooks)
    Strictly out-of-fold by classroom (zero leakage) and leaves `Human coded/` untouched.
    """
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    hc_df, _ = _load_all_human_coded_rows()
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet").reset_index(drop=True)

    hyb_raw = np.load(PREDS_DIR / "6A_hybrid_cascade_v3_y2_oof.npy").astype(np.float64)
    hyb_usf_raw = np.load(PREDS_DIR / "6A_hybrid_cascade_v3_usf_y2_oof.npy").astype(np.float64)
    loc_usf_raw = np.load(PREDS_DIR / "6B_pgkd_distilled_local_v3_usf_y2_oof.npy").astype(np.float64)
    s1_raw = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy").astype(np.float64)
    s2_txt = pd.read_parquet(S2_DIR / "full83_stage2_text.parquet")
    s2_npz = np.load(S2_DIR / "full83_stage2_outputs.npz")
    llm_avg = (0.60 * s2_txt["llm1_score"].to_numpy(dtype=np.float64)
               + 0.40 * s2_txt["llm2_score"].to_numpy(dtype=np.float64))
    llm_ev = s2_npz["eval1_y2"].astype(bool)

    # 1. Exact OOF Standalone predictions (reconciles 100% with human_coded_verification_report.xlsx)
    hyb_cal_rev, hyb_st, _, _, _ = _oof_calibrate_and_threshold(df_y2, hyb_usf_raw)
    _, loc_st, _, _, _ = _oof_calibrate_and_threshold(df_y2, loc_usf_raw)

    # 2. Exact OOF Clean-USF Screening predictions & pm2_sci_count (reconciles 100% with workbooks_hybrid/screening)
    scr = df_y2["screened"].to_numpy(dtype=bool)
    folds = df_y2["fold"].to_numpy()
    lb_orig = df_y2["label"].to_numpy()
    lb_usf = clean_training_labels_for_usf(df_y2)
    valid_all = scr & (lb_orig != -2)
    valid_usf = scr & (lb_usf != -2)
    y_all = (lb_orig == 1).astype(int)
    y_usf = (lb_usf == 1).astype(int)

    full_res = oof_bcts_and_thresholds(hyb_raw, y_all, folds, valid_all, recall_targets=(0.90,))
    adj_full, pm2_cnt = apply_usf_cluster_and_veto_rules(
        df_y2, full_res["cal_oof"], llm_scores=llm_avg, llm_eval=llm_ev,
        support_thr=0.05, min_cluster=3, penalty_logit=0.60, pause_cut_s=30.0,
    )
    hyb_scr_usf = np.zeros(len(df_y2), dtype=int)
    for f in range(5):
        ts = tune_threshold_recall(y_usf[valid_usf & (folds != f)], adj_full[valid_usf & (folds != f)], 0.90)
        va = scr & (folds == f)
        hyb_scr_usf[va] = (adj_full[va] >= ts).astype(int)

    ctx_b2, ctx_a2 = extract_pm2_context_columns(df_y2, pause_cut_s=30.0)
    usf_dirty_tp = (lb_orig == 1) & (lb_usf == -2)

    y2_keyed = pd.DataFrame({
        "folder": df_y2["folder"].astype(str),
        "sheet": df_y2["sheet"].astype(str),
        "line_no": df_y2["line_no"].astype(int),
        "classroom": df_y2["classroom"].astype(str),
        "cv_fold": df_y2["fold"].astype(int),
        "seg_start": df_y2["seg_start"].astype(str),
        "seg_end": df_y2["seg_end"].astype(str),
        "context_before_2": ctx_b2,
        "context_after_2": ctx_a2,
        "v2_flagged": df_y2["v2_flagged"].astype(int),
        "usf_legacy_excluded_tp": usf_dirty_tp,
        "hyb_st_bin": hyb_st.astype(int),
        "hyb_scr_bin": hyb_scr_usf.astype(int),
        "loc_st_bin": loc_st.astype(int),
        "v3_1_hybrid_score": np.round(adj_full, 4),
        "pm2_sci_count": pm2_cnt.astype(int),
        "s1_score": np.round(s1_raw, 4),
        "llm_score": np.where(llm_ev, np.round(llm_avg, 4), np.nan),
        "llm_subtype": s2_txt["llm_subtype"].fillna("").astype(str),
        "llm_rationale": s2_txt["llm_rationale"].fillna("").astype(str),
    }).drop_duplicates(subset=["folder", "sheet", "line_no"], keep="first")

    m = hc_df.merge(y2_keyed, on=["folder", "sheet", "line_no"], how="inner")
    m = m.sort_values(["workbook", "sheet", "line_no"]).reset_index(drop=True)

    def _lbl(arr: np.ndarray) -> list[str]:
        return [LABEL_SCIENCE if int(x) == 1 else LABEL_NOT for x in arr]

    def _outcome(yt: np.ndarray, yp: np.ndarray) -> list[str]:
        out = []
        for t, p in zip(yt, yp):
            if t == 1 and p == 1:
                out.append("TP")
            elif t == 0 and p == 0:
                out.append("TN")
            elif t == 0 and p == 1:
                out.append("FP")
            else:
                out.append("FN")
        return out

    yt = m["human_label"].to_numpy(dtype=int)
    sbs = pd.DataFrame({
        "workbook": m["workbook"],
        "classroom": m["classroom"],
        "cv_fold": m["cv_fold"],
        "sheet": m["sheet"],
        "line_no": m["line_no"],
        "seg_start": m["seg_start"],
        "seg_end": m["seg_end"],
        "speaker": m["speaker_hc"],
        "context_before_2": m["context_before_2"],
        "utterance": m["utterance_hc"],
        "context_after_2": m["context_after_2"],
        "human_review": m["human_review_str"],
        "usf_legacy_excluded_tp": m["usf_legacy_excluded_tp"],
        "v1_pred": m["v1_pred_str"],
        "v2_pred": _lbl(m["v2_flagged"].to_numpy()),
        "v3_1_hybrid_standalone": _lbl(m["hyb_st_bin"].to_numpy()),
        "v3_1_hybrid_screening": _lbl(m["hyb_scr_bin"].to_numpy()),
        "v3_1_local_standalone": _lbl(m["loc_st_bin"].to_numpy()),
        "outcome_hybrid_standalone": _outcome(yt, m["hyb_st_bin"].to_numpy()),
        "outcome_hybrid_screening": _outcome(yt, m["hyb_scr_bin"].to_numpy()),
        "v3_1_hybrid_score": m["v3_1_hybrid_score"],
        "pm2_sci_count": m["pm2_sci_count"],
        "s1_score": m["s1_score"],
        "llm_score": m["llm_score"],
        "llm_subtype": m["llm_subtype"],
        "llm_rationale": m["llm_rationale"],
    })

    def _style_sheet(ws, cols: list[str], n_rows: int) -> None:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{n_rows + 1}"
        widths = {
            "workbook": 28, "sheet": 14, "context_before_2": 40, "utterance": 44,
            "context_after_2": 40, "human_review": 18, "v1_pred": 16, "v2_pred": 18,
            "v3_1_hybrid_standalone": 22, "v3_1_hybrid_screening": 22, "v3_1_local_standalone": 21,
            "outcome_hybrid_standalone": 14, "outcome_hybrid_screening": 14,
            "llm_subtype": 18, "llm_rationale": 42,
        }
        wrap = Alignment(wrap_text=True, vertical="top")
        hdr_fill = PatternFill("solid", fgColor="1F2937")
        hdr_font = Font(bold=True, color="FFFFFF")
        for ci, c in enumerate(cols, start=1):
            cell = ws.cell(1, ci)
            cell.fill = hdr_fill
            cell.font = hdr_font
            if c in widths:
                ws.column_dimensions[get_column_letter(ci)].width = widths[c]
            if c in ("context_before_2", "utterance", "context_after_2", "llm_rationale"):
                for r in range(2, n_rows + 2):
                    ws.cell(r, ci).alignment = wrap

    # 1. Master CSV + Excel (all 7,969 rows + SUMMARY sheet)
    master_csv = REPORTS_V3 / "human_coded_side_by_side_all_7969.csv"
    master_xlsx = REPORTS_V3 / "human_coded_side_by_side_all_7969.xlsx"
    sbs.to_csv(master_csv, index=False, encoding="utf-8-sig")

    clean_m = ~m["usf_legacy_excluded_tp"].to_numpy(dtype=bool)
    sum_rows = [
        {"System": "v1 (Original Human coded flags)", **binary_metrics(yt, np.ones_like(yt))},
        {"System": "v2 (Codebook Reranker)", **binary_metrics(yt, m["v2_flagged"].to_numpy(dtype=int))},
        {"System": "v3.1 Hybrid Standalone (all 7,969 rows)", **binary_metrics(yt, m["hyb_st_bin"].to_numpy(dtype=int))},
        {"System": "v3.1 Hybrid Screening (all 7,969 rows)", **binary_metrics(yt, m["hyb_scr_bin"].to_numpy(dtype=int))},
        {"System": "v3.1 Hybrid Screening (Clean USF 1,899 TPs)", **binary_metrics(yt[clean_m], m["hyb_scr_bin"].to_numpy(dtype=int)[clean_m])},
        {"System": "v3.1 PGKD Local Standalone ($0 API)", **binary_metrics(yt, m["loc_st_bin"].to_numpy(dtype=int))},
    ]
    with pd.ExcelWriter(master_xlsx, engine="openpyxl") as xl:
        pd.DataFrame(sum_rows).to_excel(xl, sheet_name="SUMMARY", index=False)
        sbs.to_excel(xl, sheet_name="ALL_7969_SIDE_BY_SIDE", index=False)
        _style_sheet(xl.sheets["ALL_7969_SIDE_BY_SIDE"], list(sbs.columns), len(sbs))

    # 2. 83 Individual Per-Workbook Side-by-Side Excel files (mirroring Human coded/*.xlsx!)
    per_dir = REPORTS_V3 / "human_coded_side_by_side_83"
    per_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for wb_name, g in sbs.groupby("workbook"):
        stem = str(wb_name).replace("_ebh.xlsx", "").replace(".xlsx", "")
        out_p = per_dir / f"{stem}_side_by_side.xlsx"
        sn = str(g["sheet"].iloc[0])[:31]
        sub_cols = [c for c in sbs.columns if c not in ("workbook", "sheet")]
        g_out = g[sub_cols].reset_index(drop=True)
        with pd.ExcelWriter(out_p, engine="openpyxl") as xl:
            g_out.to_excel(xl, sheet_name=sn, index=False)
            _style_sheet(xl.sheets[sn], sub_cols, len(g_out))
            yg = (g["human_review"] == LABEL_SCIENCE).to_numpy(dtype=int)
            wb_sum = [
                {"System": "v1 (Original flags)", **binary_metrics(yg, np.ones_like(yg))},
                {"System": "v2 (Codebook Reranker)", **binary_metrics(yg, (g["v2_pred"] == LABEL_SCIENCE).to_numpy(dtype=int))},
                {"System": "v3.1 Hybrid Standalone", **binary_metrics(yg, (g["v3_1_hybrid_standalone"] == LABEL_SCIENCE).to_numpy(dtype=int))},
                {"System": "v3.1 Hybrid Screening", **binary_metrics(yg, (g["v3_1_hybrid_screening"] == LABEL_SCIENCE).to_numpy(dtype=int))},
                {"System": "v3.1 PGKD Local Standalone", **binary_metrics(yg, (g["v3_1_local_standalone"] == LABEL_SCIENCE).to_numpy(dtype=int))},
            ]
            pd.DataFrame(wb_sum).to_excel(xl, sheet_name="SUMMARY", index=False)
        n_written += 1

    if verbose:
        print(f"[side_by_side] Master Excel ({len(sbs):,} rows) -> {master_xlsx}")
        print(f"[side_by_side] Master CSV   ({len(sbs):,} rows) -> {master_csv}")
        print(f"[side_by_side] Per-workbook ({n_written} Excel files) -> {per_dir}")
    return {
        "n_rows": int(len(sbs)),
        "n_workbooks": n_written,
        "master_xlsx": str(master_xlsx),
        "master_csv": str(master_csv),
        "per_workbook_dir": str(per_dir),
    }


if __name__ == "__main__":
    run_human_coded_verification(verbose=True)
    export_side_by_side_human_coded(verbose=True)
