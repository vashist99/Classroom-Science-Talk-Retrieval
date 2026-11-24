"""Production CLI for TeLI Science Talk Classification v3.

Supports BOTH confirmed operating cutoffs:
  --mode screening  : High-recall human-review workbook (>=90% recall, cutting FP load sharply)
  --mode standalone : Calibrated max-F1 automated classification + EMQ/PACC classroom-rate report
  --mode both       : Emits workbooks for both modes + classroom quantification summary

Supports BOTH engines:
  --engine hybrid   : Stage-1 Local Conversation Scanner -> Stage-2 Sliding-Window LLM Judge (highest accuracy)
  --engine local    : PGKD Distilled Standalone Local Model ($0 API cost, <2 min for 1.3M turns)

Outputs exact 18-column `Human coded/*.xlsx` workbooks with deterministic `child_base` sheet naming.
"""
from __future__ import annotations

import argparse
import json
import sys
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
    infer_subtype_rule,
)
from src.v3.export_workbooks import LABEL_NOT, LABEL_SCIENCE, write_human_coded_workbook
from src.v3.metrics import (
    apply_bcts,
    emq_sld,
    oof_bcts_and_thresholds,
    pacc_estimate,
    tune_threshold_f1,
    tune_threshold_recall,
)

DATA_DIR = PROJECT_ROOT / "data" / "v3"
PREDS_DIR = PROJECT_ROOT / "reports" / "v3" / "stage1_preds"
S2_DIR = PROJECT_ROOT / "reports" / "v3" / "stage2_llm"
REPORTS_V3 = PROJECT_ROOT / "reports" / "v3"


def export_verified_y2_workbooks(
    *,
    engine: str = "hybrid",
    mode: str = "both",
    folders: list[str] | None = None,
    out_root: Path | None = None,
    science_only: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Export Human-coded-schema workbooks from the evaluated v3 OOF pipeline (zero leakage!)."""
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    s1_p = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy").astype(np.float64)

    if engine == "hybrid":
        model_key = "6A_hybrid_cascade_v3"
        scored_by_tag = "hybrid_v3"
        raw_p = np.load(PREDS_DIR / f"{model_key}_y2_oof.npy").astype(np.float64)
        s2_txt = pd.read_parquet(S2_DIR / "full83_stage2_text.parquet")
        s2_npz = np.load(S2_DIR / "full83_stage2_outputs.npz")
        llm_scores = (0.60 * s2_txt["llm1_score"].to_numpy(dtype=np.float64)
                      + 0.40 * s2_txt["llm2_score"].to_numpy(dtype=np.float64))
        llm_eval = s2_npz["eval1_y2"].astype(bool)
        llm_rats = s2_txt["llm_rationale"].fillna("").astype(str).tolist()
        llm_subs = s2_txt["llm_subtype"].fillna("").astype(str).tolist()
    else:
        model_key = "6B_pgkd_distilled_local_v3"
        scored_by_tag = "local_v3"
        raw_p = np.load(PREDS_DIR / f"{model_key}_y2_oof.npy").astype(np.float64)
        llm_scores = np.full(len(df_y2), np.nan)
        llm_eval = None
        llm_rats = [""] * len(df_y2)
        llm_subs = [""] * len(df_y2)

    # Canonical strictly fold-isolated full-transcript BCTS calibration (zero leakage!)
    df_y2 = df_y2.reset_index(drop=True)
    scr = df_y2["screened"].to_numpy(dtype=bool)
    folds = df_y2["fold"].to_numpy()
    labels = df_y2["label"].to_numpy()
    y_all = np.where(labels == 1, 1, 0)
    valid_all = scr & (labels != -2)

    full_res = oof_bcts_and_thresholds(raw_p, y_all, folds, valid_all, recall_targets=(0.90,))
    cal_p = full_res["cal_oof"]  # Unadjusted exact-intercept BCTS for unbiased classroom rates!

    # Apply USF ±2 chronological cluster rule (-0.60 logit if <3 in ±2) & Stage-2 LLM veto (-3.5 on USF traps)
    # strictly fold-isolated when tuning standalone/screening thresholds!
    adj_p, pm2_cnt = apply_usf_cluster_and_veto_rules(
        df_y2,
        cal_p,
        llm_scores=(llm_scores if engine == "hybrid" else None),
        llm_eval=llm_eval,
        support_thr=0.05,
        min_cluster=3,
        penalty_logit=0.60,
        pause_cut_s=30.0,
    )
    # Tune standalone & screening thresholds strictly OOF on CLEAN USF labels (excluding the 85
    # vetoed legacy shape/count/bare-look TPs so Screening R>=90% doesn't drag cutoff down to recapture traps!)
    lb_usf = clean_training_labels_for_usf(df_y2)
    valid_usf = scr & (lb_usf != -2)
    y_usf = (lb_usf == 1).astype(int)
    thr_stand_fold: dict[int, float] = {}
    thr_screen_fold: dict[int, float] = {}
    for f in range(5):
        tr_v = valid_usf & (folds != f)
        thr_stand_fold[f] = tune_threshold_f1(y_usf[tr_v], adj_p[tr_v])
        thr_screen_fold[f] = tune_threshold_recall(y_usf[tr_v], adj_p[tr_v], 0.90)

    # Extract pause-bounded chronological ±2 context strings from ALL turns (before screening filter!)
    ctx_before_2, ctx_after_2 = extract_pm2_context_columns(df_y2, pause_cut_s=30.0)

    modes_to_run = ["screening", "standalone"] if mode == "both" else [mode]
    out_base = out_root or (REPORTS_V3 / f"workbooks_{engine}")
    summary: dict[str, Any] = {
        "engine": engine,
        "model_key": model_key,
        "mean_thr_screening": round(float(np.mean(list(thr_screen_fold.values()))), 4),
        "mean_thr_standalone": round(float(np.mean(list(thr_stand_fold.values()))), 4),
        "usf_pm2_penalty_logit": 0.60,
        "usf_stage2_trap_veto": bool(engine == "hybrid"),
        "modes": {},
    }

    target_folders = set(folders) if folders else set(df_y2["folder"].unique().tolist())

    for m in modes_to_run:
        m_dir = out_base / m
        written = 0
        total_sci = 0
        for folder_name, g in df_y2[df_y2["screened"] & df_y2["folder"].isin(target_folders)].groupby("folder"):
            f = int(g["fold"].iloc[0])
            thr = thr_screen_fold[f] if m == "screening" else thr_stand_fold[f]
            recs = []
            all_files = sorted(g["source_file"].unique().tolist())
            for gi in g.index:
                sc = float(adj_p[gi])
                lbl = LABEL_SCIENCE if sc >= thr else LABEL_NOT
                sub = llm_subs[gi] if llm_subs[gi] else infer_subtype_rule(str(df_y2["utterance"].iat[gi]))
                recs.append({
                    "source_file": str(df_y2["source_file"].iat[gi]),
                    "child_id": str(df_y2["child_id"].iat[gi]),
                    "line_no": int(df_y2["line_no"].iat[gi]),
                    "seg_start": str(df_y2["seg_start"].iat[gi]),
                    "seg_end": str(df_y2["seg_end"].iat[gi]),
                    "speaker": str(df_y2["speaker"].iat[gi]),
                    "context_before_2": ctx_before_2[gi],
                    "utterance": str(df_y2["utterance"].iat[gi]),
                    "context_after_2": ctx_after_2[gi],
                    "confidence": None if pd.isna(df_y2["asr_conf"].iat[gi]) else round(float(df_y2["asr_conf"].iat[gi]), 3),
                    "predicted_label": lbl,
                    "review": None,
                    "score": round(sc, 4),
                    "pm2_sci_count": int(pm2_cnt[gi]),
                    "scored_by": scored_by_tag,
                    "cosine_score": round(float(s1_p[gi]), 4),
                    "llm_score": None if np.isnan(llm_scores[gi]) else round(float(llm_scores[gi]), 4),
                    "top_match_in_corpus": None,
                    "top_match_score": None,
                    "predicted_subtype": sub,
                    "rationale": llm_rats[gi] if llm_rats[gi] else None,
                    "degraded": False,
                })
                if lbl == LABEL_SCIENCE:
                    total_sci += 1
            path = write_human_coded_workbook(
                str(folder_name), recs, m_dir,
                mode=m, threshold=thr, engine=scored_by_tag,
                science_only=science_only, all_source_files=all_files,
            )
            if path is not None:
                written += 1

        summary["modes"][m] = {
            "output_dir": str(m_dir),
            "workbooks_written": written,
            "total_predicted_science": total_sci,
        }
        if verbose:
            print(f"[cli] Exported {engine}/{m}: {written} workbooks, {total_sci:,} predicted SCIENCE_TALK -> {m_dir}")

    # Also write classroom-level quantification table (Primary: Calibrated Mean Posterior + CC;
    # Secondary diagnostics: EMQ/SLD & PACC — flagged as unstable under PU-flattened 1% rates)
    q_rows = []
    for f in range(5):
        tr = valid_all & (folds != f)
        a_f, b_f = full_res["params_fold"][f]
        tr_cal = apply_bcts(raw_p[tr], a_f, b_f, is_probs=True)
        tpr_s = float(tr_cal[y_all[tr] == 1].mean())
        fpr_s = float(tr_cal[y_all[tr] == 0].mean())
        pi_tr = float(y_all[tr].mean())
        for room, g in df_y2[scr & (folds == f)].groupby("classroom"):
            idx = g.index.to_numpy()
            thr_s = thr_stand_fold[f]
            cc_n = int((adj_p[idx] >= thr_s).sum())
            mp_rate = float(cal_p[idx].mean())
            emq_rate, _ = emq_sld(cal_p[idx], pi_tr)
            pacc_rate = pacc_estimate(cal_p[idx], tpr_s, fpr_s)
            q_rows.append({
                "classroom": room,
                "fold": f,
                "n_screened": len(idx),
                "verified_v1_tp_lower_bound": int(y_all[idx].sum()),
                "verified_v1_pct_lower_bound": round(100.0 * float(y_all[idx].mean()), 2),
                "calibrated_mean_pct_primary": round(100.0 * mp_rate, 2),
                "standalone_cc_n": cc_n,
                "standalone_cc_pct": round(100.0 * cc_n / len(idx), 2),
                "screening_r90_n": int((adj_p[idx] >= thr_screen_fold[f]).sum()),
                "emq_diagnostic_pct": round(100.0 * emq_rate, 2),
                "pacc_diagnostic_pct": round(100.0 * pacc_rate, 2),
            })
    q_df = pd.DataFrame(q_rows).sort_values("classroom")
    q_path = out_base / "classroom_quantification_summary.csv"
    q_path.parent.mkdir(parents=True, exist_ok=True)
    q_df.to_csv(q_path, index=False)
    summary["classroom_quantification_csv"] = str(q_path)
    (out_base / "export_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TeLI Science Talk v3 CLI")
    p.add_argument("--engine", choices=["hybrid", "local"], default="hybrid")
    p.add_argument("--mode", choices=["screening", "standalone", "both"], default="both")
    p.add_argument("--folders", nargs="+", default=None, help="Optional subset of classroom folders")
    p.add_argument("--full-transcript", action="store_true", help="Write all rows instead of science-only")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_verified_y2_workbooks(
        engine=args.engine,
        mode=args.mode,
        folders=args.folders,
        science_only=not args.full_transcript,
        verbose=True,
    )
