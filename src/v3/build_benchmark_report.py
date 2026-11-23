"""Aggregate all v3 model benchmarks vs. v1 & v2 into unified JSON + Markdown proof reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRICS_DIR = PROJECT_ROOT / "reports" / "v3" / "metrics"
S2_DIR = PROJECT_ROOT / "reports" / "v3" / "stage2_llm"
REPORTS_V3 = PROJECT_ROOT / "reports" / "v3"

MODEL_ORDER = [
    "0A_v1_deployed_baseline",
    "0A_v2_codebook_reranker_replay",
    "0B_tfidf_isolated",
    "0B_tfidf_context",
    "0B_svm_context",
    "1A_fixed_biencoder_minilm",
    "1B_nomic_ctx_logreg",
    "1B_nomic_ctx_mlp",
    "1C_sfr_mistral_ctx_logreg",
    "1C_sfr_mistral_ctx_mlp",
    "1C_fused_nomic_sfr_ctx_logreg",
    "1C_fused_nomic_sfr_ctx_mlp",
    "1D_cross_encoder_minilm",
    "2A_bigru_episode_net",
    "2B_tcn_transformer_net",
    "2C_time_aware_hmm",
    "3A_finetuned_ettin32m_window",
    "3A_finetuned_minilm_window",
    "4A_histgbdt_stacker",
    "4B_logreg_meta_blend",
    "4C_stage1_ensemble",
    "6A_stage2_llm_primary_raw",
    "6A_stage2_llm_dual_avg",
    "6A_hybrid_cascade_v3",
    "6B_pgkd_distilled_bigru",
    "6B_pgkd_distilled_local_v3",
]

EXCLUDED_STALE = {"4D_stage1_ensemble_emq"}

MODEL_ANNOTATIONS = {
    "1A_fixed_biencoder_minilm": "†",
    "1D_cross_encoder_minilm": "†",
    "3A_finetuned_ettin32m_window": "†",
    "4A_histgbdt_stacker": "*",
    "4B_logreg_meta_blend": "*",
}


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def build_report() -> dict[str, Any]:
    # Clean up any orphaned stale EMQ metric file so reports stay spotless
    for stale in EXCLUDED_STALE:
        sp = METRICS_DIR / f"{stale}.json"
        if sp.exists():
            sp.unlink()

    loaded: dict[str, dict[str, Any]] = {}
    for p in sorted(METRICS_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        k = d.get("model", p.stem)
        if k in EXCLUDED_STALE:
            continue
        loaded[k] = d

    ordered_keys = [k for k in MODEL_ORDER if k in loaded] + [k for k in sorted(loaded) if k not in MODEL_ORDER]

    rows: list[dict[str, Any]] = []
    for k in ordered_keys:
        d = loaded[k]
        rf = d.get("reviewed_flags", {})
        st = rf.get("standalone", {})
        s90 = rf.get("screening_r90", {})
        fc = d.get("full_corpus_175k", {})
        y1 = d.get("y1_full_recall_benchmark", {})
        y1_o = y1.get("oracle_max_f1", {})
        y1_t = y1.get("at_y2_standalone_thr", {})
        y1_q = y1.get("y1_session_quantification", {})
        q = d.get("classroom_quantification", {})
        sl = d.get("slices", {})
        rows.append({
            "model": k,
            "annotation": MODEL_ANNOTATIONS.get(k, ""),
            "rev_auroc": rf.get("auroc"),
            "macro_room_auroc": rf.get("macro_classroom_auroc"),
            "rev_pr_auc": rf.get("pr_auc"),
            "rev_f1": st.get("f1"),
            "rev_prec": st.get("precision"),
            "rev_rec": st.get("recall"),
            "rev_kappa": st.get("kappa"),
            "rev_ac1": st.get("gwet_ac1"),
            "scr90_prec": s90.get("precision"),
            "scr90_rec": s90.get("recall"),
            "scr90_fp": s90.get("fp"),
            "full_f1_lb": fc.get("standalone", {}).get("f1"),
            "u0_flag_pct": fc.get("u0_flag_rate_full_standalone_pct"),
            "y1_auroc": y1.get("auroc"),
            "y1_pr_auc": y1.get("pr_auc"),
            "y1_f1_transfer": y1_t.get("f1"),
            "y1_f1_oracle": y1_o.get("f1"),
            "adult_f1": sl.get("adult", {}).get("standalone_rev", {}).get("f1"),
            "child_f1": sl.get("child", {}).get("standalone_rev", {}).get("f1"),
            "short_f1": sl.get("short_1_2", {}).get("standalone_rev", {}).get("f1"),
            "spanish_f1": sl.get("spanish_cue", {}).get("standalone_rev", {}).get("f1"),
            "quant_y2_mean_mae": q.get("calibrated_mean_posterior", {}).get("mae_pct_pts"),
            "quant_y2_mean_r": q.get("calibrated_mean_posterior", {}).get("pearson_r"),
            "quant_y2_cc_mae": q.get("classify_and_count", {}).get("mae_pct_pts"),
            "quant_y1_mean_mae": y1_q.get("calibrated_mean_posterior", {}).get("mae_pct_pts"),
            "quant_y1_mean_r": y1_q.get("calibrated_mean_posterior", {}).get("pearson_r"),
            "quant_y1_cc_mae": y1_q.get("classify_and_count", {}).get("mae_pct_pts"),
        })

    pilot = None
    p_pilot = S2_DIR / "pilot_benchmark.json"
    if p_pilot.exists():
        pilot = json.loads(p_pilot.read_text(encoding="utf-8"))

    hyb_meta = None
    p_meta = S2_DIR / "full83_hybrid_meta.json"
    if p_meta.exists():
        hyb_meta = json.loads(p_meta.read_text(encoding="utf-8"))

    summary = {
        "n_models": len(rows),
        "main_leaderboard": rows,
        "stage2_pilot": pilot,
        "hybrid_meta": hyb_meta,
    }
    REPORTS_V3.mkdir(parents=True, exist_ok=True)
    (REPORTS_V3 / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# TeLI Pre-K Science Talk v3 — End-to-End Model Benchmark Report",
        "",
        "All supervised models are evaluated strictly out-of-fold via **Stratified 5-Fold Classroom GroupKFold** (16 classrooms, 0 classroom overlap across train/val folds) using canonical fold-isolated BCTS calibration and threshold tuning. Flagship models (`4C_stage1_ensemble`, `6A_hybrid_cascade_v3`, `6B_pgkd_distilled_local_v3`) use **zero Level-2 nested-CV stacker weights**.",
        "",
        "> **Evaluation Integrity Disclosures:**",
        "> - **Y2 Reviewed Columns (`Cond. Rev *`)** measure **conditional recall/precision on the 7,969 human-reviewed v1 flags** (`1,984 TP / 5,985 FP`); they are NOT true full-transcript Y2 recall because the 166,975 unflagged turns (`U_0`) were never human-coded (`U0 Flag %` tracks background selectivity, and a blinded 600-row 5-stratum Horvitz-Thompson coder sheet is exported for true Y2 recall).",
        "> - **Y1 Full-Transcript Columns (`Y1 *`)** measure **true unbiased full-transcript recall & F1** across all 40 line-by-line coded Y1 sheets (`7,395` clean turns: 7,394 teacher + 1 child `SCI`; `210 SCI`, 0% used in supervised training). Note: 6 science-rich Y1 sheets (199/210 SCI) were used in the Stage-2 LLM model-selection pilot; all 34 remaining Y1 sheets had zero pilot exposure.",
        "> - `†` = Stage-1-gated candidate evaluation (top 18–22% label-free fold OOF). `*` = diagnostic Level-2 stacker on outer-OOF features (excluded from flagship `4C`/`6A`/`6B`).",
        "",
        "## 1. Main Leaderboard (All Model Families vs. Legacy v1 & v2)",
        "",
        "| Model | Cond. Rev AUROC | Macro-Room AUROC | Cond. Rev PR-AUC | Cond. Rev F1 (P / R) | Kappa / AC1 | Screen R≥90% Prec (FP) | U0 Flag % | Y1 Full AUROC | Y1 PR-AUC | Y1 Transfer F1 | Y1 Oracle F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        label = f"`{r['model']}`{r['annotation']}"
        f1_pr = f"{_fmt(r['rev_f1'])} ({_fmt(r['rev_prec'], 3)} / {_fmt(r['rev_rec'], 3)})"
        k_ac = f"{_fmt(r['rev_kappa'], 3)} / {_fmt(r['rev_ac1'], 3)}" if r["rev_kappa"] is not None else "—"
        s90 = f"{_fmt(r['scr90_prec'], 3)} ({r['scr90_fp']})" if r["scr90_prec"] is not None else "—"
        u0 = f"{r['u0_flag_pct']:.2f}%" if r["u0_flag_pct"] is not None else "—"
        lines.append(
            f"| {label} | {_fmt(r['rev_auroc'])} | {_fmt(r['macro_room_auroc'])} | {_fmt(r['rev_pr_auc'])} | "
            f"{f1_pr} | {k_ac} | {s90} | {u0} | {_fmt(r['y1_auroc'])} | {_fmt(r['y1_pr_auc'])} | "
            f"{_fmt(r['y1_f1_transfer'])} | {_fmt(r['y1_f1_oracle'])} |"
        )

    lines += [
        "",
        "## 2. Subgroup Slices & Classroom/Session Quantification (MAE in percentage points, Pearson r)",
        "",
        "Primary quantification uses **Calibrated Mean Posterior** (`mean(cal_p)` via exact-intercept BCTS). *(EMQ/SLD and PACC are retained in JSON/CSV as diagnostics only because PU-flattened `tpr_soft ≈ 0.28` at ~1% prevalence destabilizes label-shift ratios.)*",
        "",
        "| Model | Adult F1 | Child F1 | Short (1–2w) F1 | Spanish F1 | Y2 Cal. Mean MAE (r) | Y2 CC MAE | Y1 Cal. Mean MAE (r) | Y1 CC MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["adult_f1"] is None:
            continue
        y2m = f"{_fmt(r['quant_y2_mean_mae'], 3)} ({_fmt(r['quant_y2_mean_r'], 3)})" if r["quant_y2_mean_mae"] is not None else "—"
        y1m = f"{_fmt(r['quant_y1_mean_mae'], 3)} ({_fmt(r['quant_y1_mean_r'], 3)})" if r["quant_y1_mean_mae"] is not None else "—"
        lines.append(
            f"| `{r['model']}`{r['annotation']} | {_fmt(r['adult_f1'])} | {_fmt(r['child_f1'])} | {_fmt(r['short_f1'])} | "
            f"{_fmt(r['spanish_f1'])} | {y2m} | {_fmt(r['quant_y2_cc_mae'], 3)} | {y1m} | {_fmt(r['quant_y1_cc_mae'], 3)} |"
        )

    key_fp_models = [k for k in ("0B_tfidf_context", "1C_fused_nomic_sfr_ctx_logreg", "2A_bigru_episode_net", "4C_stage1_ensemble", "6B_pgkd_distilled_local_v3", "6A_hybrid_cascade_v3") if k in loaded]
    if key_fp_models:
        arches = sorted({a for k in key_fp_models for a in loaded[k].get("fp_archetypes_400", {}).keys()})
        if arches:
            lines += [
                "",
                "## 3. FP-400 Hand-Coded False-Positive Archetype Rejection (% Rejected at Full-Transcript Standalone Cutoff)",
                "",
                "| Archetype | N | " + " | ".join(f"`{k}`" for k in key_fp_models) + " |",
                "|---|---:|" + "---:|" * len(key_fp_models),
            ]
            for a in arches:
                n_a = next((loaded[k]["fp_archetypes_400"][a]["n"] for k in key_fp_models if a in loaded[k].get("fp_archetypes_400", {})), 0)
                vals = [
                    f"{loaded[k]['fp_archetypes_400'][a]['rejected_standalone_pct']:.1f}%"
                    if a in loaded[k].get("fp_archetypes_400", {}) else "—"
                    for k in key_fp_models
                ]
                lines.append(f"| `{a}` | {n_a} | " + " | ".join(vals) + " |")

    if pilot and pilot.get("results"):
        lines += [
            "",
            "## 4. Stage 2 Live UF NaviGator LLM Pilot Benchmark (10 Diverse Y2 + 6 Full Y1 Transcripts)",
            "",
            "| Config | Model | Effort | Y2 Rev F1 (P / R) | Y2 AUROC | Y1 Full F1 (P / R) | Y1 AUROC | Parse Err | Est. API Cost |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for cfg_name, info in pilot["results"].items():
            y2m = info.get("y2_rev_at_050", {})
            y1m = info.get("y1_full_at_050", {})
            cost = info.get("cost_usd")
            cost_s = f"\\${cost:.4f}" if isinstance(cost, (int, float)) else "—"
            lines.append(
                f"| `{cfg_name}` | `{info.get('model', 'ensemble')}` | `{info.get('reasoning_effort') or 'std'}` | "
                f"{_fmt(y2m.get('f1'))} ({_fmt(y2m.get('precision'), 3)} / {_fmt(y2m.get('recall'), 3)}) | {_fmt(info.get('y2_rev_auroc'))} | "
                f"{_fmt(y1m.get('f1'))} ({_fmt(y1m.get('precision'), 3)} / {_fmt(y1m.get('recall'), 3)}) | {_fmt(info.get('y1_full_auroc'))} | "
                f"{info.get('n_errors', 0)} | {cost_s} |"
            )

    md_path = REPORTS_V3 / "BENCHMARK_REPORT.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] Wrote {md_path} & benchmark_summary.json ({len(rows)} models)", flush=True)
    return summary


def recompute_all_saved_metrics() -> None:
    """Re-evaluate EVERY saved prediction array under the exact canonical metrics.py pipeline."""
    import numpy as np
    import pandas as pd
    from src.v3.stage1_lexical_and_embeds import DATA_DIR, PREDS_DIR, save_and_evaluate

    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    preds_map = {}
    for p2 in sorted(PREDS_DIR.glob("*_y2_oof.npy")):
        name = p2.name.replace("_y2_oof.npy", "")
        if name in EXCLUDED_STALE:
            continue
        p1 = PREDS_DIR / f"{name}_y1.npy"
        if p1.exists():
            preds_map[name] = (np.load(p2).astype(np.float32), np.load(p1).astype(np.float32))
    print(f"[report] Re-evaluating {len(preds_map)} saved models with canonical exact-intercept BCTS & Y1 quant...", flush=True)
    save_and_evaluate(preds_map, df_y2, df_y1, verbose=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-metrics", action="store_true")
    args = ap.parse_args()
    if args.recompute_metrics:
        recompute_all_saved_metrics()
    build_report()
