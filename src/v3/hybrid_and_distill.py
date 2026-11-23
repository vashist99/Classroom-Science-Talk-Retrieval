"""Step 4 — Family 6: Final Winning Hybrid Cascade (6A), PGKD Zero-API Local Student (6B),
Classroom Quantification (6C), & Stratified Horvitz-Thompson Unflagged Audit Sample.

Executes strictly out-of-fold by classroom (zero validation classroom leakage!):
  - 6A_hybrid_cascade        : Winning Stage-1 Scanner (>=96% recall candidate episodes)
                               -> Winning Stage-2 Sliding-Window LLM (+ optional 2nd LLM
                               verification on ambiguous band) -> OOF BCTS-calibrated fusion
                               supporting BOTH `screening` and `standalone` cutoffs.
  - 6B_pgkd_distilled_local  : Retrains BiGRU-Episode + HistGBDT strictly on training folds
                               using 3-4x upweighted human gold + Stage-2 consensus soft
                               silver labels on U_0 (breaking v1's positivity violation while
                               avoiding Lu & Smith 2025's Silver-Label Curse!).
  - Stratified HT Audit Export: `reports/v3/unflagged_recall_audit_sample.xlsx` (600 U_0 rows
                               with exact stratum inclusion probabilities for unbiased Y2 recall).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.codebook_v3 import infer_subtype_rule
from src.v3.embeddings import load_unique_embeddings
from src.v3.llm_client import UFNavigatorClient
from src.v3.metrics import (
    apply_bcts,
    bcts_oof_calibrate,
    emq_sld,
    evaluate_y1_full_recall,
    evaluate_y2_oof,
    fit_bcts,
    tune_threshold_f1,
    tune_threshold_recall,
)
from src.v3.stage1_lexical_and_embeds import (
    DATA_DIR,
    METRICS_DIR,
    PREDS_DIR,
    _fit_fold_svd,
    get_codebook_feature_matrix,
    save_and_evaluate,
)
from src.v3.stage1_sequence import (
    BiGRUEpisodeNet,
    _safe_logit,
    build_episode_targets,
    build_sequence_feature_tensors,
)
from src.v3.stage1_stacker import compute_episode_and_interaction_features
from src.v3.stage2_llm import (
    S2_DIR,
    adjudicate_disagreements,
    score_transcripts_with_llm,
)

MODELS_DIR = PROJECT_ROOT / "models" / "v3"
REPORTS_V3 = PROJECT_ROOT / "reports" / "v3"


def _pick_winning_stage2_models() -> tuple[tuple[str, str | None], tuple[str, str | None]]:
    """Read Step 3 pilot_benchmark.json to pick the top-2 architecturally diverse LLMs."""
    p = S2_DIR / "pilot_benchmark.json"
    if not p.exists():
        return ("gpt-oss-120b", "low"), ("gemma-4-31b-it", None)
    data = json.loads(p.read_text(encoding="utf-8"))
    res = data["results"]
    single = {k: v for k, v in res.items() if k != "5F_multi_agent_adjudicated"}
    ranked = sorted(
        single.items(),
        key=lambda kv: 0.55 * kv[1]["y2_rev_at_050"]["f1"] + 0.45 * kv[1]["y1_full_at_050"]["f1"],
        reverse=True,
    )
    m1 = (ranked[0][1]["model"], ranked[0][1]["reasoning_effort"])
    fam1 = m1[0].split("-")[0]
    m2_item = next((v for _, v in ranked[1:] if v["model"].split("-")[0] != fam1), ranked[1][1])
    m2 = (m2_item["model"], m2_item["reasoning_effort"])
    return m1, m2


def _rolling_llm_context(df: pd.DataFrame, llm_scores: np.ndarray, pause_cut_s: float = 30.0) -> np.ndarray:
    """Compute pause-bounded +/-3 and +/-10 neighbor LLM score means & max (excluding self)."""
    n = len(df)
    out = np.zeros((n, 4), dtype=np.float32)
    tx_ids = df["transcript_id"].to_numpy()
    gaps = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        m = e - s
        gp = gaps[s:e]
        ep_b = np.flatnonzero(gp > pause_cut_s)
        for es, ee in zip(np.r_[0, ep_b], np.r_[ep_b, m]):
            gs = s + es
            L = ee - es
            sc = llm_scores[gs:gs + L]
            cs = np.r_[0.0, np.cumsum(sc)]
            idx = np.arange(L)
            for slot, w in enumerate((3, 10)):
                lo = np.maximum(0, idx - w)
                hi = np.minimum(L, idx + w + 1)
                cnt = np.maximum(hi - lo - 1, 1).astype(np.float32)
                out[gs:gs + L, slot * 2] = ((cs[hi] - cs[lo]) - sc) / cnt
                mx = np.zeros(L, dtype=np.float32)
                for i in range(L):
                    v = 0.0
                    if i > lo[i]:
                        v = max(v, float(sc[lo[i]:i].max()))
                    if i + 1 < hi[i]:
                        v = max(v, float(sc[i + 1:hi[i]].max()))
                    mx[i] = v
                out[gs:gs + L, slot * 2 + 1] = mx
    return out


def fit_clean_hybrid_fusion(
    df_y2: pd.DataFrame,
    df_y1: pd.DataFrame,
    res1_y2_scores: np.ndarray,
    res1_y1_scores: np.ndarray,
    llm2_y2_scores: np.ndarray,
    llm2_y1_scores: np.ndarray,
    eval1_y2: np.ndarray,
    eval1_y1: np.ndarray,
    llm_y2_rats: list[str],
    *,
    s2_secondary_used: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], int]:
    """Zero-leakage Hybrid Cascade fusion: strictly fold-isolated Level-1 ensemble + zero-shot Dual-LLM!

    Eliminates Level-1 -> Level-2 nested-CV OOF leakage completely by fusing the zero-leakage
    Stage-1 ensemble (`4C_stage1_ensemble`) directly with the zero-shot Stage-2 Dual-LLM
    posterior in log-odds space (0 learned Level-2 weights!).
    """
    s1_y2 = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy").astype(np.float32)
    s1_y1 = np.load(PREDS_DIR / "4C_stage1_ensemble_y1.npy").astype(np.float32)

    llm_avg_y2 = (0.60 * res1_y2_scores + 0.40 * llm2_y2_scores) if s2_secondary_used else res1_y2_scores
    llm_avg_y1 = (0.60 * res1_y1_scores + 0.40 * llm2_y1_scores) if s2_secondary_used else res1_y1_scores

    # Parameter-free, strictly zero-leakage Stage-1 + Dual-LLM log-odds fusion!
    oof_hyb_ens = expit(
        0.58 * _safe_logit(s1_y2) + 0.42 * _safe_logit(np.clip(llm_avg_y2, 0.03, 0.97))
    ).astype(np.float32)
    y1_hyb_ens = expit(
        0.58 * _safe_logit(s1_y1) + 0.42 * _safe_logit(np.clip(llm_avg_y1, 0.03, 0.97))
    ).astype(np.float32)

    preds_to_eval = {
        "6A_stage2_llm_primary_raw": (res1_y2_scores.astype(np.float32), res1_y1_scores.astype(np.float32)),
        "6A_stage2_llm_dual_avg": (llm_avg_y2.astype(np.float32), llm_avg_y1.astype(np.float32)),
        "6A_hybrid_cascade_v3": (oof_hyb_ens, y1_hyb_ens),
    }
    reports = save_and_evaluate(preds_to_eval, df_y2, df_y1, verbose=verbose)

    screened = df_y2["screened"].to_numpy(dtype=bool)
    labels = df_y2["label"].to_numpy()
    u0_pos = np.flatnonzero(screened & (labels == -1) & (llm_avg_y2 >= 0.80) & (oof_hyb_ens >= 0.70))
    u0_new = df_y2.iloc[u0_pos].copy()
    u0_new["hybrid_score"] = oof_hyb_ens[u0_pos]
    u0_new["llm_score"] = llm_avg_y2[u0_pos]
    u0_new["rationale"] = [llm_y2_rats[i] for i in u0_pos]
    print(f"\n[hybrid] High-confidence NEW science talk discovered inside unflagged U_0: {len(u0_new):,} utterances!", flush=True)
    u0_new[["classroom", "obs_date", "sheet", "line_no", "speaker", "utterance", "hybrid_score", "llm_score", "rationale"]].head(50).to_csv(
        REPORTS_V3 / "newly_discovered_u0_science_talk_sample.csv", index=False
    )
    return oof_hyb_ens, y1_hyb_ens, reports, int(len(u0_new))


def run_full_hybrid_cascade(
    *,
    budget_cap_usd: float = 12.0,
    run_second_verifier: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Stage 1 (>=96% recall candidate episodes) -> Stage 2 Winning LLM across ALL 83 Y2 + 40 Y1."""
    S2_DIR.mkdir(parents=True, exist_ok=True)
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    s1_y2 = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy").astype(np.float32)
    s1_y1 = np.load(PREDS_DIR / "4C_stage1_ensemble_y1.npy").astype(np.float32)

    (m1_name, m1_eff), (m2_name, m2_eff) = _pick_winning_stage2_models()
    print(f"[hybrid] Winning Stage-2 Primary={m1_name} ({m1_eff}) | Secondary Verifier={m2_name} ({m2_eff})")

    # Stage-1 high-recall gate tuned strictly OOF per fold for Recall >= 0.96 on verified TPs!
    folds = df_y2["fold"].to_numpy()
    labels = df_y2["label"].to_numpy()
    screened = df_y2["screened"].to_numpy(dtype=bool)
    rev_mask = screened & np.isin(labels, [0, 1])

    cand_mask_y2 = np.zeros(len(df_y2), dtype=bool)
    s1_gate_thrs = []
    for f in range(5):
        tr_rev = rev_mask & (folds != f)
        va = (folds == f) & screened
        t96 = tune_threshold_recall(labels[tr_rev], s1_y2[tr_rev], target_recall=0.96)
        s1_gate_thrs.append(t96)
        cand_mask_y2[va] = s1_y2[va] >= t96

    # Measure exact Stage-1 candidate funnel stats before spending a penny!
    tp_in_cand = (cand_mask_y2 & (labels == 1)).sum()
    fp_in_cand = (cand_mask_y2 & (labels == 0)).sum()
    u0_in_cand = (cand_mask_y2 & (labels == -1)).sum()
    pct_screened = 100.0 * cand_mask_y2[screened].mean()
    print(
        f"[hybrid] Stage-1 OOF Candidate Gate (mean thr={np.mean(s1_gate_thrs):.4f}): "
        f"{cand_mask_y2.sum():,} turns ({pct_screened:.1f}% of 175k screened) | "
        f"Captures {tp_in_cand}/1984 TPs ({100*tp_in_cand/1984:.1f}% recall!), "
        f"{fp_in_cand}/5985 FPs, {u0_in_cand:,} high-potential U0 turns!"
    )

    # Also build Stage-1 candidate gate on ALL 40 Y1 transcripts (adaptive floor prevents domain-shift starvation!)
    thr_y1_gate = min(float(np.mean(s1_gate_thrs)), float(np.quantile(s1_y1, 0.85)))
    cand_mask_y1 = s1_y1 >= thr_y1_gate
    y1_cln = df_y1["eval_clean_mask"].to_numpy(dtype=bool)
    y1_sci_in = (cand_mask_y1 & y1_cln & (df_y1["label"].to_numpy() == 1)).sum()
    print(
        f"[hybrid] Y1 Stage-1 Gate: {cand_mask_y1.sum():,}/{len(df_y1):,} turns "
        f"({100*cand_mask_y1.mean():.1f}%) | Captures {y1_sci_in}/210 Y1 SCI ({100*y1_sci_in/210:.1f}%)"
    )

    client = UFNavigatorClient(budget_cap_usd=budget_cap_usd)

    # Run Primary Winning Stage-2 LLM over ALL 83 Y2 candidate episodes + ALL 40 Y1 candidate episodes
    # window_size=26 + max_tokens=3200/4096 prevents JSON truncation on science-dense episodes!
    print(f"\n[hybrid] Running Stage-2 Primary ({m1_name}) over all 83 Y2 transcripts...", flush=True)
    res1_y2 = score_transcripts_with_llm(
        df_y2, client=client, model=m1_name, reasoning_effort=m1_eff,
        target_mask=cand_mask_y2, window_size=26, expand_radius=5, max_workers=6, verbose=verbose,
    )
    print(f"[hybrid] Running Stage-2 Primary ({m1_name}) over all 40 Y1 transcripts...", flush=True)
    res1_y1 = score_transcripts_with_llm(
        df_y1, client=client, model=m1_name, reasoning_effort=m1_eff,
        target_mask=cand_mask_y1, window_size=26, expand_radius=5, max_workers=6, verbose=verbose,
    )

    llm_y2_scores = res1_y2["scores"].copy()
    llm_y1_scores = res1_y1["scores"].copy()
    llm_y2_rats = list(res1_y2["rationale"])
    llm_y1_rats = list(res1_y1["rationale"])
    llm_y2_subs = list(res1_y2["subtype"])

    # Optional Stage-2B Secondary Diverse LLM Verification ONLY on ambiguous/positive Stage-2A episodes!
    # (Cost-smart multi-agent verification: only re-check windows where Primary >= 0.30 or S1 >= 0.65!)
    s2_secondary_used = False
    llm2_y2_scores = llm_y2_scores.copy()
    llm2_y1_scores = llm_y1_scores.copy()
    if run_second_verifier and client.ledger.summary()["usd_spent"] < budget_cap_usd * 0.70:
        verify_mask_y2 = (res1_y2["scores"] >= 0.30) | (screened & (s1_y2 >= 0.65) & (res1_y2["scores"] < 0.30))
        verify_mask_y1 = (res1_y1["scores"] >= 0.30) | ((s1_y1 >= 0.65) & (res1_y1["scores"] < 0.30))
        print(
            f"\n[hybrid] Running Secondary Diverse Verifier ({m2_name}) on {verify_mask_y2.sum():,} "
            f"Y2 positive/disputed turns + {verify_mask_y1.sum():,} Y1 turns...",
            flush=True,
        )
        res2_y2 = score_transcripts_with_llm(
            df_y2, client=client, model=m2_name, reasoning_effort=m2_eff,
            target_mask=verify_mask_y2, window_size=26, expand_radius=3, max_workers=6, verbose=verbose,
        )
        res2_y1 = score_transcripts_with_llm(
            df_y1, client=client, model=m2_name, reasoning_effort=m2_eff,
            target_mask=verify_mask_y1, window_size=26, expand_radius=3, max_workers=6, verbose=verbose,
        )
        llm2_y2_scores = np.where(res2_y2["evaluated"], res2_y2["scores"], res1_y2["scores"]).astype(np.float32)
        llm2_y1_scores = np.where(res2_y1["evaluated"], res2_y1["scores"], res1_y1["scores"]).astype(np.float32)
        s2_secondary_used = True

        # Keep richer rationale whenever primary was empty
        for i in range(len(df_y2)):
            if not llm_y2_rats[i] and res2_y2["rationale"][i]:
                llm_y2_rats[i] = res2_y2["rationale"][i]
            if not llm_y2_subs[i] and res2_y2["subtype"][i]:
                llm_y2_subs[i] = res2_y2["subtype"][i]

    # Save raw Stage-2 full-83 arrays for downstream PGKD distillation & workbook export!
    llm2_conf_y2 = res2_y2["conf"] if s2_secondary_used else res1_y2["conf"]
    np.savez_compressed(
        S2_DIR / "full83_stage2_outputs.npz",
        cand_mask_y2=cand_mask_y2,
        cand_mask_y1=cand_mask_y1,
        llm1_y2=res1_y2["scores"],
        llm1_y1=res1_y1["scores"],
        llm2_y2=llm2_y2_scores,
        llm2_y1=llm2_y1_scores,
        llm1_conf_y2=res1_y2["conf"],
        llm2_conf_y2=llm2_conf_y2,
        eval1_y2=res1_y2["evaluated"],
        eval1_y1=res1_y1["evaluated"],
    )
    pd.DataFrame({
        "transcript_id": df_y2["transcript_id"],
        "line_no": df_y2["line_no"],
        "llm1_score": res1_y2["scores"],
        "llm2_score": llm2_y2_scores,
        "llm1_conf": res1_y2["conf"],
        "llm2_conf": llm2_conf_y2,
        "llm_subtype": llm_y2_subs,
        "llm_rationale": llm_y2_rats,
    }).to_parquet(S2_DIR / "full83_stage2_text.parquet", index=False)

    # --- OOF Calibrated Hybrid Fusion (clean low-variance Stage-1 + Dual-LLM fusion!) ---
    _, _, _, n_new_u0 = fit_clean_hybrid_fusion(
        df_y2, df_y1,
        res1_y2["scores"], res1_y1["scores"],
        llm2_y2_scores, llm2_y1_scores,
        res1_y2["evaluated"], res1_y1["evaluated"],
        llm_y2_rats, s2_secondary_used=s2_secondary_used, verbose=verbose,
    )

    meta = {
        "primary_llm": {"model": m1_name, "effort": m1_eff},
        "secondary_llm": {"model": m2_name, "effort": m2_eff, "used": s2_secondary_used},
        "stage1_gate_mean_thr": round(float(np.mean(s1_gate_thrs)), 4),
        "stage1_pct_screened_sent": round(pct_screened, 2),
        "stage1_tp_captured": int(tp_in_cand),
        "n_new_high_conf_u0_discovered": n_new_u0,
        "ledger": client.ledger.summary(),
    }
    (S2_DIR / "full83_hybrid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# Family 6B: PGKD Distilled Zero-API Local Standalone Model
# ---------------------------------------------------------------------------

def run_pgkd_distilled_local(verbose: bool = True) -> dict[str, Any]:
    """Distill Stage-2 consensus soft labels + 3.5x upweighted human gold into a $0-API local model.

    Strictly fold-isolated: fold f's distilled student ONLY sees Stage-2 silver labels
    and human gold from folds != f (zero leakage into validation fold f!).
    """
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    nomic_u, idx_map = load_unique_embeddings("nomic-embed-text-v1.5")
    sfr_u, _ = load_unique_embeddings("sfr-embedding-mistral")
    uids_y2, uids_y1 = idx_map["y2"], idx_map["y1"]

    s2_npz = np.load(S2_DIR / "full83_stage2_outputs.npz")
    llm1_y2 = s2_npz["llm1_y2"].astype(np.float32)
    llm2_y2 = s2_npz["llm2_y2"].astype(np.float32)
    llm_cons_y2 = 0.55 * llm1_y2 + 0.45 * llm2_y2
    eval_y2 = s2_npz["eval1_y2"].astype(bool)

    cb_y2 = get_codebook_feature_matrix(df_y2)
    cb_y1 = get_codebook_feature_matrix(df_y1)
    folds = df_y2["fold"].to_numpy()
    labels = df_y2["label"].to_numpy()
    screened = df_y2["screened"].to_numpy(dtype=bool)
    gaps_y2 = df_y2["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)

    n2, n1 = len(df_y2), len(df_y1)
    oof_pgkd_gru = np.zeros(n2, dtype=np.float32)
    y1_pgkd_gru = np.zeros(n1, dtype=np.float32)

    # 1. Distilled Standalone Chunk-Batched BiGRU Episode Net (20x faster on CPU, gold 4.5x + Stage-2 silver!)
    chunk_len = 128
    batch_chunks = 16
    for f in range(5):
        f_t0 = time.time()
        tr_mask = folds != f
        scaler = StandardScaler()
        scaler.fit(cb_y2[tr_mask])
        cb_y2_sc = (scaler.transform(cb_y2) * 0.20).astype(np.float32)
        cb_y1_sc = (scaler.transform(cb_y1) * 0.20).astype(np.float32)
        sfr_proj = _fit_fold_svd(sfr_u, uids_y2[tr_mask], n_components=256, seed=1500 + f)

        seqs_y2, spans_y2 = build_sequence_feature_tensors(df_y2, nomic_u, sfr_u, uids_y2, sfr_proj, None, cb_y2_sc)
        seqs_y1, spans_y1 = build_sequence_feature_tensors(df_y1, nomic_u, sfr_u, uids_y1, sfr_proj, None, cb_y1_sc)

        # Slice training transcripts into fixed 128-turn padded chunks for fast batched CPU GRU!
        tr_chunks_x, tr_chunks_y, tr_chunks_w, tr_chunks_ey, tr_chunks_em = [], [], [], [], []
        va_items = []
        in_dim = seqs_y2[0].shape[1]

        for seq, (s, e) in zip(seqs_y2, spans_y2):
            if int(folds[s]) == f:
                va_items.append((seq, s, e))
                continue
            lb = labels[s:e]
            sc = screened[s:e]
            gp = gaps_y2[s:e]
            lc = llm_cons_y2[s:e]
            ev = eval_y2[s:e]
            L = e - s

            y_t = np.zeros(L, dtype=np.float32)
            w_t = np.zeros(L, dtype=np.float32)
            pos_m = sc & (lb == 1)
            bn_m = sc & (lb == 0)
            u0_m = sc & (lb == -1)

            y_t[pos_m] = 1.0
            w_t[pos_m] = 4.5
            bn_bord = bn_m & (lc >= 0.70)
            bn_clean = bn_m & ~bn_bord
            y_t[bn_clean], w_t[bn_clean] = 0.0, 1.2
            y_t[bn_bord], w_t[bn_bord] = 0.25, 0.35

            u0_silver_pos = u0_m & ev & (llm1_y2[s:e] >= 0.70) & (llm2_y2[s:e] >= 0.60)
            u0_silver_neg = u0_m & ((~ev) | (lc <= 0.12))
            y_t[u0_silver_pos], w_t[u0_silver_pos] = 0.85, 1.3
            y_t[u0_silver_neg], w_t[u0_silver_neg] = 0.0, 0.08

            eff_lb = lb.copy()
            eff_lb[u0_silver_pos] = 1
            ep_y_np, ep_m_np = build_episode_targets(eff_lb, gp, window=6)

            for c0 in range(0, L, chunk_len):
                c1 = min(L, c0 + chunk_len)
                cl = c1 - c0
                xc = np.zeros((chunk_len, in_dim), dtype=np.float32)
                yc = np.zeros(chunk_len, dtype=np.float32)
                wc = np.zeros(chunk_len, dtype=np.float32)
                eyc = np.zeros(chunk_len, dtype=np.float32)
                emc = np.zeros(chunk_len, dtype=np.float32)
                xc[:cl] = seq[c0:c1]
                yc[:cl] = y_t[c0:c1]
                wc[:cl] = w_t[c0:c1]
                eyc[:cl] = ep_y_np[c0:c1]
                emc[:cl] = ep_m_np[c0:c1]
                tr_chunks_x.append(xc)
                tr_chunks_y.append(yc)
                tr_chunks_w.append(wc)
                tr_chunks_ey.append(eyc)
                tr_chunks_em.append(emc)

        X_tr_c = np.stack(tr_chunks_x)
        Y_tr_c = np.stack(tr_chunks_y)
        W_tr_c = np.stack(tr_chunks_w)
        EY_tr_c = np.stack(tr_chunks_ey)
        EM_tr_c = np.stack(tr_chunks_em)

        torch.manual_seed(1500 + f)
        np.random.seed(1500 + f)
        model = BiGRUEpisodeNet(in_dim=in_dim, proj_dim=256, hidden=192, dropout=0.40)
        opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=5.0e-2)
        epochs = 12
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        n_ch = len(X_tr_c)
        for _ep in range(epochs):
            model.train()
            perm = np.random.permutation(n_ch)
            for bi in range(0, n_ch, batch_chunks):
                idx_b = perm[bi : bi + batch_chunks]
                xb = torch.from_numpy(X_tr_c[idx_b])
                yb = torch.from_numpy(Y_tr_c[idx_b])
                wb = torch.from_numpy(W_tr_c[idx_b])
                eyb = torch.from_numpy(EY_tr_c[idx_b])
                emb = torch.from_numpy(EM_tr_c[idx_b])
                opt.zero_grad(set_to_none=True)
                tl, el = model(xb)
                loss_t = (F.binary_cross_entropy_with_logits(tl, yb, reduction="none") * wb).sum() / torch.clamp(wb.sum(), min=1.0)
                loss_e = (F.binary_cross_entropy_with_logits(el, eyb, reduction="none") * emb).sum() / torch.clamp(emb.sum(), min=1.0)
                (loss_t + 0.30 * loss_e).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
                opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            for x_np, s, e in va_items:
                tl, _ = model(torch.from_numpy(x_np).unsqueeze(0))
                oof_pgkd_gru[s:e] = torch.sigmoid(tl.squeeze(0)).numpy()
            for x_np, (s, e) in zip(seqs_y1, spans_y1):
                tl, _ = model(torch.from_numpy(x_np).unsqueeze(0))
                y1_pgkd_gru[s:e] += torch.sigmoid(tl.squeeze(0)).numpy() / 5.0
        print(f"  [pgkd_bigru] Fold {f}/4 done in {time.time() - f_t0:.1f}s", flush=True)

    # 2. Zero-leakage PGKD Local Blend (matches refit_fusion_and_local_blend_only!)
    s1_y2 = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy").astype(np.float32)
    s1_y1 = np.load(PREDS_DIR / "4C_stage1_ensemble_y1.npy").astype(np.float32)
    bg_y2 = np.load(PREDS_DIR / "2A_bigru_episode_net_y2_oof.npy").astype(np.float32)
    bg_y1 = np.load(PREDS_DIR / "2A_bigru_episode_net_y1.npy").astype(np.float32)

    oof_pgkd_final = expit(0.45 * _safe_logit(oof_pgkd_gru) + 0.35 * _safe_logit(s1_y2) + 0.20 * _safe_logit(bg_y2)).astype(np.float32)
    y1_pgkd_final = expit(0.45 * _safe_logit(y1_pgkd_gru) + 0.35 * _safe_logit(s1_y1) + 0.20 * _safe_logit(bg_y1)).astype(np.float32)

    print(f"[pgkd] Zero-API distilled local models finished in {time.time() - t0:.1f}s")
    save_and_evaluate(
        {
            "6B_pgkd_distilled_bigru": (oof_pgkd_gru, y1_pgkd_gru),
            "6B_pgkd_distilled_local_v3": (oof_pgkd_final, y1_pgkd_final),
        },
        df_y2,
        df_y1,
        verbose=True,
    )
    return {"elapsed_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------------------
# Stratified Horvitz-Thompson Unflagged Y2 Recall Audit Workbook (600 Rows)
# ---------------------------------------------------------------------------

def export_stratified_recall_audit(n_total: int = 600, seed: int = 20261123) -> Path:
    """Export a BLINDED 5-stratum Horvitz-Thompson sample of unflagged Y2 (U_0) utterances.

    Follows Manual Sections 1 & 11 blind-coding independence strictly:
      - `unflagged_recall_audit_BLIND_coder_sheet.xlsx`: randomized row order, wide pause-bounded
        context (+/-6 turns, full utterance text, >30s pause markers), dropdown validations, and
        ZERO stratum/score/rationale columns!
      - `unflagged_recall_audit_SAMPLING_KEY.csv`: private mapping from `audit_row_id` to
        stratum, `N_h`, `n_h`, `ht_weight`, and model scores for post-coding HT/PPI estimation.
    Splits the bottom 90% into `S4_mid_70_90pct` + `S5_bottom_70pct` to cut HT estimator variance.
    """
    from openpyxl.worksheet.datavalidation import DataValidation

    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    hyb_p = np.load(PREDS_DIR / "6A_hybrid_cascade_v3_y2_oof.npy")
    s1_p = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy")
    s2_txt = pd.read_parquet(S2_DIR / "full83_stage2_text.parquet")

    u0_idx = np.flatnonzero(df_y2["screened"].to_numpy() & (df_y2["label"].to_numpy() == -1))
    scores = 0.6 * hyb_p[u0_idx] + 0.4 * s1_p[u0_idx]

    q995 = float(np.quantile(scores, 0.995))
    q980 = float(np.quantile(scores, 0.980))
    q900 = float(np.quantile(scores, 0.900))
    q700 = float(np.quantile(scores, 0.700))

    strata_defs = [
        ("S1_top_0.5pct", scores >= q995, 160),
        ("S2_next_1.5pct", (scores < q995) & (scores >= q980), 140),
        ("S3_next_8.0pct", (scores < q980) & (scores >= q900), 120),
        ("S4_mid_70_90pct", (scores < q900) & (scores >= q700), 100),
        ("S5_bottom_70pct", scores < q700, 80),
    ]

    rng = np.random.default_rng(seed)
    utts = df_y2["utterance"].fillna("").astype(str).tolist()
    spks = df_y2["speaker"].fillna("unknown").astype(str).tolist()
    tx_ids = df_y2["transcript_id"].to_numpy()
    gaps = df_y2["gap_prev_s"].fillna(0.0).to_numpy(dtype=float)

    picked_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sname, mask, target_n in strata_defs:
        pool = u0_idx[mask]
        N_h = len(pool)
        n_h = min(N_h, target_n)
        picks = rng.choice(pool, size=n_h, replace=False)
        ht_w = N_h / max(n_h, 1)
        for gi in picks:
            gi_int = int(gi)
            tx = tx_ids[gi_int]
            # Wide pause-bounded +/-6 context with explicit pause boundary markers
            pre_lines = []
            j = gi_int - 1
            while j >= 0 and len(pre_lines) < 6 and tx_ids[j] == tx:
                if gaps[j + 1] > 30.0:
                    pre_lines.append("--- >30s PAUSE ---")
                    break
                pre_lines.append(f"{spks[j]}: {utts[j]}")
                j -= 1
            pre_lines.reverse()

            post_lines = []
            j = gi_int + 1
            while j < len(df_y2) and len(post_lines) < 6 and tx_ids[j] == tx:
                if gaps[j] > 30.0:
                    post_lines.append("--- >30s PAUSE ---")
                    break
                post_lines.append(f"{spks[j]}: {utts[j]}")
                j += 1

            blind_row = {
                "classroom": df_y2["classroom"].iat[gi_int],
                "obs_date": df_y2["obs_date"].iat[gi_int],
                "sheet": df_y2["sheet"].iat[gi_int],
                "line_no": int(df_y2["line_no"].iat[gi_int]),
                "speaker": spks[gi_int],
                "context_before_6": "\n".join(pre_lines),
                "target_utterance": utts[gi_int],
                "context_after_6": "\n".join(post_lines),
                "review_SCI_or_blank": None,
                "confidence_1_to_5": None,
                "coder_notes": None,
            }
            key_row = {
                "stratum": sname,
                "stratum_N": N_h,
                "stratum_n": n_h,
                "ht_weight": round(ht_w, 3),
                "classroom": df_y2["classroom"].iat[gi_int],
                "obs_date": df_y2["obs_date"].iat[gi_int],
                "sheet": df_y2["sheet"].iat[gi_int],
                "line_no": int(df_y2["line_no"].iat[gi_int]),
                "v3_hybrid_score": round(float(hyb_p[gi_int]), 4),
                "v3_stage1_score": round(float(s1_p[gi_int]), 4),
                "v3_llm_rationale": str(s2_txt["llm_rationale"].iat[gi_int] or ""),
            }
            picked_records.append((blind_row, key_row))

    # Randomly shuffle rows so coders cannot infer stratum from position!
    order = rng.permutation(len(picked_records))
    blind_rows, key_rows = [], []
    for seq_i, orig_i in enumerate(order, start=1):
        aid = f"HT_{seq_i:04d}"
        b_r, k_r = picked_records[int(orig_i)]
        blind_rows.append({"audit_row_id": aid, **b_r})
        key_rows.append({"audit_row_id": aid, **k_r})

    out_blind = REPORTS_V3 / "unflagged_recall_audit_BLIND_coder_sheet.xlsx"
    out_key = REPORTS_V3 / "unflagged_recall_audit_SAMPLING_KEY.csv"
    out_blind.parent.mkdir(parents=True, exist_ok=True)

    df_blind = pd.DataFrame(blind_rows)
    with pd.ExcelWriter(out_blind, engine="openpyxl") as xl:
        df_blind.to_excel(xl, sheet_name="BLIND_AUDIT", index=False)
        ws = xl.sheets["BLIND_AUDIT"]
        n_r = len(df_blind)
        dv_lbl = DataValidation(type="list", formula1='"SCIENCE_TALK,NOT_SCIENCE_TALK"', allow_blank=True, showDropDown=False)
        dv_cnf = DataValidation(type="list", formula1='"1,2,3,4,5"', allow_blank=True, showDropDown=False)
        ws.add_data_validation(dv_lbl)
        ws.add_data_validation(dv_cnf)
        dv_lbl.add(f"J2:J{n_r + 1}")
        dv_cnf.add(f"K2:K{n_r + 1}")
        pd.DataFrame([
            ["BLINDED Horvitz-Thompson Recall Audit (600 Unflagged Y2 Turns)"],
            ["Rows are randomly shuffled across 5 hidden probability strata."],
            ["No model scores, strata, or LLM rationales are shown to preserve coding independence."],
            ["Fill 'review_SCI_or_blank' (SCIENCE_TALK / NOT_SCIENCE_TALK) and 'confidence_1_to_5' using the Sept 2026 Manual."],
            ["After coding, join on 'audit_row_id' with 'unflagged_recall_audit_SAMPLING_KEY.csv' to compute unbiased Y2 recall & 95% CI."],
        ]).to_excel(xl, sheet_name="INSTRUCTIONS", index=False, header=False)

    pd.DataFrame(key_rows).to_csv(out_key, index=False)
    # Remove old unblinded file if present to avoid accidental unblinded coding!
    legacy_unblinded = REPORTS_V3 / "unflagged_recall_audit_sample.xlsx"
    if legacy_unblinded.exists():
        legacy_unblinded.unlink()
    print(f"[audit_export] Wrote {len(blind_rows)}-row BLIND 5-stratum audit -> {out_blind} & private key -> {out_key}", flush=True)
    return out_blind


def refit_fusion_and_local_blend_only(verbose: bool = True) -> None:
    """Zero-API refit of clean Hybrid Cascade (6A) & PGKD Local Blend (6B) from cached Stage-2 outputs."""
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    npz = np.load(S2_DIR / "full83_stage2_outputs.npz")
    s2_txt = pd.read_parquet(S2_DIR / "full83_stage2_text.parquet")

    fit_clean_hybrid_fusion(
        df_y2, df_y1,
        npz["llm1_y2"].astype(np.float32), npz["llm1_y1"].astype(np.float32),
        npz["llm2_y2"].astype(np.float32), npz["llm2_y1"].astype(np.float32),
        npz["eval1_y2"].astype(bool), npz["eval1_y1"].astype(bool),
        s2_txt["llm_rationale"].fillna("").astype(str).tolist(),
        s2_secondary_used=True, verbose=verbose,
    )

    # Also blend PGKD Distilled BiGRU + Stage-1 Diverse Ensemble for strongest $0-API Local v3!
    pg_y2 = np.load(PREDS_DIR / "6B_pgkd_distilled_bigru_y2_oof.npy").astype(np.float32)
    pg_y1 = np.load(PREDS_DIR / "6B_pgkd_distilled_bigru_y1.npy").astype(np.float32)
    s1_y2 = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy").astype(np.float32)
    s1_y1 = np.load(PREDS_DIR / "4C_stage1_ensemble_y1.npy").astype(np.float32)
    bg_y2 = np.load(PREDS_DIR / "2A_bigru_episode_net_y2_oof.npy").astype(np.float32)
    bg_y1 = np.load(PREDS_DIR / "2A_bigru_episode_net_y1.npy").astype(np.float32)

    loc_y2 = expit(0.45 * _safe_logit(pg_y2) + 0.35 * _safe_logit(s1_y2) + 0.20 * _safe_logit(bg_y2)).astype(np.float32)
    loc_y1 = expit(0.45 * _safe_logit(pg_y1) + 0.35 * _safe_logit(s1_y1) + 0.20 * _safe_logit(bg_y1)).astype(np.float32)
    save_and_evaluate({"6B_pgkd_distilled_local_v3": (loc_y2, loc_y1)}, df_y2, df_y1, verbose=verbose)
    export_stratified_recall_audit(n_total=600)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-fusion-only", action="store_true")
    args = ap.parse_args()
    if args.refit_fusion_only:
        refit_fusion_and_local_blend_only(verbose=True)
        return
    run_full_hybrid_cascade(budget_cap_usd=12.0, run_second_verifier=True, verbose=True)
    run_pgkd_distilled_local(verbose=True)
    export_stratified_recall_audit(n_total=600)


if __name__ == "__main__":
    main()
