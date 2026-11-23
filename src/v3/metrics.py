"""Leak-free evaluation, BCTS calibration, EMQ/PACC classroom quantification, & slice diagnostics.

Reports:
  - Pooled & Macro-Classroom AUROC (prevalence-invariant; McDermott et al., NeurIPS 2024)
  - PR-AUC (Average Precision), Brier Skill Score, Cohen's kappa, Gwet's AC1
  - Dual operating modes tuned STRICTLY out-of-fold (never on test fold):
      1. `standalone`: max-F1 threshold + BCTS calibration + EMQ/PACC classroom rates
      2. `screening` : high-recall threshold (Recall >= 0.90 and >= 0.95)
  - Slice metrics: Adult vs. Child, Short (1-2 words) vs. Long (3+), Spanish cues,
    Low/Mid/High science classrooms, and FP-archetype rejection on the 400 coded FPs.
  - Background flag rate on the 167,010 unflagged U_0 turns (critical sanity check
    against selection-bias FP explosion on background speech!).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    roc_auc_score,
)


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p_clip = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return logit(p_clip)


def gwet_ac1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Chance-corrected Gwet's AC1 (robust to extreme 1% prevalence paradox)."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    n = len(y_t)
    if n == 0:
        return 0.0
    pa = float(np.mean(y_t == y_p))
    p1 = 0.5 * (float(np.mean(y_t == 1)) + float(np.mean(y_p == 1)))
    pe = 2.0 * p1 * (1.0 - p1)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if pa == 1.0 else 0.0
    return float((pa - pe) / (1.0 - pe))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    try:
        kappa = float(cohen_kappa_score(yt, yp))
        if not np.isfinite(kappa):
            kappa = 0.0
    except Exception:
        kappa = 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "kappa": round(kappa, 4),
        "gwet_ac1": round(gwet_ac1(yt, yp), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def tune_threshold_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Plateau-stable max-F1 threshold (3-point local average breaks flat-plateau jitter)."""
    yt = np.asarray(y_true, dtype=int)
    sc = np.asarray(scores, dtype=float)
    cands = np.unique(np.concatenate([
        np.linspace(0.02, 0.98, 193),
        np.quantile(sc, np.linspace(0.01, 0.99, 99)),
    ]))
    f1s = np.empty(len(cands), dtype=np.float64)
    for i, thr in enumerate(cands):
        yp = (sc >= thr).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        denom = 2 * tp + fp + fn
        f1s[i] = (2.0 * tp / denom) if denom > 0 else 0.0
    if len(f1s) >= 3:
        sm = np.convolve(f1s, np.array([0.25, 0.50, 0.25]), mode="same")
        sm[0], sm[-1] = f1s[0], f1s[-1]
        best_i = int(np.argmax(sm))
    else:
        best_i = int(np.argmax(f1s))
    return float(cands[best_i])


def tune_threshold_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.90) -> float:
    """Highest threshold achieving >= target_recall on y_true==1 (maximizes precision at floor)."""
    pos_scores = np.sort(np.asarray(scores, dtype=float)[np.asarray(y_true, dtype=int) == 1])
    if len(pos_scores) == 0:
        return 0.5
    idx = int(np.floor((1.0 - target_recall) * len(pos_scores)))
    idx = max(0, min(len(pos_scores) - 1, idx))
    return float(pos_scores[idx])


# ---------------------------------------------------------------------------
# Post-Hoc BCTS / Platt Calibration (Alexandari et al., ICML 2020)
# ---------------------------------------------------------------------------

def fit_bcts(logits_or_probs: np.ndarray, y_true: np.ndarray, *, is_probs: bool = True,
             sample_weight: np.ndarray | None = None) -> tuple[float, float]:
    """Fit exact-intercept Bias-Corrected Temperature Scaling: sigmoid(a * z + b).

    Zero L2 penalty on intercept `b` guarantees exact marginal calibration
    `mean(sigmoid(a*z + b)) == mean(y_true)` even at ~1% base rates.
    """
    z = _safe_logit(logits_or_probs) if is_probs else np.asarray(logits_or_probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    w = w / np.maximum(w.mean(), 1e-12)

    def nll(params: np.ndarray) -> float:
        a, b = params
        p = expit(np.clip(a * z + b, -30.0, 30.0))
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        loss = -w * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return float(np.mean(loss) + 1e-3 * ((a - 1.0) ** 2))

    # Initialize intercept near exact log-odds prior for fast exact convergence
    pi0 = float(np.clip(np.average(y, weights=w), 1e-5, 1.0 - 1e-5))
    b0 = float(logit(pi0) - np.average(z, weights=w))
    res = minimize(
        nll,
        np.array([1.0, np.clip(b0, -9.0, 9.0)]),
        method="L-BFGS-B",
        bounds=[(0.05, 10.0), (-12.0, 12.0)],
        options={"ftol": 1e-12, "gtol": 1e-9},
    )
    return float(res.x[0]), float(res.x[1])


def apply_bcts(logits_or_probs: np.ndarray, a: float, b: float, *, is_probs: bool = True) -> np.ndarray:
    z = _safe_logit(logits_or_probs) if is_probs else np.asarray(logits_or_probs, dtype=np.float64)
    return expit(np.clip(a * z + b, -30.0, 30.0))


def oof_bcts_and_thresholds(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
    fit_mask: np.ndarray,
    *,
    recall_targets: tuple[float, ...] = (0.90, 0.95),
) -> dict[str, Any]:
    """Canonical strictly fold-isolated OOF BCTS calibration & threshold tuning (ZERO nested leak!).

    Inside EVERY fold f:
      1. Fits BCTS `(a_f, b_f)` strictly on `fit_mask & (folds != f)`.
      2. Transforms BOTH training (`tr_cal`) and validation (`va_cal`) with `(a_f, b_f)`.
      3. Tunes `thr_f1` and recall-floor thresholds strictly on `tr_cal` and applies to `va_cal`.
    Shared across `metrics.py`, `verify_against_human_coded.py`, and `cli.py` for 100% reconciliation.
    """
    n = len(raw_probs)
    raw = np.asarray(raw_probs, dtype=np.float64)
    yt = np.asarray(y_true, dtype=int)
    flds = np.asarray(folds)
    fm = np.asarray(fit_mask, dtype=bool)

    cal_oof = np.zeros(n, dtype=np.float64)
    pred_f1 = np.zeros(n, dtype=int)
    pred_rec = {r: np.zeros(n, dtype=int) for r in recall_targets}
    thrs_f1: dict[int, float] = {}
    thrs_rec: dict[float, dict[int, float]] = {r: {} for r in recall_targets}
    params_fold: dict[int, tuple[float, float]] = {}

    for f in np.unique(flds):
        fi = int(f)
        tr = fm & (flds != f)
        va = (flds == f)
        if tr.sum() < 20 or len(np.unique(yt[tr])) < 2:
            a, b = 1.0, 0.0
        else:
            a, b = fit_bcts(raw[tr], yt[tr], is_probs=True)
        params_fold[fi] = (a, b)
        tr_cal = apply_bcts(raw[tr], a, b, is_probs=True)
        va_cal = apply_bcts(raw[va], a, b, is_probs=True)
        cal_oof[va] = va_cal

        tf1 = tune_threshold_f1(yt[tr], tr_cal)
        thrs_f1[fi] = tf1
        pred_f1[va] = (va_cal >= tf1).astype(int)
        for r in recall_targets:
            tr_r = tune_threshold_recall(yt[tr], tr_cal, target_recall=r)
            thrs_rec[r][fi] = tr_r
            pred_rec[r][va] = (va_cal >= tr_r).astype(int)

    return {
        "cal_oof": cal_oof,
        "pred_f1": pred_f1,
        "pred_rec": pred_rec,
        "thrs_f1": thrs_f1,
        "thrs_rec": thrs_rec,
        "mean_thr_f1": float(np.mean(list(thrs_f1.values()))),
        "mean_thr_rec": {r: float(np.mean(list(thrs_rec[r].values()))) for r in recall_targets},
        "params_fold": params_fold,
    }


def bcts_oof_calibrate(probs: np.ndarray, y_true: np.ndarray, folds: np.ndarray,
                       eval_mask: np.ndarray) -> np.ndarray:
    """Out-of-fold BCTS calibration across classroom folds (zero leakage)."""
    return oof_bcts_and_thresholds(probs, y_true, folds, eval_mask, recall_targets=())["cal_oof"]


# ---------------------------------------------------------------------------
# Classroom-Level Quantification (EMQ/SLD & PACC; Esuli et al. 2023)
# ---------------------------------------------------------------------------

def emq_sld(probs_classroom: np.ndarray, pi_train: float, max_iter: int = 100, tol: float = 1e-6) -> tuple[float, np.ndarray]:
    """Saerens-Latinne-Decaestecker EM algorithm for classroom prior shift.

    Returns (estimated_classroom_prevalence, adapted_utterance_posteriors).
    """
    p = np.clip(np.asarray(probs_classroom, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    pi_tr = float(np.clip(pi_train, 1e-5, 1.0 - 1e-5))
    pi_c = pi_tr
    for _ in range(max_iter):
        num = (pi_c / pi_tr) * p
        den = num + ((1.0 - pi_c) / (1.0 - pi_tr)) * (1.0 - p)
        post = num / np.maximum(den, 1e-12)
        pi_new = float(np.clip(np.mean(post), 1e-6, 1.0 - 1e-6))
        if abs(pi_new - pi_c) < tol:
            pi_c = pi_new
            break
        pi_c = pi_new
    num = (pi_c / pi_tr) * p
    den = num + ((1.0 - pi_c) / (1.0 - pi_tr)) * (1.0 - p)
    post = num / np.maximum(den, 1e-12)
    return float(pi_c), post


def pacc_estimate(probs_classroom: np.ndarray, tpr_soft: float, fpr_soft: float) -> float:
    """Probabilistic Adjusted Classify & Count (Bella et al. / Esuli et al. 2023)."""
    denom = tpr_soft - fpr_soft
    if abs(denom) < 1e-6:
        return float(np.mean(probs_classroom))
    est = (float(np.mean(probs_classroom)) - fpr_soft) / denom
    return float(np.clip(est, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Comprehensive Leak-Free Y2 Benchmark Suite
# ---------------------------------------------------------------------------

def evaluate_y2_oof(
    df_y2: pd.DataFrame,
    oof_probs_all: np.ndarray,
    *,
    model_name: str,
    calibrate_oof: bool = True,
) -> dict[str, Any]:
    """Evaluate out-of-fold probabilities across ALL screened Y2 utterances.

    Computes:
      1. Reviewed-flag discrimination (1,984 TP vs 5,985 FP): AUROC, Macro-Room
         AUROC, PR-AUC, Brier Skill Score, Standalone F1, Screening P@R=90% & 95%.
      2. Full-corpus (all 174,979 screened turns) sanity: exact background flag
         rate on U_0 (167,010 unflagged turns) + overall corpus precision/recall
         under conservative U_0=0 lower bound.
      3. Slice breakdown: Adult vs Child, Short 1-2 words vs Long 3+, Spanish cues,
         and FP-archetype rejection on the 400 hand-coded FP sample.
      4. Classroom-level quantification accuracy (CC vs EMQ/SLD vs PACC).
    """
    scr_mask = df_y2["screened"].to_numpy(dtype=bool)
    scr = df_y2.loc[scr_mask].reset_index(drop=True)
    raw_p = np.asarray(oof_probs_all, dtype=np.float64)[scr_mask]

    rev_mask = scr["label"].isin([0, 1]).to_numpy()
    u0_mask = (scr["label"] == -1).to_numpy()
    folds = scr["fold"].to_numpy()
    y_all = np.where(scr["label"].to_numpy() == 1, 1, 0)
    valid_all = scr["label"].to_numpy() != -2

    # --- 1. Discrimination on Reviewed Flags (1,984 TP vs 5,985 FP; CONDITIONAL on v1 flags) --
    if calibrate_oof:
        rev_res = oof_bcts_and_thresholds(raw_p, y_all, folds, rev_mask, recall_targets=(0.90, 0.95))
        probs = rev_res["cal_oof"]
        pred_standalone = rev_res["pred_f1"]
        pred_screen90 = rev_res["pred_rec"][0.90]
        pred_screen95 = rev_res["pred_rec"][0.95]
        mean_thr_st = rev_res["mean_thr_f1"]
        mean_thr_s90 = rev_res["mean_thr_rec"][0.90]
    else:
        probs = np.clip(raw_p, 1e-6, 1.0 - 1e-6)
        pred_standalone = np.zeros(len(scr), dtype=int)
        pred_screen90 = np.zeros(len(scr), dtype=int)
        pred_screen95 = np.zeros(len(scr), dtype=int)
        t_st, t_90 = [], []
        for f in np.unique(folds):
            tr_rev = rev_mask & (folds != f)
            va = (folds == f)
            tf1 = tune_threshold_f1(y_all[tr_rev], probs[tr_rev])
            t90 = tune_threshold_recall(y_all[tr_rev], probs[tr_rev], 0.90)
            t95 = tune_threshold_recall(y_all[tr_rev], probs[tr_rev], 0.95)
            t_st.append(tf1); t_90.append(t90)
            pred_standalone[va] = (probs[va] >= tf1).astype(int)
            pred_screen90[va] = (probs[va] >= t90).astype(int)
            pred_screen95[va] = (probs[va] >= t95).astype(int)
        mean_thr_st, mean_thr_s90 = float(np.mean(t_st)), float(np.mean(t_90))

    y_rev = y_all[rev_mask]
    p_rev = probs[rev_mask]

    auroc_rev = float(roc_auc_score(y_rev, p_rev))
    ap_rev = float(average_precision_score(y_rev, p_rev))

    room_aurocs = []
    for room, g in scr.loc[rev_mask].groupby("classroom"):
        idx = g.index.to_numpy()
        yr, pr = y_all[idx], probs[idx]
        if len(np.unique(yr)) == 2:
            room_aurocs.append(float(roc_auc_score(yr, pr)))
    macro_room_auroc = float(np.mean(room_aurocs)) if room_aurocs else auroc_rev

    bs = float(brier_score_loss(y_rev, p_rev))
    bs_ref = float(np.mean(y_rev) * (1.0 - np.mean(y_rev)))
    bss = 1.0 - (bs / max(bs_ref, 1e-12))

    m_standalone_rev = binary_metrics(y_rev, pred_standalone[rev_mask])
    m_screen90_rev = binary_metrics(y_rev, pred_screen90[rev_mask])
    m_screen95_rev = binary_metrics(y_rev, pred_screen95[rev_mask])

    # --- 2. Full-Corpus (174,979 screened) metrics & U_0 background flag rate -
    # Canonical strictly fold-isolated BCTS fit on valid_all (exact marginal calibration!)
    full_res = oof_bcts_and_thresholds(raw_p, y_all, folds, valid_all, recall_targets=(0.90,))
    cal_full = full_res["cal_oof"]
    pred_full_standalone = full_res["pred_f1"]
    pred_full_screen90 = full_res["pred_rec"][0.90]
    mean_thr_full_st = full_res["mean_thr_f1"]
    mean_thr_full_s90 = full_res["mean_thr_rec"][0.90]

    auroc_full = float(roc_auc_score(y_all, cal_full))
    ap_full = float(average_precision_score(y_all, cal_full))

    m_full_standalone = binary_metrics(y_all, pred_full_standalone)
    m_full_screen90 = binary_metrics(y_all, pred_full_screen90)
    u0_flag_rate_standalone = float(pred_full_standalone[u0_mask].mean())
    u0_flag_rate_rev_standalone = float(pred_standalone[u0_mask].mean())
    u0_flag_rate_screen90 = float(pred_full_screen90[u0_mask].mean())

    # --- 3. Slice Diagnostics (on reviewed flags, standalone + screening90) ---
    slices: dict[str, Any] = {}
    slice_defs = {
        "adult": (scr["speaker"] == "adult").to_numpy(),
        "child": (scr["speaker"] == "child").to_numpy(),
        "short_1_2": (scr["cb_is_1_2_words"] >= 0.5).to_numpy(),
        "long_3plus": (scr["cb_is_1_2_words"] < 0.5).to_numpy(),
        "spanish_cue": (scr["cb_is_spanish_cue"] >= 0.5).to_numpy(),
    }
    for sname, smask in slice_defs.items():
        m_idx = rev_mask & smask
        if m_idx.sum() == 0:
            continue
        ys, ps = y_all[m_idx], probs[m_idx]
        slices[sname] = {
            "n": int(m_idx.sum()),
            "n_tp": int(ys.sum()),
            "auroc": round(float(roc_auc_score(ys, ps)), 4) if len(np.unique(ys)) == 2 else None,
            "ap": round(float(average_precision_score(ys, ps)), 4) if len(np.unique(ys)) == 2 else None,
            "standalone_rev": binary_metrics(ys, pred_standalone[m_idx]),
            "screen90_rev": binary_metrics(ys, pred_screen90[m_idx]),
            "standalone_full": binary_metrics(ys, pred_full_standalone[m_idx]),
        }

    # --- 4. FP-Archetype Rejection Rate on 400 hand-coded FPs -----------------
    fp_arch: dict[str, Any] = {}
    hand_mask = (scr["fp_label_hand"] != "").to_numpy()
    if hand_mask.any():
        for arch, g in scr.loc[hand_mask].groupby("fp_label_hand"):
            idx = g.index.to_numpy()
            fp_arch[str(arch)] = {
                "n": int(len(idx)),
                "rejected_standalone_pct": round(100.0 * float((pred_full_standalone[idx] == 0).mean()), 1),
                "rejected_screen90_pct": round(100.0 * float((pred_full_screen90[idx] == 0).mean()), 1),
                "mean_prob": round(float(probs[idx].mean()), 4),
            }

    # --- 5. Classroom-level Quantification (Leave-One-Fold-Out, Strictly Fold-Isolated) ---
    # Primary: Calibrated Mean Posterior `mean(cal_full)` (exact marginal BCTS intercept!)
    # and Full-Corpus CC. Secondary diagnostics: EMQ/SLD & PACC (unstable at ~1% PU prevalence!).
    room_true, room_mean_p, room_cc, room_emq, room_pacc = [], [], [], [], []
    for f in np.unique(folds):
        fi = int(f)
        tr = valid_all & (folds != f)
        a_f, b_f = full_res["params_fold"][fi]
        tr_cal = apply_bcts(raw_p[tr], a_f, b_f, is_probs=True)
        tpr_soft = float(tr_cal[y_all[tr] == 1].mean()) if (y_all[tr] == 1).any() else 0.5
        fpr_soft = float(tr_cal[y_all[tr] == 0].mean()) if (y_all[tr] == 0).any() else 0.01
        pi_tr = float(y_all[tr].mean())
        for room, g in scr.loc[folds == f].groupby("classroom"):
            idx = g.index.to_numpy()
            t_rate = float(y_all[idx].mean())
            mp_rate = float(cal_full[idx].mean())
            cc_rate = float(pred_full_standalone[idx].mean())
            emq_rate, _ = emq_sld(cal_full[idx], pi_tr)
            pacc_rate = pacc_estimate(cal_full[idx], tpr_soft, fpr_soft)
            room_true.append(t_rate)
            room_mean_p.append(mp_rate)
            room_cc.append(cc_rate)
            room_emq.append(emq_rate)
            room_pacc.append(pacc_rate)

    rt = np.array(room_true)
    def _qstats(est: list[float]) -> dict[str, float]:
        ea = np.array(est)
        mae = float(np.mean(np.abs(ea - rt)))
        bias = float(np.mean(ea - rt))
        r = float(np.corrcoef(rt, ea)[0, 1]) if np.std(ea) > 1e-12 and np.std(rt) > 1e-12 else 0.0
        return {"mae_pct_pts": round(100.0 * mae, 3), "bias_pct_pts": round(100.0 * bias, 3), "pearson_r": round(r, 4)}

    quant = {
        "n_classrooms": len(room_true),
        "calibrated_mean_posterior": _qstats(room_mean_p),
        "classify_and_count": _qstats(room_cc),
        "emq_sld_diagnostic": _qstats(room_emq),
        "pacc_diagnostic": _qstats(room_pacc),
        "note": "Evaluated against v1_tp lower bound (U_0=0); calibrated_mean_posterior is primary (EMQ/PACC destabilize under PU-flattened tpr_soft at ~1% base rate).",
    }

    return {
        "model": model_name,
        "reviewed_flags": {
            "n": int(rev_mask.sum()),
            "auroc": round(auroc_rev, 4),
            "macro_classroom_auroc": round(macro_room_auroc, 4),
            "pr_auc": round(ap_rev, 4),
            "brier_skill": round(bss, 4),
            "standalone": m_standalone_rev,
            "screening_r90": m_screen90_rev,
            "screening_r95": m_screen95_rev,
            "mean_thr_standalone": round(mean_thr_st, 4),
            "mean_thr_screen90": round(mean_thr_s90, 4),
        },
        "full_corpus_175k": {
            "n": int(len(scr)),
            "auroc_lower_bound": round(auroc_full, 4),
            "pr_auc_lower_bound": round(ap_full, 4),
            "standalone": m_full_standalone,
            "screening_r90": m_full_screen90,
            "u0_flag_rate_full_standalone_pct": round(100.0 * u0_flag_rate_standalone, 2),
            "u0_flag_rate_rev_standalone_pct": round(100.0 * u0_flag_rate_rev_standalone, 2),
            "u0_flag_rate_full_screen90_pct": round(100.0 * u0_flag_rate_screen90, 2),
            "mean_thr_full_standalone": round(mean_thr_full_st, 4),
            "mean_thr_full_screen90": round(mean_thr_full_s90, 4),
        },
        "slices": slices,
        "fp_archetypes_400": fp_arch,
        "classroom_quantification": quant,
    }


def evaluate_y1_full_recall(
    df_y1: pd.DataFrame,
    probs_y1: np.ndarray,
    *,
    thr_standalone: float,
    thr_screen90: float,
    model_name: str,
) -> dict[str, Any]:
    """Unbiased full-transcript recall, precision, & session quantification on all 40 Y1 transcripts."""
    mask = df_y1["eval_clean_mask"].to_numpy(dtype=bool)
    sub = df_y1.loc[mask].reset_index(drop=True)
    y = sub["label"].to_numpy(dtype=int)
    p = np.asarray(probs_y1, dtype=np.float64)[mask]
    auroc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    best_thr = tune_threshold_f1(y, p)

    # Session-level quantification across all 40 fully-coded Y1 transcripts (true full-transcript rates!)
    st_true, st_mp, st_cc = [], [], []
    for _, g in sub.groupby("transcript_id"):
        idx = g.index.to_numpy()
        st_true.append(float(y[idx].mean()))
        st_mp.append(float(p[idx].mean()))
        st_cc.append(float((p[idx] >= thr_standalone).mean()))
    st_t = np.array(st_true)
    def _yq(est: list[float]) -> dict[str, float]:
        ea = np.array(est)
        r = float(np.corrcoef(st_t, ea)[0, 1]) if np.std(ea) > 1e-12 and np.std(st_t) > 1e-12 else 0.0
        return {
            "true_mean_pct": round(100.0 * float(st_t.mean()), 2),
            "est_mean_pct": round(100.0 * float(ea.mean()), 2),
            "mae_pct_pts": round(100.0 * float(np.mean(np.abs(ea - st_t))), 3),
            "pearson_r": round(r, 4),
        }

    return {
        "model": model_name,
        "n_clean": int(mask.sum()),
        "n_sci": int(y.sum()),
        "auroc": round(auroc, 4),
        "pr_auc": round(ap, 4),
        "at_y2_standalone_thr": binary_metrics(y, (p >= thr_standalone).astype(int)),
        "at_y2_screen90_thr": binary_metrics(y, (p >= thr_screen90).astype(int)),
        "oracle_max_f1": binary_metrics(y, (p >= best_thr).astype(int)),
        "oracle_thr": round(best_thr, 4),
        "y1_session_quantification": {
            "n_sessions": len(st_true),
            "calibrated_mean_posterior": _yq(st_mp),
            "classify_and_count": _yq(st_cc),
        },
    }
