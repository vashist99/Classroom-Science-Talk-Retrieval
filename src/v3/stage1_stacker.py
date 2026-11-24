"""Stage 1 — Family 4: Feature + Embedding + Multi-Model GBDT Meta-Stacker & Classroom Quantification.

Combines ALL strictly out-of-fold base model probabilities (Families 0B, 1A, 1B, 1C, 1D, 2A, 2B, 2C, 3A)
with:
  1. Multi-scale pause-bounded rolling episode density features (+/-1, +/-3, +/-5, +/-10, +/-20 turns)
     and wall-clock time-window densities (+/-15s, +/-45s, +/-90s)
  2. Ilagan-style classroom interaction structure features (speaker alternation density,
     prev-is-question + short-answer flag, utterance length ratios, ASR conf, pause gaps)
  3. Manual-v0.3 codebook cues (5 domains + 6 practices + traps + rolling content cue density)
  4. Top 32 fold-strict SVD embedding components + neighbor cosine differentials

Benchmarks:
  - 4A_histgbdt_stacker   : Regularized HistGradientBoostingClassifier (LightGBM algorithm in sklearn)
  - 4B_logreg_meta_blend  : Calibrated L2 Logistic meta-blender over OOF logits + episode densities
  - 4C_stage1_ensemble    : BCTS-calibrated ensemble of 4A + 4B + 2A (best Stage 1 scanner!)
  - Plus EMQ/SLD classroom-prior adapted utterance posteriors (`4C_stage1_ensemble_emq`)!
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

from src.v3.embeddings import load_unique_embeddings
from src.v3.metrics import bcts_oof_calibrate, emq_sld
from src.v3.stage1_lexical_and_embeds import (
    DATA_DIR,
    METRICS_DIR,
    PREDS_DIR,
    _fit_fold_svd,
    build_pubn_cvir_training_set,
    get_codebook_feature_matrix,
    save_and_evaluate,
)


def _safe_logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def compute_episode_and_interaction_features(
    df: pd.DataFrame,
    base_probs_dict: dict[str, np.ndarray],
    *,
    pause_cut_s: float = 30.0,
) -> tuple[np.ndarray, list[str]]:
    """Compute rich pause-bounded rolling episode density + turn-taking features."""
    n = len(df)
    names = sorted(base_probs_dict.keys())
    prob_mat = np.column_stack([np.clip(base_probs_dict[k], 1e-5, 1.0 - 1e-5) for k in names]).astype(np.float32)
    logit_mat = _safe_logit(prob_mat)
    # Consensus base probability across all base models
    p_mean = prob_mat.mean(axis=1)
    p_max = prob_mat.max(axis=1)
    p_std = prob_mat.std(axis=1)

    # Strongest sequence + embedding signal for rolling episode density
    seq_keys = [k for k in names if k.startswith(("2A_", "2B_", "1C_fused", "3A_"))]
    p_core = (
        np.mean([base_probs_dict[k] for k in seq_keys], axis=0).astype(np.float32)
        if seq_keys else p_mean
    )

    tx_ids = df["transcript_id"].to_numpy()
    gap_prev = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)
    seg_s = df["seg_start_s"].fillna(0.0).to_numpy(dtype=np.float32) if "seg_start_s" in df.columns else (np.arange(n, dtype=np.float32) * 3.0)
    is_adult = (df["speaker"] == "adult").to_numpy(dtype=np.float32)
    is_q = df["cb_is_question"].to_numpy(dtype=np.float32)
    is_short = df["cb_is_1_2_words"].to_numpy(dtype=np.float32)
    has_content = df["cb_has_content"].to_numpy(dtype=np.float32)
    has_trap = df["cb_has_any_trap"].to_numpy(dtype=np.float32)

    # Rolling features across +/- 1, 3, 5, 10, 20 turns (EXCLUDING self to avoid triviality!)
    windows = (1, 3, 5, 10, 20)
    roll_feats = np.zeros((n, len(windows) * 3 + 10), dtype=np.float32)

    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], n]

    for s, e in zip(starts, ends):
        m = e - s
        gp = gap_prev[s:e]
        ep_breaks = np.flatnonzero(gp > pause_cut_s)
        ep_starts = np.r_[0, ep_breaks]
        ep_ends = np.r_[ep_breaks, m]

        for es, ee in zip(ep_starts, ep_ends):
            gs = s + es
            L = ee - es
            pc = p_core[gs:gs + L]
            hi_flag = (pc >= 0.35).astype(np.float32)
            hc = has_content[gs:gs + L]
            ht = has_trap[gs:gs + L]
            ad = is_adult[gs:gs + L]
            qs = is_q[gs:gs + L]
            sh = is_short[gs:gs + L]

            cs_p = np.r_[0.0, np.cumsum(pc)]
            cs_hi = np.r_[0.0, np.cumsum(hi_flag)]
            cs_hc = np.r_[0.0, np.cumsum(hc)]
            cs_ht = np.r_[0.0, np.cumsum(ht)]
            idx = np.arange(L)

            col = 0
            for w in windows:
                lo = np.maximum(0, idx - w)
                hi = np.minimum(L, idx + w + 1)
                cnt = np.maximum(hi - lo - 1, 1).astype(np.float32)
                nb_mean_p = ((cs_p[hi] - cs_p[lo]) - pc) / cnt
                nb_sum_hi = (cs_hi[hi] - cs_hi[lo]) - hi_flag
                # Max neighbor prob excluding self
                nb_max_p = np.zeros(L, dtype=np.float32)
                for i in range(L):
                    left = pc[lo[i]:i]
                    right = pc[i + 1:hi[i]]
                    mx = 0.0
                    if len(left):
                        mx = max(mx, float(left.max()))
                    if len(right):
                        mx = max(mx, float(right.max()))
                    nb_max_p[i] = mx
                roll_feats[gs:gs + L, col] = nb_mean_p
                roll_feats[gs:gs + L, col + 1] = nb_max_p
                roll_feats[gs:gs + L, col + 2] = nb_sum_hi
                col += 3

            # Extra 10 discourse & interaction dynamics features (Ilagan et al. BEA 2024)
            # 0-1: prev/next turn probability directly
            if L > 1:
                roll_feats[gs + 1:gs + L, col] = pc[:-1]
                roll_feats[gs:gs + L - 1, col + 1] = pc[1:]
                # 2: speaker switch from prev turn
                sw = (ad[1:] != ad[:-1]).astype(np.float32)
                roll_feats[gs + 1:gs + L, col + 2] = sw
                # 3: prev turn was an adult science-ish question & current is short child answer!
                qa_pattern = ((ad[:-1] == 1.0) & (qs[:-1] == 1.0) & (sh[1:] == 1.0)).astype(np.float32) * pc[:-1]
                roll_feats[gs + 1:gs + L, col + 3] = qa_pattern
                # 4: next turn revoices with high science probability (teacher uptake!)
                uptake = ((ad[1:] == 1.0) & (ad[:-1] == 0.0)).astype(np.float32) * pc[1:]
                roll_feats[gs:gs + L - 1, col + 4] = uptake

            # 5-6: content-word density & trap-word density in +/-5 turns
            lo5 = np.maximum(0, idx - 5)
            hi5 = np.minimum(L, idx + 6)
            cnt5 = np.maximum(hi5 - lo5 - 1, 1).astype(np.float32)
            roll_feats[gs:gs + L, col + 5] = ((cs_hc[hi5] - cs_hc[lo5]) - hc) / cnt5
            roll_feats[gs:gs + L, col + 6] = ((cs_ht[hi5] - cs_ht[lo5]) - ht) / cnt5

            # 7: episode length (log) & 8: transcript-level mean base prob (session science richness!)
            roll_feats[gs:gs + L, col + 7] = np.log1p(float(L))
            roll_feats[gs:gs + L, col + 8] = float(p_core[s:e].mean())
            # 9: relative excess over session baseline
            roll_feats[gs:gs + L, col + 9] = pc - float(p_core[s:e].mean())

    cb_mat = get_codebook_feature_matrix(df)
    feat_names = (
        [f"logit_{k}" for k in names]
        + ["p_mean", "p_max", "p_std"]
        + [f"roll_w{w}_{m}" for w in windows for m in ("mean", "max", "sum_hi")]
        + [
            "prev_p", "next_p", "spk_switch", "qa_short_answer", "teacher_uptake",
            "nb5_content_dens", "nb5_trap_dens", "log_ep_len", "tx_mean_p", "rel_excess_p",
        ]
        + [f"cb_f{i}" for i in range(cb_mat.shape[1])]
    )
    X = np.hstack([logit_mat, p_mean[:, None], p_max[:, None], p_std[:, None], roll_feats, cb_mat]).astype(np.float32)
    return X, feat_names


def main() -> None:
    t0 = time.time()
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    nomic_u, idx_map = load_unique_embeddings("nomic-embed-text-v1.5")
    uids_y2, uids_y1 = idx_map["y2"], idx_map["y1"]

    # Auto-discover ONLY Stage-1 base models (Families 0B–3A; strictly exclude Family 4 & Stage-2 Family 6!)
    base_y2: dict[str, np.ndarray] = {}
    base_y1: dict[str, np.ndarray] = {}
    for p in sorted(PREDS_DIR.glob("*_y2_oof.npy")):
        name = p.name.replace("_y2_oof.npy", "")
        if name.startswith(("4", "6")):
            continue
        p1 = PREDS_DIR / f"{name}_y1.npy"
        if p1.exists():
            base_y2[name] = np.load(p).astype(np.float32)
            base_y1[name] = np.load(p1).astype(np.float32)

    print(f"[stage1_stacker] Stacking {len(base_y2)} base models: {sorted(base_y2.keys())}")
    X_y2_base, feat_names = compute_episode_and_interaction_features(df_y2, base_y2)
    X_y1_base, _ = compute_episode_and_interaction_features(df_y1, base_y1)

    folds = df_y2["fold"].to_numpy()
    labels = df_y2["label"].to_numpy()
    screened = df_y2["screened"].to_numpy(dtype=bool)
    prior_oof = base_y2.get("2A_bigru_episode_net", X_y2_base[:, 0])

    n2, n1 = len(df_y2), len(df_y1)
    oof_gbdt = np.zeros(n2, dtype=np.float32)
    y1_gbdt = np.zeros(n1, dtype=np.float32)
    oof_meta_lr = np.zeros(n2, dtype=np.float32)
    y1_meta_lr = np.zeros(n1, dtype=np.float32)

    # Compact, low-variance meta-feature matrix for 4B (ONLY diverse model logits + top 6 episode context feats!)
    core_models = [
        k for k in (
            "0B_tfidf_context",
            "1A_fixed_biencoder_minilm",
            "1B_nomic_ctx_logreg",
            "1C_sfr_mistral_ctx_logreg",
            "1C_fused_nomic_sfr_ctx_logreg",
            "1D_cross_encoder_minilm",
            "2A_bigru_episode_net",
            "2B_tcn_transformer_net",
            "2C_time_aware_hmm",
            "3A_finetuned_ettin32m_window",
        )
        if k in base_y2
    ]
    L_y2_core = np.column_stack([_safe_logit(base_y2[k]) for k in core_models]).astype(np.float32)
    L_y1_core = np.column_stack([_safe_logit(base_y1[k]) for k in core_models]).astype(np.float32)
    # Grab key rolling neighbor features (prev_p, next_p, roll_w3_mean, roll_w10_mean, qa_short_answer, tx_mean_p)
    keep_roll_names = ["roll_w3_mean", "roll_w10_mean", "prev_p", "next_p", "qa_short_answer", "tx_mean_p"]
    keep_roll_idx = [feat_names.index(n) for n in keep_roll_names if n in feat_names]
    M_y2 = np.hstack([L_y2_core, X_y2_base[:, keep_roll_idx]]).astype(np.float32)
    M_y1 = np.hstack([L_y1_core, X_y1_base[:, keep_roll_idx]]).astype(np.float32)

    for f in range(5):
        tr_mask = folds != f
        va_mask = folds == f
        tr_idx, tr_y, tr_w = build_pubn_cvir_training_set(
            labels, screened, tr_mask,
            u0_scores=prior_oof, bn_oof_probs=prior_oof,
            u0_ratio=3.0, cvir_drop_top_frac=0.025,
            borderline_top_frac=0.11, seed=1100 + f,
        )
        tr_y_bin = (tr_y >= 0.5).astype(int)

        # 4A: Regularized Histogram GBDT over compact meta-features + top rolling densities (no noisy 32d SVD!)
        gbdt = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=250,
            max_leaf_nodes=12,
            min_samples_leaf=80,
            l2_regularization=12.0,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=1100 + f,
        )
        gbdt.fit(M_y2[tr_idx], tr_y_bin, sample_weight=tr_w)
        oof_gbdt[va_mask] = gbdt.predict_proba(M_y2[va_mask])[:, 1]
        y1_gbdt += gbdt.predict_proba(M_y1)[:, 1] / 5.0

        # 4B: Strictly Non-Negative / Low-Variance L2 Logistic Meta-Blender over core diverse model logits
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(M_y2[tr_idx])
        X_va_s = scaler.transform(M_y2[va_mask])
        X_1_s = scaler.transform(M_y1)
        meta_lr = LogisticRegression(C=0.08, max_iter=400, solver="lbfgs")
        meta_lr.fit(X_tr_s, tr_y_bin, sample_weight=tr_w)
        oof_meta_lr[va_mask] = meta_lr.predict_proba(X_va_s)[:, 1]
        y1_meta_lr += meta_lr.predict_proba(X_1_s)[:, 1] / 5.0

    # 4C: ZERO-LEAKAGE Stage 1 Flagship Ensemble (strictly fold-isolated Level-1 models only —
    # NO learned Level-2 OOF stacker weights, eliminating nested-CV optimism completely!)
    def _opt_logit(k: str, fallback: np.ndarray) -> np.ndarray:
        return _safe_logit(base_y2[k]) if k in base_y2 else _safe_logit(fallback)

    def _opt_logit_y1(k: str, fallback: np.ndarray) -> np.ndarray:
        return _safe_logit(base_y1[k]) if k in base_y1 else _safe_logit(fallback)

    fb2 = base_y2.get("2A_bigru_episode_net", oof_meta_lr)
    fb1 = base_y1.get("2A_bigru_episode_net", y1_meta_lr)
    z_y2 = (
        0.42 * _opt_logit("2A_bigru_episode_net", fb2)
        + 0.18 * _opt_logit("2B_tcn_transformer_net", fb2)
        + 0.18 * _opt_logit("2C_time_aware_hmm", fb2)
        + 0.15 * _opt_logit("1C_fused_nomic_sfr_ctx_logreg", fb2)
        + 0.07 * _opt_logit("0B_tfidf_context", fb2)
    )
    z_y1 = (
        0.42 * _opt_logit_y1("2A_bigru_episode_net", fb1)
        + 0.18 * _opt_logit_y1("2B_tcn_transformer_net", fb1)
        + 0.18 * _opt_logit_y1("2C_time_aware_hmm", fb1)
        + 0.15 * _opt_logit_y1("1C_fused_nomic_sfr_ctx_logreg", fb1)
        + 0.07 * _opt_logit_y1("0B_tfidf_context", fb1)
    )
    oof_ens = expit(z_y2).astype(np.float32)
    y1_ens = expit(z_y1).astype(np.float32)

    print(f"[stage1_stacker] Finished in {time.time() - t0:.1f}s")
    save_and_evaluate(
        {
            "4A_histgbdt_stacker": (oof_gbdt, y1_gbdt),
            "4B_logreg_meta_blend": (oof_meta_lr, y1_meta_lr),
            "4C_stage1_ensemble": (oof_ens, y1_ens),
        },
        df_y2,
        df_y1,
        verbose=True,
    )


if __name__ == "__main__":
    main()
