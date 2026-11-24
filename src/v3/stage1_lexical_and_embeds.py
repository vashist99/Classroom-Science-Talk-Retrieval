"""Stage 1 — Families 0 & 1: Baselines, Lexical/Codebook Models, Fixed Bi-Encoder, & Multi-Scale Frozen Embeddings.

Implements & evaluates on identical 5-fold Classroom GroupKFold CV + untouched Y1 recall benchmark:
  - 0A_v1_replay              : Historical v1 system replay (24.9% precision on flags)
  - 0A_v2_replay              : Historical v2 deploy_v2 replay (57.4% overlap prec, 20.7% recall)
  - 0B_tfidf_isolated         : Word(1-3)+Char(3-5) TF-IDF + Manual-v0.3 cues + LogReg (isolated)
  - 0B_tfidf_context          : Pause-bounded [-5,+3] contextual TF-IDF + Manual-v0.3 cues + LogReg
  - 0B_svm_context            : Pause-bounded [-5,+3] contextual TF-IDF + Calibrated LinearSVM
  - 1A_fixed_biencoder        : Fixed nomic bi-encoder (fixing biencoder_8.py's 99.94% HN-drop bug!)
  - 1B_nomic_ctx_logreg       : Frozen nomic-embed-v1.5 (classification:) pause-bounded multi-scale context LogReg (PUbN+CVIR)
  - 1B_nomic_ctx_mlp          : Frozen nomic-embed-v1.5 pause-bounded multi-scale context MLP (PUbN+CVIR + asymmetric bN noise softening)
  - 1C_sfr_mistral_ctx_logreg : Frozen sfr-embedding-mistral (4096-d) pause-bounded multi-scale context LogReg (PUbN+CVIR)
  - 1C_sfr_mistral_ctx_mlp    : Frozen sfr-embedding-mistral + nomic fused multi-scale context MLP (PUbN+CVIR + borderline softening)

Saves all OOF predictions (Y2 full 184,351 rows) + Y1 predictions (11,690 rows) under
`reports/v3/stage1_preds/` so Stage 1 Sequence models & GBDT Stackers can inspect/stack cleanly!
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.codebook_v3 import clean_training_labels_for_usf
from src.v3.embeddings import build_pause_bounded_context_matrix, load_unique_embeddings
from src.v3.metrics import (
    apply_bcts,
    evaluate_y1_full_recall,
    evaluate_y2_oof,
    fit_bcts,
)

DATA_DIR = PROJECT_ROOT / "data" / "v3"
PREDS_DIR = PROJECT_ROOT / "reports" / "v3" / "stage1_preds"
METRICS_DIR = PROJECT_ROOT / "reports" / "v3" / "metrics"

torch.set_num_threads(min(10, max(4, torch.get_num_threads())))


def get_codebook_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    # Keep post-BCTS veto flags (`cb_trap_shape_only`, `cb_trap_count_only`, `cb_trap_look_no_content`)
    # out of the linear/sequence feature matrix so masking dirty legacy TPs cannot cause degenerate
    # -infinity weights on zero-positive columns! Those 3 columns power the USF Stage-2 veto layer.
    _VETO_ONLY = {"cb_trap_shape_only", "cb_trap_count_only", "cb_trap_look_no_content"}
    cols = [c for c in df.columns if c.startswith("cb_") and c not in _VETO_ONLY]
    arr = df[cols].to_numpy(dtype=np.float32)
    # Add timing & ASR features cleanly
    gp = np.log1p(np.clip(df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32), 0.0, 600.0))[:, None]
    gn = np.log1p(np.clip(df["gap_next_s"].fillna(0.0).to_numpy(dtype=np.float32), 0.0, 600.0))[:, None]
    p_cut = (df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32) > 30.0).astype(np.float32)[:, None]
    conf = df["asr_conf"].fillna(0.75).to_numpy(dtype=np.float32)[:, None]
    return np.hstack([arr, gp, gn, p_cut, conf])


def build_pause_bounded_window_texts(
    df: pd.DataFrame,
    *,
    before: int = 5,
    after: int = 3,
    pause_cut_s: float = 30.0,
) -> tuple[list[str], list[str]]:
    """Return (target_texts, context_only_texts) respecting transcript & silence bounds."""
    n = len(df)
    utts = df["utterance"].fillna("").astype(str).tolist()
    spks = df["speaker"].fillna("unknown").astype(str).tolist()
    tx_ids = df["transcript_id"].to_numpy()
    gap_prev = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)

    target_texts = [f"{s} {u}" for s, u in zip(spks, utts)]
    ctx_texts = [""] * n

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
            for i in range(L):
                lo = max(0, i - before)
                hi = min(L, i + after + 1)
                parts = []
                for j in range(lo, hi):
                    if j == i:
                        continue
                    parts.append(utts[gs + j])
                ctx_texts[gs + i] = " ".join(parts)
    return target_texts, ctx_texts


# ---------------------------------------------------------------------------
# PUbN + CVIR + Asymmetric Borderline Soft-Target Sample Builder
# ---------------------------------------------------------------------------

def build_pubn_cvir_training_set(
    labels: np.ndarray,
    screened: np.ndarray,
    train_mask: np.ndarray,
    *,
    u0_scores: np.ndarray | None = None,
    bn_oof_probs: np.ndarray | None = None,
    u0_ratio: float = 4.0,
    cvir_drop_top_frac: float = 0.025,
    borderline_top_frac: float = 0.11,
    borderline_soft_y: float = 0.25,
    borderline_weight: float = 0.35,
    u0_weight: float = 0.35,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct debiased (indices, soft_targets, sample_weights) for a training fold.

    Implements:
      - PUbN (Hsieh et al. ICML 2019): keep verified P_biased (y=1) + hard bN_biased
        explicitly balanced against background U_0 so background FP rate stays low.
      - CVIR (Garg et al. NeurIPS 2021): mask out top `cvir_drop_top_frac` (2.5%)
        highest-scoring U_0 turns so missed positives in U_0 are NOT forced to y=0!
      - Asymmetric Borderline Softening (Plank 2022 / RoLT 2021): soften the top
        ~11% highest-OOF-probability rows inside bN_biased (matching the empirically
        measured 11% borderline_science slice) to soft target y=0.25, weight=0.35.
    """
    rng = np.random.default_rng(seed)
    tr_scr = train_mask & screened

    pos_idx = np.flatnonzero(tr_scr & (labels == 1))
    bn_idx = np.flatnonzero(tr_scr & (labels == 0))
    u0_idx = np.flatnonzero(tr_scr & (labels == -1))

    # 1. CVIR masking on U_0: drop top 2.5% suspect positives from negative pool
    if u0_scores is not None and len(u0_idx) > 100 and cvir_drop_top_frac > 0:
        cutoff = np.quantile(u0_scores[u0_idx], 1.0 - cvir_drop_top_frac)
        u0_clean = u0_idx[u0_scores[u0_idx] < cutoff]
    else:
        u0_clean = u0_idx

    n_u0_sample = min(len(u0_clean), int(round(u0_ratio * (len(pos_idx) + len(bn_idx)))))
    u0_pick = rng.choice(u0_clean, size=n_u0_sample, replace=False) if n_u0_sample > 0 else np.array([], dtype=int)

    # 2. Asymmetric borderline softening inside bN_biased
    bn_y = np.zeros(len(bn_idx), dtype=np.float32)
    bn_w = np.ones(len(bn_idx), dtype=np.float32)
    if bn_oof_probs is not None and len(bn_idx) > 50 and borderline_top_frac > 0:
        b_cut = np.quantile(bn_oof_probs[bn_idx], 1.0 - borderline_top_frac)
        is_bord = bn_oof_probs[bn_idx] >= b_cut
        bn_y[is_bord] = borderline_soft_y
        bn_w[is_bord] = borderline_weight

    # 3. Class-balanced positive weighting (keeps probabilities clean for BCTS!)
    pos_y = np.ones(len(pos_idx), dtype=np.float32)
    u0_y = np.zeros(len(u0_pick), dtype=np.float32)
    u0_w = np.full(len(u0_pick), u0_weight, dtype=np.float32)

    total_neg_mass = float(bn_w.sum() + u0_w.sum())
    # Moderate positive upweight (sqrt-balanced avoids overconfident probability saturation)
    pos_w_val = float(np.clip(np.sqrt(total_neg_mass / max(len(pos_idx), 1)) * 1.5, 1.5, 8.0))
    pos_w = np.full(len(pos_idx), pos_w_val, dtype=np.float32)

    all_idx = np.concatenate([pos_idx, bn_idx, u0_pick])
    all_y = np.concatenate([pos_y, bn_y, u0_y])
    all_w = np.concatenate([pos_w, bn_w, u0_w])
    return all_idx, all_y, all_w


# ---------------------------------------------------------------------------
# Family 0A: Historical v1 & v2 Replays
# ---------------------------------------------------------------------------

def run_v1_v2_replays(df_y2: pd.DataFrame) -> list[dict[str, Any]]:
    """Exact baseline metrics of existing v1 and v2 systems on our Y2 dataset."""
    scr = df_y2[df_y2["screened"]].reset_index(drop=True)
    rev = scr[scr["label"].isin([0, 1])]
    y_rev = rev["label"].to_numpy(dtype=int)

    # v1: flagged every row in rev (precision=24.9%, conditional recall=100% by construction)
    v1_flag_all = scr["v1_flagged"].to_numpy(dtype=bool)
    v1_tp = int((scr.loc[v1_flag_all, "label"] == 1).sum())
    v1_fp = int((scr.loc[v1_flag_all, "label"] == 0).sum())

    # v2: reports/y2_scores_codebook_v2 flags
    v2_flag_rev = rev["v2_flagged"].to_numpy(dtype=int)
    v2_tp = int(((y_rev == 1) & (v2_flag_rev == 1)).sum())
    v2_fp = int(((y_rev == 0) & (v2_flag_rev == 1)).sum())
    v2_fn = int(((y_rev == 1) & (v2_flag_rev == 0)).sum())
    v2_prec = v2_tp / max(v2_tp + v2_fp, 1)
    v2_rec = v2_tp / max(v2_tp + v2_fn, 1)
    v2_f1 = 2 * v2_prec * v2_rec / max(v2_prec + v2_rec, 1e-12)
    v2_u0_rate = float(scr.loc[scr["label"] == -1, "v2_flagged"].mean())

    return [
        {
            "model": "0A_v1_deployed_baseline",
            "reviewed_flags": {
                "n": len(rev),
                "auroc": None,
                "pr_auc": None,
                "standalone": {
                    "precision": round(v1_tp / max(v1_tp + v1_fp, 1), 4),
                    "recall": 1.0,
                    "f1": round(2 * (v1_tp / max(v1_tp + v1_fp, 1)) / (1 + v1_tp / max(v1_tp + v1_fp, 1)), 4),
                    "tp": v1_tp, "fp": v1_fp, "fn": 0, "tn": 0,
                },
            },
            "note": "v1 flagged 7,969 reviewed rows (precision 24.9%). Recall=1.0 is conditional on v1 flags only.",
        },
        {
            "model": "0A_v2_codebook_reranker_replay",
            "reviewed_flags": {
                "n": len(rev),
                "auroc": None,
                "pr_auc": None,
                "standalone": {
                    "precision": round(v2_prec, 4),
                    "recall": round(v2_rec, 4),
                    "f1": round(v2_f1, 4),
                    "tp": v2_tp, "fp": v2_fp, "fn": v2_fn, "tn": int(((y_rev == 0) & (v2_flag_rev == 0)).sum()),
                },
            },
            "full_corpus_175k": {
                "u0_flag_rate_full_standalone_pct": round(100.0 * v2_u0_rate, 2),
                "v2_total_flags_on_83_sheets": int(scr["v2_flagged"].sum()),
            },
            "note": "v2 precision jumped to 57.4% on overlap, but 10x pairwise budget starvation dropped 79.3% of verified TPs.",
        },
    ]


# ---------------------------------------------------------------------------
# Family 0B: Lexical + Codebook Baselines (Isolated & Context TF-IDF + LogReg/SVM)
# ---------------------------------------------------------------------------

def run_lexical_models(
    df_y2: pd.DataFrame,
    df_y1: pd.DataFrame,
    *,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Train 0B_tfidf_isolated, 0B_tfidf_context, and 0B_svm_context via 5-fold Classroom CV."""
    t0 = time.time()
    n2, n1 = len(df_y2), len(df_y1)
    folds = df_y2["fold"].to_numpy()
    labels = clean_training_labels_for_usf(df_y2)
    screened = df_y2["screened"].to_numpy(dtype=bool)

    cb_y2 = get_codebook_feature_matrix(df_y2)
    cb_y1 = get_codebook_feature_matrix(df_y1)

    t2, c2 = build_pause_bounded_window_texts(df_y2, before=5, after=3, pause_cut_s=30.0)
    t1, c1 = build_pause_bounded_window_texts(df_y1, before=5, after=3, pause_cut_s=30.0)

    oof_iso = np.zeros(n2, dtype=np.float32)
    oof_ctx = np.zeros(n2, dtype=np.float32)
    oof_svm = np.zeros(n2, dtype=np.float32)
    y1_iso = np.zeros(n1, dtype=np.float32)
    y1_ctx = np.zeros(n1, dtype=np.float32)
    y1_svm = np.zeros(n1, dtype=np.float32)

    for f in range(5):
        tr_mask = folds != f
        va_mask = folds == f
        tr_idx, tr_y, tr_w = build_pubn_cvir_training_set(
            labels, screened, tr_mask, u0_ratio=3.5, seed=100 + f
        )
        tr_y_bin = (tr_y >= 0.5).astype(int)

        # Fit TF-IDF strictly on training fold text (zero leakage!)
        vec_tw = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=35000, sublinear_tf=True)
        vec_tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=25000, sublinear_tf=True)
        vec_cw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=25000, sublinear_tf=True)

        X_tr_tw = vec_tw.fit_transform([t2[i] for i in tr_idx])
        X_tr_tc = vec_tc.fit_transform([t2[i] for i in tr_idx])
        X_tr_cw = vec_cw.fit_transform([c2[i] for i in tr_idx])

        scaler = StandardScaler()
        cb_tr = scaler.fit_transform(cb_y2[tr_idx])
        cb_va = scaler.transform(cb_y2[va_mask])
        cb_1 = scaler.transform(cb_y1)

        va_indices = np.flatnonzero(va_mask)
        X_va_tw = vec_tw.transform([t2[i] for i in va_indices])
        X_va_tc = vec_tc.transform([t2[i] for i in va_indices])
        X_va_cw = vec_cw.transform([c2[i] for i in va_indices])

        X_1_tw = vec_tw.transform(t1)
        X_1_tc = vec_tc.transform(t1)
        X_1_cw = vec_cw.transform(c1)

        # 1) Isolated TF-IDF + Codebook LogReg
        X_tr_iso = sparse.hstack([X_tr_tw, X_tr_tc, sparse.csr_matrix(cb_tr * 0.25)], format="csr")
        X_va_iso = sparse.hstack([X_va_tw, X_va_tc, sparse.csr_matrix(cb_va * 0.25)], format="csr")
        X_1_iso = sparse.hstack([X_1_tw, X_1_tc, sparse.csr_matrix(cb_1 * 0.25)], format="csr")

        clf_iso = LogisticRegression(C=1.5, max_iter=500, solver="lbfgs")
        clf_iso.fit(X_tr_iso, tr_y_bin, sample_weight=tr_w)
        oof_iso[va_mask] = clf_iso.predict_proba(X_va_iso)[:, 1]
        y1_iso += clf_iso.predict_proba(X_1_iso)[:, 1] / 5.0

        # 2) Context-Window TF-IDF + Codebook LogReg
        X_tr_ctx = sparse.hstack([X_tr_tw, X_tr_tc, X_tr_cw * 0.85, sparse.csr_matrix(cb_tr * 0.25)], format="csr")
        X_va_ctx = sparse.hstack([X_va_tw, X_va_tc, X_va_cw * 0.85, sparse.csr_matrix(cb_va * 0.25)], format="csr")
        X_1_ctx = sparse.hstack([X_1_tw, X_1_tc, X_1_cw * 0.85, sparse.csr_matrix(cb_1 * 0.25)], format="csr")

        clf_ctx = LogisticRegression(C=1.2, max_iter=500, solver="lbfgs")
        clf_ctx.fit(X_tr_ctx, tr_y_bin, sample_weight=tr_w)
        oof_ctx[va_mask] = clf_ctx.predict_proba(X_va_ctx)[:, 1]
        y1_ctx += clf_ctx.predict_proba(X_1_ctx)[:, 1] / 5.0

        # 3) Context-Window Calibrated LinearSVM (BCTS on train decision scores)
        svm = LinearSVC(C=0.3, max_iter=1500, dual="auto", random_state=42 + f)
        svm.fit(X_tr_ctx, tr_y_bin, sample_weight=tr_w)
        tr_dec = svm.decision_function(X_tr_ctx)
        a_s, b_s = fit_bcts(tr_dec, tr_y_bin, is_probs=False, sample_weight=tr_w)
        oof_svm[va_mask] = apply_bcts(svm.decision_function(X_va_ctx), a_s, b_s, is_probs=False)
        y1_svm += apply_bcts(svm.decision_function(X_1_ctx), a_s, b_s, is_probs=False) / 5.0

    if verbose:
        print(f"[stage1] Lexical models (0B) finished in {time.time() - t0:.1f}s")
    return {
        "0B_tfidf_isolated": (oof_iso, y1_iso),
        "0B_tfidf_context": (oof_ctx, y1_ctx),
        "0B_svm_context": (oof_svm, y1_svm),
    }


# ---------------------------------------------------------------------------
# PyTorch Contextual MLP with Soft-Target BCE + Weight Decay + Cosine Schedule
# ---------------------------------------------------------------------------

class ContextMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 384, dropout: float = 0.35) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_context_mlp_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_va: np.ndarray,
    X_y1: np.ndarray,
    *,
    hidden: int = 384,
    epochs: int = 7,
    batch_size: int = 512,
    lr: float = 8e-4,
    dropout: float = 0.50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ContextMLP(X_tr.shape[1], hidden=hidden, dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=8e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    n = len(X_tr)
    model.train()
    for _ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            xb = torch.from_numpy(X_tr[b])
            yb = torch.from_numpy(y_tr[b])
            wb = torch.from_numpy(w_tr[b])
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = (loss_fn(logits, yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    def _predict(X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.from_numpy(X[i : i + 2048])
                out[i : i + 2048] = torch.sigmoid(model(xb)).numpy()
        return out

    return _predict(X_va), _predict(X_y1)


# ---------------------------------------------------------------------------
# Family 1B & 1C: Multi-Scale Frozen Embeddings (Nomic + SFR-Mistral 4096d)
# ---------------------------------------------------------------------------

def _fit_fold_svd(unique_emb: np.ndarray, train_uids: np.ndarray, n_components: int = 256, seed: int = 42) -> np.ndarray:
    """Fit TruncatedSVD strictly on unique embeddings present in the training fold."""
    uniq_tr = np.unique(train_uids)
    # Subsample up to 45k unique training vectors for fast exact randomized SVD on CPU
    if len(uniq_tr) > 45000:
        rng = np.random.default_rng(seed)
        uniq_tr = rng.choice(uniq_tr, size=45000, replace=False)
    svd = TruncatedSVD(n_components=n_components, n_iter=5, random_state=seed)
    svd.fit(unique_emb[uniq_tr].astype(np.float32))
    return svd.components_.T.astype(np.float32)  # (dim, n_components)


def _compute_inner_oof_prior(
    X_unique: np.ndarray,
    uids: np.ndarray,
    labels: np.ndarray,
    screened: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    seed: int = 42,
) -> np.ndarray:
    """Strictly fold-isolated inner-OOF linear probe on folds != outer_fold (ZERO outer leakage!)."""
    n = len(labels)
    inner_probs = np.full(n, 0.05, dtype=np.float32)
    inner_folds = [g for g in range(5) if g != outer_fold]
    X_low = X_unique[:, : min(256, X_unique.shape[1])].astype(np.float32)
    for g in inner_folds:
        in_tr = (folds != outer_fold) & (folds != g)
        in_va = folds == g
        idx, y, w = build_pubn_cvir_training_set(
            labels, screened, in_tr, u0_ratio=3.0, seed=seed + g
        )
        clf = LogisticRegression(C=0.6, max_iter=250, solver="lbfgs")
        clf.fit(X_low[uids[idx]], (y >= 0.5).astype(int), sample_weight=w)
        va_pos = np.flatnonzero(in_va)
        inner_probs[va_pos] = clf.predict_proba(X_low[uids[va_pos]])[:, 1].astype(np.float32)
    return inner_probs


def run_nomic_context_models(
    df_y2: pd.DataFrame,
    df_y1: pd.DataFrame,
    *,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Train 1B (nomic-embed-v1.5) contextual Linear & MLP models with fold-isolated inner-OOF CVIR."""
    n2, n1 = len(df_y2), len(df_y1)
    folds = df_y2["fold"].to_numpy()
    labels = clean_training_labels_for_usf(df_y2)
    screened = df_y2["screened"].to_numpy(dtype=bool)
    cb_y2 = get_codebook_feature_matrix(df_y2)
    cb_y1 = get_codebook_feature_matrix(df_y1)

    nomic_u, idx_map = load_unique_embeddings("nomic-embed-text-v1.5")
    uids_y2 = idx_map["y2"]
    uids_y1 = idx_map["y1"]

    inner_priors = {
        f: _compute_inner_oof_prior(nomic_u, uids_y2, labels, screened, folds, f, seed=180 + 10 * f)
        for f in range(5)
    }

    t0 = time.time()
    X_nom_y2_ctx = build_pause_bounded_context_matrix(df_y2, nomic_u, uids_y2, pause_cut_s=30.0, pca_dim=256)
    X_nom_y1_ctx = build_pause_bounded_context_matrix(df_y1, nomic_u, uids_y1, pause_cut_s=30.0, pca_dim=256)

    oof_nom_lr = np.zeros(n2, dtype=np.float32)
    y1_nom_lr = np.zeros(n1, dtype=np.float32)
    oof_nom_mlp = np.zeros(n2, dtype=np.float32)
    y1_nom_mlp = np.zeros(n1, dtype=np.float32)

    for f in range(5):
        tr_mask = folds != f
        va_mask = folds == f
        tr_idx, tr_y, tr_w = build_pubn_cvir_training_set(
            labels, screened, tr_mask,
            u0_scores=inner_priors[f], bn_oof_probs=inner_priors[f],
            u0_ratio=4.0, seed=200 + f,
        )
        scaler = StandardScaler()
        cb_tr = scaler.fit_transform(cb_y2[tr_idx]).astype(np.float32) * 0.20
        cb_va = scaler.transform(cb_y2[va_mask]).astype(np.float32) * 0.20
        cb_1 = scaler.transform(cb_y1).astype(np.float32) * 0.20

        # Concatenate Matryoshka contextual features (6*256+5 = 1541) + full 768d target + codebook
        X_tr = np.hstack([X_nom_y2_ctx[tr_idx], nomic_u[uids_y2[tr_idx]].astype(np.float32), cb_tr])
        X_va = np.hstack([X_nom_y2_ctx[va_mask], nomic_u[uids_y2[va_mask]].astype(np.float32), cb_va])
        X_1 = np.hstack([X_nom_y1_ctx, nomic_u[uids_y1].astype(np.float32), cb_1])

        lr = LogisticRegression(C=0.5, max_iter=400, solver="lbfgs")
        lr.fit(X_tr, (tr_y >= 0.5).astype(int), sample_weight=tr_w)
        oof_nom_lr[va_mask] = lr.predict_proba(X_va)[:, 1]
        y1_nom_lr += lr.predict_proba(X_1)[:, 1] / 5.0

        p_va, p_1 = _train_context_mlp_fold(X_tr, tr_y, tr_w, X_va, X_1, hidden=384, epochs=12, seed=200 + f)
        oof_nom_mlp[va_mask] = p_va
        y1_nom_mlp += p_1 / 5.0

    if verbose:
        print(f"[stage1] 1B nomic-embed-v1.5 context LogReg + MLP finished in {time.time() - t0:.1f}s")
    return {
        "1B_nomic_ctx_logreg": (oof_nom_lr, y1_nom_lr),
        "1B_nomic_ctx_mlp": (oof_nom_mlp, y1_nom_mlp),
    }


def run_sfr_context_models(
    df_y2: pd.DataFrame,
    df_y1: pd.DataFrame,
    *,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Train 1C (sfr-embedding-mistral 4096d + fused nomic+sfr) with inner-OOF 4096d probe & SVD."""
    n2, n1 = len(df_y2), len(df_y1)
    folds = df_y2["fold"].to_numpy()
    labels = clean_training_labels_for_usf(df_y2)
    screened = df_y2["screened"].to_numpy(dtype=bool)
    cb_y2 = get_codebook_feature_matrix(df_y2)
    cb_y1 = get_codebook_feature_matrix(df_y1)

    nomic_u, idx_map = load_unique_embeddings("nomic-embed-text-v1.5")
    sfr_u, _ = load_unique_embeddings("sfr-embedding-mistral")
    uids_y2 = idx_map["y2"]
    uids_y1 = idx_map["y1"]

    inner_priors = {
        f: _compute_inner_oof_prior(nomic_u, uids_y2, labels, screened, folds, f, seed=180 + 10 * f)
        for f in range(5)
    }
    X_nom_y2_ctx = build_pause_bounded_context_matrix(df_y2, nomic_u, uids_y2, pause_cut_s=30.0, pca_dim=256)
    X_nom_y1_ctx = build_pause_bounded_context_matrix(df_y1, nomic_u, uids_y1, pause_cut_s=30.0, pca_dim=256)

    t1 = time.time()
    oof_sfr_lr = np.zeros(n2, dtype=np.float32)
    y1_sfr_lr = np.zeros(n1, dtype=np.float32)
    oof_sfr_mlp = np.zeros(n2, dtype=np.float32)
    y1_sfr_mlp = np.zeros(n1, dtype=np.float32)
    oof_fused_lr = np.zeros(n2, dtype=np.float32)
    y1_fused_lr = np.zeros(n1, dtype=np.float32)
    oof_fused_mlp = np.zeros(n2, dtype=np.float32)
    y1_fused_mlp = np.zeros(n1, dtype=np.float32)

    for f in range(5):
        tr_mask = folds != f
        va_mask = folds == f
        tr_idx, tr_y, tr_w = build_pubn_cvir_training_set(
            labels, screened, tr_mask,
            u0_scores=inner_priors[f], bn_oof_probs=inner_priors[f],
            u0_ratio=4.0, seed=300 + f,
        )
        # Fold-strict 256-d SVD on SFR-Mistral 4096-d unique training embeddings
        proj = _fit_fold_svd(sfr_u, uids_y2[tr_mask], n_components=256, seed=300 + f)
        X_sfr_y2_ctx = build_pause_bounded_context_matrix(df_y2, sfr_u, uids_y2, pause_cut_s=30.0, proj_matrix=proj)
        X_sfr_y1_ctx = build_pause_bounded_context_matrix(df_y1, sfr_u, uids_y1, pause_cut_s=30.0, proj_matrix=proj)

        # Full 4096-d target-only linear score via INNER 4-classroom-fold OOF (ZERO in-sample overfit!)
        tr_folds = folds[tr_idx]
        d4096_tr = np.zeros((len(tr_idx), 1), dtype=np.float32)
        for g in [x for x in range(5) if x != f]:
            in_tr = tr_folds != g
            in_va = tr_folds == g
            lr_in = LogisticRegression(C=0.25, max_iter=250, solver="lbfgs")
            lr_in.fit(
                sfr_u[uids_y2[tr_idx[in_tr]]].astype(np.float32),
                (tr_y[in_tr] >= 0.5).astype(int),
                sample_weight=tr_w[in_tr],
            )
            d4096_tr[in_va, 0] = lr_in.decision_function(
                sfr_u[uids_y2[tr_idx[in_va]]].astype(np.float32)
            ).astype(np.float32) * 0.25

        lr_4096 = LogisticRegression(C=0.25, max_iter=300, solver="lbfgs")
        lr_4096.fit(sfr_u[uids_y2[tr_idx]].astype(np.float32), (tr_y >= 0.5).astype(int), sample_weight=tr_w)

        def _pred_4096(uids: np.ndarray) -> np.ndarray:
            p = np.empty(len(uids), dtype=np.float32)
            for i in range(0, len(uids), 8192):
                p[i : i + 8192] = lr_4096.decision_function(sfr_u[uids[i : i + 8192]].astype(np.float32))
            return p[:, None]

        d4096_va = _pred_4096(uids_y2[va_mask]) * 0.25
        d4096_1 = _pred_4096(uids_y1) * 0.25

        scaler = StandardScaler()
        cb_tr = scaler.fit_transform(cb_y2[tr_idx]).astype(np.float32) * 0.20
        cb_va = scaler.transform(cb_y2[va_mask]).astype(np.float32) * 0.20
        cb_1 = scaler.transform(cb_y1).astype(np.float32) * 0.20

        X_tr = np.hstack([X_sfr_y2_ctx[tr_idx], d4096_tr, cb_tr])
        X_va = np.hstack([X_sfr_y2_ctx[va_mask], d4096_va, cb_va])
        X_1 = np.hstack([X_sfr_y1_ctx, d4096_1, cb_1])

        lr = LogisticRegression(C=0.25, max_iter=400, solver="lbfgs")
        lr.fit(X_tr, (tr_y >= 0.5).astype(int), sample_weight=tr_w)
        oof_sfr_lr[va_mask] = lr.predict_proba(X_va)[:, 1]
        y1_sfr_lr += lr.predict_proba(X_1)[:, 1] / 5.0

        p_va, p_1 = _train_context_mlp_fold(X_tr, tr_y, tr_w, X_va, X_1, hidden=384, epochs=7, lr=8e-4, dropout=0.50, seed=300 + f)
        oof_sfr_mlp[va_mask] = p_va
        y1_sfr_mlp += p_1 / 5.0

        # Fused Nomic + SFR-Mistral Context Linear & MLP (complementary representation spaces!)
        X_tr_f = np.hstack([X_tr, X_nom_y2_ctx[tr_idx], nomic_u[uids_y2[tr_idx]].astype(np.float32)])
        X_va_f = np.hstack([X_va, X_nom_y2_ctx[va_mask], nomic_u[uids_y2[va_mask]].astype(np.float32)])
        X_1_f = np.hstack([X_1, X_nom_y1_ctx, nomic_u[uids_y1].astype(np.float32)])

        lr_f = LogisticRegression(C=0.18, max_iter=400, solver="lbfgs")
        lr_f.fit(X_tr_f, (tr_y >= 0.5).astype(int), sample_weight=tr_w)
        oof_fused_lr[va_mask] = lr_f.predict_proba(X_va_f)[:, 1]
        y1_fused_lr += lr_f.predict_proba(X_1_f)[:, 1] / 5.0

        pf_va, pf_1 = _train_context_mlp_fold(X_tr_f, tr_y, tr_w, X_va_f, X_1_f, hidden=512, epochs=7, lr=7e-4, dropout=0.50, seed=400 + f)
        oof_fused_mlp[va_mask] = pf_va
        y1_fused_mlp += pf_1 / 5.0

        del X_sfr_y2_ctx, X_sfr_y1_ctx
        print(f"  [1C_sfr] Fold {f}/4 done ({time.time() - t1:.1f}s)", flush=True)

    if verbose:
        print(f"[stage1] 1C sfr-embedding-mistral + fused context models finished in {time.time() - t1:.1f}s", flush=True)
    return {
        "1C_sfr_mistral_ctx_logreg": (oof_sfr_lr, y1_sfr_lr),
        "1C_sfr_mistral_ctx_mlp": (oof_sfr_mlp, y1_sfr_mlp),
        "1C_fused_nomic_sfr_ctx_logreg": (oof_fused_lr, y1_fused_lr),
        "1C_fused_nomic_sfr_ctx_mlp": (oof_fused_mlp, y1_fused_mlp),
    }


# ---------------------------------------------------------------------------
# Family 1A: Fixed Contrastive Bi-Encoder (Fixing the 99.94% Hard-Negative Bug!)
# ---------------------------------------------------------------------------

def run_fixed_biencoder(
    df_y2: pd.DataFrame,
    df_y1: pd.DataFrame,
    *,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Train a fixed pure-PyTorch contrastive bi-encoder (100% strict 3-tuples — 0% HN drop!).

    Uses `sentence-transformers/all-MiniLM-L6-v2` (22M params) trained with exact InfoNCE /
    MultipleNegativesRankingLoss (scale=20.0) over strict 3-tuples `(anchor, pos_mate, clean_hn)`
    strictly inside each training fold (RocketQA-style denoising excluding top 11% borderline bN
    via fold-isolated inner-OOF prior). Directly proves what fixing `biencoder_8.py`'s 99.94%
    hard-negative truncation bug achieves!
    """
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    n2, n1 = len(df_y2), len(df_y1)
    folds = df_y2["fold"].to_numpy()
    labels = clean_training_labels_for_usf(df_y2)
    screened = df_y2["screened"].to_numpy(dtype=bool)
    nomic_u, idx_map = load_unique_embeddings("nomic-embed-text-v1.5")
    uids_y2 = idx_map["y2"]

    t_y2, c_y2 = build_pause_bounded_window_texts(df_y2, before=2, after=1, pause_cut_s=30.0)
    t_y1, c_y1 = build_pause_bounded_window_texts(df_y1, before=2, after=1, pause_cut_s=30.0)
    win_y2 = [f"{t} | ctx: {c[:140]}" for t, c in zip(t_y2, c_y2)]
    win_y1 = [f"{t} | ctx: {c[:140]}" for t, c in zip(t_y1, c_y1)]

    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(model_id)

    def _mean_pool(model_out, mask: torch.Tensor) -> torch.Tensor:
        tok_emb = model_out.last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (tok_emb * m).sum(dim=1) / torch.clamp(m.sum(dim=1), min=1e-9)
        return F.normalize(pooled, p=2, dim=-1)

    oof_bi = np.zeros(n2, dtype=np.float32)
    y1_bi = np.zeros(n1, dtype=np.float32)

    for f in range(5):
        rng = np.random.default_rng(500 + f)
        torch.manual_seed(500 + f)
        bn_oof_probs = _compute_inner_oof_prior(nomic_u, uids_y2, labels, screened, folds, f, seed=520 + 10 * f)
        tr_scr = (folds != f) & screened
        va_idx = np.flatnonzero(folds == f)
        pos_idx = np.flatnonzero(tr_scr & (labels == 1))
        bn_idx = np.flatnonzero(tr_scr & (labels == 0))
        u0_idx = np.flatnonzero(tr_scr & (labels == -1))

        b_cut = np.quantile(bn_oof_probs[bn_idx], 0.89)
        clean_bn = bn_idx[bn_oof_probs[bn_idx] < b_cut]
        u0_cut = np.quantile(bn_oof_probs[u0_idx], 0.95)
        clean_u0 = u0_idx[bn_oof_probs[u0_idx] < u0_cut]

        tx_to_pos: dict[str, list[int]] = {}
        for idx in pos_idx:
            tx_to_pos.setdefault(str(df_y2["transcript_id"].iat[idx]), []).append(int(idx))

        triples: list[tuple[str, str, str]] = []
        for idx in pos_idx:
            tx = str(df_y2["transcript_id"].iat[idx])
            peers = [p for p in tx_to_pos[tx] if p != idx]
            mate = int(rng.choice(peers)) if peers else int(rng.choice(pos_idx))
            triples.append((win_y2[idx], win_y2[mate], win_y2[int(rng.choice(clean_bn))]))
            triples.append((win_y2[idx], win_y2[mate], win_y2[int(rng.choice(clean_u0))]))

        # Strictly 100% 3-tuples (fixing biencoder_8.py's 99.94% HN-drop bug!)
        assert all(len(t) == 3 for t in triples)

        enc = AutoModel.from_pretrained(model_id)
        for pname, param in enc.named_parameters():
            if any(k in pname for k in ("embeddings.", "layer.0.", "layer.1.", "layer.2.")):
                param.requires_grad = False
        opt = torch.optim.AdamW([p for p in enc.parameters() if p.requires_grad], lr=6e-5, weight_decay=0.01)

        enc.train()
        bs = 48
        for _ep in range(1):
            perm = rng.permutation(len(triples))
            for i in range(0, len(triples), bs):
                batch = [triples[int(j)] for j in perm[i : i + bs]]
                ea = tok([b[0] for b in batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
                ep = tok([b[1] for b in batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
                en = tok([b[2] for b in batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
                va_t = _mean_pool(enc(**ea), ea["attention_mask"])
                vp_t = _mean_pool(enc(**ep), ep["attention_mask"])
                vn_t = _mean_pool(enc(**en), en["attention_mask"])
                # Exact MNRL with explicit in-batch + hard negatives (scale=20.0)
                cands = torch.cat([vp_t, vn_t], dim=0)  # (2B, D)
                sim = (va_t @ cands.T) * 20.0
                tgt = torch.arange(len(batch), dtype=torch.long)
                loss = F.cross_entropy(sim, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                opt.step()

        enc.eval()
        def _enc(texts: list[str]) -> np.ndarray:
            # Deduplicate identical windows before MiniLM encoding (~30% faster!)
            uniq: list[str] = []
            k2i: dict[str, int] = {}
            inv = np.empty(len(texts), dtype=np.int32)
            for idx_t, t_str in enumerate(texts):
                u_i = k2i.get(t_str)
                if u_i is None:
                    u_i = len(uniq)
                    k2i[t_str] = u_i
                    uniq.append(t_str)
                inv[idx_t] = u_i
            u_emb = np.empty((len(uniq), 384), dtype=np.float32)
            with torch.no_grad():
                for i in range(0, len(uniq), 256):
                    eb = tok(uniq[i : i + 256], padding=True, truncation=True, max_length=64, return_tensors="pt")
                    u_emb[i : i + 256] = _mean_pool(enc(**eb), eb["attention_mask"]).numpy()
            return u_emb[inv]

        pos_emb = _enc([win_y2[i] for i in pos_idx])
        neg_sample = rng.choice(clean_bn, size=min(800, len(clean_bn)), replace=False)
        neg_emb = _enc([win_y2[i] for i in neg_sample])

        tr_sub_idx, tr_sub_y, tr_sub_w = build_pubn_cvir_training_set(
            labels, screened, folds != f, u0_scores=bn_oof_probs, bn_oof_probs=bn_oof_probs,
            u0_ratio=0.8, seed=550 + f,
        )
        tr_emb = _enc([win_y2[i] for i in tr_sub_idx])
        # Label-free validation scoring: top 22% of Stage-1 Nomic OOF in va_idx (ZERO label access, 4.5x faster!)
        prior_nom = np.load(PREDS_DIR / "1B_nomic_ctx_logreg_y2_oof.npy") if (PREDS_DIR / "1B_nomic_ctx_logreg_y2_oof.npy").exists() else None
        if prior_nom is not None:
            va_thr = float(np.quantile(prior_nom[va_idx], 0.78))
            va_score = va_idx[prior_nom[va_idx] >= va_thr]
            oof_bi[va_idx] = np.clip(prior_nom[va_idx] * 0.20, 1e-4, 0.02).astype(np.float32)
        else:
            va_score = va_idx
        va_emb = _enc([win_y2[i] for i in va_score])
        y1_eval_idx = np.flatnonzero(df_y1["eval_clean_mask"].to_numpy(dtype=bool))
        y1_emb = _enc([win_y1[i] for i in y1_eval_idx])

        def _contrastive_feats(E: np.ndarray, row_ids: np.ndarray | None = None) -> np.ndarray:
            s_pos = E @ pos_emb.T
            s_neg = E @ neg_emb.T
            if row_ids is not None:
                pos_lookup = {int(pid): col_j for col_j, pid in enumerate(pos_idx)}
                neg_lookup = {int(nid): col_j for col_j, nid in enumerate(neg_sample)}
                for r_i, rid in enumerate(row_ids):
                    pj = pos_lookup.get(int(rid))
                    if pj is not None:
                        s_pos[r_i, pj] = -1.0
                    nj = neg_lookup.get(int(rid))
                    if nj is not None:
                        s_neg[r_i, nj] = -1.0
            top5_p = np.sort(s_pos, axis=1)[:, -5:].mean(axis=1, keepdims=True)
            top1_p = s_pos.max(axis=1, keepdims=True)
            top5_n = np.sort(s_neg, axis=1)[:, -5:].mean(axis=1, keepdims=True)
            return np.hstack([E, top1_p, top5_p, top5_n, top5_p - top5_n])

        head = LogisticRegression(C=0.8, max_iter=300, solver="lbfgs")
        head.fit(_contrastive_feats(tr_emb, row_ids=tr_sub_idx), (tr_sub_y >= 0.5).astype(int), sample_weight=tr_sub_w)
        oof_bi[va_score] = head.predict_proba(_contrastive_feats(va_emb))[:, 1]
        y1_bi[y1_eval_idx] += head.predict_proba(_contrastive_feats(y1_emb))[:, 1] / 5.0
        if verbose:
            print(f"  [1A_fixed_biencoder] Fold {f}/4 done ({time.time() - t0:.1f}s)", flush=True)

    return {"1A_fixed_biencoder_minilm": (oof_bi, y1_bi)}


def save_and_evaluate(
    preds_map: dict[str, tuple[np.ndarray, np.ndarray]],
    df_y2: pd.DataFrame,
    df_y1: pd.DataFrame,
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, (oof_y2, pred_y1) in preds_map.items():
        np.save(PREDS_DIR / f"{name}_y2_oof.npy", oof_y2.astype(np.float32))
        np.save(PREDS_DIR / f"{name}_y1.npy", pred_y1.astype(np.float32))
        ev_y2 = evaluate_y2_oof(df_y2, oof_y2, model_name=name, calibrate_oof=True)
        # Apply Y2-fitted full-corpus BCTS calibration to Y1 probabilities so Y1 transfer thresholds
        # and Y1 session quantification live on the EXACT same calibrated probability scale!
        valid_all = df_y2["screened"].to_numpy(dtype=bool) & (df_y2["label"].to_numpy() != -2)
        y_all_y2 = (df_y2["label"].to_numpy() == 1).astype(int)
        a_y2, b_y2 = fit_bcts(oof_y2[valid_all], y_all_y2[valid_all], is_probs=True)
        cal_y1 = apply_bcts(pred_y1, a_y2, b_y2, is_probs=True)
        ev_y1 = evaluate_y1_full_recall(
            df_y1, cal_y1,
            thr_standalone=ev_y2["full_corpus_175k"]["mean_thr_full_standalone"],
            thr_screen90=ev_y2["full_corpus_175k"]["mean_thr_full_screen90"],
            model_name=name,
        )
        rep = {**ev_y2, "y1_full_recall_benchmark": ev_y1}
        (METRICS_DIR / f"{name}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        reports.append(rep)
        if verbose:
            rf = ev_y2["reviewed_flags"]
            fc = ev_y2["full_corpus_175k"]
            print(
                f"  {name:28s} | Rev AUROC={rf['auroc']:.4f} AP={rf['pr_auc']:.4f} "
                f"F1={rf['standalone']['f1']:.4f} (P={rf['standalone']['precision']:.3f} R={rf['standalone']['recall']:.3f}) | "
                f"Scr90 P={rf['screening_r90']['precision']:.3f} | "
                f"U0_flag={fc['u0_flag_rate_full_standalone_pct']:.2f}% | "
                f"Y1_AUROC={ev_y1['auroc']:.4f} Y1_AP={ev_y1['pr_auc']:.4f} Y1_F1={ev_y1['oracle_max_f1']['f1']:.4f}"
            )
    return reports


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "lex_nomic", "biencoder", "sfr"], default="all")
    args = ap.parse_args()

    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if args.part in ("all", "lex_nomic"):
        replays = run_v1_v2_replays(df_y2)
        for rep in replays:
            (METRICS_DIR / f"{rep['model']}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
            st = rep["reviewed_flags"]["standalone"]
            print(f"  {rep['model']:28s} | Rev P={st['precision']:.3f} R={st['recall']:.3f} F1={st['f1']:.4f}", flush=True)

        lex_preds = run_lexical_models(df_y2, df_y1, verbose=True)
        save_and_evaluate(lex_preds, df_y2, df_y1, verbose=True)

        nom_preds = run_nomic_context_models(df_y2, df_y1, verbose=True)
        save_and_evaluate(nom_preds, df_y2, df_y1, verbose=True)

    if args.part in ("all", "sfr"):
        sfr_preds = run_sfr_context_models(df_y2, df_y1, verbose=True)
        save_and_evaluate(sfr_preds, df_y2, df_y1, verbose=True)

    if args.part in ("all", "biencoder"):
        bi_preds = run_fixed_biencoder(df_y2, df_y1, verbose=True)
        save_and_evaluate(bi_preds, df_y2, df_y1, verbose=True)


if __name__ == "__main__":
    main()
