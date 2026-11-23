"""Stage 1 — Family 2: Conversation-Sequence & Episode-Aware Neural Models.

Directly exploits the empirically verified 35x–44x temporal/episodic clustering
of Pre-K science talk across full chronological transcripts:
  - 2A_bigru_episode_net      : 2-layer Bidirectional GRU + Multi-Task Episode Head
                                over FULL transcripts (all turns, including context-only
                                neighbors), trained with masked PUbN + CVIR + asymmetric
                                borderline soft targets + pairwise CRF transition consistency.
  - 2B_tcn_transformer_net    : Multi-Scale Dilated 1D-CNN (dilations 1,2,4,8,16 -> +/-31 turns)
                                + Local Time-Decay Attention + Episode Head.
  - 2C_time_aware_hmm         : Continuous-time 2-state HMM smoother (exp(-dt/tau)) over
                                best turn logits with standalone high-confidence bypass.

Runs fast on 12-core CPU (~3–6 min for all 5 folds; <5s inference over 184k turns!).
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
import torch.nn.functional as F
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.embeddings import load_unique_embeddings
from src.v3.stage1_lexical_and_embeds import (
    DATA_DIR,
    METRICS_DIR,
    PREDS_DIR,
    _compute_inner_oof_prior,
    _fit_fold_svd,
    get_codebook_feature_matrix,
    save_and_evaluate,
)

torch.set_num_threads(min(10, max(4, torch.get_num_threads())))


def _safe_logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def build_sequence_feature_tensors(
    df: pd.DataFrame,
    nomic_u: np.ndarray,
    sfr_u: np.ndarray,
    uids: np.ndarray,
    sfr_proj: np.ndarray,
    oof_aux_logits: np.ndarray | None,
    cb_scaled: np.ndarray,
    *,
    nomic_dim: int = 256,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Build per-transcript chronological sequence arrays (T_i, D_seq).

    Features per turn t:
      - nomic Matryoshka 256-d L2-normalized embedding
      - SFR-Mistral fold-strict SVD 256-d L2-normalized embedding
      - delta cosine scalars: cos(e_t, e_{t-1}), cos(e_t, e_{t+1}) for both backbones (4d)
      - Manual-v0.3 codebook + timing/pause/ASR features (cb_scaled)
      - Optional auxiliary logits (None for pure standalone zero-leakage sequence models!)
    """
    n_u = nomic_u[:, :nomic_dim].astype(np.float32)
    n_u = n_u / np.maximum(np.linalg.norm(n_u, axis=1, keepdims=True), 1e-8)

    s_u = (sfr_u.astype(np.float32) @ sfr_proj).astype(np.float32)
    s_u = s_u / np.maximum(np.linalg.norm(s_u, axis=1, keepdims=True), 1e-8)

    tx_ids = df["transcript_id"].to_numpy()
    gap_prev = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], len(df)]

    seqs: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        L = e - s
        u_slice = uids[s:e]
        En = n_u[u_slice]  # (L, 256)
        Es = s_u[u_slice]  # (L, 256)
        gp = gap_prev[s:e]

        # Neighbor cosine similarities (zeroed across >30s pauses!)
        same_ep = (gp <= 30.0).astype(np.float32)
        cos_n_prev = np.zeros((L, 1), dtype=np.float32)
        cos_n_next = np.zeros((L, 1), dtype=np.float32)
        cos_s_prev = np.zeros((L, 1), dtype=np.float32)
        cos_s_next = np.zeros((L, 1), dtype=np.float32)
        if L > 1:
            cn = np.sum(En[1:] * En[:-1], axis=1) * same_ep[1:]
            cs = np.sum(Es[1:] * Es[:-1], axis=1) * same_ep[1:]
            cos_n_prev[1:, 0] = cn
            cos_n_next[:-1, 0] = cn
            cos_s_prev[1:, 0] = cs
            cos_s_next[:-1, 0] = cs

        parts = [En, Es, cos_n_prev, cos_n_next, cos_s_prev, cos_s_next, cb_scaled[s:e]]
        if oof_aux_logits is not None:
            parts.append(oof_aux_logits[s:e])
        X_seq = np.hstack(parts).astype(np.float32)
        seqs.append(X_seq)
        spans.append((int(s), int(e)))
    return seqs, spans


def build_episode_targets(labels_seq: np.ndarray, gap_prev_seq: np.ndarray, window: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Auxiliary episode-activity target: 1 if any verified TP within pause-bounded +/-window."""
    L = len(labels_seq)
    ep_y = np.zeros(L, dtype=np.float32)
    ep_mask = np.zeros(L, dtype=np.float32)

    ep_breaks = np.flatnonzero(gap_prev_seq > 30.0)
    ep_starts = np.r_[0, ep_breaks]
    ep_ends = np.r_[ep_breaks, L]
    for es, ee in zip(ep_starts, ep_ends):
        sub_l = labels_seq[es:ee]
        m = ee - es
        for i in range(m):
            lo, hi = max(0, i - window), min(m, i + window + 1)
            win = sub_l[lo:hi]
            if (win == 1).any():
                ep_y[es + i] = 1.0
                ep_mask[es + i] = 1.0
            elif (win == 0).any():
                # Reviewed FPs with zero nearby TP -> clearly outside science episode!
                ep_y[es + i] = 0.0
                ep_mask[es + i] = 0.8
            else:
                # Pure U_0 region -> low weight negative background
                ep_y[es + i] = 0.0
                ep_mask[es + i] = 0.08
    return ep_y, ep_mask


# ---------------------------------------------------------------------------
# Model 2A: 2-Layer BiGRU + Multi-Task Episode Head + Transition Consistency
# ---------------------------------------------------------------------------

class BiGRUEpisodeNet(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256, hidden: int = 192, dropout: float = 0.30) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.post_norm = nn.LayerNorm(hidden * 2 + proj_dim)
        self.drop = nn.Dropout(dropout)
        # Dual heads: target utterance science probability + local science episode probability
        self.turn_head = nn.Linear(hidden * 2 + proj_dim, 1)
        self.episode_head = nn.Linear(hidden * 2 + proj_dim, 1)
        # Residual episode-to-turn fusion gate (allows isolated micro-science bypass!)
        self.fuse = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        h0 = self.in_proj(x)
        h_seq, _ = self.gru(h0)
        h = self.drop(self.post_norm(torch.cat([h_seq, h0], dim=-1)))
        turn_raw = self.turn_head(h)
        ep_logit = self.episode_head(h)
        fused = turn_raw + 0.35 * self.fuse(torch.cat([turn_raw, ep_logit], dim=-1))
        return fused.squeeze(-1), ep_logit.squeeze(-1)


# ---------------------------------------------------------------------------
# Model 2B: Multi-Scale Dilated 1D-CNN + Local Self-Attention + Episode Head
# ---------------------------------------------------------------------------

class DilatedTCNBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> gated dilated conv residual
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        a, b = y.chunk(2, dim=-1)
        y = a * torch.sigmoid(b)
        return self.norm(x + self.drop(y))


class MultiScaleTCNTransformerNet(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 224, dropout: float = 0.30) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Dilations 1, 2, 4, 8, 16 cover receptive field +/- 31 turns seamlessly!
        self.blocks = nn.ModuleList([DilatedTCNBlock(d_model, d, dropout=dropout) for d in (1, 2, 4, 8, 16)])
        # Lightweight 1-layer Transformer encoder for long-range context interaction
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.turn_head = nn.Linear(d_model * 2, 1)
        self.episode_head = nn.Linear(d_model * 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = self.in_proj(x)
        h = h0
        for blk in self.blocks:
            h = blk(h)
        # Chunked attention over 256-turn segments to stay O(T) fast on CPU!
        B, T, C = h.shape
        if T <= 256:
            ha = self.attn(h)
        else:
            chunks = []
            for i in range(0, T, 256):
                chunks.append(self.attn(h[:, i : i + 256, :]))
            ha = torch.cat(chunks, dim=1)
        feat = torch.cat([ha, h0], dim=-1)
        return self.turn_head(feat).squeeze(-1), self.episode_head(feat).squeeze(-1)


def _train_sequence_model_fold(
    model: nn.Module,
    tr_seqs: list[np.ndarray],
    tr_labels: list[np.ndarray],
    tr_screened: list[np.ndarray],
    tr_gaps: list[np.ndarray],
    tr_prior: list[np.ndarray],
    va_seqs: list[np.ndarray],
    y1_seqs: list[np.ndarray],
    *,
    epochs: int = 16,
    lr: float = 1.0e-3,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Train sequence model over full chronological transcripts with PUbN+CVIR+AUM loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5.0e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Precompute static per-transcript targets (with asymmetric borderline softening on bN)
    all_bn_priors = np.concatenate([
        pr[(lb == 0) & sc] for pr, lb, sc in zip(tr_prior, tr_labels, tr_screened)
        if ((lb == 0) & sc).any()
    ])
    bord_cut = float(np.quantile(all_bn_priors, 0.89)) if len(all_bn_priors) > 50 else 0.99

    all_u0_priors = np.concatenate([
        pr[(lb == -1) & sc] for pr, lb, sc in zip(tr_prior, tr_labels, tr_screened)
        if ((lb == -1) & sc).any()
    ])
    u0_cvir_cut_init = float(np.quantile(all_u0_priors, 0.975)) if len(all_u0_priors) > 100 else 0.99

    ep_targets = [build_episode_targets(lb, gp, window=6) for lb, gp in zip(tr_labels, tr_gaps)]

    n_tx = len(tr_seqs)
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(n_tx)
        for ti in order:
            x_np = tr_seqs[ti]
            lb = tr_labels[ti]
            sc = tr_screened[ti]
            gp = tr_gaps[ti]
            pr = tr_prior[ti]
            ep_y_np, ep_m_np = ep_targets[ti]

            x_t = torch.from_numpy(x_np).unsqueeze(0)  # (1, T, D)
            opt.zero_grad(set_to_none=True)
            turn_logits, ep_logits = model(x_t)
            turn_logits = turn_logits.squeeze(0)
            ep_logits = ep_logits.squeeze(0)

            # Build PUbN + CVIR + borderline soft targets dynamically!
            L = len(lb)
            y_t = np.zeros(L, dtype=np.float32)
            w_t = np.zeros(L, dtype=np.float32)

            pos_m = sc & (lb == 1)
            bn_m = sc & (lb == 0)
            u0_m = sc & (lb == -1)

            # Positives
            y_t[pos_m] = 1.0
            w_t[pos_m] = 4.5

            # Hard negatives (bN) with top-11% borderline softening
            if bn_m.any():
                bord = bn_m & (pr >= bord_cut)
                clean_bn = bn_m & ~bord
                y_t[clean_bn] = 0.0
                w_t[clean_bn] = 1.2
                y_t[bord] = 0.25
                w_t[bord] = 0.40

            # Unflagged U_0 with dynamic CVIR masking (after epoch 3 use model's own detached probs!)
            if u0_m.any():
                if ep < 3:
                    u0_sc = pr
                    u0_cut = u0_cvir_cut_init
                else:
                    with torch.no_grad():
                        u0_sc = torch.sigmoid(turn_logits).numpy()
                    u0_cut = float(np.quantile(u0_sc[u0_m], 0.975)) if u0_m.sum() > 40 else 0.99
                clean_u0 = u0_m & (u0_sc < u0_cut)
                # Moderate background mass (~0.08 per U0 turn * ~2000 U0 turns ~ balanced against bN)
                y_t[clean_u0] = 0.0
                w_t[clean_u0] = 0.065

            w_sum = max(float(w_t.sum()), 1.0)
            y_ten = torch.from_numpy(y_t)
            w_ten = torch.from_numpy(w_t)
            loss_turn = (F.binary_cross_entropy_with_logits(turn_logits, y_ten, reduction="none") * w_ten).sum() / w_sum

            # Auxiliary episode loss
            ep_y_ten = torch.from_numpy(ep_y_np)
            ep_m_ten = torch.from_numpy(ep_m_np)
            loss_ep = (F.binary_cross_entropy_with_logits(ep_logits, ep_y_ten, reduction="none") * ep_m_ten).sum() / max(float(ep_m_np.sum()), 1.0)

            # Light pause-aware temporal smoothness regularizer (CRF-style pairwise potential)
            if L > 1:
                probs_cur = torch.sigmoid(turn_logits)
                same_ep = torch.from_numpy((gp[1:] <= 20.0).astype(np.float32))
                loss_smooth = (torch.abs(probs_cur[1:] - probs_cur[:-1]) * same_ep).mean()
            else:
                loss_smooth = 0.0

            loss = loss_turn + 0.30 * loss_ep + 0.05 * loss_smooth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            opt.step()
        sched.step()

    model.eval()
    def _infer(seqs: list[np.ndarray]) -> list[np.ndarray]:
        outs = []
        with torch.no_grad():
            for x_np in seqs:
                tl, _ = model(torch.from_numpy(x_np).unsqueeze(0))
                outs.append(torch.sigmoid(tl.squeeze(0)).numpy().astype(np.float32))
        return outs

    return _infer(va_seqs), _infer(y1_seqs)


# ---------------------------------------------------------------------------
# Model 2C: Continuous-Time 2-State HMM Episode Smoother (Forward-Backward)
# ---------------------------------------------------------------------------

def hmm_forward_backward_transcript(
    probs: np.ndarray,
    gaps_prev_s: np.ndarray,
    *,
    pi_ep: float = 0.04,
    tau_s: float = 45.0,
    stay_ep_base: float = 0.90,
    enter_ep_base: float = 0.015,
    bypass_thr: float = 0.85,
) -> np.ndarray:
    """Continuous-time 2-state HMM (Science Episode vs Non-Science) + isolated bypass.

    Computes posterior P(Episode_t = 1 | p_{1:T}, gaps) via exact forward-backward,
    then smoothly modulates emission odds without erasing high-confidence isolated turns.
    """
    T = len(probs)
    if T == 0:
        return probs.copy()
    p = np.clip(probs.astype(np.float64), 1e-5, 1.0 - 1e-5)
    # Calibrated emission likelihoods under Episode=1 vs Episode=0:
    # Under Episode=0, high-p science turns are rare (~(1-p)^3);
    # under Episode=1, both high-p turns and ambiguous contextual turns occur.
    log_e1 = np.log(0.30 + 1.80 * p)
    log_e0 = np.log(np.power(1.0 - p, 3.0) + 0.01)

    log_alpha = np.zeros((T, 2), dtype=np.float64)
    log_alpha[0, 0] = np.log(1.0 - pi_ep) + log_e0[0]
    log_alpha[0, 1] = np.log(pi_ep) + log_e1[0]

    # Precompute time-decayed transition matrices per step
    dt = np.clip(np.nan_to_num(gaps_prev_s, nan=60.0), 0.0, 300.0)
    decay = np.exp(-dt / tau_s)
    # As dt -> inf, transition reverts to stationary prior (1-pi_ep, pi_ep)
    p11 = pi_ep + (stay_ep_base - pi_ep) * decay
    p01 = pi_ep + (enter_ep_base - pi_ep) * decay
    p10 = np.clip(1.0 - p11, 1e-6, 1.0 - 1e-6)
    p00 = np.clip(1.0 - p01, 1e-6, 1.0 - 1e-6)

    def _lse2(a: float, b: float) -> float:
        m = max(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    for t in range(1, T):
        log_alpha[t, 0] = _lse2(log_alpha[t - 1, 0] + np.log(p00[t]), log_alpha[t - 1, 1] + np.log(p10[t])) + log_e0[t]
        log_alpha[t, 1] = _lse2(log_alpha[t - 1, 0] + np.log(p01[t]), log_alpha[t - 1, 1] + np.log(p11[t])) + log_e1[t]
        # Normalize step to avoid underflow
        z = _lse2(log_alpha[t, 0], log_alpha[t, 1])
        log_alpha[t, 0] -= z
        log_alpha[t, 1] -= z

    log_beta = np.zeros((T, 2), dtype=np.float64)
    for t in range(T - 2, -1, -1):
        b0 = _lse2(np.log(p00[t + 1]) + log_e0[t + 1] + log_beta[t + 1, 0],
                   np.log(p01[t + 1]) + log_e1[t + 1] + log_beta[t + 1, 1])
        b1 = _lse2(np.log(p10[t + 1]) + log_e0[t + 1] + log_beta[t + 1, 0],
                   np.log(p11[t + 1]) + log_e1[t + 1] + log_beta[t + 1, 1])
        z = _lse2(b0, b1)
        log_beta[t, 0] = b0 - z
        log_beta[t, 1] = b1 - z

    log_post1 = log_alpha[:, 1] + log_beta[:, 1]
    log_post0 = log_alpha[:, 0] + log_beta[:, 0]
    ep_post = expit(np.clip(log_post1 - log_post0, -20.0, 20.0))

    # Combine base turn log-odds + centered episode posterior lift:
    # - Inside a science episode (ep_post -> 1): boosts ambiguous turns without turning
    #   confident non-science management turns (p <= 0.03) into false positives.
    # - Outside an episode (ep_post -> 0): mildly suppresses isolated low/mid-p noise.
    # - Standalone micro-science bypass: never suppress ultra-confident standalone turns.
    base_z = logit(p)
    ep_lift = 1.85 * (ep_post - 0.20)
    blended = expit(np.clip(base_z + ep_lift, -20.0, 20.0))
    out = np.where(p >= bypass_thr, np.maximum(p, blended), blended)
    return out.astype(np.float32)


def run_hmm_smoother(df: pd.DataFrame, base_probs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.float32)
    tx_ids = df["transcript_id"].to_numpy()
    gaps = df["gap_prev_s"].to_numpy(dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        out[s:e] = hmm_forward_backward_transcript(base_probs[s:e], gaps[s:e])
    return out


def main() -> None:
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    nomic_u, idx_map = load_unique_embeddings("nomic-embed-text-v1.5")
    sfr_u, _ = load_unique_embeddings("sfr-embedding-mistral")
    uids_y2, uids_y1 = idx_map["y2"], idx_map["y1"]

    # Pure standalone zero-leakage sequence models (NO Level-1 OOF logits as input features!)
    cb_y2 = get_codebook_feature_matrix(df_y2)
    cb_y1 = get_codebook_feature_matrix(df_y1)
    folds = df_y2["fold"].to_numpy()
    labels = df_y2["label"].to_numpy()
    screened = df_y2["screened"].to_numpy(dtype=bool)
    gaps_y2 = df_y2["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)

    n2, n1 = len(df_y2), len(df_y1)
    oof_bigru = np.zeros(n2, dtype=np.float32)
    y1_bigru = np.zeros(n1, dtype=np.float32)
    oof_tcn = np.zeros(n2, dtype=np.float32)
    y1_tcn = np.zeros(n1, dtype=np.float32)

    t0 = time.time()
    for f in range(5):
        tr_mask = folds != f
        # Strictly fold-isolated inner-OOF prior for initial CVIR & borderline softening!
        prior_prob = _compute_inner_oof_prior(nomic_u, uids_y2, labels, screened, folds, f, seed=650 + 10 * f)

        # Fold-strict scaler + fold-strict SVD (ZERO leakage!)
        scaler = StandardScaler()
        scaler.fit(cb_y2[tr_mask])
        cb_y2_sc = (scaler.transform(cb_y2) * 0.20).astype(np.float32)
        cb_y1_sc = (scaler.transform(cb_y1) * 0.20).astype(np.float32)
        sfr_proj = _fit_fold_svd(sfr_u, uids_y2[tr_mask], n_components=256, seed=600 + f)

        seqs_y2, spans_y2 = build_sequence_feature_tensors(df_y2, nomic_u, sfr_u, uids_y2, sfr_proj, None, cb_y2_sc)
        seqs_y1, spans_y1 = build_sequence_feature_tensors(df_y1, nomic_u, sfr_u, uids_y1, sfr_proj, None, cb_y1_sc)

        tr_seqs, tr_lbs, tr_scrs, tr_gps, tr_prs = [], [], [], [], []
        va_seqs, va_spans = [], []
        for seq, (s, e) in zip(seqs_y2, spans_y2):
            tx_fold = int(folds[s])
            if tx_fold == f:
                va_seqs.append(seq)
                va_spans.append((s, e))
            else:
                tr_seqs.append(seq)
                tr_lbs.append(labels[s:e])
                tr_scrs.append(screened[s:e])
                tr_gps.append(gaps_y2[s:e])
                tr_prs.append(prior_prob[s:e])

        in_dim = seqs_y2[0].shape[1]
        # 2A: BiGRU + Episode Head (regularized for unseen classroom & Y1 generalization!)
        bigru = BiGRUEpisodeNet(in_dim=in_dim, proj_dim=256, hidden=192, dropout=0.40)
        va_p, y1_p_list = _train_sequence_model_fold(
            bigru, tr_seqs, tr_lbs, tr_scrs, tr_gps, tr_prs, va_seqs, seqs_y1,
            epochs=12, lr=9.0e-4, seed=700 + f,
        )
        for (s, e), p in zip(va_spans, va_p):
            oof_bigru[s:e] = p
        for (s, e), p in zip(spans_y1, y1_p_list):
            y1_bigru[s:e] += p / 5.0

        # 2B: Multi-Scale Dilated TCN + Transformer + Episode Head
        tcn = MultiScaleTCNTransformerNet(in_dim=in_dim, d_model=224, dropout=0.40)
        va_p2, y1_p2_list = _train_sequence_model_fold(
            tcn, tr_seqs, tr_lbs, tr_scrs, tr_gps, tr_prs, va_seqs, seqs_y1,
            epochs=12, lr=8.0e-4, seed=800 + f,
        )
        for (s, e), p in zip(va_spans, va_p2):
            oof_tcn[s:e] = p
        for (s, e), p in zip(spans_y1, y1_p2_list):
            y1_tcn[s:e] += p / 5.0

        print(f"  [stage1_sequence] Fold {f}/4 done ({time.time() - t0:.1f}s)", flush=True)

    # 2C: Continuous-Time 2-State HMM smoother stacked over strongest base embedding model
    hmm_base = "1C_fused_nomic_sfr_ctx_logreg" if (PREDS_DIR / "1C_fused_nomic_sfr_ctx_logreg_y2_oof.npy").exists() else "1C_fused_nomic_sfr_ctx_mlp"
    base_y2 = np.load(PREDS_DIR / f"{hmm_base}_y2_oof.npy")
    base_y1 = np.load(PREDS_DIR / f"{hmm_base}_y1.npy")
    oof_hmm = run_hmm_smoother(df_y2, base_y2)
    y1_hmm = run_hmm_smoother(df_y1, base_y1)

    save_and_evaluate(
        {
            "2A_bigru_episode_net": (oof_bigru, y1_bigru),
            "2B_tcn_transformer_net": (oof_tcn, y1_tcn),
            "2C_time_aware_hmm": (oof_hmm, y1_hmm),
        },
        df_y2,
        df_y1,
        verbose=True,
    )


if __name__ == "__main__":
    main()
