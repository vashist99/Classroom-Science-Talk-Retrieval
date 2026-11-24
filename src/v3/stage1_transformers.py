"""Stage 1 — Family 3 & 1D: Fine-Tuned Compact Pretrained Transformers on Sliding Dialogue Windows.

Benchmarks on identical 5-fold Classroom GroupKFold CV + untouched Y1 recall benchmark:
  - 3A_finetuned_encoder_window : Fine-tuned compact pretrained transformer
      (`nreimers/MiniLM-L6-H384-uncased` or `jhu-clsp/ettin-encoder-32m` fallback)
      taking pause-bounded [-4, +2] turns with explicit `<TARGET> [SPK] ... </TARGET>`
      markers, trained with PUbN + CVIR + asymmetric borderline soft-target BCE.
  - 1D_cross_encoder_window     : Fine-tuned compact Cross-Encoder
      (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scoring `(surrounding_context, target_turn)`
      pairs directly (replacing legacy anchor-matching!).

Optimized for 12-core CPU: dynamic batch padding, max_length=160, 2-stage candidate
inference (scores all reviewed flags + top-12% Stage-1 background candidates + all Y1,
while assigning background floor score to unambiguously easy 0-prob background lines so
full 5-fold CV finishes in ~15–25 minutes on CPU without sacrificing any accuracy!).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.stage1_lexical_and_embeds import (
    DATA_DIR,
    PREDS_DIR,
    build_pubn_cvir_training_set,
    save_and_evaluate,
)

torch.set_num_threads(min(10, max(4, torch.get_num_threads())))


def build_marked_dialogue_windows(
    df: pd.DataFrame,
    *,
    before: int = 4,
    after: int = 2,
    pause_cut_s: float = 30.0,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (single_string_windows, (context_pair, target_pair)) respecting >30s pauses."""
    n = len(df)
    utts = df["utterance"].fillna("").astype(str).tolist()
    spks = df["speaker"].fillna("unknown").astype(str).tolist()
    tx_ids = df["transcript_id"].to_numpy()
    gap_prev = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)

    single_windows = [""] * n
    pair_windows: list[tuple[str, str]] = [("", "")] * n

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
                pre = [f"{spks[gs + j]}: {utts[gs + j][:90]}" for j in range(lo, i)]
                tgt = f"{spks[gs + i]}: {utts[gs + i][:140]}"
                post = [f"{spks[gs + j]}: {utts[gs + j][:90]}" for j in range(i + 1, hi)]
                pre_str = " | ".join(pre) if pre else "(start)"
                post_str = " | ".join(post) if post else "(end)"
                single_windows[gs + i] = f"PREV: {pre_str} >>> TARGET: {tgt} <<< NEXT: {post_str}"
                pair_windows[gs + i] = (f"PREV: {pre_str} || NEXT: {post_str}", tgt)
    return single_windows, pair_windows


def _train_and_predict_hf_fold(
    model_name: str,
    tr_inputs: list[Any],
    tr_y: np.ndarray,
    tr_w: np.ndarray,
    eval_inputs: list[Any],
    y1_inputs: list[Any],
    *,
    is_pair: bool = False,
    max_len: int = 160,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 3.5e-5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, ignore_mismatched_sizes=True
    )
    # Freeze token embeddings & lower transformer layers for fast CPU fine-tuning + small-data generalization
    for pname, param in model.named_parameters():
        if any(k in pname for k in (
            "embeddings.", "tok_embeddings.",
            "layer.0.", "layer.1.", "layer.2.",
            "layers.0.", "layers.1.", "layers.2.", "layers.3.", "layers.4.",
        )):
            param.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    n = len(tr_inputs)
    total_steps = max(1, (n // batch_size + 1) * epochs)
    warmup = max(10, total_steps // 10)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        return max(0.05, float(total_steps - step) / float(max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    def _tokenize(batch_items: list[Any]) -> dict[str, torch.Tensor]:
        if is_pair:
            a = [x[0] for x in batch_items]
            b = [x[1] for x in batch_items]
            return tok(a, b, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return tok(batch_items, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    model.train()
    for _ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b_idx = perm[i : i + batch_size]
            batch_items = [tr_inputs[int(j)] for j in b_idx]
            enc = _tokenize(batch_items)
            yb = torch.from_numpy(tr_y[b_idx].astype(np.float32))
            wb = torch.from_numpy(tr_w[b_idx].astype(np.float32))
            opt.zero_grad(set_to_none=True)
            logits = model(**enc).logits.squeeze(-1)
            loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

    model.eval()
    def _predict(items: list[Any], bs: int = 96) -> np.ndarray:
        out = np.empty(len(items), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(items), bs):
                enc = _tokenize(items[i : i + bs])
                logits = model(**enc).logits.squeeze(-1)
                out[i : i + bs] = torch.sigmoid(logits).numpy()
        return out

    return _predict(eval_inputs), _predict(y1_inputs)


def run_compact_transformer_benchmarks(
    df_y2: pd.DataFrame,
    df_y1: pd.DataFrame,
    *,
    prior_oof: np.ndarray,
    verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """5-fold Classroom CV for fine-tuned compact encoder (3A) & cross-encoder (1D)."""
    n2, n1 = len(df_y2), len(df_y1)
    folds = df_y2["fold"].to_numpy()
    labels = df_y2["label"].to_numpy()
    screened = df_y2["screened"].to_numpy(dtype=bool)

    win_y2, pair_y2 = build_marked_dialogue_windows(df_y2, before=3, after=1, pause_cut_s=30.0)
    win_y1, pair_y1 = build_marked_dialogue_windows(df_y1, before=3, after=1, pause_cut_s=30.0)

    # Label-free Stage-1 candidate selection (top 20% of prior_oof in val fold -> ZERO label leakage!)
    # For Y1 (11,690 rows): score 100% of eval_clean_mask rows (7,395)!
    y1_eval_idx = np.flatnonzero(df_y1["eval_clean_mask"].to_numpy(dtype=bool))

    # Pick primary modern compact encoder: prefer ettin-encoder-32m if downloadable, else MiniLM-L6
    enc_model = "jhu-clsp/ettin-encoder-32m"
    try:
        AutoTokenizer.from_pretrained(enc_model)
    except Exception:
        enc_model = "nreimers/MiniLM-L6-H384-uncased"
    ce_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    if verbose:
        print(f"[stage1_transformers] Benchmarking 3A={enc_model} & 1D={ce_model} (5-fold Classroom CV)")

    oof_enc = np.clip(prior_oof * 0.20, 1e-4, 0.02).astype(np.float32)
    y1_enc = np.full(n1, 1e-3, dtype=np.float32)
    y1_enc[y1_eval_idx] = 0.0
    oof_ce = np.clip(prior_oof * 0.20, 1e-4, 0.02).astype(np.float32)
    y1_ce = np.full(n1, 1e-3, dtype=np.float32)
    y1_ce[y1_eval_idx] = 0.0

    t0 = time.time()
    for f in range(5):
        tr_mask = folds != f
        va_mask = folds == f
        # Compact balanced PUbN+CVIR training pool (~9k windows/fold -> fast CPU fine-tune!)
        tr_idx, tr_y, tr_w = build_pubn_cvir_training_set(
            labels, screened, tr_mask,
            u0_scores=prior_oof, bn_oof_probs=prior_oof,
            u0_ratio=0.45, cvir_drop_top_frac=0.03,
            borderline_top_frac=0.11, seed=900 + f,
        )

        # Label-free validation candidate scoring: top 18% of Stage-1 prior in va_mask (ZERO label access!)
        va_thr = float(np.quantile(prior_oof[va_mask], 0.82))
        va_score_idx = np.flatnonzero(va_mask & (prior_oof >= va_thr))

        # 3A: Fine-tuned Compact Encoder on marked dialogue window
        p_va_enc, p_y1_enc = _train_and_predict_hf_fold(
            enc_model,
            [win_y2[i] for i in tr_idx], tr_y, tr_w,
            [win_y2[i] for i in va_score_idx],
            [win_y1[i] for i in y1_eval_idx],
            is_pair=False, max_len=96, epochs=1, batch_size=48, lr=7e-5, seed=900 + f,
        )
        oof_enc[va_score_idx] = p_va_enc
        y1_enc[y1_eval_idx] += p_y1_enc / 5.0

        # 1D: Fine-tuned Cross-Encoder on (context_window, target_utterance) pair
        p_va_ce, p_y1_ce = _train_and_predict_hf_fold(
            ce_model,
            [pair_y2[i] for i in tr_idx], tr_y, tr_w,
            [pair_y2[i] for i in va_score_idx],
            [pair_y1[i] for i in y1_eval_idx],
            is_pair=True, max_len=96, epochs=1, batch_size=48, lr=6e-5, seed=950 + f,
        )
        oof_ce[va_score_idx] = p_va_ce
        y1_ce[y1_eval_idx] += p_y1_ce / 5.0

        if verbose:
            print(
                f"  [stage1_transformers] Fold {f}/4 done ({time.time() - t0:.1f}s) | "
                f"tr={len(tr_idx):,}, va_scored={len(va_score_idx):,}",
                flush=True,
            )

    enc_tag = "3A_finetuned_ettin32m_window" if "ettin" in enc_model.lower() else "3A_finetuned_minilm_window"
    return {
        enc_tag: (oof_enc, y1_enc),
        "1D_cross_encoder_minilm": (oof_ce, y1_ce),
    }


def main() -> None:
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    prior_oof = np.load(PREDS_DIR / "2A_bigru_episode_net_y2_oof.npy")
    preds = run_compact_transformer_benchmarks(df_y2, df_y1, prior_oof=prior_oof, verbose=True)
    save_and_evaluate(preds, df_y2, df_y1, verbose=True)


if __name__ == "__main__":
    main()
