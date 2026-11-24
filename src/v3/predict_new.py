"""End-to-end production training & inference CLI for brand-new unseen Pre-K classroom transcripts.

Subcommands:
  1. `train-artifacts`:
     - Runs a clean 5-fold Classroom GroupKFold CV of the EXACT portable Nomic-only local
       pipeline (`0B_tfidf_context` + `1B_nomic_ctx_logreg` + `portable_nomic_pgkd_bigru`)
       AND the EXACT portable Hybrid Cascade (`0.58 * logit(portable_local) + 0.42 * logit(llm)`),
       calibrating ALL BCTS parameters and `screening`/`standalone` cutoffs strictly OOF from
       those exact portable predictions (zero mismatch, zero hardcoded thresholds!).
     - Fits final full-data models on 100% of verified Y2 data (+ PGKD silver) and saves them
       under `models/v3/`.
  2. `score`:
     - Loads ANY brand-new folder of raw USF classroom `.xlsx` transcripts (handling exact USF
       column headers `Segment Start Time`, `Segment End Time`, `Transcription (Confidence)`,
       `Speaker`, `"-"` placeholder filtering, and `screened` gating while keeping all turns
       for chronological context).
     - Runs `--engine hybrid` (Stage-1 -> Live Stage-2 Dual-LLM at `window_size=26`) or
       `--engine local` ($0 LLM calls) and writes exact 18-column `Human coded/*.xlsx`
       workbooks in `--mode screening|standalone|both` plus Calibrated Mean Posterior rate!
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.codebook_v3 import (
    apply_usf_cluster_and_veto_rules,
    clean_training_labels_for_usf,
    extract_codebook_features,
    extract_pm2_context_columns,
    infer_subtype_rule,
)
from src.v3.dataset import parse_transcription, to_seconds
from src.v3.embeddings import build_pause_bounded_context_matrix, load_unique_embeddings
from src.v3.export_workbooks import LABEL_NOT, LABEL_SCIENCE, child_id_from_filename, write_human_coded_workbook
from src.v3.llm_client import UFNavigatorClient
from src.v3.metrics import (
    apply_bcts,
    evaluate_y1_full_recall,
    evaluate_y2_oof,
    fit_bcts,
    oof_bcts_and_thresholds,
    tune_threshold_f1,
    tune_threshold_recall,
)
from src.v3.stage1_lexical_and_embeds import (
    DATA_DIR,
    PREDS_DIR,
    build_pause_bounded_window_texts,
    get_codebook_feature_matrix,
)
from src.v3.stage1_sequence import BiGRUEpisodeNet, _safe_logit, build_episode_targets
from src.v3.stage2_llm import S2_DIR, score_transcripts_with_llm

MODELS_DIR = PROJECT_ROOT / "models" / "v3"


def _embed_unique_nomic(df: pd.DataFrame, client: UFNavigatorClient) -> tuple[np.ndarray, np.ndarray]:
    """Embed unique `(speaker, utterance)` keys via UF NaviGator `nomic-embed-text-v1.5`."""
    spks = df["speaker"].fillna("unknown").astype(str).tolist()
    utts = df["utterance"].fillna("").astype(str).tolist()
    keys = [f"{s}|||{u}" for s, u in zip(spks, utts)]
    uniq_keys: list[str] = []
    key_to_id: dict[str, int] = {}
    uids = np.empty(len(df), dtype=np.int32)
    for i, k in enumerate(keys):
        idx = key_to_id.get(k)
        if idx is None:
            idx = len(uniq_keys)
            key_to_id[k] = idx
            uniq_keys.append(k)
        uids[i] = idx

    texts = []
    for k in uniq_keys:
        s, u = k.split("|||", 1)
        texts.append(f"classification: [{s}] {u}"[:1500])

    dim = 768
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    bs = 256
    for i in range(0, len(texts), bs):
        vecs = client.embed_batch(model="nomic-embed-text-v1.5", texts=texts[i : i + bs])
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
        mat[i : i + len(arr)] = arr
    return mat, uids


def _build_nomic_only_sequences(
    df: pd.DataFrame, nomic_u: np.ndarray, uids: np.ndarray, cb_scaled: np.ndarray
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Build portable per-transcript sequence tensors using nomic-embed-v1.5 (256d) + codebook."""
    n_u = nomic_u[:, :256].astype(np.float32)
    n_u /= np.maximum(np.linalg.norm(n_u, axis=1, keepdims=True), 1e-8)
    tx_ids = df["transcript_id"].to_numpy()
    gap_prev = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], len(df)]
    seqs, spans = [], []
    for s, e in zip(starts, ends):
        L = e - s
        En = n_u[uids[s:e]]
        gp = gap_prev[s:e]
        same_ep = (gp <= 30.0).astype(np.float32)
        c_prev = np.zeros((L, 1), dtype=np.float32)
        c_next = np.zeros((L, 1), dtype=np.float32)
        if L > 1:
            cn = np.sum(En[1:] * En[:-1], axis=1) * same_ep[1:]
            c_prev[1:, 0] = cn
            c_next[:-1, 0] = cn
        seqs.append(np.hstack([En, c_prev, c_next, cb_scaled[s:e]]).astype(np.float32))
        spans.append((int(s), int(e)))
    return seqs, spans


def _train_chunked_nomic_bigru(
    seqs: list[np.ndarray],
    spans: list[tuple[int, int]],
    labels: np.ndarray,
    scr: np.ndarray,
    gaps: np.ndarray,
    has_pgkd: bool,
    llm1: np.ndarray,
    llm2: np.ndarray,
    llm_cons: np.ndarray,
    eval_s2: np.ndarray,
    *,
    include_span_fn,
    epochs: int = 7,
    proj_dim: int = 160,
    hidden: int = 128,
    seed: int = 2026,
) -> BiGRUEpisodeNet:
    """Train chunk-batched BiGRUEpisodeNet on selected transcripts (zero weight on pad tails!)."""
    in_dim = seqs[0].shape[1]
    chunk_len = 96
    cx_list, cy_list, cw_list, cey_list, cem_list = [], [], [], [], []
    for x_np, (s, e) in zip(seqs, spans):
        if not include_span_fn(s, e):
            continue
        lb, sc, gp = labels[s:e], scr[s:e], gaps[s:e]
        L = e - s
        y_t = np.zeros(L, dtype=np.float32)
        w_t = np.zeros(L, dtype=np.float32)
        y_t[sc & (lb == 1)], w_t[sc & (lb == 1)] = 1.0, 4.5
        bn_m = sc & (lb == 0)
        if has_pgkd:
            bord = bn_m & (llm_cons[s:e] >= 0.70)
            y_t[bn_m & ~bord], w_t[bn_m & ~bord] = 0.0, 1.2
            y_t[bord], w_t[bord] = 0.25, 0.35
            u0_m = sc & (lb == -1)
            sp = u0_m & eval_s2[s:e] & (llm1[s:e] >= 0.70) & (llm2[s:e] >= 0.60)
            sn = u0_m & ((~eval_s2[s:e]) | (llm_cons[s:e] <= 0.12))
            y_t[sp], w_t[sp] = 0.85, 1.3
            y_t[sn], w_t[sn] = 0.0, 0.08
            eff_lb = lb.copy()
            eff_lb[sp] = 1
        else:
            y_t[bn_m], w_t[bn_m] = 0.0, 1.2
            y_t[sc & (lb == -1)], w_t[sc & (lb == -1)] = 0.0, 0.06
            eff_lb = lb
        ep_y, ep_m = build_episode_targets(eff_lb, gp, window=6)
        for c0 in range(0, L, chunk_len):
            c1 = min(L, c0 + chunk_len)
            cl = c1 - c0
            xc = np.zeros((chunk_len, in_dim), dtype=np.float32)
            yc = np.zeros(chunk_len, dtype=np.float32)
            wc = np.zeros(chunk_len, dtype=np.float32)  # pad positions stay 0.0 weight!
            eyc = np.zeros(chunk_len, dtype=np.float32)
            emc = np.zeros(chunk_len, dtype=np.float32)
            xc[:cl], yc[:cl], wc[:cl] = x_np[c0:c1], y_t[c0:c1], w_t[c0:c1]
            eyc[:cl], emc[:cl] = ep_y[c0:c1], ep_m[c0:c1]
            cx_list.append(xc); cy_list.append(yc); cw_list.append(wc); cey_list.append(eyc); cem_list.append(emc)

    X_c, Y_c, W_c = np.stack(cx_list), np.stack(cy_list), np.stack(cw_list)
    EY_c, EM_c = np.stack(cey_list), np.stack(cem_list)
    torch.manual_seed(seed)
    np.random.seed(seed)
    bigru = BiGRUEpisodeNet(in_dim=in_dim, proj_dim=proj_dim, hidden=hidden, dropout=0.40)
    opt_g = torch.optim.AdamW(bigru.parameters(), lr=1.2e-3, weight_decay=5.0e-2)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs)
    bigru.train()
    for _ in range(epochs):
        perm = np.random.permutation(len(X_c))
        for bi in range(0, len(X_c), 32):
            ib = perm[bi : bi + 32]
            opt_g.zero_grad(set_to_none=True)
            tl, el = bigru(torch.from_numpy(X_c[ib]))
            wb, emb = torch.from_numpy(W_c[ib]), torch.from_numpy(EM_c[ib])
            lt = (F.binary_cross_entropy_with_logits(tl, torch.from_numpy(Y_c[ib]), reduction="none") * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            le = (F.binary_cross_entropy_with_logits(el, torch.from_numpy(EY_c[ib]), reduction="none") * emb).sum() / torch.clamp(emb.sum(), min=1.0)
            (lt + 0.30 * le).backward()
            torch.nn.utils.clip_grad_norm_(bigru.parameters(), 1.5)
            opt_g.step()
        sched_g.step()
    bigru.eval()
    return bigru


def train_deployable_artifacts(verbose: bool = True) -> Path:
    """Fit EXACT portable bundle (`0B_tfidf_ctx` + `1B_nomic_ctx_logreg` + `portable_nomic_pgkd_bigru`)
    with 5-fold OOF calibration & thresholds for BOTH `local` and `hybrid` engines!
    """
    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    nomic_u, idx_map = load_unique_embeddings("nomic-embed-text-v1.5")
    uids_y2, uids_y1 = idx_map["y2"], idx_map["y1"]

    scr = df_y2["screened"].to_numpy(dtype=bool)
    folds = df_y2["fold"].to_numpy()
    labels_eval = df_y2["label"].to_numpy()
    labels = clean_training_labels_for_usf(df_y2)  # Cleaned STRICTLY for training folds!
    gaps = df_y2["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)
    cb_y2 = get_codebook_feature_matrix(df_y2)
    cb_y1 = get_codebook_feature_matrix(df_y1)

    s2_npz_path = S2_DIR / "full83_stage2_outputs.npz"
    has_pgkd = s2_npz_path.exists()
    if has_pgkd:
        npz = np.load(s2_npz_path)
        llm1 = npz["llm1_y2"].astype(np.float32)
        llm2 = npz["llm2_y2"].astype(np.float32)
        llm1_y1 = npz["llm1_y1"].astype(np.float32)
        llm2_y1 = npz["llm2_y1"].astype(np.float32)
        llm_cons = 0.60 * llm1 + 0.40 * llm2
        llm_cons_y1 = 0.60 * llm1_y1 + 0.40 * llm2_y1
        eval_s2 = npz["eval1_y2"].astype(bool)
        eval_s2_y1 = npz["eval1_y1"].astype(bool)
    else:
        llm1 = llm2 = llm_cons = np.zeros(len(df_y2), dtype=np.float32)
        llm1_y1 = llm2_y1 = llm_cons_y1 = np.zeros(len(df_y1), dtype=np.float32)
        eval_s2 = np.zeros(len(df_y2), dtype=bool)
        eval_s2_y1 = np.zeros(len(df_y1), dtype=bool)

    # 1. Exact 5-fold OOF of portable Nomic-only bundle (TF-IDF + Nomic LogReg + PGKD BiGRU)
    #    AND save EVERY held-out classroom fold model under `models/v3/fold_models/` so scoring
    #    any of the 650 Y2 transcripts automatically uses the held-out fold that NEVER saw that classroom!
    fold_dir = MODELS_DIR / "fold_models"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _VETO_ONLY = {"cb_trap_shape_only", "cb_trap_count_only", "cb_trap_look_no_content"}
    cb_cols_keep = [c for c in df_y2.columns if c.startswith("cb_") and c not in _VETO_ONLY]

    t_txt, c_txt = build_pause_bounded_window_texts(df_y2, before=5, after=3, pause_cut_s=30.0)
    X_nom_ctx_all = build_pause_bounded_context_matrix(df_y2, nomic_u, uids_y2, pause_cut_s=30.0, pca_dim=256)

    oof_port_gru = np.zeros(len(df_y2), dtype=np.float32)
    y1_port_gru = np.zeros(len(df_y1), dtype=np.float32)
    for f in range(5):
        tr_m = folds != f
        sc_f = StandardScaler().fit(cb_y2[tr_m])
        cb2_f = (sc_f.transform(cb_y2) * 0.20).astype(np.float32)
        cb1_f = (sc_f.transform(cb_y1) * 0.20).astype(np.float32)
        seqs2_f, spans2_f = _build_nomic_only_sequences(df_y2, nomic_u, uids_y2, cb2_f)
        seqs1_f, spans1_f = _build_nomic_only_sequences(df_y1, nomic_u, uids_y1, cb1_f)
        g_f = _train_chunked_nomic_bigru(
            seqs2_f, spans2_f, labels, scr, gaps, has_pgkd, llm1, llm2, llm_cons, eval_s2,
            include_span_fn=lambda s, e, ff=f: int(folds[s]) != ff,
            epochs=7, proj_dim=160, hidden=128, seed=2100 + f,
        )
        with torch.no_grad():
            for x_np, (s, e) in zip(seqs2_f, spans2_f):
                if int(folds[s]) == f:
                    tl, _ = g_f(torch.from_numpy(x_np).unsqueeze(0))
                    oof_port_gru[s:e] = torch.sigmoid(tl.squeeze(0)).numpy()
            for x_np, (s, e) in zip(seqs1_f, spans1_f):
                tl, _ = g_f(torch.from_numpy(x_np).unsqueeze(0))
                y1_port_gru[s:e] += torch.sigmoid(tl.squeeze(0)).numpy() / 5.0

        # Also fit & save fold-strict TF-IDF + Nomic LogReg + BiGRU so Y2 sibling files have ZERO same-room leakage!
        pos_f = np.flatnonzero(tr_m & scr & (labels == 1))
        bn_f = np.flatnonzero(tr_m & scr & (labels == 0))
        if has_pgkd:
            u0_spos_f = np.flatnonzero(tr_m & scr & (labels == -1) & eval_s2 & (llm1 >= 0.70) & (llm2 >= 0.60))
            u0_sneg_f = np.flatnonzero(tr_m & scr & (labels == -1) & ((~eval_s2) | (llm_cons <= 0.12)))
        else:
            u0_spos_f = np.array([], dtype=int)
            u0_sneg_f = np.flatnonzero(tr_m & scr & (labels == -1))
        rng_f = np.random.default_rng(2200 + f)
        u0_pick_f = rng_f.choice(u0_sneg_f, size=min(len(u0_sneg_f), 4 * (len(pos_f) + len(bn_f))), replace=False)
        tr_idx_f = np.concatenate([pos_f, u0_spos_f, bn_f, u0_pick_f])
        tr_y_f = np.concatenate([np.ones(len(pos_f)), np.ones(len(u0_spos_f)), np.zeros(len(bn_f)), np.zeros(len(u0_pick_f))]).astype(int)
        bn_wf = np.where(llm_cons[bn_f] >= 0.70, 0.35, 1.0) if has_pgkd else np.ones(len(bn_f))
        tr_w_f = np.concatenate([np.full(len(pos_f), 3.5), np.full(len(u0_spos_f), 1.0), bn_wf, np.full(len(u0_pick_f), 0.30)]).astype(np.float32)

        vec_tw_f = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=35000, sublinear_tf=True)
        vec_tc_f = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=25000, sublinear_tf=True)
        vec_cw_f = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=25000, sublinear_tf=True)
        X_lex_f = sparse.hstack([
            vec_tw_f.fit_transform([t_txt[i] for i in tr_idx_f]),
            vec_tc_f.fit_transform([t_txt[i] for i in tr_idx_f]),
            vec_cw_f.fit_transform([c_txt[i] for i in tr_idx_f]) * 0.85,
            sparse.csr_matrix(cb2_f[tr_idx_f]),
        ], format="csr")
        clf_lex_f = LogisticRegression(C=1.2, max_iter=500, solver="lbfgs").fit(X_lex_f, tr_y_f, sample_weight=tr_w_f)
        X_nom_f = np.hstack([X_nom_ctx_all[tr_idx_f], nomic_u[uids_y2[tr_idx_f]].astype(np.float32), cb2_f[tr_idx_f]])
        clf_nom_f = LogisticRegression(C=0.4, max_iter=400, solver="lbfgs").fit(X_nom_f, tr_y_f, sample_weight=tr_w_f)

        with (fold_dir / f"portable_models_fold_{f}.pkl").open("wb") as fh:
            pickle.dump({
                "scaler": sc_f, "cb_cols": cb_cols_keep,
                "vec_tw": vec_tw_f, "vec_tc": vec_tc_f, "vec_cw": vec_cw_f,
                "clf_lex": clf_lex_f, "clf_nomic": clf_nom_f,
            }, fh)
        torch.save(
            {"in_dim": seqs2_f[0].shape[1], "proj_dim": 160, "hidden": 128, "state_dict": g_f.state_dict()},
            fold_dir / f"pgkd_bigru_fold_{f}.pt",
        )
        if verbose:
            print(f"  [deploy_oof] Saved held-out classroom fold {f}/4 bundle ({time.time() - t0:.1f}s)", flush=True)

    # Combine with ALREADY strictly fold-isolated OOFs of 1B_nomic_ctx_logreg & 0B_tfidf_context!
    p_nom_lr_oof = np.load(PREDS_DIR / "1B_nomic_ctx_logreg_y2_oof.npy").astype(np.float32)
    p_nom_lr_y1 = np.load(PREDS_DIR / "1B_nomic_ctx_logreg_y1.npy").astype(np.float32)
    p_lex_oof = np.load(PREDS_DIR / "0B_tfidf_context_y2_oof.npy").astype(np.float32)
    p_lex_y1 = np.load(PREDS_DIR / "0B_tfidf_context_y1.npy").astype(np.float32)

    oof_port_local = expit(
        0.55 * _safe_logit(oof_port_gru) + 0.33 * _safe_logit(p_nom_lr_oof) + 0.12 * _safe_logit(p_lex_oof)
    ).astype(np.float32)
    y1_port_local = expit(
        0.55 * _safe_logit(y1_port_gru) + 0.33 * _safe_logit(p_nom_lr_y1) + 0.12 * _safe_logit(p_lex_y1)
    ).astype(np.float32)

    # Exact portable Hybrid Cascade OOF (same zero-leakage log-odds fusion formula used at inference!)
    oof_port_hyb = expit(
        0.58 * _safe_logit(oof_port_local) + 0.42 * _safe_logit(np.clip(llm_cons, 0.03, 0.97))
    ).astype(np.float32)
    y1_port_hyb = expit(
        0.58 * _safe_logit(y1_port_local) + 0.42 * _safe_logit(np.clip(llm_cons_y1, 0.03, 0.97))
    ).astype(np.float32)

    # Evaluate exact portable OOF metrics & calibrate exact BCTS + USF ±2 cluster/veto thresholds!
    valid_all = scr & (labels_eval != -2)
    y_all = (labels_eval == 1).astype(int)
    rev_m = scr & np.isin(labels_eval, [0, 1])

    # Strictly fold-isolated OOF BCTS -> USF ±2 cluster (-0.60 logit) & Stage-2 LLM veto -> OOF thresholds!
    oof_loc_res = oof_bcts_and_thresholds(oof_port_local, y_all, folds, valid_all, recall_targets=(0.90,))
    oof_hyb_res = oof_bcts_and_thresholds(oof_port_hyb, y_all, folds, valid_all, recall_targets=(0.90,))

    cal_oof_loc_adj, _ = apply_usf_cluster_and_veto_rules(
        df_y2, oof_loc_res["cal_oof"], llm_scores=None, llm_eval=None,
        support_thr=0.05, min_cluster=3, penalty_logit=0.60, pause_cut_s=30.0,
    )
    cal_oof_hyb_adj, _ = apply_usf_cluster_and_veto_rules(
        df_y2, oof_hyb_res["cal_oof"],
        llm_scores=(llm_cons if has_pgkd else None),
        llm_eval=(eval_s2 if has_pgkd else None),
        support_thr=0.05, min_cluster=3, penalty_logit=0.60, pause_cut_s=30.0,
    )

    ev_loc = evaluate_y2_oof(df_y2, cal_oof_loc_adj, model_name="portable_local_v3_usf", calibrate_oof=False)
    ev_hyb = evaluate_y2_oof(df_y2, cal_oof_hyb_adj, model_name="portable_hybrid_v3_usf", calibrate_oof=False)

    # Tune standalone & screening thresholds strictly OOF on CLEAN USF labels (`labels != -2`, excluding
    # the 85 vetoed legacy shape/count/bare-look TPs so Screening R>=90% doesn't collapse to recapture traps!)
    valid_usf = scr & (labels != -2)
    y_usf = (labels == 1).astype(int)

    a_loc, b_loc = fit_bcts(oof_port_local[valid_all], y_all[valid_all], is_probs=True)
    cal_loc_all = apply_bcts(oof_port_local, a_loc, b_loc, is_probs=True)
    thr_stand_loc = float(np.mean([tune_threshold_f1(y_usf[valid_usf & (folds != f)], cal_oof_loc_adj[valid_usf & (folds != f)]) for f in range(5)]))
    thr_screen_loc = float(np.mean([tune_threshold_recall(y_usf[valid_usf & (folds != f)], cal_oof_loc_adj[valid_usf & (folds != f)], 0.90) for f in range(5)]))
    thr_s1_gate = tune_threshold_recall(y_usf[rev_m & valid_usf], cal_loc_all[rev_m & valid_usf], 0.96)

    a_hyb, b_hyb = fit_bcts(oof_port_hyb[valid_all], y_all[valid_all], is_probs=True)
    thr_stand_hyb = float(np.mean([tune_threshold_f1(y_usf[valid_usf & (folds != f)], cal_oof_hyb_adj[valid_usf & (folds != f)]) for f in range(5)]))
    thr_screen_hyb = float(np.mean([tune_threshold_recall(y_usf[valid_usf & (folds != f)], cal_oof_hyb_adj[valid_usf & (folds != f)], 0.90) for f in range(5)]))

    y1_loc_adj, _ = apply_usf_cluster_and_veto_rules(
        df_y1, apply_bcts(y1_port_local, a_loc, b_loc, is_probs=True),
        llm_scores=None, llm_eval=None, support_thr=0.05, min_cluster=3, penalty_logit=0.60, pause_cut_s=30.0,
    )
    y1_hyb_adj, _ = apply_usf_cluster_and_veto_rules(
        df_y1, apply_bcts(y1_port_hyb, a_hyb, b_hyb, is_probs=True),
        llm_scores=(llm_cons_y1 if has_pgkd else None),
        llm_eval=(eval_s2_y1 if has_pgkd else None),
        support_thr=0.05, min_cluster=3, penalty_logit=0.60, pause_cut_s=30.0,
    )
    ev_loc_y1 = evaluate_y1_full_recall(
        df_y1, y1_loc_adj,
        thr_standalone=thr_stand_loc, thr_screen90=thr_screen_loc, model_name="portable_local_v3_usf",
    )
    ev_hyb_y1 = evaluate_y1_full_recall(
        df_y1, y1_hyb_adj,
        thr_standalone=thr_stand_hyb, thr_screen90=thr_screen_hyb, model_name="portable_hybrid_v3_usf",
    )
    if verbose:
        print(
            f"  [deploy_oof] Portable Local  | Rev AUC={ev_loc['reviewed_flags']['auroc']:.4f} "
            f"F1={ev_loc['reviewed_flags']['standalone']['f1']:.4f} | Y1 AUC={ev_loc_y1['auroc']:.4f} "
            f"AP={ev_loc_y1['pr_auc']:.4f} TransferF1={ev_loc_y1['at_y2_standalone_thr']['f1']:.4f}",
            flush=True,
        )
        print(
            f"  [deploy_oof] Portable Hybrid | Rev AUC={ev_hyb['reviewed_flags']['auroc']:.4f} "
            f"F1={ev_hyb['reviewed_flags']['standalone']['f1']:.4f} | Y1 AUC={ev_hyb_y1['auroc']:.4f} "
            f"AP={ev_hyb_y1['pr_auc']:.4f} TransferF1={ev_hyb_y1['at_y2_standalone_thr']['f1']:.4f}",
            flush=True,
        )

    # 2. Fit full-data models on 100% of verified Y2 data (+ PGKD silver) for brand-new unseen classrooms!
    scaler = StandardScaler()
    cb_sc = (scaler.fit_transform(cb_y2) * 0.20).astype(np.float32)

    pos_i = np.flatnonzero(scr & (labels == 1))
    bn_i = np.flatnonzero(scr & (labels == 0))
    if has_pgkd:
        u0_spos = np.flatnonzero(scr & (labels == -1) & eval_s2 & (llm1 >= 0.70) & (llm2 >= 0.60))
        u0_sneg = np.flatnonzero(scr & (labels == -1) & ((~eval_s2) | (llm_cons <= 0.12)))
    else:
        u0_spos = np.array([], dtype=int)
        u0_sneg = np.flatnonzero(scr & (labels == -1))

    rng = np.random.default_rng(2026)
    u0_pick = rng.choice(u0_sneg, size=min(len(u0_sneg), 4 * (len(pos_i) + len(bn_i))), replace=False)
    tr_idx = np.concatenate([pos_i, u0_spos, bn_i, u0_pick])
    tr_y_bin = np.concatenate([np.ones(len(pos_i)), np.ones(len(u0_spos)), np.zeros(len(bn_i)), np.zeros(len(u0_pick))]).astype(int)
    bn_w = np.where(llm_cons[bn_i] >= 0.70, 0.35, 1.0) if has_pgkd else np.ones(len(bn_i))
    tr_w = np.concatenate([np.full(len(pos_i), 3.5), np.full(len(u0_spos), 1.0), bn_w, np.full(len(u0_pick), 0.30)]).astype(np.float32)

    # Full-data 0B Contextual TF-IDF LogReg
    vec_tw = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=35000, sublinear_tf=True)
    vec_tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=25000, sublinear_tf=True)
    vec_cw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=25000, sublinear_tf=True)
    X_lex = sparse.hstack([
        vec_tw.fit_transform([t_txt[i] for i in tr_idx]),
        vec_tc.fit_transform([t_txt[i] for i in tr_idx]),
        vec_cw.fit_transform([c_txt[i] for i in tr_idx]) * 0.85,
        sparse.csr_matrix(cb_sc[tr_idx]),
    ], format="csr")
    clf_lex = LogisticRegression(C=1.2, max_iter=500, solver="lbfgs").fit(X_lex, tr_y_bin, sample_weight=tr_w)

    # Full-data 1B Nomic Multi-Scale Context LogReg (replaces ContextMLP -> +0.16 Y1 PR-AUC!)
    X_nom_full = np.hstack([X_nom_ctx_all[tr_idx], nomic_u[uids_y2[tr_idx]].astype(np.float32), cb_sc[tr_idx]])
    clf_nomic = LogisticRegression(C=0.4, max_iter=400, solver="lbfgs").fit(X_nom_full, tr_y_bin, sample_weight=tr_w)

    # Full-data Portable PGKD BiGRU-Episode Net
    seqs_all, spans_all = _build_nomic_only_sequences(df_y2, nomic_u, uids_y2, cb_sc)
    bigru = _train_chunked_nomic_bigru(
        seqs_all, spans_all, labels, scr, gaps, has_pgkd, llm1, llm2, llm_cons, eval_s2,
        include_span_fn=lambda s, e: True, epochs=8, proj_dim=160, hidden=128, seed=2026,
    )

    # Pick winning Stage-2 Primary + Secondary LLMs from meta
    primary_llm = {"model": "gemma-4-31b-it", "effort": None}
    secondary_llm = {"model": "gpt-oss-20b", "effort": "low"}
    if (S2_DIR / "full83_hybrid_meta.json").exists():
        meta_s2 = json.loads((S2_DIR / "full83_hybrid_meta.json").read_text(encoding="utf-8"))
        primary_llm = meta_s2.get("primary_llm", primary_llm)
        secondary_llm = meta_s2.get("secondary_llm", secondary_llm)

    _VETO_ONLY = {"cb_trap_shape_only", "cb_trap_count_only", "cb_trap_look_no_content"}
    with (MODELS_DIR / "portable_models.pkl").open("wb") as fh:
        pickle.dump({
            "scaler": scaler,
            "cb_cols": [c for c in df_y2.columns if c.startswith("cb_") and c not in _VETO_ONLY],
            "vec_tw": vec_tw,
            "vec_tc": vec_tc,
            "vec_cw": vec_cw,
            "clf_lex": clf_lex,
            "clf_nomic": clf_nomic,
        }, fh)
    torch.save(
        {"in_dim": seqs_all[0].shape[1], "proj_dim": 160, "hidden": 128, "state_dict": bigru.state_dict()},
        MODELS_DIR / "pgkd_bigru.pt",
    )
    # Remove obsolete ContextMLP artifact if present
    (MODELS_DIR / "nomic_ctx_mlp.pt").unlink(missing_ok=True)
    (MODELS_DIR / "lexical_and_scaler.pkl").unlink(missing_ok=True)

    # Save exact classroom -> fold map so `score` routes ANY Y2 training classroom to its held-out fold!
    room_to_fold = {
        str(room): int(g["fold"].iloc[0])
        for room, g in df_y2.groupby("classroom")
    }

    cfg = {
        "bundle_version": "v3.1_usf_pm2_veto_5fold_leakfree",
        "has_pgkd_distillation": has_pgkd,
        "has_heldout_fold_models": True,
        "classroom_to_fold": room_to_fold,
        "local_weights": {"pgkd_bigru": 0.55, "nomic_ctx_logreg": 0.33, "tfidf_ctx_logreg": 0.12},
        "hybrid_weights": {"stage1_local_logit": 0.58, "stage2_llm_logit": 0.42},
        "usf_pm2_cluster": {"support_thr": 0.05, "min_cluster": 3, "penalty_logit": 0.60, "pause_cut_s": 30.0},
        "bcts_local": {"a": a_loc, "b": b_loc},
        "bcts_hybrid": {"a": a_hyb, "b": b_hyb},
        "thr_local_standalone": round(thr_stand_loc, 4),
        "thr_local_screening_r90": round(thr_screen_loc, 4),
        "thr_stage1_llm_gate_r96": round(thr_s1_gate, 4),
        "thr_hybrid_standalone": round(thr_stand_hyb, 4),
        "thr_hybrid_screening_r90": round(thr_screen_hyb, 4),
        "llm_window_size": 26,
        "pi_train_y2": round(float(y_all[valid_all].mean()), 6),
        "primary_llm": primary_llm,
        "secondary_llm": secondary_llm,
        "portable_oof_benchmark": {
            "local_y2_rev_auroc": ev_loc["reviewed_flags"]["auroc"],
            "local_y2_rev_f1": ev_loc["reviewed_flags"]["standalone"]["f1"],
            "local_y1_auroc": ev_loc_y1["auroc"],
            "local_y1_pr_auc": ev_loc_y1["pr_auc"],
            "local_y1_transfer_f1": ev_loc_y1["at_y2_standalone_thr"]["f1"],
            "hybrid_y2_rev_auroc": ev_hyb["reviewed_flags"]["auroc"],
            "hybrid_y2_rev_f1": ev_hyb["reviewed_flags"]["standalone"]["f1"],
            "hybrid_y1_auroc": ev_hyb_y1["auroc"],
            "hybrid_y1_pr_auc": ev_hyb_y1["pr_auc"],
            "hybrid_y1_transfer_f1": ev_hyb_y1["at_y2_standalone_thr"]["f1"],
            "note": "Portable hybrid OOF uses Stage-2 cached scores inside 4C's R96 gate (13.9% of turns); live scoring applies portable Stage-1 R96 gate directly.",
        },
    }
    (MODELS_DIR / "deploy_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if verbose:
        print(f"[deploy] Trained & calibrated portable production bundle -> {MODELS_DIR} in {time.time() - t0:.1f}s", flush=True)
    return MODELS_DIR


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Case-insensitive exact or substring column match for real USF transcript schemas."""
    low = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for cand in candidates:
        for lc, orig in low.items():
            if cand in lc:
                return orig
    return None


def load_raw_transcript_folder(folder_path: Path) -> pd.DataFrame:
    """Read all focal `.xlsx` transcripts in a new classroom folder into chronological DataFrame.

    Matches real USF Y2 raw Excel headers (`Segment Start Time`, `Segment End Time`,
    `Transcription (Confidence)`, `Speaker`) as well as simplified headers, skips `"-"`
    placeholder turns, keeps `unknown`/low-confidence turns for sequence context, and sets
    `screened = (speaker in {"adult", "child"}) & (asr_conf is None or asr_conf >= 0.40)`.
    """
    files = sorted([p for p in folder_path.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise FileNotFoundError(f"No .xlsx transcript files found in {folder_path}")

    rows: list[dict[str, Any]] = []
    for fpath in files:
        cid = child_id_from_filename(fpath.name)
        df = pd.read_excel(fpath, engine="openpyxl")
        col_text = _find_col(df, ("transcription (confidence)", "transcription", "utterance", "text"))
        if col_text is None:
            raise ValueError(f"Could not find transcription column in {fpath.name}: columns={list(df.columns)}")
        col_spk = _find_col(df, ("speaker",))
        col_s = _find_col(df, ("segment start time", "segment start", "seg_start", "start time"))
        col_e = _find_col(df, ("segment end time", "segment end", "seg_end", "end time"))

        tx_id = f"{folder_path.name}::{fpath.name}"
        prev_end = None
        for idx, r in df.iterrows():
            text, conf = parse_transcription(r[col_text])
            if not text or text == "-":
                continue
            spk = str(r[col_spk]).strip().lower() if col_spk and pd.notna(r[col_spk]) else "unknown"
            if spk not in ("adult", "child"):
                spk = "unknown"
            ss_raw = "" if not col_s or pd.isna(r[col_s]) else str(r[col_s])
            se_raw = "" if not col_e or pd.isna(r[col_e]) else str(r[col_e])
            ss = to_seconds(ss_raw)
            se = to_seconds(se_raw)
            gap_prev = 0.0 if (prev_end is None or np.isnan(ss) or np.isnan(prev_end)) else max(0.0, ss - prev_end)
            if not np.isnan(se):
                prev_end = se
            is_screened = (spk in ("adult", "child")) and (conf is None or conf >= 0.40)
            cb = extract_codebook_features(text, spk)
            rows.append({
                "transcript_id": tx_id,
                "folder": folder_path.name,
                "source_file": fpath.name,
                "child_id": cid,
                "line_no": int(idx) + 2,
                "seg_start": ss_raw,
                "seg_end": se_raw,
                "seg_start_s": ss,
                "gap_prev_s": gap_prev,
                "gap_next_s": 0.0,
                "speaker": spk,
                "screened": bool(is_screened),
                "utterance": text,
                "asr_conf": conf,
                **{f"cb_{k}": v for k, v in cb.items()},
            })
    out = pd.DataFrame(rows)
    if len(out):
        tx = out["transcript_id"].to_numpy()
        gp = out["gap_prev_s"].to_numpy(dtype=np.float32)
        gn = np.zeros_like(gp)
        same = tx[:-1] == tx[1:]
        gn[:-1] = np.where(same, gp[1:], 0.0)
        out["gap_next_s"] = gn
    return out


def _extract_classroom_id(folder_name: str) -> str | None:
    import re
    m = re.match(r"Classroom(\w+?)_(\d{6})", str(folder_name), flags=re.I)
    return m.group(1) if m else None


def score_new_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    engine: str = "hybrid",
    mode: str = "both",
    budget_cap_usd: float = 5.0,
    science_only: bool = True,
    use_secondary_verifier: bool = True,
    force_full_model: bool = False,
) -> dict[str, Any]:
    """Score a classroom transcript folder and write 21-column Human coded workbooks.

    Zero-leakage classroom routing:
      - If `input_dir` matches one of the 83 Y2 training classrooms (e.g., scoring the other
        567 sibling-child Y2 files!), automatically loads `models/v3/fold_models/fold_{f}`
        which NEVER saw a single utterance from that classroom during training!
      - If `input_dir` is a brand-new unseen classroom (or `force_full_model=True`), loads
        the full 83-room bundle (`models/v3/portable_models.pkl` + `pgkd_bigru.pt`).
    """
    if not (MODELS_DIR / "deploy_config.json").exists():
        raise FileNotFoundError(f"Missing {MODELS_DIR / 'deploy_config.json'}. Run `train-artifacts` first.")

    cfg = json.loads((MODELS_DIR / "deploy_config.json").read_text(encoding="utf-8"))
    room_id = _extract_classroom_id(input_dir.name)
    room_to_fold = cfg.get("classroom_to_fold", {})
    fold_dir = MODELS_DIR / "fold_models"

    if (
        not force_full_model
        and room_id is not None
        and room_id in room_to_fold
        and (fold_dir / f"portable_models_fold_{room_to_fold[room_id]}.pkl").exists()
        and (fold_dir / f"pgkd_bigru_fold_{room_to_fold[room_id]}.pt").exists()
    ):
        held_f = int(room_to_fold[room_id])
        pkl_path = fold_dir / f"portable_models_fold_{held_f}.pkl"
        gru_path = fold_dir / f"pgkd_bigru_fold_{held_f}.pt"
        model_selection = f"held_out_fold_{held_f} (zero same-classroom leakage for Y2 classroom {room_id})"
    else:
        pkl_path = MODELS_DIR / "portable_models.pkl"
        gru_path = MODELS_DIR / "pgkd_bigru.pt"
        model_selection = f"full_83_room_bundle ({'forced' if force_full_model else 'brand-new unseen classroom'})"

    with pkl_path.open("rb") as fh:
        bundle = pickle.load(fh)

    df = load_raw_transcript_folder(input_dir)
    if len(df) == 0:
        raise ValueError(f"Loaded 0 valid utterances from {input_dir}")

    client = UFNavigatorClient(budget_cap_usd=budget_cap_usd)
    nomic_u, uids = _embed_unique_nomic(df, client)

    # Align codebook feature columns strictly to training bundle order
    cb_cols = bundle["cb_cols"]
    for c in cb_cols:
        if c not in df.columns:
            df[c] = 0.0
    cb_core = df[cb_cols].to_numpy(dtype=np.float32)
    gp = np.log1p(np.clip(df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32), 0.0, 600.0))[:, None]
    gn = np.log1p(np.clip(df["gap_next_s"].fillna(0.0).to_numpy(dtype=np.float32), 0.0, 600.0))[:, None]
    p_cut = (df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32) > 30.0).astype(np.float32)[:, None]
    conf = df["asr_conf"].fillna(0.75).to_numpy(dtype=np.float32)[:, None]
    cb = np.hstack([cb_core, gp, gn, p_cut, conf])
    cb_sc = (bundle["scaler"].transform(cb) * 0.20).astype(np.float32)

    # 1. Contextual TF-IDF LogReg
    t_txt, c_txt = build_pause_bounded_window_texts(df, before=5, after=3, pause_cut_s=30.0)
    X_lex = sparse.hstack([
        bundle["vec_tw"].transform(t_txt),
        bundle["vec_tc"].transform(t_txt),
        bundle["vec_cw"].transform(c_txt) * 0.85,
        sparse.csr_matrix(cb_sc),
    ], format="csr")
    p_lex = bundle["clf_lex"].predict_proba(X_lex)[:, 1].astype(np.float32)

    # 2. Nomic Multi-Scale Context LogReg
    X_nom_ctx = build_pause_bounded_context_matrix(df, nomic_u, uids, pause_cut_s=30.0, pca_dim=256)
    X_nom_full = np.hstack([X_nom_ctx, nomic_u[uids].astype(np.float32), cb_sc])
    p_nom = bundle["clf_nomic"].predict_proba(X_nom_full)[:, 1].astype(np.float32)

    # 3. Portable PGKD BiGRU-Episode Net (held-out classroom fold or full bundle!)
    seqs, spans = _build_nomic_only_sequences(df, nomic_u, uids, cb_sc)
    ck_g = torch.load(gru_path, map_location="cpu", weights_only=True)
    bigru = BiGRUEpisodeNet(
        in_dim=ck_g["in_dim"],
        proj_dim=int(ck_g.get("proj_dim", 256)),
        hidden=int(ck_g.get("hidden", 192)),
        dropout=0.40,
    )
    bigru.load_state_dict(ck_g["state_dict"])
    bigru.eval()
    p_gru = np.zeros(len(df), dtype=np.float32)
    with torch.no_grad():
        for x_np, (s, e) in zip(seqs, spans):
            tl, _ = bigru(torch.from_numpy(x_np).unsqueeze(0))
            p_gru[s:e] = torch.sigmoid(tl.squeeze(0)).numpy()

    # Exact calibrated portable local ensemble
    w_loc = cfg["local_weights"]
    raw_local = expit(
        w_loc["pgkd_bigru"] * _safe_logit(p_gru)
        + w_loc["nomic_ctx_logreg"] * _safe_logit(p_nom)
        + w_loc["tfidf_ctx_logreg"] * _safe_logit(p_lex)
    )
    cal_local = apply_bcts(raw_local, cfg["bcts_local"]["a"], cfg["bcts_local"]["b"], is_probs=True)

    screened_mask = df["screened"].to_numpy(dtype=bool)
    llm_scores = np.full(len(df), np.nan)
    llm_eval_mask: np.ndarray | None = None
    llm_rats = [""] * len(df)
    llm_subs = [""] * len(df)
    cal_unadjusted = cal_local.copy()
    scored_by = "local_v3"

    if engine == "hybrid":
        cand_mask = screened_mask & (cal_local >= float(cfg["thr_stage1_llm_gate_r96"]))
        w_size = int(cfg.get("llm_window_size", 26))
        pllm = cfg["primary_llm"]
        res1 = score_transcripts_with_llm(
            df, client=client, model=pllm["model"], reasoning_effort=pllm.get("effort"),
            target_mask=cand_mask, window_size=w_size, expand_radius=5, max_workers=4, verbose=True,
        )
        llm_comb = res1["scores"].copy()
        llm_rats = list(res1["rationale"])
        llm_subs = list(res1["subtype"])

        sllm = cfg.get("secondary_llm")
        if use_secondary_verifier and sllm and sllm.get("model"):
            v_mask = (res1["scores"] >= 0.30) | (screened_mask & (raw_local >= 0.65) & (res1["scores"] < 0.30))
            if v_mask.any():
                res2 = score_transcripts_with_llm(
                    df, client=client, model=sllm["model"], reasoning_effort=sllm.get("effort"),
                    target_mask=v_mask, window_size=w_size, expand_radius=3, max_workers=4, verbose=True,
                )
                l2 = np.where(res2["evaluated"], res2["scores"], res1["scores"])
                llm_comb = 0.60 * res1["scores"] + 0.40 * l2
                for i in range(len(df)):
                    if not llm_rats[i] and res2["rationale"][i]:
                        llm_rats[i] = res2["rationale"][i]
                    if not llm_subs[i] and res2["subtype"][i]:
                        llm_subs[i] = res2["subtype"][i]

        llm_eval_mask = res1["evaluated"].astype(bool)
        llm_scores = np.where(llm_eval_mask, llm_comb, np.nan)
        w_h = cfg["hybrid_weights"]
        raw_hyb = expit(
            w_h["stage1_local_logit"] * _safe_logit(raw_local)
            + w_h["stage2_llm_logit"] * _safe_logit(np.clip(llm_comb, 0.03, 0.97))
        )
        cal_unadjusted = apply_bcts(raw_hyb, cfg["bcts_hybrid"]["a"], cfg["bcts_hybrid"]["b"], is_probs=True)
        scored_by = "hybrid_v3"

    # Apply USF ±2 chronological cluster rule (-0.60 logit if <3 turns in ±2) & Stage-2 LLM trap veto!
    pm2_cfg = cfg.get("usf_pm2_cluster", {"support_thr": 0.05, "min_cluster": 3, "penalty_logit": 0.60, "pause_cut_s": 30.0})
    final_scores, pm2_cnt = apply_usf_cluster_and_veto_rules(
        df,
        cal_unadjusted,
        llm_scores=(np.nan_to_num(llm_scores, nan=0.02) if engine == "hybrid" else None),
        llm_eval=llm_eval_mask,
        support_thr=float(pm2_cfg["support_thr"]),
        min_cluster=int(pm2_cfg["min_cluster"]),
        penalty_logit=float(pm2_cfg["penalty_logit"]),
        pause_cut_s=float(pm2_cfg["pause_cut_s"]),
    )
    # Build chronological ±2 context strings from ALL turns (before screening filter!)
    ctx_before_2, ctx_after_2 = extract_pm2_context_columns(df, pause_cut_s=float(pm2_cfg["pause_cut_s"]))

    modes_to_run = ["screening", "standalone"] if mode == "both" else [mode]
    all_files = sorted(df["source_file"].unique().tolist())
    out_summary: dict[str, Any] = {
        "folder": input_dir.name,
        "model_selection": model_selection,
        "n_turns_total": int(len(df)),
        "n_turns_screened": int(screened_mask.sum()),
        "engine": engine,
        "modes": {},
    }

    for m in modes_to_run:
        if engine == "hybrid":
            thr = float(cfg["thr_hybrid_screening_r90"] if m == "screening" else cfg["thr_hybrid_standalone"])
        else:
            thr = float(cfg["thr_local_screening_r90"] if m == "screening" else cfg["thr_local_standalone"])
        recs = []
        n_sci = 0
        for gi in range(len(df)):
            sc = float(final_scores[gi])
            # Only screened adult/child >=0.40-conf turns can be flagged SCIENCE_TALK!
            lbl = LABEL_SCIENCE if (screened_mask[gi] and sc >= thr) else LABEL_NOT
            if lbl == LABEL_SCIENCE:
                n_sci += 1
            sub = llm_subs[gi] if llm_subs[gi] else infer_subtype_rule(str(df["utterance"].iat[gi]))
            recs.append({
                "source_file": str(df["source_file"].iat[gi]),
                "child_id": str(df["child_id"].iat[gi]),
                "line_no": int(df["line_no"].iat[gi]),
                "seg_start": str(df["seg_start"].iat[gi]),
                "seg_end": str(df["seg_end"].iat[gi]),
                "speaker": str(df["speaker"].iat[gi]),
                "context_before_2": ctx_before_2[gi],
                "utterance": str(df["utterance"].iat[gi]),
                "context_after_2": ctx_after_2[gi],
                "confidence": None if pd.isna(df["asr_conf"].iat[gi]) else round(float(df["asr_conf"].iat[gi]), 3),
                "predicted_label": lbl,
                "review": None,
                "score": round(sc, 4),
                "pm2_sci_count": int(pm2_cnt[gi]),
                "scored_by": scored_by,
                "cosine_score": round(float(cal_local[gi]), 4),
                "llm_score": None if np.isnan(llm_scores[gi]) else round(float(llm_scores[gi]), 4),
                "top_match_in_corpus": None,
                "top_match_score": None,
                "predicted_subtype": sub,
                "rationale": llm_rats[gi] if llm_rats[gi] else None,
                "degraded": False,
            })
        wb_path = write_human_coded_workbook(
            input_dir.name, recs, output_dir / m,
            mode=m, threshold=thr, engine=scored_by,
            science_only=science_only, all_source_files=all_files,
        )
        out_summary["modes"][m] = {"workbook": str(wb_path), "threshold": thr, "predicted_science": n_sci}

    cal_mean_pct = round(100.0 * float(cal_unadjusted[screened_mask].mean()), 2) if screened_mask.any() else 0.0
    out_summary["calibrated_mean_science_pct"] = cal_mean_pct
    out_summary["ledger"] = client.ledger.summary()
    print(json.dumps(out_summary, indent=2))
    return out_summary


def main() -> None:
    ap = argparse.ArgumentParser(description="TeLI Science Talk v3 Production Trainer & Unseen-Folder Scorer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train-artifacts", help="Train and save calibrated portable production bundle in models/v3/")
    sc = sub.add_parser("score", help="Score a brand-new unseen classroom transcript folder")
    sc.add_argument("--input-dir", type=Path, required=True)
    sc.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "v3" / "new_predictions")
    sc.add_argument("--engine", choices=["hybrid", "local"], default="hybrid")
    sc.add_argument("--mode", choices=["screening", "standalone", "both"], default="both")
    sc.add_argument("--budget", type=float, default=5.0)
    sc.add_argument("--no-secondary", action="store_true", help="Use only Primary LLM in hybrid mode")
    sc.add_argument("--full-transcript", action="store_true")
    sc.add_argument("--force-full-model", action="store_true", help="Force full 83-room bundle even if classroom is in Y2 training rooms")

    args = ap.parse_args()
    if args.cmd == "train-artifacts":
        train_deployable_artifacts(verbose=True)
    else:
        score_new_folder(
            args.input_dir, args.output_dir,
            engine=args.engine, mode=args.mode,
            budget_cap_usd=args.budget,
            science_only=not args.full_transcript,
            use_secondary_verifier=not args.no_secondary,
            force_full_model=args.force_full_model,
        )


if __name__ == "__main__":
    main()
