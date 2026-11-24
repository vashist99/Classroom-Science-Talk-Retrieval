"""One-time deduplicated, resumable multi-backbone embedding cache for TeLI v3.

Embeds all unique (speaker, utterance) strings across Y2 (all 83 focal transcripts)
and Y1 (40 Coding Transcripts sheets) + Y1 seed via UF NaviGator API:
  1. `nomic-embed-text-v1.5` (768-d, `classification: [speaker] utterance` prefix)
  2. `sfr-embedding-mistral` (4096-d Mistral-7B embeddings)

Stores compact float16 unique matrices + int32 row-index maps under
`data/v3/embeddings/` so 185k+ turns fit comfortably in <1.5 GB RAM on a 16 GB laptop.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.llm_client import UFNavigatorClient

DATA_DIR = PROJECT_ROOT / "data" / "v3"
EMB_DIR = DATA_DIR / "embeddings"


def make_canonical_keys(df: pd.DataFrame) -> list[str]:
    spks = df["speaker"].fillna("unknown").astype(str).str.strip().str.lower().tolist()
    utts = df["utterance"].fillna("").astype(str).str.strip().tolist()
    return [f"[{s}] {u}" for s, u in zip(spks, utts)]


def build_unique_vocab(data_dir: Path = DATA_DIR) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate (speaker, utterance) across Y2 + Y1 + seed into one shared table."""
    y2 = pd.read_parquet(data_dir / "y2_utterances.parquet", columns=["speaker", "utterance"])
    y1 = pd.read_parquet(data_dir / "y1_coding_transcripts.parquet", columns=["speaker", "utterance"])
    seed = pd.read_parquet(data_dir / "y1_seed.parquet", columns=["utterance"])
    seed["speaker"] = "adult"

    k_y2 = make_canonical_keys(y2)
    k_y1 = make_canonical_keys(y1)
    k_sd = make_canonical_keys(seed)

    vocab_map: dict[str, int] = {}
    vocab: list[str] = []

    def _encode_keys(keys: list[str]) -> np.ndarray:
        idx = np.empty(len(keys), dtype=np.int32)
        for i, k in enumerate(keys):
            uid = vocab_map.get(k)
            if uid is None:
                uid = len(vocab)
                vocab_map[k] = uid
                vocab.append(k)
            idx[i] = uid
        return idx

    idx_y2 = _encode_keys(k_y2)
    idx_y1 = _encode_keys(k_y1)
    idx_sd = _encode_keys(k_sd)
    return vocab, idx_y2, idx_y1, idx_sd


def _l2_normalize_f16(arr: np.ndarray) -> np.ndarray:
    arr32 = arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr32, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (arr32 / norms).astype(np.float16)


def _save_npy_safe(path: Path, arr: np.ndarray, retries: int = 10) -> None:
    """Save .npy retrying transiently on Windows/OneDrive file locks."""
    for attempt in range(retries):
        try:
            np.save(path, arr)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 + attempt)


def compute_and_cache_embeddings(
    model: str,
    *,
    prefix: str = "",
    batch_size: int = 128,
    max_workers: int = 8,
    budget_cap_usd: float = 5.0,
    data_dir: Path = DATA_DIR,
    emb_dir: Path = EMB_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Embed all unique vocab strings in parallel with resumable checkpointing."""
    emb_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model.replace("/", "_").replace(".", "_")
    vocab_path = emb_dir / "vocab.parquet"
    idx_path = emb_dir / "row_indices.npz"
    out_npy = emb_dir / f"{safe_name}_unique_f16.npy"
    ckpt_npy = emb_dir / f"{safe_name}_ckpt_f16.npy"
    ckpt_done = emb_dir / f"{safe_name}_ckpt_done.npy"
    meta_json = emb_dir / f"{safe_name}_meta.json"

    if vocab_path.exists() and idx_path.exists():
        vocab = pd.read_parquet(vocab_path)["key"].tolist()
    else:
        vocab, idx_y2, idx_y1, idx_sd = build_unique_vocab(data_dir)
        pd.DataFrame({"uid": np.arange(len(vocab), dtype=np.int32), "key": vocab}).to_parquet(
            vocab_path, index=False
        )
        np.savez_compressed(idx_path, y2=idx_y2, y1=idx_y1, seed=idx_sd)
        if verbose:
            print(
                f"[embed] Built shared unique vocab: {len(vocab):,} unique keys "
                f"(from Y2={len(idx_y2):,} + Y1={len(idx_y1):,} + seed={len(idx_sd):,})"
            )

    if out_npy.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        if meta.get("n_vocab") == len(vocab):
            if verbose:
                print(f"[embed] Reusing cached {out_npy.name} ({meta['n_vocab']:,} x {meta['dim']})")
            return meta

    client = UFNavigatorClient(budget_cap_usd=budget_cap_usd)
    n_vocab = len(vocab)

    # Probe 1 item to determine embedding dimension
    probe = client.embed_batch(model=model, texts=[prefix + vocab[0]])
    dim = len(probe[0])
    if verbose:
        print(f"[embed] Starting {model} (dim={dim}, n_vocab={n_vocab:,}, batch={batch_size}, workers={max_workers})")

    if ckpt_npy.exists() and ckpt_done.exists():
        mat = np.load(ckpt_npy)
        done = np.load(ckpt_done)
        if mat.shape != (n_vocab, dim) or len(done) != n_vocab:
            mat = np.zeros((n_vocab, dim), dtype=np.float16)
            done = np.zeros(n_vocab, dtype=bool)
    else:
        mat = np.zeros((n_vocab, dim), dtype=np.float16)
        done = np.zeros(n_vocab, dtype=bool)

    todo_indices = np.flatnonzero(~done)
    batches: list[np.ndarray] = [
        todo_indices[i : i + batch_size] for i in range(0, len(todo_indices), batch_size)
    ]
    if verbose:
        print(f"[embed] Remaining items: {len(todo_indices):,} ({len(batches):,} batches)")

    t0 = time.time()
    completed_batches = 0

    def _run_batch(idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        texts = [prefix + vocab[int(i)] for i in idxs]
        vecs = client.embed_batch(model=model, texts=texts)
        arr = _l2_normalize_f16(np.asarray(vecs, dtype=np.float32))
        return idxs, arr

    # Process in waves of 64 batches so we checkpoint cleanly and stay RAM-safe
    wave_size = max(16, max_workers * 8)
    for w_start in range(0, len(batches), wave_size):
        wave = batches[w_start : w_start + wave_size]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_run_batch, b) for b in wave]
            for fut in as_completed(futs):
                idxs, arr = fut.result()
                mat[idxs] = arr
                done[idxs] = True
                completed_batches += 1
        if not done.all():
            _save_npy_safe(ckpt_npy, mat)
            _save_npy_safe(ckpt_done, done)
        if verbose:
            pct = 100.0 * done.mean()
            dt = time.time() - t0
            led = client.ledger.summary()
            print(
                f"  [{model}] {done.sum():,}/{n_vocab:,} ({pct:.1f}%) | "
                f"{dt:.1f}s | ${led['usd_spent']:.4f}",
                flush=True,
            )

    assert done.all(), f"Incomplete embedding matrix: {(~done).sum()} missing"
    _save_npy_safe(out_npy, mat)
    for p in (ckpt_npy, ckpt_done):
        for _ in range(5):
            try:
                p.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(1.5)

    meta = {
        "model": model,
        "prefix": prefix,
        "n_vocab": n_vocab,
        "dim": dim,
        "dtype": "float16",
        "path": str(out_npy),
        "elapsed_s": round(time.time() - t0, 2),
        "ledger": client.ledger.summary(),
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"[embed] Saved {out_npy.name} ({n_vocab:,} x {dim}) in {meta['elapsed_s']}s | ${meta['ledger']['usd_spent']:.4f}")
    return meta


def load_unique_embeddings(model: str, emb_dir: Path = EMB_DIR) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    safe_name = model.replace("/", "_").replace(".", "_")
    mat = np.load(emb_dir / f"{safe_name}_unique_f16.npy", mmap_mode="r")
    idx_npz = np.load(emb_dir / "row_indices.npz")
    return mat, {"y2": idx_npz["y2"], "y1": idx_npz["y1"], "seed": idx_npz["seed"]}


def build_pause_bounded_context_matrix(
    df: pd.DataFrame,
    unique_emb: np.ndarray,
    row_uids: np.ndarray,
    *,
    pause_cut_s: float = 30.0,
    pca_dim: int | None = None,
    proj_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Build rich pause-bounded multi-scale contextual embedding features per turn.

    For each turn t inside its chronological transcript:
      - e_t (projected or raw)
      - e_{t-1}, e_{t+1} (zeroed if gap > pause_cut_s or transcript boundary)
      - mean(e_{t-3..t+3}), mean(e_{t-10..t+10}), mean(e_{t-20..t+20}) (pause-bounded!)
      - cosine similarities: cos(e_t, e_{t-1}), cos(e_t, e_{t+1}),
        cos(e_t, mean_{±3}), cos(e_t, mean_{±10}), cos(e_t, mean_{±20})
    Returns float32 array of shape (N, 6*D + 5).
    """
    n = len(df)
    if proj_matrix is not None:
        # Project unique vocab once -> huge RAM & speed win!
        u_proj = (unique_emb.astype(np.float32) @ proj_matrix).astype(np.float32)
        norms = np.maximum(np.linalg.norm(u_proj, axis=1, keepdims=True), 1e-8)
        u_proj = u_proj / norms
        d = u_proj.shape[1]
    else:
        u_proj = unique_emb.astype(np.float32)
        if pca_dim is not None and pca_dim < u_proj.shape[1]:
            # Matryoshka truncation + L2 renorm (valid for nomic-embed-text-v1.5!)
            u_proj = u_proj[:, :pca_dim].copy()
            norms = np.maximum(np.linalg.norm(u_proj, axis=1, keepdims=True), 1e-8)
            u_proj = u_proj / norms
        d = u_proj.shape[1]

    out = np.zeros((n, 6 * d + 5), dtype=np.float32)
    tx_ids = df["transcript_id"].to_numpy()
    gap_prev = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=np.float32)

    # Identify contiguous transcript spans
    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], n]

    for s, e in zip(starts, ends):
        m = e - s
        E = u_proj[row_uids[s:e]]  # (m, d)
        gp = gap_prev[s:e]

        # Split transcript into pause-bounded episodes where gap > pause_cut_s
        ep_breaks = np.flatnonzero(gp > pause_cut_s)
        ep_starts = np.r_[0, ep_breaks]
        ep_ends = np.r_[ep_breaks, m]

        for es, ee in zip(ep_starts, ep_ends):
            L = ee - es
            sub = E[es:ee]  # (L, d)
            gs = s + es
            ge = s + ee

            # 0: target e_t
            out[gs:ge, 0:d] = sub

            # 1: prev e_{t-1}
            if L > 1:
                out[gs + 1 : ge, d : 2 * d] = sub[:-1]
                out[gs : ge - 1, 2 * d : 3 * d] = sub[1:]

            # Prefix sums for O(L) exact sliding-window means!
            csum = np.zeros((L + 1, d), dtype=np.float32)
            np.cumsum(sub, axis=0, out=csum[1:])
            idx = np.arange(L)

            for slot, w in enumerate((3, 10, 20), start=3):
                lo = np.maximum(0, idx - w)
                hi = np.minimum(L, idx + w + 1)
                cnt = np.maximum(hi - lo - 1, 1)[:, None].astype(np.float32)
                # Exclude self from neighbor mean!
                nb_sum = (csum[hi] - csum[lo]) - sub
                nb_mean = nb_sum / cnt
                # L2 normalize non-zero means
                nn = np.linalg.norm(nb_mean, axis=1, keepdims=True)
                nb_mean = np.where(nn > 1e-8, nb_mean / np.maximum(nn, 1e-8), 0.0)
                out[gs:ge, slot * d : (slot + 1) * d] = nb_mean

            # 5 cosine scalars: cos(e_t, prev/next/w3/w10/w20)
            base_c = 6 * d
            for k_off in range(5):
                nb_vec = out[gs:ge, (k_off + 1) * d : (k_off + 2) * d]
                out[gs:ge, base_c + k_off] = np.sum(sub * nb_vec, axis=1)

    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["nomic-embed-text-v1.5", "sfr-embedding-mistral"])
    p.add_argument("--budget", type=float, default=5.0)
    args = p.parse_args()

    for m in args.models:
        if m == "nomic-embed-text-v1.5":
            compute_and_cache_embeddings(m, prefix="classification: ", batch_size=256, max_workers=4, budget_cap_usd=args.budget)
        else:
            compute_and_cache_embeddings(m, prefix="", batch_size=128, max_workers=4, budget_cap_usd=args.budget)
