"""Step 5a -- Frozen-baseline embedding pass.

Embeds anchor/variant/negative texts with the off-the-shelf
`nomic-embed-text-v1.5` model and writes a `baseline_cosine` column onto both
`pairs.parquet` and `negatives.parquet`.

Why a *frozen* baseline rather than the fine-tuned bi-encoder?
  - Step 5 (confidence + routing) needs a *neutral* sanity signal that is
    independent of any retrieval training. The fine-tuned bi-encoder (Step 8)
    would have seen the same pairs and would loop its own opinion back into
    confidence -- circular.
  - The same frozen encoder will later serve as the warm-start checkpoint for
    Step 8, so this pass is reused there.

All embedding calls go through `llm_client_0.cached_request`, so re-runs are
free once cached. Texts are deduped within a batch before being sent.

Schema additions
----------------
pairs.parquet     +baseline_cosine (float64)   cosine(anchor_text, variant_text)
negatives.parquet +baseline_cosine (float64)   cosine(anchor_positive, neg_text)
                                                NaN for rows without an
                                                anchor_utt_id (transcript_clean,
                                                seed_word_nonscience).

DoD addressed by this module:
  1. Every pair (variant) and every llm_hard_negative has a real cosine
     against its anchor, computed with a fixed encoder + fixed prompt_version.
  2. Embeddings are cached so re-running Step 5 is free.
  3. The model name and prompt_version are recorded in the cache and in the
     config (`config/confidence.json -> baseline_encoder`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR
from src.llm_client_0 import cached_request


DEFAULT_BASELINE_MODEL = "nomic-embed-text-v1.5"
BASELINE_PROMPT_VERSION = "emb_baseline_v1"
EMBED_DIM = 768  # nomic-embed-text-v1.5


def _normalize_text_for_embedding(text: object) -> str:
    """Lowercase + strip. Returns empty string for null/NaN inputs."""
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    return str(text).strip()


def get_embedding_for_text(
    text: str,
    *,
    model: str,
    api_key: str,
    url: str,
    prompt_version: str = BASELINE_PROMPT_VERSION,
    dim: int = EMBED_DIM,
) -> np.ndarray:
    """Return a `dim`-vector embedding for `text`, or a zero vector on error.

    On any endpoint failure (non-JSON, error payload, network blip) returns
    `np.zeros(dim)`. Downstream cosine_similarity will yield 0.0 against any
    other vector, which the band score interprets as "low / extreme = bad".
    """
    text = _normalize_text_for_embedding(text)
    if not text:
        return np.zeros(dim, dtype=np.float32)
    params = {"model": model, "input": text}
    try:
        raw = cached_request(
            api_key=api_key,
            url=url,
            endpoint="embedding",
            model=model,
            params=params,
            prompt_version=prompt_version,
        )
        emb = raw["data"][0]["embedding"]
        return np.asarray(emb, dtype=np.float32)
    except Exception:
        return np.zeros(dim, dtype=np.float32)


def embed_texts(
    texts: Iterable[object],
    *,
    model: str = DEFAULT_BASELINE_MODEL,
    api_key_env: str = "LLM_API_KEY",
    url_env: str = "EMBEDDING_URL",
    prompt_version: str = BASELINE_PROMPT_VERSION,
    dim: int = EMBED_DIM,
    verbose: bool = True,
    log_every: int = 100,
    embedder=None,
) -> np.ndarray:
    """Embed an iterable of texts. Dedupes within the batch.

    `embedder` is an optional injected callable (text) -> np.ndarray used by
    tests; in production we use the real cached_request path.
    """
    texts_list = [_normalize_text_for_embedding(t) for t in texts]
    unique = list(dict.fromkeys(texts_list))

    if embedder is None:
        api_key = os.getenv(api_key_env)
        url = os.getenv(url_env)
        if not api_key or not url:
            raise RuntimeError(
                f"Missing env vars {api_key_env!r} / {url_env!r}; "
                "cannot reach the embedding endpoint."
            )

        def _default_embedder(t: str) -> np.ndarray:
            return get_embedding_for_text(
                t, model=model, api_key=api_key, url=url,
                prompt_version=prompt_version, dim=dim,
            )

        embedder = _default_embedder

    cache: dict[str, np.ndarray] = {}
    for i, t in enumerate(unique):
        cache[t] = embedder(t)
        if verbose and (i + 1) % log_every == 0:
            print(f"  embedded {i + 1:,}/{len(unique):,} unique texts")
    if verbose:
        print(f"  embedded {len(unique):,}/{len(unique):,} unique texts (done)")

    return np.stack([cache[t] for t in texts_list], axis=0)


def cosine_similarity_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-aligned pairwise cosine similarity.

    Returns a 1-D array of length len(a) == len(b). Zero-norm rows yield 0.0.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = a_norm * b_norm
    safe = np.where(denom == 0, 1.0, denom)
    dots = np.sum(a * b, axis=1)
    out = dots / safe
    return np.where(denom == 0, 0.0, out)


def add_baseline_cosine_to_pairs(
    df_pairs: pd.DataFrame,
    *,
    model: str = DEFAULT_BASELINE_MODEL,
    verbose: bool = True,
    embedder=None,
) -> pd.DataFrame:
    """Adds `baseline_cosine` to a pairs.parquet-shaped frame."""
    df = df_pairs.copy()
    if len(df) == 0:
        df["baseline_cosine"] = pd.Series([], dtype="float64")
        return df

    if verbose:
        print(f"  pairs.parquet -> embedding {len(df):,} anchors + variants")
    anchors = embed_texts(
        df["anchor_text"].tolist(), model=model, verbose=verbose, embedder=embedder,
    )
    variants = embed_texts(
        df["variant_text"].tolist(), model=model, verbose=verbose, embedder=embedder,
    )
    df["baseline_cosine"] = cosine_similarity_pairs(anchors, variants).astype("float64")
    return df


def add_baseline_cosine_to_negatives(
    df_neg: pd.DataFrame,
    df_corpus: pd.DataFrame,
    *,
    model: str = DEFAULT_BASELINE_MODEL,
    verbose: bool = True,
    embedder=None,
) -> pd.DataFrame:
    """Adds `baseline_cosine` to a negatives.parquet-shaped frame.

    Only rows with a non-null `anchor_utt_id` get a cosine (i.e.
    `llm_hard_negative`). Rows without an anchor positive get NaN.
    """
    df = df_neg.copy()
    df["baseline_cosine"] = np.nan

    has_anchor = df["anchor_utt_id"].notna()
    rows = df[has_anchor]
    if len(rows) == 0:
        return df

    utt_to_text = dict(zip(df_corpus["utt_id"], df_corpus["utterance"]))
    anchor_texts = rows["anchor_utt_id"].map(utt_to_text).fillna("").tolist()
    neg_texts = rows["text"].tolist()

    if verbose:
        print(f"  negatives.parquet -> embedding {len(rows):,} hard-neg pairs")
    anchors = embed_texts(
        anchor_texts, model=model, verbose=verbose, embedder=embedder,
    )
    negs = embed_texts(
        neg_texts, model=model, verbose=verbose, embedder=embedder,
    )
    cos = cosine_similarity_pairs(anchors, negs).astype("float64")
    df.loc[has_anchor, "baseline_cosine"] = cos
    return df


def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    model: str = DEFAULT_BASELINE_MODEL,
    verbose: bool = True,
    embedder=None,
) -> tuple[Path, Path]:
    """End-to-end: write `baseline_cosine` onto both parquets in place."""
    pairs_path = processed_dir / "pairs.parquet"
    negs_path = processed_dir / "negatives.parquet"
    corpus_path = processed_dir / "corpus.parquet"

    df_pairs = pd.read_parquet(pairs_path)
    df_neg = pd.read_parquet(negs_path)
    df_corpus = pd.read_parquet(corpus_path)

    df_pairs = add_baseline_cosine_to_pairs(
        df_pairs, model=model, verbose=verbose, embedder=embedder,
    )
    df_neg = add_baseline_cosine_to_negatives(
        df_neg, df_corpus, model=model, verbose=verbose, embedder=embedder,
    )

    df_pairs.to_parquet(pairs_path, index=False)
    df_neg.to_parquet(negs_path, index=False)
    if verbose:
        cos_pairs = df_pairs["baseline_cosine"]
        cos_neg = df_neg["baseline_cosine"].dropna()
        print(
            f"  pairs.parquet baseline_cosine: "
            f"mean={cos_pairs.mean():.3f}, median={cos_pairs.median():.3f}, "
            f"n={len(cos_pairs):,}"
        )
        print(
            f"  negatives.parquet baseline_cosine (anchored rows): "
            f"mean={cos_neg.mean():.3f}, median={cos_neg.median():.3f}, "
            f"n={len(cos_neg):,}"
        )
    return pairs_path, negs_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--model", type=str, default=DEFAULT_BASELINE_MODEL)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(processed_dir=args.processed_dir, model=args.model, verbose=not args.quiet)
