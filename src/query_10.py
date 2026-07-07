"""Step 10 — Query-time pipeline (bi-encoder retrieval + LLM re-rank).

Wraps Track A (Step 8 fine-tuned bi-encoder + anchor-bank embeddings) and
Track B (Step 9 gpt-oss-120b pair re-ranker) into one callable:

    classify(phrase) -> {label, score, ranked_candidates, degraded, ...}

Flow:
  1. Embed the phrase (query prefix) and cosine-search the normalized anchor
     bank -> top-k `(utt_id, utterance, bi_score)`.
  2. Score all k `(phrase, candidate)` pairs with the gpt-oss-120b re-ranker,
     in parallel, under a per-phrase wall-clock cap.
  3. score = max over the k LLM scores (a phrase is SCIENCE_TALK if it strongly
     matches ANY science anchor); label = SCIENCE_TALK iff score >= threshold;
     ranked_candidates sorted by llm_score (stable -> deterministic).

DoD addressed:
  * Latency budget: cosine search is sub-ms; query embed timed separately;
    per-phrase wall-clock cap recorded in config.
  * Caching: reuses the Step 9 cache; a dev re-eval makes ZERO new calls.
  * Determinism: temperature=0 + pinned prompt_version + stable sort -> two
    calls on the same phrase yield identical scores and ordering.
  * Lift over Track A: precision@1 reranked vs cosine on dev (ceiling-limited
    diagnostic on the easy dev set; promoted to a gate at Y2).
  * Graceful degradation: endpoint failure -> cosine-only ranking with a loud
    `degraded=true` flag (deployment never silently changes behavior).

Usage:
    python src/query_10.py --smoke      # capped real dev eval
    python src/query_10.py --full       # full dev eval
    python src/query_10.py --stub       # offline plumbing, zero network
    python src/query_10.py --phrase "what do you think will happen to the ice?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR

DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "query.json"

LABEL_SCIENCE = "SCIENCE_TALK"
LABEL_NOT = "NOT_SCIENCE_TALK"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


class QueryPipeline:
    """Loads Track A + Track B once; exposes `classify(phrase)`."""

    def __init__(
        self,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        *,
        config_path: Path = DEFAULT_CONFIG_PATH,
        caller=None,
        model=None,
        verbose: bool = True,
    ) -> None:
        import os
        from sentence_transformers import SentenceTransformer
        from src import reranker_9 as r9
        from src.biencoder_8 import load_config as load_bicfg

        self.processed_dir = Path(processed_dir)
        self.config = load_config(config_path)
        self.verbose = verbose
        self.top_k = self.config.get("top_k", 20)
        self.aggregation = self.config.get("aggregation", "max")
        self.score_threshold = self.config.get("score_threshold", 0.5)
        self.degraded_cosine_threshold = self.config.get("degraded_cosine_threshold", 0.5)
        self.max_workers = self.config.get("max_workers", 8)
        self.wall_cap = self.config.get("per_phrase_wall_clock_cap_s", 60)
        self.prompt_version = self.config.get("prompt_version", "reranker_v1")

        # --- Track A: fine-tuned bi-encoder + anchor bank -------------------
        bicfg = load_bicfg(_PROJECT_ROOT / self.config["biencoder_config"])
        self.query_prefix = bicfg.get("query_prefix", "")
        bi_out = bicfg.get("output", {})
        model_dir = _PROJECT_ROOT / bi_out.get("model_dir", "models/biencoder")
        emb_path = _PROJECT_ROOT / bi_out.get(
            "corpus_embeddings", "data/processed/corpus_embeddings.npy")
        meta_path = _PROJECT_ROOT / bi_out.get(
            "corpus_embeddings_meta", "data/processed/corpus_embeddings_meta.parquet")
        if not model_dir.exists() or not emb_path.exists():
            raise RuntimeError(
                "Step 10 requires Step 8 artifacts (models/biencoder + "
                "corpus_embeddings.npy). Run `python src/biencoder_8.py` first.")
        self.model = SentenceTransformer(str(model_dir), trust_remote_code=True, device="cpu")
        self.corpus_emb = np.load(emb_path).astype(np.float32)
        meta = pd.read_parquet(meta_path)
        self.meta_utt_ids = meta["utt_id"].astype(str).tolist()
        self.meta_utts = meta["utterance"].astype(str).tolist()

        # subtype lookup (for richer candidates; not required by the DoD schema)
        corpus = pd.read_parquet(self.processed_dir / "corpus.parquet")
        self.subtype_map = {}
        if "subtype" in corpus.columns:
            self.subtype_map = dict(
                zip(corpus["utt_id"].astype(str), corpus["subtype"]))

        # --- Track B: re-ranker prompt + caller -----------------------------
        rrcfg = r9.load_config(_PROJECT_ROOT / self.config["reranker_config"])
        self.max_retries = rrcfg.get("max_retries", 1)
        self.system_prompt = r9.build_system_prompt(
            self.processed_dir,
            template_path=_PROJECT_ROOT / rrcfg["prompt_file"])
        self.ledger = r9.CallLedger()
        self._r9 = r9
        if caller is not None:
            self.caller = caller
        else:
            mdl = model or os.getenv(self.config["model_env"])
            if not mdl:
                raise RuntimeError(f"env {self.config['model_env']} not set.")
            self.caller = r9.make_real_caller(
                mdl, temperature=rrcfg.get("temperature", 0.0),
                max_tokens=rrcfg.get("max_tokens", 800), ledger=self.ledger)

    # ---------------------------------------------------------------- retrieve
    def _embed_and_search(self, phrase: str, top_k: int):
        from src.biencoder_8 import _encode
        t0 = time.perf_counter()
        q = _encode(self.model, [phrase], self.query_prefix)[0]
        embed_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        sims = self.corpus_emb @ q
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        idx = idx[np.argsort(-sims[idx])]
        search_s = time.perf_counter() - t1
        cands = [{
            "utt_id": self.meta_utt_ids[i],
            "utterance": self.meta_utts[i],
            "bi_score": float(sims[i]),
            "subtype": self.subtype_map.get(self.meta_utt_ids[i]),
        } for i in idx]
        return cands, embed_s, search_s

    def retrieve(self, phrase: str, top_k: int | None = None) -> list[dict]:
        cands, _, _ = self._embed_and_search(phrase, top_k or self.top_k)
        return cands

    # ------------------------------------------------------------------ rerank
    def _score_one(self, phrase: str, cand: dict) -> dict:
        try:
            s, rat = self._r9.score_pair_full(
                phrase, cand["utterance"], caller=self.caller,
                system_prompt=self.system_prompt, prompt_version=self.prompt_version,
                max_retries=self.max_retries, row_id=cand["utt_id"], verbose=False)
            return {"llm_score": s, "rationale": rat, "error": None}
        except self._r9.RerankerParseError:
            return {"llm_score": None, "rationale": "parse_failed", "error": "parse"}
        except Exception as e:  # noqa: BLE001 - endpoint/network failure -> degrade
            return {"llm_score": None, "rationale": f"error:{type(e).__name__}",
                    "error": "network"}

    def rerank(self, phrase: str, candidates: list[dict]) -> list[dict]:
        results: list[dict | None] = [None] * len(candidates)
        if self.max_workers > 1 and len(candidates) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futs = {ex.submit(self._score_one, phrase, c): i
                        for i, c in enumerate(candidates)}
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
        else:
            results = [self._score_one(phrase, c) for c in candidates]
        return results  # type: ignore[return-value]

    # ---------------------------------------------------------------- classify
    def classify(self, phrase: str, *, top_k: int | None = None) -> dict:
        top_k = top_k or self.top_k
        t0 = time.perf_counter()
        cands, embed_s, search_s = self._embed_and_search(phrase, top_k)
        scored = self.rerank(phrase, cands)
        elapsed = time.perf_counter() - t0

        merged = [{
            "utt_id": c["utt_id"], "utterance": c["utterance"],
            "llm_score": r["llm_score"], "bi_score": c["bi_score"],
            "rationale": r["rationale"], "subtype": c["subtype"],
        } for c, r in zip(cands, scored)]

        successes = [m["llm_score"] for m in merged if m["llm_score"] is not None]
        degraded = len(successes) == 0

        if degraded:
            ranked = sorted(merged, key=lambda d: (-d["bi_score"], d["utt_id"]))
            score = float(ranked[0]["bi_score"]) if ranked else 0.0
            label = LABEL_SCIENCE if score >= self.degraded_cosine_threshold else LABEL_NOT
            if self.verbose:
                print(f"[query] degraded=true (LLM endpoint unavailable); "
                      f"cosine-only label for {phrase!r}")
        else:
            ranked = sorted(
                merged,
                key=lambda d: (-(d["llm_score"] if d["llm_score"] is not None else -1.0),
                               -d["bi_score"], d["utt_id"]))
            score = (max(successes) if self.aggregation == "max"
                     else float(np.mean(successes)))
            label = LABEL_SCIENCE if score >= self.score_threshold else LABEL_NOT

        over_budget = elapsed > self.wall_cap
        if over_budget and self.verbose:
            print(f"[query] WARNING over wall-clock cap ({elapsed:.1f}s > "
                  f"{self.wall_cap}s) for {phrase!r}")

        return {
            "phrase": phrase,
            "label": label,
            "score": score,
            "degraded": degraded,
            "over_budget": over_budget,
            "ranked_candidates": ranked,
            "timing": {"total_s": elapsed, "embed_s": embed_s, "search_s": search_s},
        }

    def reset_ledger(self) -> None:
        self.ledger.hits = 0
        self.ledger.misses = 0


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_PIPELINE: QueryPipeline | None = None


def classify(phrase: str, *, processed_dir: Path = DEFAULT_PROCESSED_DIR,
             config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Lazily build a shared pipeline and classify one phrase."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = QueryPipeline(processed_dir, config_path=config_path, verbose=False)
    return _PIPELINE.classify(phrase)


# ---------------------------------------------------------------------------
# Dev evaluation (DoD checks)
# ---------------------------------------------------------------------------

def build_dev_queries(frames: dict, *, n_pos: int, n_neg: int, seed: int = 42):
    """Dev phrases: positives (variant -> gold anchor) and negatives (text)."""
    import random
    from src.biencoder_8 import _split_ids
    rng = random.Random(seed)
    corpus, pairs, negs, splits = (
        frames["corpus"], frames["pairs"], frames["negatives"], frames["splits"])

    pos: list[dict] = []
    val_pair_ids = _split_ids(splits, "pair", "val")
    if len(pairs) > 0:
        ep = pairs[pairs["pair_id"].astype(str).isin(val_pair_ids)]
        for _, p in ep.iterrows():
            if pd.isna(p.get("variant_text")):
                continue
            pos.append({"phrase": str(p["variant_text"]),
                        "gold_anchor": str(p["anchor_id"])})

    neg: list[dict] = []
    val_neg_ids = _split_ids(splits, "negative", "val")
    if len(negs) > 0:
        en = negs[negs["neg_id"].astype(str).isin(val_neg_ids)]
        for _, nrow in en.iterrows():
            neg.append({"phrase": str(nrow["text"])})

    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n_pos], neg[:n_neg]


def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    mode: str = "smoke",
    use_stub: bool = False,
    verbose: bool = True,
) -> dict:
    from src.biencoder_8 import load_verified_frames

    config = load_config(config_path)
    budget = config.get(mode, config.get("smoke", {}))
    n_pos, n_neg = budget.get("n_pos", 5), budget.get("n_neg", 3)
    cap = budget.get("max_new_calls", 250)

    caller = __import__("src.reranker_9", fromlist=["stub_caller"]).stub_caller if use_stub else None
    pipe = QueryPipeline(processed_dir, config_path=config_path,
                         caller=caller, verbose=verbose)
    frames = load_verified_frames(processed_dir)
    pos, neg = build_dev_queries(frames, n_pos=n_pos, n_neg=n_neg)

    projected = (len(pos) + len(neg)) * pipe.top_k
    if verbose:
        print(f"Mode={mode} | {len(pos)} pos + {len(neg)} neg phrases | "
              f"projected <= {projected} calls (cap {cap}) | stub={use_stub}")
    if not use_stub and projected > cap:
        raise RuntimeError(f"Projected {projected} > cap {cap}; lower n_pos/n_neg "
                           f"or raise '{mode}.max_new_calls'.")

    # --- First pass: classify all dev phrases -------------------------------
    pipe.reset_ledger()
    pos_res = [pipe.classify(p["phrase"]) for p in pos]
    neg_res = [pipe.classify(p["phrase"]) for p in neg]
    new_first = pipe.ledger.misses

    # --- Caching: re-classify, expect ZERO new calls ------------------------
    pipe.reset_ledger()
    for p in pos:
        pipe.classify(p["phrase"])
    for p in neg:
        pipe.classify(p["phrase"])
    new_second = pipe.ledger.misses
    caching_pass = (new_second == 0)

    # --- Determinism: classify the first phrase twice, compare --------------
    determinism_pass = None
    if pos:
        a = pipe.classify(pos[0]["phrase"])
        b = pipe.classify(pos[0]["phrase"])
        determinism_pass = (
            a["score"] == b["score"]
            and [c["utt_id"] for c in a["ranked_candidates"]]
            == [c["utt_id"] for c in b["ranked_candidates"]])

    # --- Lift diagnostic: precision@1 reranked vs cosine (positives) --------
    cos_hits, rr_hits, n_lift = 0, 0, 0
    for p, res in zip(pos, pos_res):
        gold = p["gold_anchor"]
        cos = pipe.retrieve(p["phrase"])
        if not cos:
            continue
        n_lift += 1
        cos_hits += int(cos[0]["utt_id"] == gold)
        if res["ranked_candidates"]:
            rr_hits += int(res["ranked_candidates"][0]["utt_id"] == gold)
    lift = {
        "n": n_lift,
        "cosine_p@1": (cos_hits / n_lift) if n_lift else None,
        "reranked_p@1": (rr_hits / n_lift) if n_lift else None,
        "note": "ceiling-limited on the easy dev set; promoted to a gate at Y2.",
    }

    # --- Classification sanity at the untuned threshold (diagnostic) --------
    pos_correct = sum(r["label"] == LABEL_SCIENCE for r in pos_res)
    neg_correct = sum(r["label"] == LABEL_NOT for r in neg_res)

    # --- Graceful degradation: swap in a failing caller ---------------------
    degradation_pass = None
    if pos and not use_stub:
        def _down(system, user, prompt_version):
            raise ConnectionError("simulated endpoint outage")
        orig = pipe.caller
        pipe.caller = _down
        dres = pipe.classify(pos[0]["phrase"])
        pipe.caller = orig
        degradation_pass = bool(
            dres["degraded"] is True
            and dres["label"] in (LABEL_SCIENCE, LABEL_NOT)
            and all(c["llm_score"] is None for c in dres["ranked_candidates"]))

    # --- Latency ------------------------------------------------------------
    all_res = pos_res + neg_res
    latency = {
        "mean_search_ms": float(np.mean([r["timing"]["search_s"] for r in all_res]) * 1000) if all_res else None,
        "mean_embed_ms": float(np.mean([r["timing"]["embed_s"] for r in all_res]) * 1000) if all_res else None,
        "mean_total_s": float(np.mean([r["timing"]["total_s"] for r in all_res])) if all_res else None,
        "wall_clock_cap_s": pipe.wall_cap,
        "any_over_budget": any(r["over_budget"] for r in all_res),
    }

    dod_passed = bool(caching_pass and determinism_pass
                      and (degradation_pass or use_stub))

    report = {
        "version": config["version"],
        "mode": mode,
        "stub": use_stub,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "top_k": pipe.top_k,
        "aggregation": pipe.aggregation,
        "score_threshold": pipe.score_threshold,
        "caching": {"new_calls_first_pass": new_first,
                    "new_calls_second_pass": new_second, "passed": caching_pass},
        "determinism": {"passed": determinism_pass},
        "graceful_degradation": {"passed": degradation_pass},
        "latency": latency,
        "retrieval_lift_diagnostic": lift,
        "classification_sanity": {
            "pos_recall": (pos_correct / len(pos)) if pos else None,
            "neg_recall": (neg_correct / len(neg)) if neg else None,
            "note": "at the UNTUNED threshold; Step 11 tunes it.",
        },
        "cost_ledger": {"new_calls_total": new_first, "cap": cap},
        "dod_passed": dod_passed,
    }

    out_path = _PROJECT_ROOT / config["output"]["report_json"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    if verbose:
        _print_report(report, out_path)
    return report


def _print_report(report: dict, path: Path) -> None:
    print("\n=== Step 10 query-time pipeline eval ===")
    c = report["caching"]
    print(f"  Caching       : {c['new_calls_first_pass']} new (1st) -> "
          f"{c['new_calls_second_pass']} new (2nd) "
          f"{'PASS' if c['passed'] else 'FAIL'}")
    print(f"  Determinism   : {'PASS' if report['determinism']['passed'] else 'FAIL'}")
    dg = report["graceful_degradation"]["passed"]
    print(f"  Degradation   : {'PASS' if dg else ('SKIP' if dg is None else 'FAIL')}")
    lat = report["latency"]
    if lat["mean_search_ms"] is not None:
        print(f"  Latency       : search {lat['mean_search_ms']:.2f} ms | "
              f"embed {lat['mean_embed_ms']:.1f} ms | total {lat['mean_total_s']:.2f} s/phrase")
    li = report["retrieval_lift_diagnostic"]
    print(f"  Lift (diag)   : cosine p@1={li['cosine_p@1']} -> "
          f"reranked p@1={li['reranked_p@1']} (n={li['n']}, ceiling-limited)")
    cs = report["classification_sanity"]
    print(f"  Class sanity  : pos_recall={cs['pos_recall']} neg_recall={cs['neg_recall']} "
          f"(untuned threshold)")
    print(f"  Step 10 DoD   : {'PASS' if report['dod_passed'] else 'FAIL'}")
    print(f"  Wrote {path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--smoke", action="store_true", help="Capped real dev eval (default).")
    p.add_argument("--full", action="store_true", help="Full dev eval.")
    p.add_argument("--stub", action="store_true", help="Offline plumbing, no network.")
    p.add_argument("--phrase", type=str, default=None, help="Classify a single phrase and exit.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.phrase:
        res = QueryPipeline(args.processed_dir, config_path=args.config).classify(args.phrase)
        slim = {k: res[k] for k in ("label", "score", "degraded")}
        slim["top3"] = [
            {"utt": c["utterance"][:60], "llm": c["llm_score"], "bi": round(c["bi_score"], 3)}
            for c in res["ranked_candidates"][:3]]
        print(json.dumps(slim, indent=2, default=str))
    else:
        run(args.processed_dir, config_path=args.config,
            mode="full" if args.full else "smoke", use_stub=args.stub,
            verbose=not args.quiet)
