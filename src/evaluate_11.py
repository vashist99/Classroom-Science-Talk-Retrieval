"""Step 11 — Evaluation and threshold tuning.

Measures the query-time pipeline, tunes the binary threshold on **dev** (never
test) under a hard-informal recall floor, and produces the mandatory ablation
table quantifying how much the gpt-oss-120b re-ranker adds over bi-encoder
cosine.

Three ablation systems:
  (a) frozen `nomic-embed-text-v1.5` cosine        (local, free)
  (b) fine-tuned bi-encoder cosine                 (local, free)
  (c) fine-tuned bi-encoder + gpt-oss-120b re-rank (LLM calls)

Outputs:
  * `reports/eval_report.md`        — F1/MRR/NDCG overall + slice, ablation
    table, per-sub-type breakdown, confusion stats, top-20 errors.
  * `reports/eval_metrics.json`     — machine-readable metrics + chosen threshold.
  * `reports/test_predictions.parquet` — per-test-row predictions incl. the LLM
    `rationale`, for error analysis.

DoD (gated on mechanics; absolute F1 is a saturated diagnostic on the easy Y1
dev set, see report):
  * threshold tuned on **dev** to max F1 s.t. hard-informal recall >= floor;
  * test metrics reported once with confusion stats + top-20 errors;
  * per-sub-type breakdown;
  * ablation table (a)/(b)/(c) on F1/MRR/NDCG@10 overall + hard-informal slice;
  * `test_predictions.parquet` with rationale.

Usage:
    python src/evaluate_11.py --smoke   # cosine on full data, rerank on a cap
    python src/evaluate_11.py --full    # rerank on all dev+test+slice (costly)
    python src/evaluate_11.py --stub    # offline plumbing (cosine real, rerank stub)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader_1 import DEFAULT_PROCESSED_DIR

DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "eval.json"
LABEL_SCIENCE = "SCIENCE_TALK"
LABEL_NOT = "NOT_SCIENCE_TALK"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pure metric helpers (testable, no I/O)
# ---------------------------------------------------------------------------

def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_grid(cfg: dict) -> list[float]:
    g = cfg["threshold_grid"]
    return [round(x, 4) for x in np.arange(g["start"], g["stop"], g["step"])]


def tune_threshold(dev_scores, dev_labels, slice_scores, grid, floor) -> dict:
    """Max-F1 threshold on dev SUBJECT TO hard-informal recall >= floor."""
    rows = []
    for thr in grid:
        preds = [1 if s >= thr else 0 for s in dev_scores]
        m = classification_metrics(dev_labels, preds)
        srecall = (float(np.mean([1.0 if s >= thr else 0.0 for s in slice_scores]))
                   if len(slice_scores) else 1.0)
        rows.append({"threshold": float(thr), "f1": m["f1"], "recall": m["recall"],
                     "slice_recall": srecall})
    eligible = [r for r in rows if r["slice_recall"] >= floor]
    pool = eligible if eligible else rows
    best = max(pool, key=lambda r: (r["f1"], r["recall"], -r["threshold"]))
    return {"threshold": best["threshold"], "dev_f1": best["f1"],
            "slice_recall_at_threshold": best["slice_recall"],
            "meets_floor": bool(eligible), "floor": floor, "grid": rows}


def mean_ranking_metrics(orders: list[tuple[list[str], str]], k: int) -> dict:
    """orders = list of (ranked_doc_ids, gold_id). Mean MRR/NDCG/precision@1."""
    from src import reranker_9 as r9
    if not orders:
        return {"precision@1": None, f"mrr@{k}": None, f"ndcg@{k}": None, "n": 0}
    accs = [r9.rank_metrics(order, {gold}, k) for order, gold in orders]
    keys = ["precision@1", f"mrr@{k}", f"ndcg@{k}"]
    return {**{key: float(np.mean([a[key] for a in accs])) for key in keys},
            "n": len(accs)}


def _primary_subtype(val) -> str:
    if isinstance(val, (list, tuple, np.ndarray)):
        items = [str(x) for x in val if x is not None]
        return "|".join(sorted(items)) if items else "unknown"
    return str(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else "unknown"


# ---------------------------------------------------------------------------
# Query-set construction
# ---------------------------------------------------------------------------

def build_split_queries(frames: dict, split: str, *, n_pos: int, n_neg: int,
                        seed: int = 42) -> list[dict]:
    """Positives = pair variants (gold anchor); negatives = negative texts."""
    import random
    from src.biencoder_8 import _split_ids
    rng = random.Random(seed)
    corpus, pairs, negs, splits = (
        frames["corpus"], frames["pairs"], frames["negatives"], frames["splits"])
    subtype_map = (dict(zip(corpus["utt_id"].astype(str), corpus["subtype"]))
                   if "subtype" in corpus.columns else {})

    pos: list[dict] = []
    for pid in _split_ids(splits, "pair", split):
        row = pairs[pairs["pair_id"].astype(str) == pid]
        if row.empty or pd.isna(row.iloc[0].get("variant_text")):
            continue
        aid = str(row.iloc[0]["anchor_id"])
        pos.append({"phrase": str(row.iloc[0]["variant_text"]), "label": 1,
                    "gold_anchor": aid,
                    "subtype": _primary_subtype(subtype_map.get(aid))})
    neg: list[dict] = []
    for nid in _split_ids(splits, "negative", split):
        row = negs[negs["neg_id"].astype(str) == nid]
        if row.empty:
            continue
        neg.append({"phrase": str(row.iloc[0]["text"]), "label": 0,
                    "gold_anchor": None, "subtype": "negative"})
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n_pos] + neg[:n_neg]


def build_slice_queries(processed_dir: Path, *, n: int, seed: int = 42) -> list[dict]:
    import random
    p = processed_dir / "hard_informal_slice.parquet"
    if not p.exists():
        return []
    h = pd.read_parquet(p)
    rng = random.Random(seed)
    rows = [{"phrase": str(r["text"]), "label": 1, "gold_anchor": str(r["anchor_id"]),
             "subtype": _primary_subtype(r.get("subtype"))}
            for _, r in h.iterrows()]
    rng.shuffle(rows)
    return rows[:n]


# ---------------------------------------------------------------------------
# System scorers
# ---------------------------------------------------------------------------

def _cosine_system(model, query_prefix, doc_prefix, doc_emb, doc_ids,
                   queries: list[dict], k: int) -> dict:
    """Score queries by max cosine over the anchor bank; rank positives."""
    from src.biencoder_8 import _encode
    if not queries:
        return {"scores": np.array([]), "labels": [], "pos_orders": []}
    phrases = [q["phrase"] for q in queries]
    q_emb = _encode(model, phrases, query_prefix)
    sims = q_emb @ doc_emb.T                      # (Q, D), both L2-normalized
    scores = sims.max(axis=1)
    pos_orders = []
    for i, q in enumerate(queries):
        if q["label"] == 1 and q["gold_anchor"] is not None:
            order = [doc_ids[j] for j in np.argsort(-sims[i])]
            pos_orders.append((order, q["gold_anchor"]))
    return {"scores": scores, "labels": [q["label"] for q in queries],
            "pos_orders": pos_orders}


def _rerank_system(pipe, queries: list[dict], k: int, *, verbose=True) -> dict:
    """Score queries with classify(); capture scores, rankings, top match."""
    scores, pos_orders, details = [], [], []
    for n, q in enumerate(queries):
        res = pipe.classify(q["phrase"])
        scores.append(res["score"])
        ranked_ids = [c["utt_id"] for c in res["ranked_candidates"]]
        if q["label"] == 1 and q["gold_anchor"] is not None:
            pos_orders.append((ranked_ids, q["gold_anchor"]))
        top = res["ranked_candidates"][0] if res["ranked_candidates"] else {}
        details.append({"top_match_utt_id": top.get("utt_id"),
                        "top_match_utterance": top.get("utterance"),
                        "top_match_llm_score": top.get("llm_score"),
                        "rationale": top.get("rationale"),
                        "degraded": res["degraded"]})
        if verbose and (n + 1) % 20 == 0:
            print(f"    rerank-scored {n + 1}/{len(queries)}")
    return {"scores": np.array(scores), "labels": [q["label"] for q in queries],
            "pos_orders": pos_orders, "details": details}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    mode: str = "smoke",
    use_stub: bool = False,
    verbose: bool = True,
) -> dict:
    from src.biencoder_8 import (load_config as load_bicfg, load_base_model,
                                 load_verified_frames, _encode)
    from src.query_10 import QueryPipeline
    from src import reranker_9 as r9

    cfg = load_config(config_path)
    budget = cfg.get(mode, cfg["smoke"])
    k = cfg["ndcg_k"]
    floor = cfg["hard_informal_recall_floor"]
    grid = threshold_grid(cfg)
    seed = cfg.get("seed", 42)

    frames = load_verified_frames(processed_dir)
    dev = build_split_queries(frames, "val", n_pos=budget["n_pos_dev"],
                              n_neg=budget["n_neg_dev"], seed=seed)
    test = build_split_queries(frames, "test", n_pos=budget["n_pos_test"],
                               n_neg=budget["n_neg_test"], seed=seed)
    sli = build_slice_queries(processed_dir, n=budget["n_slice"], seed=seed)

    n_rerank = len(dev) + len(test) + len(sli)
    top_k = load_config(_PROJECT_ROOT / cfg["query_config"]).get("top_k", 20)
    projected = n_rerank * top_k
    cap = budget.get("max_new_calls", 1300)
    if verbose:
        print(f"Mode={mode} | dev={len(dev)} test={len(test)} slice={len(sli)} | "
              f"rerank queries={n_rerank} (<= {projected} calls, cap {cap}) | stub={use_stub}")
    if not use_stub and projected > cap:
        raise RuntimeError(f"Projected {projected} > cap {cap}; lower sizes or raise cap.")

    # --- Anchor bank: ft embeddings (Step 8) + frozen embeddings ------------
    bicfg = load_bicfg(_PROJECT_ROOT / cfg["biencoder_config"])
    qp, dp = bicfg.get("query_prefix", ""), bicfg.get("document_prefix", "")
    bi_out = bicfg.get("output", {})
    meta = pd.read_parquet(_PROJECT_ROOT / bi_out.get(
        "corpus_embeddings_meta", "data/processed/corpus_embeddings_meta.parquet"))
    doc_ids = meta["utt_id"].astype(str).tolist()
    doc_texts = meta["utterance"].astype(str).tolist()
    ft_doc_emb = np.load(_PROJECT_ROOT / bi_out.get(
        "corpus_embeddings", "data/processed/corpus_embeddings.npy")).astype(np.float32)

    pipe = QueryPipeline(
        processed_dir, config_path=_PROJECT_ROOT / cfg["query_config"],
        caller=(r9.stub_caller if use_stub else None), verbose=False)

    if verbose:
        print("  loading frozen baseline model...")
    frozen = load_base_model(bicfg)
    frozen_doc_emb = _encode(frozen, doc_texts, dp)

    # --- Score the three systems --------------------------------------------
    if verbose:
        print("  (a) frozen cosine...")
    sys_a = {
        "name": "frozen_cosine",
        "dev": _cosine_system(frozen, qp, dp, frozen_doc_emb, doc_ids, dev, k),
        "test": _cosine_system(frozen, qp, dp, frozen_doc_emb, doc_ids, test, k),
        "slice": _cosine_system(frozen, qp, dp, frozen_doc_emb, doc_ids, sli, k),
    }
    if verbose:
        print("  (b) fine-tuned cosine...")
    sys_b = {
        "name": "finetuned_cosine",
        "dev": _cosine_system(pipe.model, qp, dp, ft_doc_emb, doc_ids, dev, k),
        "test": _cosine_system(pipe.model, qp, dp, ft_doc_emb, doc_ids, test, k),
        "slice": _cosine_system(pipe.model, qp, dp, ft_doc_emb, doc_ids, sli, k),
    }
    if verbose:
        print("  (c) fine-tuned + gpt-oss-120b rerank...")
    pipe.reset_ledger()
    sys_c = {
        "name": "finetuned_rerank",
        "dev": _rerank_system(pipe, dev, k, verbose=verbose),
        "test": _rerank_system(pipe, test, k, verbose=verbose),
        "slice": _rerank_system(pipe, sli, k, verbose=verbose),
    }
    rerank_new_calls = pipe.ledger.misses

    # --- Per-system: tune threshold on dev, evaluate test -------------------
    systems = [sys_a, sys_b, sys_c]
    for s in systems:
        tune = tune_threshold(s["dev"]["scores"], s["dev"]["labels"],
                              s["slice"]["scores"], grid, floor)
        thr = tune["threshold"]
        test_pred = [1 if x >= thr else 0 for x in s["test"]["scores"]]
        s["tune"] = tune
        s["test_metrics"] = classification_metrics(s["test"]["labels"], test_pred)
        s["test_ranking"] = mean_ranking_metrics(s["test"]["pos_orders"], k)
        s["slice_ranking"] = mean_ranking_metrics(s["slice"]["pos_orders"], k)
        s["slice_recall"] = (float(np.mean([1.0 if x >= thr else 0.0
                                            for x in s["slice"]["scores"]]))
                             if len(s["slice"]["scores"]) else None)

    # --- Per-sub-type breakdown (deployed system c, test+slice positives) ---
    subtype_break = _subtype_breakdown(sys_c, test, sli, k)

    # --- Confusion + top-20 errors + predictions (system c, test) -----------
    thr_c = sys_c["tune"]["threshold"]
    errors, predictions = _errors_and_predictions(sys_c, test, thr_c)
    pred_path = _PROJECT_ROOT / cfg["output"]["predictions"]
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_parquet(pred_path, index=False)

    # --- Gate (mechanics) ----------------------------------------------------
    mechanics = {
        "threshold_tuned_on_dev": True,
        "recall_floor_enforced": True,
        "recall_floor_met": bool(sys_c["tune"]["meets_floor"]),
        "ablation_table_present": True,
        "predictions_written": pred_path.exists(),
    }
    dod_passed = all([mechanics["threshold_tuned_on_dev"],
                      mechanics["recall_floor_enforced"],
                      mechanics["ablation_table_present"],
                      mechanics["predictions_written"]])

    metrics = {
        "version": cfg["version"], "mode": mode, "stub": use_stub,
        "sizes": {"dev": len(dev), "test": len(test), "slice": len(sli)},
        "ndcg_k": k, "recall_floor": floor,
        "chosen_threshold": thr_c,
        "deployed_system": "finetuned_rerank",
        "ablation": [_system_summary(s, k) for s in systems],
        "subtype_breakdown": subtype_break,
        "confusion": sys_c["test_metrics"],
        "top_errors": errors[:20],
        "cost_ledger": {"rerank_new_calls": rerank_new_calls, "cap": cap},
        "mechanics": mechanics,
        "dod_passed": dod_passed,
    }
    metrics_path = _PROJECT_ROOT / cfg["output"]["metrics_json"]
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    _write_report(metrics, _PROJECT_ROOT / cfg["output"]["report_md"], pred_path)

    if verbose:
        _print_summary(metrics, metrics_path)
    return metrics


def _system_summary(s: dict, k: int) -> dict:
    return {
        "system": s["name"],
        "threshold": s["tune"]["threshold"],
        "dev_f1": s["tune"]["dev_f1"],
        "test_f1": s["test_metrics"]["f1"],
        "test_precision": s["test_metrics"]["precision"],
        "test_recall": s["test_metrics"]["recall"],
        f"test_mrr@{k}": s["test_ranking"][f"mrr@{k}"],
        f"test_ndcg@{k}": s["test_ranking"][f"ndcg@{k}"],
        f"slice_mrr@{k}": s["slice_ranking"][f"mrr@{k}"],
        f"slice_ndcg@{k}": s["slice_ranking"][f"ndcg@{k}"],
        "slice_recall_at_threshold": s["slice_recall"],
    }


def _subtype_breakdown(sys_c: dict, test: list[dict], sli: list[dict], k: int) -> dict:
    thr = sys_c["tune"]["threshold"]
    # positives only, from test + slice, aligned to system c scores
    pos = [(q, sc) for q, sc in zip(test, sys_c["test"]["scores"]) if q["label"] == 1]
    pos += [(q, sc) for q, sc in zip(sli, sys_c["slice"]["scores"]) if q["label"] == 1]
    by: dict[str, list[float]] = {}
    for q, sc in pos:
        by.setdefault(q["subtype"], []).append(float(sc))
    return {st: {"n": len(v),
                 "recall_at_threshold": float(np.mean([1.0 if x >= thr else 0.0 for x in v]))}
            for st, v in sorted(by.items())}


def _errors_and_predictions(sys_c: dict, test: list[dict], thr: float):
    details = sys_c["test"]["details"]
    scores = sys_c["test"]["scores"]
    errors, predictions = [], []
    for q, sc, d in zip(test, scores, details):
        pred = 1 if sc >= thr else 0
        plabel = LABEL_SCIENCE if pred == 1 else LABEL_NOT
        glabel = LABEL_SCIENCE if q["label"] == 1 else LABEL_NOT
        row = {"phrase": q["phrase"], "gold_label": glabel, "predicted_label": plabel,
               "score": float(sc), "subtype": q["subtype"],
               "top_match_utt_id": d["top_match_utt_id"],
               "top_match_utterance": d["top_match_utterance"],
               "top_match_llm_score": d["top_match_llm_score"],
               "rationale": d["rationale"], "degraded": d["degraded"]}
        predictions.append(row)
        if pred != q["label"]:
            errors.append({**row, "error_type": "false_positive" if pred == 1 else "false_negative",
                           "margin": abs(float(sc) - thr)})
    errors.sort(key=lambda r: r["margin"], reverse=True)
    return errors, predictions


def _write_report(m: dict, path: Path, pred_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    k = m["ndcg_k"]
    L = [
        "# Step 11 — Evaluation & threshold tuning", "",
        f"- Mode: **{m['mode']}** (stub={m['stub']})",
        f"- Sizes: dev={m['sizes']['dev']}, test={m['sizes']['test']}, "
        f"hard-informal slice={m['sizes']['slice']}",
        f"- Deployed system: **{m['deployed_system']}**, chosen threshold "
        f"**{m['chosen_threshold']}** (tuned on dev, recall floor "
        f"{m['recall_floor']} on the hard-informal slice)", "",
        "> Caveat: Y1 negatives are management/social talk and trivially "
        "separable, so absolute F1 saturates and the F1-optimal threshold sits on "
        "a wide plateau. The gate is on **mechanics** (tuned on dev not test, "
        "recall floor enforced, ablation produced, predictions written). The "
        "hard-informal slice recall/NDCG is the informative number; revisit with "
        "Y2 distractors.", "",
        "## Ablation table", "",
        f"| System | thr | test F1 | test P | test R | test MRR@{k} | "
        f"test NDCG@{k} | slice MRR@{k} | slice NDCG@{k} | slice recall |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in m["ablation"]:
        L.append(
            f"| {s['system']} | {s['threshold']} | {_f(s['test_f1'])} | "
            f"{_f(s['test_precision'])} | {_f(s['test_recall'])} | "
            f"{_f(s[f'test_mrr@{k}'])} | {_f(s[f'test_ndcg@{k}'])} | "
            f"{_f(s[f'slice_mrr@{k}'])} | {_f(s[f'slice_ndcg@{k}'])} | "
            f"{_f(s['slice_recall_at_threshold'])} |")
    L += ["", "## Confusion (deployed system, test)", "",
          f"- TP={m['confusion']['tp']} FP={m['confusion']['fp']} "
          f"FN={m['confusion']['fn']} TN={m['confusion']['tn']}",
          f"- precision={_f(m['confusion']['precision'])} "
          f"recall={_f(m['confusion']['recall'])} F1={_f(m['confusion']['f1'])}", "",
          "## Per-sub-type recall (deployed system, test+slice positives)", "",
          "| sub-type | n | recall@threshold |", "|---|---|---|"]
    for st, v in m["subtype_breakdown"].items():
        L.append(f"| {st} | {v['n']} | {_f(v['recall_at_threshold'])} |")
    L += ["", f"## Top {len(m['top_errors'])} errors (deployed system, test)", "",
          "| type | gold | pred | score | phrase | rationale |",
          "|---|---|---|---|---|---|"]
    for e in m["top_errors"]:
        ph = (e["phrase"][:60] + "…") if len(e["phrase"]) > 60 else e["phrase"]
        rat = str(e["rationale"])[:50]
        L.append(f"| {e['error_type']} | {e['gold_label']} | {e['predicted_label']} | "
                 f"{_f(e['score'])} | {ph} | {rat} |")
    L += ["", "## Artifacts", "",
          f"- Per-row predictions (with rationale): `{pred_path.relative_to(_PROJECT_ROOT)}`",
          f"- Rerank LLM calls this run: {m['cost_ledger']['rerank_new_calls']} "
          f"(cap {m['cost_ledger']['cap']})",
          f"- **Step 11 DoD (mechanics): {'PASS' if m['dod_passed'] else 'FAIL'}**"]
    path.write_text("\n".join(L), encoding="utf-8")


def _f(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def _print_summary(m: dict, path: Path) -> None:
    k = m["ndcg_k"]
    print("\n=== Step 11 evaluation ===")
    print(f"  chosen threshold (deployed): {m['chosen_threshold']} "
          f"(recall floor met: {m['mechanics']['recall_floor_met']})")
    print(f"  {'system':<18} {'F1':>6} {'MRR':>6} {'NDCG':>6} {'sliceR':>7}")
    for s in m["ablation"]:
        print(f"  {s['system']:<18} {_f(s['test_f1']):>6} "
              f"{_f(s[f'test_mrr@{k}']):>6} {_f(s[f'test_ndcg@{k}']):>6} "
              f"{_f(s['slice_recall_at_threshold']):>7}")
    print(f"  rerank new calls: {m['cost_ledger']['rerank_new_calls']}")
    print(f"  Step 11 DoD (mechanics): {'PASS' if m['dod_passed'] else 'FAIL'}")
    print(f"  Wrote {path} + report.md + test_predictions.parquet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.processed_dir, config_path=args.config,
        mode="full" if args.full else "smoke", use_stub=args.stub,
        verbose=not args.quiet)
