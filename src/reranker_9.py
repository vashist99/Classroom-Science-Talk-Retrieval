"""Step 9 — Track B: zero-shot LLM pair re-ranker (`gpt-oss-120b`).

A prompted, *non-fine-tuned* re-ranker that scores a `(query, candidate)` pair
on a continuous 0-1 "shared scientific-practice" scale. Used at query time
(Step 10) to reorder the bi-encoder's top-k candidates.

Design (DoD):
  * versioned, source-controlled prompt (`prompts/reranker_v{n}.txt`) that
    injects the operational definitions from `category_defs.parquet`, the
    sub-type taxonomy, a strict JSON schema, and 3 in-context examples;
  * `score_pair(query, candidate) -> float[0,1]` with structured-output parsing
    that FAILS LOUDLY (logs row id), one automatic retry, cache-by-(model,
    prompt_version, params), and deterministic decoding (temperature=0);
  * calibration: AUROC on a gold-labeled 100-pair audit, prompt stability
    (Spearman rho between two prompt phrasings), and a retrieval-lift diagnostic;
  * a cost ledger (new vs cached calls) with a config cap.

Gate (this revision): AUROC >= 0.85 AND Spearman rho >= 0.90. The dev retrieval
set is near-ceiling after Step 8, so retrieval lift + hard-informal lift are
reported as DIAGNOSTICS and promoted to hard gates once Y2 distractors exist.

Usage:
    python src/reranker_9.py --smoke       # capped real calibration (~200 calls)
    python src/reranker_9.py --full        # full dev calibration (~few thousand)
    python src/reranker_9.py --stub        # offline plumbing check, zero network
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data_loader_1 import DEFAULT_PROCESSED_DIR

load_dotenv()

DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "reranker.json"

# Caller signature: (system_prompt, user_message, prompt_version) -> raw text
Caller = Callable[[str, str, str], str]


class RerankerParseError(ValueError):
    """Raised when the model output cannot be parsed into a score."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def build_operational_definitions(df_cat: pd.DataFrame) -> str:
    """Format the operational science-practice definitions from category_defs."""
    lines: list[str] = []
    for _, r in df_cat.iterrows():
        typ = str(r.get("type", ""))
        if not typ.lower().startswith("category"):
            continue
        label = str(r.get("label", "")).strip()
        definition = str(r.get("definition", "")).strip()
        if label and definition:
            lines.append(f"- {label}: {definition}")
    if not lines:  # fallback: use whatever definitions exist
        for _, r in df_cat.iterrows():
            label = str(r.get("label", "")).strip()
            definition = str(r.get("definition", "")).strip()
            if label and definition:
                lines.append(f"- {label}: {definition}")
    return "\n".join(lines)


def build_system_prompt(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    template_path: Path,
) -> str:
    df_cat = pd.read_parquet(processed_dir / "category_defs.parquet")
    defs = build_operational_definitions(df_cat)
    tmpl = Path(template_path).read_text(encoding="utf-8")
    return tmpl.replace("{operational_definitions}", defs)


def build_user_message(query: str, candidate: str) -> str:
    return (
        f'Phrase A (query): "{query}"\n'
        f'Phrase B (candidate): "{candidate}"\n\n'
        "Return ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# Output parsing (fails loudly)
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def parse_score(text: str) -> tuple[float, str]:
    """Parse `{"score":..,"rationale":..}` or `<score>..</score>`; else raise."""
    if text:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "score" in obj:
                    return _clamp01(float(obj["score"])), str(obj.get("rationale", ""))
            except (ValueError, TypeError):
                pass
        m2 = re.search(r"<score>\s*([0-9]*\.?[0-9]+)\s*</score>", text)
        if m2:
            return _clamp01(float(m2.group(1))), ""
    raise RerankerParseError(f"unparsable model output: {text[:160]!r}")


# ---------------------------------------------------------------------------
# Callers (real endpoint + deterministic stub)
# ---------------------------------------------------------------------------

class CallLedger:
    """Counts cached hits vs new (network) calls for the cost DoD."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    def classify(self, *, model: str, params: dict, prompt_version: str) -> None:
        from src.llm_client_0 import _make_cache_key, _cache_path
        key = _make_cache_key(prompt_version, model, "completion", params)
        if _cache_path(key).exists():
            self.hits += 1
        else:
            self.misses += 1


# Markers in an endpoint error body that mean "this key's budget/credential is
# spent" -> rotate to the next key. Rate-limit/transient blips are NOT here so we
# retry the same key instead of burning through the pool.
_QUOTA_AUTH_MARKERS = (
    "quota", "budget", "credit", "balance", "insufficient", "exceeded",
    "spending limit", "payment", "unauthor", "invalid api key", "invalid_api_key",
    "expired", "forbidden", "402", "401", "403",
)


def _collect_api_keys(primary_env: str) -> list[str]:
    """Ordered, de-duped keys from LLM_API_KEY (+ optional comma list) then
    LLM_API_KEY_2, LLM_API_KEY_3, ... (contiguous)."""
    keys: list[str] = []
    seen: set[str] = set()

    def _add(raw: str | None) -> None:
        if not raw:
            return
        for k in raw.split(","):
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                keys.append(k)

    _add(os.getenv(primary_env))
    i = 2
    while (val := os.getenv(f"{primary_env}_{i}")) is not None:
        _add(val)
        i += 1
    return keys


def _response_kind(raw) -> str:
    """Classify an endpoint response: 'ok' | 'quota_auth' | 'transient'."""
    if isinstance(raw, dict) and raw.get("error"):
        blob = json.dumps(raw["error"]).lower()
        if any(m in blob for m in _QUOTA_AUTH_MARKERS):
            return "quota_auth"
        return "transient"
    try:
        content = raw["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return "transient"
    return "ok" if content.strip() else "transient"


def make_real_caller(
    model: str,
    *,
    api_key_env: str = "LLM_API_KEY",
    completion_url_env: str = "COMPLETION_URL",
    temperature: float = 0.0,
    max_tokens: int = 200,
    ledger: CallLedger | None = None,
    verbose: bool = True,
) -> Caller:
    """Build a caller hitting the UF endpoint via `cached_request`.

    Supports automatic API-key rotation: when a key's budget/credential is
    exhausted (endpoint returns a quota/auth error), the caller advances to the
    next key in the pool and retries. Provide extra keys as `LLM_API_KEY_2`,
    `LLM_API_KEY_3`, ... (or a comma-separated list in `LLM_API_KEY`). Because the
    response cache is keyed on the request (not the key), already-scored pairs
    stay free across rotation.
    """
    from src.llm_client_0 import cached_request, _make_cache_key, _cache_path

    keys = _collect_api_keys(api_key_env)
    url = os.getenv(completion_url_env)
    if not keys or not url:
        raise RuntimeError(
            f"Real reranker mode requires env vars {api_key_env} and {completion_url_env}."
        )
    if verbose and len(keys) > 1:
        print(f"[reranker] {len(keys)} API keys available for rotation.")

    state = {"idx": 0}  # sticky current-key index across calls

    def _call(system: str, user: str, prompt_version: str) -> str:
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if ledger is not None:
            ledger.classify(model=model, params=params, prompt_version=prompt_version)
        cpath = _cache_path(_make_cache_key(prompt_version, model, "completion", params))

        while state["idx"] < len(keys):
            key = keys[state["idx"]]
            try:
                raw = cached_request(
                    api_key=key, url=url, endpoint="completion",
                    model=model, params=params, prompt_version=prompt_version,
                )
            except Exception as e:  # noqa: BLE001 - network/JSON blip -> transient
                cpath.unlink(missing_ok=True)
                if verbose:
                    print(f"[reranker] request error ({type(e).__name__}); transient retry.")
                return ""

            kind = _response_kind(raw)
            if kind == "ok":
                return raw["choices"][0]["message"]["content"]

            # Bad body was cached by cached_request; purge so the next attempt
            # (this key retried, or the next key) re-hits the network.
            cpath.unlink(missing_ok=True)

            if kind == "quota_auth":
                if verbose:
                    print(f"[reranker] key #{state['idx'] + 1}/{len(keys)} depleted/"
                          f"rejected; rotating to next key.")
                state["idx"] += 1
                continue  # try next key immediately
            return ""  # transient -> let score_pair_full retry the same key

        if verbose:
            print("[reranker] all API keys exhausted; degrading (cosine-only).")
        return ""

    return _call


def stub_caller(system: str, user: str, prompt_version: str) -> str:
    """Deterministic offline caller: lexical-overlap score. For tests/plumbing."""
    phrases = re.findall(r'"([^"]*)"', user)
    a = set(phrases[0].lower().split()) if len(phrases) > 0 else set()
    b = set(phrases[1].lower().split()) if len(phrases) > 1 else set()
    jac = len(a & b) / len(a | b) if (a | b) else 0.0
    return json.dumps({"score": round(jac, 3), "rationale": "stub lexical overlap"})


# ---------------------------------------------------------------------------
# score_pair
# ---------------------------------------------------------------------------

def score_pair_full(
    query: str,
    candidate: str,
    *,
    caller: Caller,
    system_prompt: str,
    prompt_version: str,
    max_retries: int = 1,
    row_id: str | None = None,
    verbose: bool = True,
) -> tuple[float, str]:
    """Score a pair; retry once on parse failure; raise loudly if still bad."""
    user = build_user_message(query, candidate)
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        text = caller(system_prompt, user, prompt_version)
        try:
            return parse_score(text)
        except RerankerParseError as e:
            last = e
            if verbose:
                print(f"[reranker] parse failure (row={row_id}, "
                      f"attempt {attempt + 1}/{max_retries + 1}): {e}")
    raise RerankerParseError(
        f"row={row_id}: unparsable after {max_retries + 1} attempts: {last}"
    )


def score_pair(
    query: str,
    candidate: str,
    *,
    model_id: str | None = None,
    prompt_version: str | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> float:
    """Public convenience: real-endpoint score in [0,1] (rationale logged)."""
    config = load_config(config_path)
    model = model_id or os.getenv(config["model_env"])
    pv = prompt_version or config["prompt_version"]
    system_prompt = build_system_prompt(
        processed_dir, template_path=_PROJECT_ROOT / config["prompt_file"])
    caller = make_real_caller(
        model, temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 200))
    s, _ = score_pair_full(
        query, candidate, caller=caller, system_prompt=system_prompt,
        prompt_version=pv, max_retries=config.get("max_retries", 1))
    return s


# ---------------------------------------------------------------------------
# Metrics (dependency-free)
# ---------------------------------------------------------------------------

def auroc(labels: list[int], scores: list[float]) -> float | None:
    """Rank-based AUROC (Mann-Whitney U with tie-aware average ranks)."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rho = Pearson correlation of ranks."""
    if len(x) < 2 or len(x) != len(y):
        return None

    def _rank(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _dcg(rels: list[int]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels: list[int], k: int) -> float:
    ideal = sorted(ranked_rels, reverse=True)
    idcg = _dcg(ideal[:k])
    return float(_dcg(ranked_rels[:k]) / idcg) if idcg > 0 else 0.0


def rank_metrics(order_ids: list[str], relevant: set, k: int) -> dict:
    """precision@1, mrr@k, ndcg@k for a single ranked list."""
    rels = [1 if d in relevant else 0 for d in order_ids]
    rr = 0.0
    for i, rel in enumerate(rels[:k]):
        if rel:
            rr = 1.0 / (i + 1)
            break
    return {
        "precision@1": float(rels[0]) if rels else 0.0,
        f"mrr@{k}": rr,
        f"ndcg@{k}": ndcg_at_k(rels, k),
    }


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def _utt_text_map(corpus: pd.DataFrame) -> dict:
    return dict(zip(corpus["utt_id"].astype(str), corpus["utterance"].astype(str)))


def build_pair_audit(frames: dict, *, n: int = 100, split: str = "val",
                     seed: int = 42) -> list[dict]:
    """Gold-labeled pairs: (anchor, its variant)=1, (anchor, hard/neg)=0."""
    import random
    from src.biencoder_8 import _split_ids
    rng = random.Random(seed)

    corpus, pairs, negs, splits = (
        frames["corpus"], frames["pairs"], frames["negatives"], frames["splits"])
    utt_text = _utt_text_map(corpus)

    pair_ids = _split_ids(splits, "pair", split)
    pos: list[dict] = []
    anchors_used: list[str] = []
    if len(pairs) > 0:
        ep = pairs[pairs["pair_id"].astype(str).isin(pair_ids)]
        for _, p in ep.iterrows():
            aid = str(p["anchor_id"])
            anchor = utt_text.get(aid)
            if not anchor or pd.isna(p["variant_text"]):
                continue
            pos.append({"query": anchor, "candidate": str(p["variant_text"]),
                        "label": 1, "anchor_id": aid})
            anchors_used.append(anchor)

    neg: list[dict] = []
    if len(negs) > 0 and anchors_used:
        neg_ids = _split_ids(splits, "negative", split)
        en = negs[negs["neg_id"].astype(str).isin(neg_ids)]
        neg_texts = en["text"].astype(str).tolist()
        rng.shuffle(neg_texts)
        for i, ntext in enumerate(neg_texts):
            anchor = anchors_used[i % len(anchors_used)]
            neg.append({"query": anchor, "candidate": ntext, "label": 0,
                        "anchor_id": None})

    half = max(1, n // 2)
    rng.shuffle(pos)
    rng.shuffle(neg)
    audit = pos[:half] + neg[:half]
    rng.shuffle(audit)
    return audit


def score_audit(audit: list[dict], *, caller: Caller, system_prompt: str,
                prompt_version: str, max_retries: int = 1,
                verbose: bool = True) -> dict:
    labels, scores = [], []
    for i, row in enumerate(audit):
        s, _ = score_pair_full(
            row["query"], row["candidate"], caller=caller,
            system_prompt=system_prompt, prompt_version=prompt_version,
            max_retries=max_retries, row_id=f"audit_{i}", verbose=verbose)
        labels.append(int(row["label"]))
        scores.append(s)
    return {"n": len(audit), "auroc": auroc(labels, scores),
            "labels": labels, "scores": scores}


def run_stability(frames: dict, *, caller: Caller, system_a: str, system_b: str,
                  pv_a: str, pv_b: str, n: int = 50, seed: int = 42,
                  max_retries: int = 1, verbose: bool = True) -> dict:
    """Spearman rho between two prompt phrasings on the same pairs."""
    audit = build_pair_audit(frames, n=n * 2, seed=seed)[:n]
    sa, sb = [], []
    for i, row in enumerate(audit):
        a, _ = score_pair_full(row["query"], row["candidate"], caller=caller,
                               system_prompt=system_a, prompt_version=pv_a,
                               max_retries=max_retries, row_id=f"stab_a_{i}",
                               verbose=verbose)
        b, _ = score_pair_full(row["query"], row["candidate"], caller=caller,
                               system_prompt=system_b, prompt_version=pv_b,
                               max_retries=max_retries, row_id=f"stab_b_{i}",
                               verbose=verbose)
        sa.append(a)
        sb.append(b)
    return {"n": len(audit), "spearman": spearman(sa, sb),
            "scores_a": sa, "scores_b": sb}


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
    from src.biencoder_8 import load_verified_frames

    config = load_config(config_path)
    budget = config.get(mode, config.get("smoke", {}))
    pv_a = config["prompt_version"]
    pv_b = pv_a + "b"

    frames = load_verified_frames(processed_dir)
    system_a = build_system_prompt(
        processed_dir, template_path=_PROJECT_ROOT / config["prompt_file"])
    system_b = build_system_prompt(
        processed_dir, template_path=_PROJECT_ROOT / config["prompt_file_paraphrase"])

    # --- caller + projected-cost guard --------------------------------------
    ledger = CallLedger()
    if use_stub:
        caller: Caller = stub_caller
    else:
        model = os.getenv(config["model_env"])
        if not model:
            raise RuntimeError(f"env {config['model_env']} not set (gpt-oss-120b name).")
        caller = make_real_caller(
            model, temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 200), ledger=ledger)

    n_audit = budget.get("audit_pairs", 100)
    n_stab = budget.get("stability_pairs", 50)
    rerank_sample = budget.get("rerank_sample", 0)
    cap = budget.get("max_new_calls", 400)
    projected = n_audit + 2 * n_stab + rerank_sample * budget.get("top_k", 20)
    if verbose:
        print(f"Mode={mode} | projected max calls ~{projected} (cap {cap}) | "
              f"stub={use_stub}")
    if not use_stub and projected > cap:
        raise RuntimeError(
            f"Projected {projected} calls exceeds cap {cap}; raise '{mode}.max_new_calls' "
            f"or lower audit/stability sizes.")

    mr = config.get("max_retries", 1)

    # --- AUROC (score sanity) -----------------------------------------------
    audit = build_pair_audit(frames, n=n_audit, seed=config["audit"]["seed"])
    audit_res = score_audit(audit, caller=caller, system_prompt=system_a,
                            prompt_version=pv_a, max_retries=mr, verbose=verbose)

    # --- Prompt stability ----------------------------------------------------
    stab_res = run_stability(
        frames, caller=caller, system_a=system_a, system_b=system_b,
        pv_a=pv_a, pv_b=pv_b, n=n_stab, seed=config["stability"]["seed"],
        max_retries=mr, verbose=verbose)

    # --- Retrieval lift diagnostic (optional; --full) -----------------------
    retrieval = None
    if rerank_sample > 0 and not use_stub:
        retrieval = rerank_diagnostic(
            frames, caller=caller, system_prompt=system_a, prompt_version=pv_a,
            sample=rerank_sample, top_k=budget.get("top_k", 20),
            processed_dir=processed_dir, max_retries=mr, verbose=verbose)

    # --- Gate ----------------------------------------------------------------
    gate = config["gate"]
    au = audit_res["auroc"]
    rho = stab_res["spearman"]
    auroc_pass = au is not None and au >= gate["min_auroc"]
    stab_pass = rho is not None and rho >= gate["min_spearman"]
    passed = bool(auroc_pass and stab_pass)

    report = {
        "version": config["version"],
        "prompt_version": pv_a,
        "mode": mode,
        "stub": use_stub,
        "auroc": {"value": au, "n": audit_res["n"], "min": gate["min_auroc"],
                  "passed": auroc_pass},
        "stability": {"spearman": rho, "n": stab_res["n"], "min": gate["min_spearman"],
                      "passed": stab_pass},
        "retrieval_lift_diagnostic": retrieval,
        "cost_ledger": {"new_calls": ledger.misses, "cached_calls": ledger.hits,
                        "total": ledger.total, "cap": cap},
        "dod_passed": passed,
    }

    out = config["output"]
    report_path = _PROJECT_ROOT / out["report_json"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    _write_calibration_md(report, _PROJECT_ROOT / out["calibration_md"], config)

    if verbose:
        _print_report(report, report_path)
    return report


def rerank_diagnostic(frames: dict, *, caller: Caller, system_prompt: str,
                      prompt_version: str, sample: int, top_k: int,
                      processed_dir: Path, max_retries: int = 1,
                      verbose: bool = True) -> dict:
    """Bi-encoder top-k vs LLM-reranked order on a dev sample (diagnostic)."""
    import random
    from sentence_transformers import SentenceTransformer
    from src.biencoder_8 import (_encode, _split_ids, load_config as _bcfg,
                                 verified_positive_corpus)

    bcfg = _bcfg()
    model_dir = _PROJECT_ROOT / bcfg["output"]["model_dir"]
    if not model_dir.exists():
        return {"skipped": "no fine-tuned bi-encoder"}
    st = SentenceTransformer(str(model_dir), trust_remote_code=True, device="cpu")
    qp, dp = bcfg.get("query_prefix", ""), bcfg.get("document_prefix", "")

    corpus, pairs, negs, splits = (
        frames["corpus"], frames["pairs"], frames["negatives"], frames["splits"])
    pos = verified_positive_corpus(corpus)
    anchor_ids = pos["utt_id"].astype(str).tolist()
    anchor_text = pos["utterance"].astype(str).tolist()

    neg_ids_split = _split_ids(splits, "negative", "val")
    en = negs[negs["neg_id"].astype(str).isin(neg_ids_split)]
    neg_ids = en["neg_id"].astype(str).tolist()
    neg_text = en["text"].astype(str).tolist()

    doc_ids = anchor_ids + neg_ids
    doc_text = anchor_text + neg_text
    doc_emb = _encode(st, doc_text, dp)
    text_by_id = dict(zip(doc_ids, doc_text))

    val_pair_ids = _split_ids(splits, "pair", "val")
    ep = pairs[pairs["pair_id"].astype(str).isin(val_pair_ids)]
    rng = random.Random(42)
    rows = ep.to_dict("records")
    rng.shuffle(rows)
    rows = rows[:sample]

    base_acc, rr_acc = [], []
    for qi, p in enumerate(rows):
        aid = str(p["anchor_id"])
        if aid not in text_by_id:
            continue
        q_emb = _encode(st, [str(p["variant_text"])], qp)[0]
        sims = doc_emb @ q_emb
        top_idx = np.argsort(-sims)[:top_k]
        cand_ids = [doc_ids[i] for i in top_idx]
        base_acc.append(rank_metrics(cand_ids, {aid}, top_k))
        llm_scores = []
        for ci in cand_ids:
            s, _ = score_pair_full(str(p["variant_text"]), text_by_id[ci],
                                   caller=caller, system_prompt=system_prompt,
                                   prompt_version=prompt_version,
                                   max_retries=max_retries,
                                   row_id=f"rr_{qi}_{ci}", verbose=verbose)
            llm_scores.append(s)
        rer_ids = [cid for _, cid in sorted(zip(llm_scores, cand_ids), reverse=True)]
        rr_acc.append(rank_metrics(rer_ids, {aid}, top_k))

    def _mean(accs, key):
        vals = [a[key] for a in accs if key in a]
        return float(np.mean(vals)) if vals else None

    keys = ["precision@1", f"mrr@{top_k}", f"ndcg@{top_k}"]
    return {
        "n_queries": len(base_acc),
        "bi_encoder": {k: _mean(base_acc, k) for k in keys},
        "reranked": {k: _mean(rr_acc, k) for k in keys},
    }


def _print_report(report: dict, path: Path) -> None:
    print("\n=== Step 9 re-ranker calibration ===")
    a = report["auroc"]
    s = report["stability"]
    av = f"{a['value']:.4f}" if isinstance(a["value"], float) else "n/a"
    sv = f"{s['spearman']:.4f}" if isinstance(s["spearman"], float) else "n/a"
    print(f"  AUROC (n={a['n']})        : {av} (>= {a['min']}) "
          f"{'PASS' if a['passed'] else 'FAIL'}")
    print(f"  Stability rho (n={s['n']}): {sv} (>= {s['min']}) "
          f"{'PASS' if s['passed'] else 'FAIL'}")
    if report.get("retrieval_lift_diagnostic"):
        print(f"  Retrieval lift (diagnostic): {report['retrieval_lift_diagnostic']}")
    c = report["cost_ledger"]
    print(f"  Cost ledger: {c['new_calls']} new / {c['cached_calls']} cached "
          f"(cap {c['cap']})")
    print(f"  Step 9 DoD (AUROC + stability): "
          f"{'PASS' if report['dod_passed'] else 'FAIL'}")
    print(f"  Wrote {path}")


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def _write_calibration_md(report: dict, path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a, s, c = report["auroc"], report["stability"], report["cost_ledger"]
    lines = [
        "# Re-ranker calibration", "",
        f"- **Prompt version**: `{report['prompt_version']}` "
        f"(`{config['prompt_file']}`)",
        f"- **Model**: `{config['model_env']}` (gpt-oss-120b), temperature "
        f"{config.get('temperature', 0.0)}, up to {config.get('max_retries', 1)} "
        f"retries on parse failure (purges cache on transient empty completions)",
        f"- **Mode**: {report['mode']} (stub={report['stub']})", "",
        "## Gate (this revision)", "",
        f"| Metric | Value | Threshold | Result |",
        f"|---|---|---|---|",
        f"| AUROC (score sanity, n={a['n']}) | {_fmt(a['value'])} | >= {a['min']} | "
        f"{'PASS' if a['passed'] else 'FAIL'} |",
        f"| Spearman rho (prompt stability, n={s['n']}) | {_fmt(s['spearman'])} | "
        f">= {s['min']} | {'PASS' if s['passed'] else 'FAIL'} |", "",
        f"**Step 9 DoD: {'PASS' if report['dod_passed'] else 'FAIL'}**", "",
        "## Diagnostics (not gated until Y2)", "",
        "Dev retrieval is near-ceiling after Step 8, so retrieval lift cannot "
        "show meaningful headroom yet. Reported for the record; promoted to a "
        "hard gate once Y2 distractors exist.", "",
        f"- Retrieval lift: {report.get('retrieval_lift_diagnostic')}", "",
        "## Cost ledger", "",
        f"- New (network) calls: {c['new_calls']}",
        f"- Cached calls: {c['cached_calls']}",
        f"- Cap: {c['cap']}",
        "- Re-running after a code-only change makes zero new calls (cache hit).",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--smoke", action="store_true", help="Capped real calibration (default).")
    p.add_argument("--full", action="store_true", help="Full dev calibration (costly).")
    p.add_argument("--stub", action="store_true", help="Offline plumbing check, no network.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mode = "full" if args.full else "smoke"
    run(
        args.processed_dir,
        config_path=args.config,
        mode=mode,
        use_stub=args.stub,
        verbose=not args.quiet,
    )
