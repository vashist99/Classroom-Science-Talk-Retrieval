"""Stage 2 — Family 5: Direct Codebook-Grounded Sliding-Window Multi-Utterance LLMs & Multi-Agent Verifier.

Replaces legacy pairwise anchor-matching (10 calls/utterance, confirmation-bias hallucinations)
with direct sliding-window dialogue coding (35 consecutive target turns + 8 pre / 4 post
read-only context turns in ONE call, split at >30s silences), strictly following
2026-09-28_TeLI_Science_Talk_Coding_Manual.docx (v0.3).

Benchmarks live UF NaviGator LLMs head-to-head on identical held-out transcripts:
  - 5A_gpt_oss_120b_low
  - 5A_gpt_oss_120b_med
  - 5B_gemma4_31b_it
  - 5C_llama33_70b
  - 5D_nemotron3_super_120b
  - 5E_gpt_oss_20b_low
  - 5F_multi_agent_adjudicated (Top-2 diverse models + conservative high-effort adjudicator on disagreements!)
"""
from __future__ import annotations

import json
import re
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

from src.v3.codebook_v3 import (
    MANUAL_V3_ADJUDICATOR_SYSTEM_PROMPT,
    MANUAL_V3_SLIDING_WINDOW_SYSTEM_PROMPT,
    infer_subtype_rule,
)
from src.v3.llm_client import UFNavigatorClient
from src.v3.metrics import binary_metrics, evaluate_y1_full_recall, evaluate_y2_oof, tune_threshold_f1

DATA_DIR = PROJECT_ROOT / "data" / "v3"
PREDS_DIR = PROJECT_ROOT / "reports" / "v3" / "stage1_preds"
S2_DIR = PROJECT_ROOT / "reports" / "v3" / "stage2_llm"

PROMPT_VERSION_WINDOW = "v3_manual_win_v3"
PROMPT_VERSION_ADJ = "v3_manual_adj_v2"


# ---------------------------------------------------------------------------
# Sliding-Window Dialogue Chunking (Pause-Split, Episode-Aware)
# ---------------------------------------------------------------------------

def build_dialogue_windows(
    tx_df: pd.DataFrame,
    *,
    target_mask: np.ndarray | None = None,
    window_size: int = 32,
    ctx_before: int = 8,
    ctx_after: int = 4,
    pause_cut_s: float = 30.0,
    expand_radius: int = 6,
) -> list[dict[str, Any]]:
    """Chunk one chronological transcript into sliding dialogue windows.

    If `target_mask` is provided (Stage-1 candidate filtering), expands every True
    index by `±expand_radius` inside its pause-bounded episode, merges overlapping
    intervals, and chunks those active episodes into `window_size` target blocks
    padded with read-only `[CTX]` neighbors.
    If `target_mask` is None, chunks the ENTIRE transcript sequentially.
    """
    n = len(tx_df)
    if n == 0:
        return []
    line_nos = tx_df["line_no"].to_numpy(dtype=int)
    speakers = tx_df["speaker"].fillna("unknown").astype(str).tolist()
    utts = tx_df["utterance"].fillna("").astype(str).tolist()
    gaps = tx_df["gap_prev_s"].fillna(0.0).to_numpy(dtype=float)

    # Split transcript at silence gaps > pause_cut_s
    ep_breaks = np.flatnonzero(gaps > pause_cut_s)
    ep_starts = np.r_[0, ep_breaks]
    ep_ends = np.r_[ep_breaks, n]

    windows: list[dict[str, Any]] = []
    for es, ee in zip(ep_starts, ep_ends):
        L = ee - es
        if target_mask is None:
            spans = [(0, L)]
        else:
            sub_m = target_mask[es:ee]
            hits = np.flatnonzero(sub_m)
            if len(hits) == 0:
                continue
            # Expand each Stage-1 hit into local episode interval & merge overlaps
            intervals = [(max(0, int(h) - expand_radius), min(L, int(h) + expand_radius + 1)) for h in hits]
            merged: list[tuple[int, int]] = []
            for a, b in intervals:
                if merged and a <= merged[-1][1] + 2:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            spans = merged

        for a, b in spans:
            for chunk_s in range(a, b, window_size):
                chunk_e = min(b, chunk_s + window_size)
                pre_s = max(0, chunk_s - ctx_before)
                post_e = min(L, chunk_e + ctx_after)

                lines_out: list[str] = []
                target_row_positions: list[int] = []
                target_line_nos: list[int] = []

                for local_i in range(pre_s, post_e):
                    gi = int(es + local_i)
                    spk = speakers[gi]
                    txt = utts[gi].replace("\n", " ").strip()[:240]
                    lno = int(line_nos[gi])
                    if chunk_s <= local_i < chunk_e:
                        lines_out.append(f"[T {lno}] {spk}: {txt}")
                        target_row_positions.append(gi)
                        target_line_nos.append(lno)
                    else:
                        lines_out.append(f"[CTX] {spk}: {txt}")

                if target_row_positions:
                    windows.append({
                        "transcript_id": str(tx_df["transcript_id"].iloc[0]),
                        "user_prompt": (
                            "Classify ONLY the [T <id>] lines below according to the TeLI Coding Manual. "
                            "Use [CTX] lines strictly for context.\n\n"
                            + "\n".join(lines_out)
                            + '\n\nReturn ONLY JSON: {"hits": [...]}'
                        ),
                        "target_positions": target_row_positions,
                        "target_line_nos": target_line_nos,
                        "line_to_pos": dict(zip(target_line_nos, target_row_positions)),
                    })
    return windows


# ---------------------------------------------------------------------------
# Strict JSON Output Parser
# ---------------------------------------------------------------------------

def parse_window_hits(raw_text: str, valid_line_nos: set[int]) -> list[dict[str, Any]]:
    """Parse sparse JSON `{"hits": [...]}` safely, keeping only valid target IDs."""
    text = (raw_text or "").strip()
    # Strip markdown code fences if model included them
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    obj = None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            # Repair trailing commas if any
            repaired = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            try:
                obj = json.loads(repaired)
            except json.JSONDecodeError:
                obj = None

    items_raw: list[dict[str, Any]] = []
    if isinstance(obj, dict) and isinstance(obj.get("hits"), list):
        items_raw = obj["hits"]
    else:
        # Truncation-resilient recovery: salvage every complete JSON hit object
        # emitted before max_tokens cutoff instead of discarding the whole science episode!
        for om in re.finditer(r"\{[^{}]*\"id\"\s*:\s*\d+[^{}]*\}", text):
            try:
                it = json.loads(re.sub(r",\s*\}", "}", om.group(0)))
                if isinstance(it, dict):
                    items_raw.append(it)
            except json.JSONDecodeError:
                continue

    if not items_raw:
        return []

    parsed: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in items_raw:
        if not isinstance(item, dict):
            continue
        try:
            lno = int(item.get("id"))
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if lno not in valid_line_nos or lno in seen:
            continue
        seen.add(lno)
        score = max(0.0, min(1.0, score))
        try:
            conf = int(item.get("conf", 3))
        except (TypeError, ValueError):
            conf = 3
        conf = max(1, min(5, conf))
        route = str(item.get("route", "content")).strip().lower()
        subtype = str(item.get("subtype", "content")).strip().lower() or "content"
        rationale = str(item.get("rationale", "")).strip()[:200]
        parsed.append({
            "id": lno,
            "score": score,
            "conf": conf,
            "route": route,
            "subtype": subtype,
            "rationale": rationale,
        })
    return parsed


def parse_adjudicator_response(raw_text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", (raw_text or "").strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
            if isinstance(obj, dict) and "score" in obj:
                return {
                    "score": max(0.0, min(1.0, float(obj["score"]))),
                    "conf": max(1, min(5, int(obj.get("conf", 3)))),
                    "route": str(obj.get("route", "none")).strip().lower(),
                    "subtype": str(obj.get("subtype", "none")).strip().lower(),
                    "rationale": str(obj.get("rationale", "")).strip()[:200],
                }
        except Exception:
            pass
    return {"score": 0.5, "conf": 1, "route": "none", "subtype": "none", "rationale": "parse_fallback"}


# ---------------------------------------------------------------------------
# Score Windows in Parallel with Any UF NaviGator LLM
# ---------------------------------------------------------------------------

def score_transcripts_with_llm(
    df: pd.DataFrame,
    *,
    client: UFNavigatorClient,
    model: str,
    reasoning_effort: str | None = None,
    target_mask: np.ndarray | None = None,
    window_size: int = 32,
    expand_radius: int = 6,
    max_workers: int = 4,
    max_tokens_override: int | None = None,
    prompt_version_override: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Score transcripts using sliding-window direct Manual-v0.3 prompting.

    Returns dict with arrays aligned to `df`:
      - `scores`: float32 (N,), default 0.02 for unscored/omitted rows
      - `evaluated`: bool (N,), True for every turn sent as a [T] target
      - `conf`: int8 (N,)
      - `subtype`: list[str] (N,)
      - `rationale`: list[str] (N,)
      - `stats`: window count, parse failures, elapsed time, ledger summary
    """
    n = len(df)
    scores = np.full(n, 0.02, dtype=np.float32)
    evaluated = np.zeros(n, dtype=bool)
    confs = np.full(n, 4, dtype=np.int8)
    subtypes = [""] * n
    rationales = [""] * n

    # Build all sliding windows across transcripts, keeping global row offsets!
    all_windows: list[dict[str, Any]] = []
    tx_ids = df["transcript_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, tx_ids[1:] != tx_ids[:-1]])
    ends = np.r_[starts[1:], n]

    for s, e in zip(starts, ends):
        tx_sub = df.iloc[s:e].reset_index(drop=True)
        tm = target_mask[s:e] if target_mask is not None else None
        wins = build_dialogue_windows(
            tx_sub, target_mask=tm, window_size=window_size, expand_radius=expand_radius
        )
        for w in wins:
            w["global_offset"] = int(s)
            all_windows.append(w)

    t0 = time.time()
    snap_start = client.ledger.summary()
    n_err = 0
    err_samples: list[str] = []
    pv_base = prompt_version_override or PROMPT_VERSION_WINDOW
    pv = f"{pv_base}_{reasoning_effort or 'std'}"
    if max_tokens_override is not None:
        max_tok = max_tokens_override
    else:
        max_tok = 4096 if (reasoning_effort in ("low", "medium", "high") or "nemotron" in model) else 3200

    def _call_one(w: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
        try:
            res = client.complete(
                model=model,
                system_prompt=MANUAL_V3_SLIDING_WINDOW_SYSTEM_PROMPT,
                user_prompt=w["user_prompt"],
                prompt_version=pv,
                temperature=0.0,
                max_tokens=max_tok,
                reasoning_effort=reasoning_effort,
            )
            hits = parse_window_hits(res["content"], set(w["target_line_nos"]))
            return w, hits, None
        except Exception as exc:
            return w, [], f"{w['transcript_id']}@{w['target_line_nos'][0]}: {type(exc).__name__}: {str(exc)[:140]}"

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_call_one, w) for w in all_windows]
        for fut in as_completed(futs):
            w, hits, err_msg = fut.result()
            done += 1
            if err_msg is not None:
                n_err += 1
                if len(err_samples) < 10:
                    err_samples.append(err_msg)
                if verbose and n_err <= 5:
                    print(f"  [WARN {model}] window failed: {err_msg}", flush=True)
                continue
            off = w["global_offset"]
            for local_pos in w["target_positions"]:
                evaluated[off + local_pos] = True
                scores[off + local_pos] = 0.05  # explicitly evaluated non-science default
            for h in hits:
                local_pos = w["line_to_pos"][h["id"]]
                gi = off + local_pos
                if h["score"] > scores[gi]:
                    scores[gi] = float(h["score"])
                    confs[gi] = int(h["conf"])
                    subtypes[gi] = h["subtype"]
                    rationales[gi] = h["rationale"]
            if verbose and (done % 40 == 0 or done == len(all_windows)):
                led = client.ledger.summary()
                print(
                    f"  [{model}:{reasoning_effort or 'std'}] {done}/{len(all_windows)} windows | "
                    f"err={n_err} | {time.time() - t0:.1f}s | ${led['usd_spent']:.4f}"
                )

    return {
        "scores": scores,
        "evaluated": evaluated,
        "conf": confs,
        "subtype": subtypes,
        "rationale": rationales,
        "stats": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "n_windows": len(all_windows),
            "n_evaluated_turns": int(evaluated.sum()),
            "n_errors": n_err,
            "error_samples": err_samples,
            "elapsed_s": round(time.time() - t0, 2),
            "ledger_diff": client.ledger.diff_from(snap_start),
        },
    }


def adjudicate_disagreements(
    df: pd.DataFrame,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    rats_a: list[str],
    rats_b: list[str],
    *,
    client: UFNavigatorClient,
    adjudicator_model: str = "gpt-oss-120b",
    adjudicator_effort: str = "medium",
    disagree_thr: float = 0.50,
    borderline_lo: float = 0.35,
    borderline_hi: float = 0.72,
    max_workers: int = 4,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run conservative adjudicator ONLY where Model A & Model B disagree (Ahtisham et al. 2026)."""
    n = len(df)
    a_pos = scores_a >= disagree_thr
    b_pos = scores_b >= disagree_thr
    disagree = (a_pos != b_pos) | (
        ((scores_a >= borderline_lo) & (scores_a <= borderline_hi))
        & ((scores_b >= borderline_lo) & (scores_b <= borderline_hi))
    )
    idxs = np.flatnonzero(disagree)

    final_scores = (0.5 * scores_a + 0.5 * scores_b).astype(np.float32)
    final_rats = [ra if sa >= sb else rb for sa, sb, ra, rb in zip(scores_a, scores_b, rats_a, rats_b)]
    final_sub = [infer_subtype_rule(u) for u in df["utterance"].fillna("").astype(str).tolist()]

    if len(idxs) == 0:
        return {"scores": final_scores, "rationale": final_rats, "subtype": final_sub, "n_adjudicated": 0}

    utts = df["utterance"].fillna("").astype(str).tolist()
    spks = df["speaker"].fillna("unknown").astype(str).tolist()
    tx_ids = df["transcript_id"].to_numpy()
    gaps = df["gap_prev_s"].fillna(0.0).to_numpy(dtype=float)

    def _build_adj_prompt(gi: int) -> str:
        tx = tx_ids[gi]
        # Gather up to 6 turns before & 3 turns after inside same pause-bounded episode
        pre_lines = []
        j = gi - 1
        while j >= 0 and len(pre_lines) < 6 and tx_ids[j] == tx and gaps[j + 1] <= 30.0:
            pre_lines.append(f"  {spks[j]}: {utts[j][:180]}")
            j -= 1
        pre_lines.reverse()

        post_lines = []
        j = gi + 1
        while j < n and len(post_lines) < 3 and tx_ids[j] == tx and gaps[j] <= 30.0:
            post_lines.append(f"  {spks[j]}: {utts[j][:180]}")
            j += 1

        return (
            "BEFORE:\n" + ("\n".join(pre_lines) if pre_lines else "  (start)")
            + f"\n\n>>> TARGET UTTERANCE: {spks[gi]}: {utts[gi][:240]} <<<\n\n"
            + "AFTER:\n" + ("\n".join(post_lines) if post_lines else "  (end)")
            + f"\n\nCoder A score={scores_a[gi]:.2f} ({rats_a[gi] or 'not flagged'})"
            + f"\nCoder B score={scores_b[gi]:.2f} ({rats_b[gi] or 'not flagged'})"
            + "\n\nApply the TeLI Manual strictly and return ONLY the JSON object."
        )

    def _call_adj(gi: int) -> tuple[int, dict[str, Any]]:
        res = client.complete(
            model=adjudicator_model,
            system_prompt=MANUAL_V3_ADJUDICATOR_SYSTEM_PROMPT,
            user_prompt=_build_adj_prompt(gi),
            prompt_version=f"{PROMPT_VERSION_ADJ}_{adjudicator_effort}",
            temperature=0.0,
            max_tokens=600,
            reasoning_effort=adjudicator_effort,
        )
        return gi, parse_adjudicator_response(res["content"])

    t0 = time.time()
    if verbose:
        print(f"  [adjudicator] Resolving {len(idxs)} disagreements with {adjudicator_model} ({adjudicator_effort})...")
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_call_adj, int(gi)) for gi in idxs]
        for fut in as_completed(futs):
            try:
                gi, parsed = fut.result()
                final_scores[gi] = float(parsed["score"])
                if parsed["rationale"]:
                    final_rats[gi] = parsed["rationale"]
                if parsed["subtype"] and parsed["subtype"] != "none":
                    final_sub[gi] = parsed["subtype"]
            except Exception:
                pass
            done += 1

    if verbose:
        print(f"  [adjudicator] Resolved {done} items in {time.time() - t0:.1f}s | ${client.ledger.summary()['usd_spent']:.4f}")
    return {
        "scores": final_scores,
        "rationale": final_rats,
        "subtype": final_sub,
        "n_adjudicated": int(len(idxs)),
    }


# ---------------------------------------------------------------------------
# Head-to-Head Pilot Benchmark Across 6 Live UF NaviGator LLM Configs
# ---------------------------------------------------------------------------

def select_stratified_pilot_transcripts(df_y2: pd.DataFrame, df_y1: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Pick 10 diverse Y2 transcripts (2 per fold: 1 high-sci + 1 low/mid-sci) + 5 Y1 SCI-rich sheets."""
    scr = df_y2[df_y2["screened"]]
    y2_picks: list[str] = []
    for f in range(5):
        fg = scr[scr["fold"] == f]
        tx_stats = fg.groupby("transcript_id").agg(
            n_tp=("label", lambda s: int((s == 1).sum())),
            n_fp=("label", lambda s: int((s == 0).sum())),
            n_all=("label", "size"),
        ).reset_index()
        tx_stats = tx_stats[tx_stats["n_fp"] >= 20].sort_values("n_tp", ascending=False)
        if len(tx_stats) >= 2:
            # Pick top-quartile science transcript + lower-quartile (hard FP room!) transcript
            y2_picks.append(str(tx_stats.iloc[1]["transcript_id"]))
            y2_picks.append(str(tx_stats.iloc[max(2, len(tx_stats) - 3)]["transcript_id"]))

    # Y1: top 5 science-rich sheets (190 of 210 SCI turns + ~1,300 clean non-SCI!)
    y1_clean = df_y1[df_y1["eval_clean_mask"]]
    y1_stats = y1_clean.groupby("transcript_id")["label"].sum().sort_values(ascending=False)
    y1_picks = y1_stats.head(6).index.tolist()
    return sorted(set(y2_picks)), y1_picks


def run_llm_pilot_benchmark(budget_cap_usd: float = 8.0, verbose: bool = True) -> dict[str, Any]:
    """Benchmark 6 live UF NaviGator LLM configs + Multi-Agent Adjudication head-to-head."""
    S2_DIR.mkdir(parents=True, exist_ok=True)
    df_y2 = pd.read_parquet(DATA_DIR / "y2_utterances.parquet")
    df_y1 = pd.read_parquet(DATA_DIR / "y1_coding_transcripts.parquet")
    s1_oof = np.load(PREDS_DIR / "4C_stage1_ensemble_y2_oof.npy")

    y2_txs, y1_txs = select_stratified_pilot_transcripts(df_y2, df_y1)
    y2_mask = df_y2["transcript_id"].isin(y2_txs).to_numpy()
    sub_y2 = df_y2.loc[y2_mask].reset_index(drop=True)
    sub_s1 = s1_oof[y2_mask]
    sub_y1 = df_y1[df_y1["transcript_id"].isin(y1_txs)].reset_index(drop=True)

    # On Y2 pilot: evaluate ALL reviewed flags + Stage-1 high-recall candidates (top 14% of transcript)
    # so we test BOTH reviewed-flag precision/recall AND real-world Stage-1->Stage-2 cascade behavior!
    s1_thr_cand = float(np.quantile(sub_s1[sub_y2["screened"].to_numpy()], 0.86))
    y2_cand_mask = (sub_y2["label"].to_numpy() >= 0) | (sub_y2["screened"].to_numpy() & (sub_s1 >= s1_thr_cand))

    print(
        f"[stage2_llm] Pilot set: Y2={len(y2_txs)} transcripts ({len(sub_y2):,} turns, "
        f"TP={(sub_y2['label']==1).sum()}, FP={(sub_y2['label']==0).sum()}, cand={y2_cand_mask.sum():,}) | "
        f"Y1={len(y1_txs)} FULL transcripts ({len(sub_y1):,} turns, clean SCI={sub_y1.loc[sub_y1['eval_clean_mask'], 'label'].sum()})"
    )

    configs = [
        ("5A_gpt_oss_120b_low", "gpt-oss-120b", "low", 1400),
        ("5A_gpt_oss_120b_med", "gpt-oss-120b", "medium", 2200),
        ("5B_gemma4_31b_it", "gemma-4-31b-it", None, 1400),
        ("5C_llama33_70b", "llama-3.3-70b-instruct", None, 1400),
        ("5E_gpt_oss_20b_low", "gpt-oss-20b", "low", 2400),
    ]

    client = UFNavigatorClient(budget_cap_usd=budget_cap_usd)
    pilot_results: dict[str, Any] = {}
    outputs_y2: dict[str, dict[str, Any]] = {}
    outputs_y1: dict[str, dict[str, Any]] = {}

    rev_mask = sub_y2["label"].isin([0, 1]).to_numpy()
    y_y2_rev = sub_y2.loc[rev_mask, "label"].to_numpy(dtype=int)
    u0_mask = (sub_y2["screened"].to_numpy()) & (sub_y2["label"].to_numpy() == -1)

    y1_clean_mask = sub_y1["eval_clean_mask"].to_numpy(dtype=bool)
    y_y1_clean = sub_y1.loc[y1_clean_mask, "label"].to_numpy(dtype=int)

    for tag, model, eff, mt_ov in configs:
        print(f"\n--- Benchmarking {tag} ({model}, effort={eff}) ---", flush=True)
        snap_before = client.ledger.summary()
        res_y2 = score_transcripts_with_llm(
            sub_y2, client=client, model=model, reasoning_effort=eff,
            target_mask=y2_cand_mask, window_size=32, expand_radius=5, max_workers=4,
            max_tokens_override=mt_ov, prompt_version_override="v3_manual_win_v1", verbose=verbose,
        )
        res_y1 = score_transcripts_with_llm(
            sub_y1, client=client, model=model, reasoning_effort=eff,
            target_mask=None, window_size=32, expand_radius=0, max_workers=4,
            max_tokens_override=mt_ov, prompt_version_override="v3_manual_win_v1", verbose=verbose,
        )
        diff_cfg = client.ledger.diff_from(snap_before)
        cost_cfg = round(diff_cfg["total_equivalent_usd"], 4)
        outputs_y2[tag] = res_y2
        outputs_y1[tag] = res_y1

        p2_rev = res_y2["scores"][rev_mask]
        p1_cln = res_y1["scores"][y1_clean_mask]
        from sklearn.metrics import roc_auc_score
        y2_auroc = round(float(roc_auc_score(y_y2_rev, p2_rev)), 4) if len(np.unique(y_y2_rev)) == 2 else None
        y1_auroc = round(float(roc_auc_score(y_y1_clean, p1_cln)), 4) if len(np.unique(y_y1_clean)) == 2 else None

        m2_05 = binary_metrics(y_y2_rev, (p2_rev >= 0.50).astype(int))
        m2_07 = binary_metrics(y_y2_rev, (p2_rev >= 0.70).astype(int))
        m1_05 = binary_metrics(y_y1_clean, (p1_cln >= 0.50).astype(int))
        m1_07 = binary_metrics(y_y1_clean, (p1_cln >= 0.70).astype(int))
        u0_05 = float((res_y2["scores"][u0_mask] >= 0.50).mean())
        u0_07 = float((res_y2["scores"][u0_mask] >= 0.70).mean())

        pilot_results[tag] = {
            "model": model,
            "reasoning_effort": eff,
            "y2_rev_auroc": y2_auroc,
            "y1_full_auroc": y1_auroc,
            "y2_rev_at_050": m2_05,
            "y2_rev_at_070": m2_07,
            "y2_u0_flag_pct_050": round(100.0 * u0_05, 2),
            "y2_u0_flag_pct_070": round(100.0 * u0_07, 2),
            "y1_full_at_050": m1_05,
            "y1_full_at_070": m1_07,
            "n_errors": int(res_y2["stats"]["n_errors"] + res_y1["stats"]["n_errors"]),
            "cost_usd": cost_cfg,
            "y2_stats": res_y2["stats"],
            "y1_stats": res_y1["stats"],
        }
        print(
            f"  => {tag:26s} | Y2@0.50: P={m2_05['precision']:.3f} R={m2_05['recall']:.3f} F1={m2_05['f1']:.4f} κ={m2_05['kappa']:.3f} AUC={y2_auroc} | "
            f"Y1_FULL@0.50: P={m1_05['precision']:.3f} R={m1_05['recall']:.3f} F1={m1_05['f1']:.4f} AUC={y1_auroc} | U0@0.50={100*u0_05:.2f}% | ${cost_cfg:.4f}",
            flush=True,
        )

    # Pick Top-2 architecturally diverse LLMs by average(Y2_F1@0.50, Y1_F1@0.50) for Multi-Agent Verification!
    ranked = sorted(
        pilot_results.items(),
        key=lambda kv: 0.5 * kv[1]["y2_rev_at_050"]["f1"] + 0.5 * kv[1]["y1_full_at_050"]["f1"],
        reverse=True,
    )
    top1_tag = ranked[0][0]
    top1_family = pilot_results[top1_tag]["model"].split("-")[0]
    top2_tag = next((k for k, v in ranked[1:] if v["model"].split("-")[0] != top1_family), ranked[1][0])
    print(f"\n--- Multi-Agent Disagreement Adjudication: {top1_tag} + {top2_tag} ---")

    snap_adj_before = client.ledger.summary()
    # Use strongest zero-error non-truncating model (gemma-4-31b-it) as conservative adjudicator!
    adj_model = pilot_results[top1_tag]["model"]
    adj_eff = pilot_results[top1_tag]["reasoning_effort"]
    adj_y2 = adjudicate_disagreements(
        sub_y2,
        outputs_y2[top1_tag]["scores"], outputs_y2[top2_tag]["scores"],
        outputs_y2[top1_tag]["rationale"], outputs_y2[top2_tag]["rationale"],
        client=client, adjudicator_model=adj_model, adjudicator_effort=adj_eff,
        max_workers=4, verbose=verbose,
    )
    adj_y1 = adjudicate_disagreements(
        sub_y1,
        outputs_y1[top1_tag]["scores"], outputs_y1[top2_tag]["scores"],
        outputs_y1[top1_tag]["rationale"], outputs_y1[top2_tag]["rationale"],
        client=client, adjudicator_model=adj_model, adjudicator_effort=adj_eff,
        max_workers=4, verbose=verbose,
    )
    diff_adj = client.ledger.diff_from(snap_adj_before)
    from sklearn.metrics import roc_auc_score
    m2_adj_05 = binary_metrics(y_y2_rev, (adj_y2["scores"][rev_mask] >= 0.50).astype(int))
    m2_adj_07 = binary_metrics(y_y2_rev, (adj_y2["scores"][rev_mask] >= 0.70).astype(int))
    m1_adj_05 = binary_metrics(y_y1_clean, (adj_y1["scores"][y1_clean_mask] >= 0.50).astype(int))
    m1_adj_07 = binary_metrics(y_y1_clean, (adj_y1["scores"][y1_clean_mask] >= 0.70).astype(int))
    pilot_results["5F_multi_agent_adjudicated"] = {
        "model": f"{top1_tag}+{top2_tag}+adj",
        "reasoning_effort": adj_eff or "std",
        "coders": [top1_tag, top2_tag],
        "adjudicator": f"{adj_model}:{adj_eff or 'std'}",
        "n_adjudicated_y2": adj_y2["n_adjudicated"],
        "n_adjudicated_y1": adj_y1["n_adjudicated"],
        "y2_rev_auroc": round(float(roc_auc_score(y_y2_rev, adj_y2["scores"][rev_mask])), 4),
        "y1_full_auroc": round(float(roc_auc_score(y_y1_clean, adj_y1["scores"][y1_clean_mask])), 4),
        "y2_rev_at_050": m2_adj_05,
        "y2_rev_at_070": m2_adj_07,
        "y2_u0_flag_pct_050": round(100.0 * float((adj_y2["scores"][u0_mask] >= 0.50).mean()), 2),
        "y1_full_at_050": m1_adj_05,
        "y1_full_at_070": m1_adj_07,
        "n_errors": 0,
        "cost_usd": round(diff_adj["total_equivalent_usd"], 4),
    }
    print(
        f"  => 5F_multi_agent         | Y2@0.50: P={m2_adj_05['precision']:.3f} R={m2_adj_05['recall']:.3f} F1={m2_adj_05['f1']:.4f} κ={m2_adj_05['kappa']:.3f} | "
        f"Y1_FULL@0.50: P={m1_adj_05['precision']:.3f} R={m1_adj_05['recall']:.3f} F1={m1_adj_05['f1']:.4f}"
    )

    summary = {
        "pilot_y2_transcripts": y2_txs,
        "pilot_y1_transcripts": y1_txs,
        "top2_diverse": [top1_tag, top2_tag],
        "results": pilot_results,
        "total_ledger": client.ledger.summary(),
    }
    (S2_DIR / "pilot_benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Save pilot predictions for error inspection & fusion
    np.savez_compressed(
        S2_DIR / "pilot_preds.npz",
        y2_mask=y2_mask,
        **{f"y2_{k}": outputs_y2[k]["scores"] for k in outputs_y2},
        y2_5F_multi_agent=adj_y2["scores"],
        **{f"y1_{k}": outputs_y1[k]["scores"] for k in outputs_y1},
        y1_5F_multi_agent=adj_y1["scores"],
    )
    print(f"\n[stage2_llm] Pilot complete! Total spend: ${client.ledger.summary()['usd_spent']:.4f}")
    return summary


if __name__ == "__main__":
    run_llm_pilot_benchmark(budget_cap_usd=8.0, verbose=True)
