"""Run error analyses 1–3 and write artifacts under error_analysis/output/."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from codebook import SCIENCE_CONTENT_LEXICON, assign_fp_label, gate_flags
from hand_codes import HAND_CODES, apply_hand_code
from load_reviews import load_all_flagged, reviewed_only

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SEED = 20260831
SAMPLE_N = 400

HIGH_ROOMS = {"83", "92", "67"}
LOW_ROOMS = {"63", "14", "46", "21"}


def _pct(num: float, den: float) -> float | None:
    if not den:
        return None
    return round(100.0 * num / den, 1)


def _band(classroom: str) -> str:
    if classroom in HIGH_ROOMS:
        return "high"
    if classroom in LOW_ROOMS:
        return "low"
    return "mid"


def stratified_fp_sample(fps: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """100 per speaker × scored_by cell, spread across classrooms."""
    rng = np.random.default_rng(seed)
    cells = [
        ("adult", "llm"),
        ("adult", "cosine"),
        ("child", "llm"),
        ("child", "cosine"),
    ]
    per_cell = n // len(cells)
    parts: list[pd.DataFrame] = []
    leftover = n
    for speaker, scored in cells:
        cell = fps[
            (fps["speaker_norm"] == speaker) & (fps["scored_by_norm"] == scored)
        ]
        take = min(per_cell, len(cell), leftover)
        if take == 0:
            continue
        # sample across classrooms: take at least 1 from each room if possible
        picked_idx: list[int] = []
        rooms = list(cell["classroom"].unique())
        rng.shuffle(rooms)
        remaining = take
        # first pass: 1 per room
        for room in rooms:
            if remaining <= 0:
                break
            pool = cell[cell["classroom"] == room]
            if pool.empty:
                continue
            choice = rng.choice(pool.index.to_numpy(), size=1, replace=False)
            picked_idx.extend(choice.tolist())
            remaining -= 1
        # fill rest proportional / random from unused
        unused = cell.index.difference(picked_idx)
        if remaining > 0 and len(unused) > 0:
            extra = rng.choice(
                unused.to_numpy(),
                size=min(remaining, len(unused)),
                replace=False,
            )
            picked_idx.extend(extra.tolist())
        parts.append(cell.loc[picked_idx])
        leftover -= len(picked_idx)
    sample = pd.concat(parts).drop_duplicates()
    # top up if a cell was short
    if len(sample) < n:
        unused = fps.loc[~fps.index.isin(sample.index)]
        need = min(n - len(sample), len(unused))
        if need:
            extra_idx = rng.choice(unused.index.to_numpy(), size=need, replace=False)
            sample = pd.concat([sample, unused.loc[extra_idx]])
    return sample.head(n).copy()


def confusion_counts(sub: pd.DataFrame) -> dict:
    tp = int((sub["review_norm"] == "verified").sum())
    fp = int((sub["review_norm"] == "false_positive").sum())
    n = tp + fp
    return {"n": n, "tp": tp, "fp": fp, "precision_pct": _pct(tp, n)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_all_flagged()
    rev = reviewed_only(raw)
    rev["is_tp"] = rev["review_norm"].eq("verified")
    rev["room_band"] = rev["classroom"].map(_band)
    fps = rev[rev["review_norm"] == "false_positive"].copy()
    tps = rev[rev["review_norm"] == "verified"].copy()

    # ------------------------------------------------------------------
    # Analysis 1 — stratified 400 FP taxonomy
    # ------------------------------------------------------------------
    sample = stratified_fp_sample(fps, SAMPLE_N, SEED)
    sample["fp_label_rule"] = sample["utt"].map(assign_fp_label)
    sample["fp_label"] = [
        apply_hand_code(u, lab) for u, lab in zip(sample["utt"], sample["fp_label_rule"])
    ]
    sample["label_source"] = [
        "hand" if u.strip() in HAND_CODES else "rule" for u in sample["utt"]
    ]
    sample["room_band"] = sample["classroom"].map(_band)
    sample["gates"] = sample["utt"].map(gate_flags)
    for key in ["look_exact", "count_exact", "shape_color_exact",
                "conservative_exact", "short_1_2",
                "any_proposed_gate", "science_holdout"]:
        sample[key] = sample["gates"].map(lambda d, k=key: d[k])
    sample = sample.drop(columns=["gates"])

    keep_cols = [
        "workbook", "sheet", "classroom", "obs_date", "room_band",
        "speaker_norm", "scored_by_norm", "score_num", "cosine_num",
        "llm_num", "n_words", "subtype_norm", "utt", "fp_label", "fp_label_rule",
        "label_source",
        "look_exact", "count_exact", "shape_color_exact",
        "conservative_exact", "short_1_2",
        "any_proposed_gate", "science_holdout",
    ]
    coded = sample[keep_cols].sort_values(
        ["fp_label", "speaker_norm", "scored_by_norm", "classroom"]
    )
    coded.to_csv(OUT / "fp_sample_400_coded.csv", index=False)

    label_counts = coded["fp_label"].value_counts().to_dict()
    label_by_speaker = (
        pd.crosstab(coded["fp_label"], coded["speaker_norm"], margins=True)
        .to_dict()
    )
    label_by_path = (
        pd.crosstab(coded["fp_label"], coded["scored_by_norm"], margins=True)
        .to_dict()
    )
    label_by_band = (
        pd.crosstab(coded["fp_label"], coded["room_band"], margins=True)
        .to_dict()
    )

    # examples per label (up to 6 unique)
    examples: dict[str, list[str]] = {}
    for lab, g in coded.groupby("fp_label"):
        utts = list(dict.fromkeys(g["utt"].tolist()))[:6]
        examples[lab] = utts

    # apply same coder to ALL FPs (for population estimate vs sample)
    fps = fps.copy()
    fps["fp_label"] = fps["utt"].map(assign_fp_label)
    pop_label_counts = fps["fp_label"].value_counts().to_dict()

    # ------------------------------------------------------------------
    # Analysis 2 — cosine fallback
    # ------------------------------------------------------------------
    cosine = rev[rev["scored_by_norm"] == "cosine"]
    llm = rev[rev["scored_by_norm"] == "llm"]
    cosine_only = confusion_counts(cosine)
    llm_only = confusion_counts(llm)
    overall = confusion_counts(rev)
    drop_cosine = confusion_counts(llm)  # remaining if cosine flags removed

    # cosine score bins among cosine-scored reviewed rows
    cosine = cosine.copy()
    bins = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
    cosine["cbin"] = pd.cut(cosine["cosine_num"], bins=bins, right=False)
    bin_rows = []
    for interval, g in cosine.groupby("cbin", observed=False):
        bin_rows.append({
            "bin": str(interval),
            **confusion_counts(g),
        })

    # raise threshold: keep cosine flags only if cosine >= t, plus all LLM
    thresholds = [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.55]
    thresh_rows = []
    for t in thresholds:
        keep = rev[
            (rev["scored_by_norm"] == "llm")
            | ((rev["scored_by_norm"] == "cosine") & (rev["cosine_num"] >= t))
        ]
        row = confusion_counts(keep)
        lost_tp = int(((rev["scored_by_norm"] == "cosine")
                       & (rev["cosine_num"] < t)
                       & rev["is_tp"]).sum())
        dropped = len(rev) - row["n"]
        row.update({
            "cosine_min": t,
            "dropped_flags": dropped,
            "lost_tp_among_flagged": lost_tp,
        })
        thresh_rows.append(row)

    cosine_tps = tps[tps["scored_by_norm"] == "cosine"]
    cosine_tp_examples = list(dict.fromkeys(cosine_tps["utt"].tolist()))[:15]
    cosine_tp_len = {
        "n": len(cosine_tps),
        "mean_words": round(float(cosine_tps["n_words"].mean()), 2) if len(cosine_tps) else None,
        "median_words": float(cosine_tps["n_words"].median()) if len(cosine_tps) else None,
        "n_short_1_2": int((cosine_tps["n_words"] <= 2).sum()),
        "n_science_holdout": int(cosine_tps["utt"].map(
            lambda u: gate_flags(u)["science_holdout"]
        ).sum()),
    }

    # ------------------------------------------------------------------
    # Analysis 3 — pre-Track-B gates + science-name hold-out
    # ------------------------------------------------------------------
    gate_df = rev.copy()
    unpacked = gate_df["utt"].map(gate_flags).apply(pd.Series)
    gate_df = pd.concat([gate_df, unpacked], axis=1)

    gate_names = [
        "look_exact",
        "count_exact",
        "shape_color_exact",
        "conservative_exact",
        "short_1_2",
        "any_proposed_gate",
    ]
    gate_rows = []
    for name in gate_names:
        hit = gate_df[gate_df[name]]
        remain = gate_df[~gate_df[name]]
        hit_c = confusion_counts(hit)
        rem_c = confusion_counts(remain)
        holdout_hit = int(gate_df.loc[gate_df[name], "science_holdout"].sum()) if False else 0
        # TPs that would be gated (lost among flagged)
        lost_tp = int((hit["is_tp"]).sum())
        holdout_saved = int(gate_df.loc[gate_df["science_holdout"] & gate_df["is_tp"]].shape[0]) if name == "any_proposed_gate" else None
        gate_rows.append({
            "gate": name,
            "flagged_removed": hit_c["n"],
            "removed_tp": hit_c["tp"],
            "removed_fp": hit_c["fp"],
            "precision_of_removed_pct": hit_c["precision_pct"],
            "remaining_n": rem_c["n"],
            "remaining_tp": rem_c["tp"],
            "remaining_fp": rem_c["fp"],
            "remaining_precision_pct": rem_c["precision_pct"],
            "lost_tp_among_flagged": lost_tp,
        })

    # hold-out inventory: short TPs that are science names (would have been gated without hold-out)
    short_tps = tps[tps["n_words"] <= 2]
    short_tp_holdout = short_tps[short_tps["utt"].map(
        lambda u: gate_flags(u)["science_holdout"]
    )]
    short_tp_not = short_tps[~short_tps["utt"].map(
        lambda u: gate_flags(u)["science_holdout"]
    )]
    holdout_examples = list(dict.fromkeys(short_tp_holdout["utt"].tolist()))[:20]
    short_tp_lost_examples = list(dict.fromkeys(short_tp_not["utt"].tolist()))[:20]

    # naive short gate WITHOUT hold-out (to show the cost)
    naive_short = rev[rev["n_words"] <= 2]
    naive_remain = rev[rev["n_words"] > 2]
    naive_short_c = confusion_counts(naive_short)
    naive_remain_c = confusion_counts(naive_remain)

    # combined: drop cosine OR any proposed gate
    combo_drop = gate_df[
        (gate_df["scored_by_norm"] == "cosine") | gate_df["any_proposed_gate"]
    ]
    combo_keep = gate_df[
        (gate_df["scored_by_norm"] != "cosine") & ~gate_df["any_proposed_gate"]
    ]
    combo = {
        "dropped": confusion_counts(combo_drop),
        "remaining": confusion_counts(combo_keep),
    }
    cons_drop = gate_df[
        (gate_df["scored_by_norm"] == "cosine") | gate_df["conservative_exact"]
    ]
    cons_keep = gate_df[
        (gate_df["scored_by_norm"] != "cosine") & ~gate_df["conservative_exact"]
    ]
    combo_conservative = {
        "dropped": confusion_counts(cons_drop),
        "remaining": confusion_counts(cons_keep),
    }

    # one-word TP inventory for the report
    one_word_tps = tps[tps["n_words"] == 1]
    one_word_tp_examples = list(dict.fromkeys(one_word_tps["utt"].tolist()))[:25]

    summary = {
        "n_flagged": int(len(raw)),
        "n_reviewed": int(len(rev)),
        "n_tp": int(rev["is_tp"].sum()),
        "n_fp": int((~rev["is_tp"]).sum()),
        "overall_precision_pct": overall["precision_pct"],
        "sample": {
            "n": int(len(coded)),
            "seed": SEED,
            "design": "100 per speaker × scored_by cell, classrooms spread",
            "speaker": coded["speaker_norm"].value_counts().to_dict(),
            "scored_by": coded["scored_by_norm"].value_counts().to_dict(),
            "room_band": coded["room_band"].value_counts().to_dict(),
            "classroom": coded["classroom"].value_counts().to_dict(),
            "label_counts": label_counts,
            "label_by_speaker": {
                k: {kk: int(vv) for kk, vv in v.items()}
                if isinstance(v, dict) else v
                for k, v in pd.crosstab(coded["fp_label"], coded["speaker_norm"]).to_dict().items()
            },
            "label_by_path": {
                k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in pd.crosstab(coded["fp_label"], coded["scored_by_norm"]).to_dict().items()
            },
            "label_by_band": {
                k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in pd.crosstab(coded["fp_label"], coded["room_band"]).to_dict().items()
            },
            "examples": examples,
            "n_hand_coded": int((coded["label_source"] == "hand").sum()) if "label_source" in coded.columns else 0,
        },
        "population_fp_labels": pop_label_counts,
        "n_population_fp": int(len(fps)),
        "cosine": {
            "cosine_only": cosine_only,
            "llm_only": llm_only,
            "drop_all_cosine": drop_cosine,
            "score_bins": bin_rows,
            "threshold_sweep": thresh_rows,
            "cosine_tp_profile": cosine_tp_len,
            "cosine_tp_examples": cosine_tp_examples,
        },
        "gates": {
            "per_gate": gate_rows,
            "naive_short_no_holdout": {
                "removed": naive_short_c,
                "remaining": naive_remain_c,
            },
            "holdout_lexicon_size": len(SCIENCE_CONTENT_LEXICON),
            "short_tp_n": int(len(short_tps)),
            "short_tp_holdout_n": int(len(short_tp_holdout)),
            "short_tp_still_gated_n": int(len(short_tp_not)),
            "holdout_examples": holdout_examples,
            "short_tp_lost_examples": short_tp_lost_examples,
            "one_word_tp_n": int(len(one_word_tps)),
            "one_word_tp_examples": one_word_tp_examples,
            "cosine_or_gate": combo,
            "cosine_or_conservative": combo_conservative,
        },
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # also a compact label table
    pd.DataFrame(
        [{"label": k, "sample_n": label_counts.get(k, 0),
          "population_fp_n": pop_label_counts.get(k, 0)}
         for k in sorted(set(label_counts) | set(pop_label_counts),
                         key=lambda x: -label_counts.get(x, 0))]
    ).to_csv(OUT / "fp_label_counts.csv", index=False)

    pd.DataFrame(thresh_rows).to_csv(OUT / "cosine_threshold_sweep.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT / "gate_sweep.csv", index=False)

    print(f"reviewed={len(rev)} tp={rev['is_tp'].sum()} fp={(~rev['is_tp']).sum()}")
    print(f"sample={len(coded)}")
    print("sample labels:")
    print(coded["fp_label"].value_counts().to_string())
    print("population FP labels:")
    print(fps["fp_label"].value_counts().to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
