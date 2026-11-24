"""Comprehensive offline unit tests for TeLI Science Talk v3 pipeline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.codebook_v3 import (
    MANUAL_V3_SLIDING_WINDOW_SYSTEM_PROMPT,
    apply_usf_cluster_and_veto_rules,
    clean_training_labels_for_usf,
    extract_codebook_features,
    extract_pm2_context_columns,
    infer_subtype_rule,
)
from src.v3.dataset import assign_stratified_classroom_folds, parse_transcription, to_seconds
from src.v3.export_workbooks import (
    HUMAN_CODED_COLS,
    LABEL_NOT,
    LABEL_SCIENCE,
    assign_deterministic_sheet_names,
    write_human_coded_workbook,
)
from src.v3.llm_client import CostLedger, UFNavigatorClient
from src.v3.metrics import (
    apply_bcts,
    binary_metrics,
    emq_sld,
    fit_bcts,
    gwet_ac1,
    pacc_estimate,
    tune_threshold_f1,
    tune_threshold_recall,
)
from src.v3.stage1_sequence import hmm_forward_backward_transcript
from src.v3.stage2_llm import build_dialogue_windows, parse_adjudicator_response, parse_window_hits


def test_codebook_aligns_with_sept_2026_manual():
    # Rule 6: shapes & math alone must be flagged as traps, NOT science content
    f_shape = extract_codebook_features("What shape is this? Circle.", "adult")
    assert f_shape["trap_shape_color"] > 0
    assert f_shape["c_content_total"] == 0

    # Rule 2: management with science object triggers management trap
    f_mgmt = extract_codebook_features("Put the magnet away and line up.", "adult")
    assert f_mgmt["trap_mgmt"] > 0

    # Route A + B: physical science force + causal/prediction
    f_sci = extract_codebook_features("What do you think will happen if the magnet pulls it?", "adult")
    assert f_sci["c_phys"] >= 2
    assert f_sci["p_pred"] >= 1
    assert f_sci["has_both_content_and_practice"] == 1.0

    # Spanish/bilingual science cues
    f_es = extract_codebook_features("Mira las semillas, la planta necesita agua y sol.", "adult")
    assert f_es["is_spanish_cue"] == 1.0
    assert f_es["c_es_sci"] >= 2

    # System prompt explicitly excludes shapes/geometry, casual comfort, & has no Y2 verbatim leak
    assert "STRUCTURE_SHAPE_SPACE" not in MANUAL_V3_SLIDING_WINDOW_SYSTEM_PROMPT
    assert "What shape is this?" in MANUAL_V3_SLIDING_WINDOW_SYSTEM_PROMPT
    assert "cold walking in this morning" in MANUAL_V3_SLIDING_WINDOW_SYSTEM_PROMPT
    assert "mama owl baby" not in MANUAL_V3_SLIDING_WINDOW_SYSTEM_PROMPT
    assert infer_subtype_rule("Because baby birds can't fly, their wings aren't ready yet.") == "content|causal_reasoning"

    # Manual v0.3 growth & deduplication checks (no double-counting gravity/soil/bulb across domains!)
    f_grow = extract_codebook_features("Look how the seedling grows taller in soil with gravity.", "adult")
    assert f_grow["c_life"] >= 2
    assert f_grow["c_env"] == 0  # soil belongs strictly to Earth/Space, not double-counted in Env
    assert f_grow["c_earth"] == 1  # soil only; gravity belongs strictly to Physical
    assert f_grow["c_phys"] == 1
    f_food = extract_codebook_features("Eat your chicken nuggets and look at the butterfly on my shirt.", "child")
    assert f_food["trap_pretend_food"] >= 2


def test_dataset_helpers_and_stratified_group_folds():
    assert parse_transcription("It is sunny outside. (84.5%)") == ("It is sunny outside.", pytest.approx(0.845))
    assert parse_transcription("Cloudy.") == ("Cloudy.", None)
    assert to_seconds("01:02:03.5") == pytest.approx(3723.5)
    assert to_seconds("02:10") == pytest.approx(130.0)

    # Stratified classroom folds must have zero classroom overlap
    rows = []
    for r in range(10):
        for i in range(100):
            rows.append({"classroom": f"C{r:02d}", "label": 1 if i < (r + 1) else 0})
    df = pd.DataFrame(rows)
    fmap = assign_stratified_classroom_folds(df, n_folds=5)
    assert len(fmap) == 10
    assert set(fmap.values()) == {0, 1, 2, 3, 4}
    df["fold"] = df["classroom"].map(fmap)
    for f in range(5):
        tr = set(df.loc[df["fold"] != f, "classroom"])
        va = set(df.loc[df["fold"] == f, "classroom"])
        assert tr.isdisjoint(va)


def test_llm_client_atomic_cache_and_never_caches_errors(tmp_path):
    client = UFNavigatorClient(cache_dir=tmp_path, budget_cap_usd=1.0, keys=["k1"], completion_url="http://fake")
    # Verify CostLedger tracking
    client.ledger.record_network("gpt-oss-120b", 100_000, 50_000)
    s = client.ledger.summary()
    assert s["network_calls"] == 1
    assert s["usd_spent"] == pytest.approx(0.0135, rel=1e-3)

    # Atomic cache round-trip
    p = client._cache_path("chat_test", "abc12345")
    client._write_cache_atomic(p, {"content": '{"hits": []}'})
    assert client._read_cache(p)["content"] == '{"hits": []}'

    # Classification of 429 vs 401
    assert client._classify_error(429, {"error": "rate limit"}) == "rate_limit"
    assert client._classify_error(401, {"error": "invalid api key"}) == "quota_auth"


def test_metrics_bcts_emq_pacc():
    rng = np.random.default_rng(7)
    y = np.array([1] * 200 + [0] * 1800)
    logits = np.concatenate([rng.normal(2.0, 1.0, 200), rng.normal(-2.0, 1.0, 1800)])
    probs = 1.0 / (1.0 + np.exp(-logits))

    a, b = fit_bcts(probs, y, is_probs=True)
    cal = apply_bcts(probs, a, b, is_probs=True)
    assert 0.0 <= cal.min() and cal.max() <= 1.0

    # EMQ should shift prevalence estimate upward when classroom has higher true positive rate
    room_high = np.concatenate([cal[:100], cal[200:500]])  # 25% true positives vs 10% train prior
    pi_emq, post = emq_sld(room_high, pi_train=0.10)
    assert pi_emq > 0.18
    assert post.shape == room_high.shape

    tpr_s = float(cal[y == 1].mean())
    fpr_s = float(cal[y == 0].mean())
    pi_pacc = pacc_estimate(room_high, tpr_s, fpr_s)
    assert abs(pi_pacc - 0.25) < 0.05

    # Exact marginal calibration check even at 1% rare-event base rate (zero b^2 shrinkage!)
    y_rare = np.array([1] * 30 + [0] * 2970)
    z_rare = np.concatenate([rng.normal(2.2, 1.0, 30), rng.normal(-2.2, 1.0, 2970)])
    p_rare = 1.0 / (1.0 + np.exp(-z_rare))
    ar, br = fit_bcts(p_rare, y_rare, is_probs=True)
    cal_rare = apply_bcts(p_rare, ar, br, is_probs=True)
    assert abs(float(cal_rare.mean()) - float(y_rare.mean())) < 1e-4

    thr_f1 = tune_threshold_f1(y, cal)
    thr_r90 = tune_threshold_recall(y, cal, 0.90)
    assert thr_r90 <= thr_f1 + 0.2
    m = binary_metrics(y, (cal >= thr_r90).astype(int))
    assert m["recall"] >= 0.90
    assert gwet_ac1(y, y) == pytest.approx(1.0)


def test_hmm_smoother_boosts_episodes_and_preserves_isolated_bypass():
    # Episode of high probs around index 5, isolated ultra-confident 0.96 spike at index 25
    probs = np.full(30, 0.03, dtype=np.float32)
    probs[4:9] = [0.75, 0.35, 0.85, 0.40, 0.80]
    probs[25] = 0.96
    gaps = np.full(30, 2.0, dtype=np.float64)
    gaps[15] = 120.0  # 2-minute pause separating episodes
    sm = hmm_forward_backward_transcript(probs, gaps)
    # Turn 5 inside science episode should be boosted relative to raw 0.35
    assert sm[5] > 0.35
    # Ultra-confident standalone turn 25 preserved by bypass (>= 0.94)
    assert sm[25] >= 0.95


def test_stage2_windowing_and_json_parser():
    df = pd.DataFrame({
        "transcript_id": ["tx1"] * 10,
        "line_no": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "speaker": ["adult", "child"] * 5,
        "utterance": [f"turn {i}" for i in range(10)],
        "gap_prev_s": [0.0, 2.0, 2.0, 2.0, 2.0, 65.0, 2.0, 2.0, 2.0, 2.0],
    })
    # Silence gap > 30s at index 5 MUST split into 2 separate windows (no cross-episode bleed!)
    wins = build_dialogue_windows(df, target_mask=None, window_size=32, pause_cut_s=30.0)
    assert len(wins) == 2
    assert wins[0]["target_line_nos"] == [10, 11, 12, 13, 14]
    assert wins[1]["target_line_nos"] == [15, 16, 17, 18, 19]

    # Parser ignores hallucinated / CTX line_nos not in valid_line_nos
    raw = '```json\n{"hits": [{"id": 11, "score": 0.9, "conf": 5, "route": "content", "subtype": "content", "rationale": "weather"}, {"id": 999, "score": 0.95}]}\n```'
    hits = parse_window_hits(raw, {10, 11, 12})
    assert len(hits) == 1
    assert hits[0]["id"] == 11 and hits[0]["score"] == pytest.approx(0.9)

    # Truncation-resilient parser salvages complete hit objects before max_tokens cutoff!
    trunc = '{"hits": [{"id": 10, "score": 0.91, "conf": 5, "route": "content", "subtype": "content", "rationale": "magnet"}, {"id": 11, "score": 0.85, "conf":'
    hits_trunc = parse_window_hits(trunc, {10, 11, 12})
    assert len(hits_trunc) == 1 and hits_trunc[0]["id"] == 10

    adj = parse_adjudicator_response('{"score": 0.15, "conf": 5, "route": "none", "subtype": "none", "rationale": "bare look"}')
    assert adj["score"] == pytest.approx(0.15)


def test_export_workbooks_exact_schema_and_fixes_child_base_collision(tmp_path):
    # Two suffix-less adult/base files: file A has 0 science hits, file B has 1 science hit.
    # Deterministic pre-assignment MUST give file B `child_base_1`, NOT steal `child_base`!
    f_a = "98-46-143_111825_sync_spkr2_focal_au.xlsx"
    f_b = "98-46-294_111825_sync_spkr2_focal_au.xlsx"
    f_c = "98-46-143-141_111825_sync_spkr2_focal_au.xlsx"
    smap = assign_deterministic_sheet_names([f_a, f_b, f_c])
    assert smap[f_a] == "child_base"
    assert smap[f_b] == "child_base_1"
    assert smap[f_c] == "child_141"

    records = [
        {
            "source_file": f_a, "child_id": "base", "line_no": 5, "seg_start": "0:00:05",
            "seg_end": "0:00:07", "speaker": "adult", "utterance": "Sit down.",
            "confidence": 0.9, "predicted_label": LABEL_NOT, "score": 0.02,
            "scored_by": "local_v3", "cosine_score": 0.02, "llm_score": None,
            "top_match_in_corpus": None, "top_match_score": None,
            "predicted_subtype": "", "rationale": "", "degraded": False,
        },
        {
            "source_file": f_b, "child_id": "base", "line_no": 12, "seg_start": "0:01:12",
            "seg_end": "0:01:15", "speaker": "adult", "utterance": "Push and pull are forces.",
            "confidence": 0.95, "predicted_label": LABEL_SCIENCE, "score": 0.94,
            "scored_by": "hybrid_v3", "cosine_score": 0.88, "llm_score": 0.95,
            "top_match_in_corpus": None, "top_match_score": None,
            "predicted_subtype": "content", "rationale": "physical science force concept", "degraded": False,
        },
    ]
    out = write_human_coded_workbook(
        "Classroom46_111825", records, tmp_path, mode="screening", threshold=0.35,
        all_source_files=[f_a, f_b, f_c],
    )
    assert out is not None and out.exists()
    xl = pd.ExcelFile(out, engine="openpyxl")
    assert "child_base_1" in xl.sheet_names
    assert "child_base" not in xl.sheet_names  # f_a had 0 hits, and f_b did NOT steal its name!
    assert "INSTRUCTIONS" in xl.sheet_names
    df_out = xl.parse("child_base_1")
    assert list(df_out.columns) == HUMAN_CODED_COLS
    assert int(df_out["line_no"].iloc[0]) == 12


def test_pubn_cvir_zero_leakage_and_borderline_softening():
    from src.v3.stage1_lexical_and_embeds import build_pubn_cvir_training_set

    n = 1000
    labels = np.full(n, -1, dtype=int)
    labels[:40] = 1
    labels[40:140] = 0
    screened = np.ones(n, dtype=bool)
    train_mask = np.ones(n, dtype=bool)
    train_mask[900:] = False  # held-out validation fold
    scores = np.linspace(0.0, 1.0, n, dtype=np.float32)

    idx, y, w = build_pubn_cvir_training_set(
        labels, screened, train_mask,
        u0_scores=scores, bn_oof_probs=scores,
        u0_ratio=2.0, cvir_drop_top_frac=0.05, borderline_top_frac=0.10, seed=7,
    )
    # Strictly zero held-out validation indices!
    assert (idx < 900).all()
    # Top 5% highest-scoring U0 turns must be excluded by CVIR
    u0_train = np.flatnonzero(train_mask & (labels == -1))
    top_u0_cut = np.quantile(scores[u0_train], 0.95)
    picked_u0 = idx[labels[idx] == -1]
    assert (scores[picked_u0] < top_u0_cut).all()
    # Top 10% bN turns must receive soft target 0.25 & reduced weight 0.35
    assert np.isclose(y, 0.25).sum() >= 8
    assert np.isclose(w[np.isclose(y, 0.25)], 0.35).all()


def test_load_raw_transcript_folder_and_pause_gaps(tmp_path):
    from src.v3.predict_new import load_raw_transcript_folder

    folder = tmp_path / "Classroom99_010126"
    folder.mkdir()
    # Match EXACT real USF raw focal `.xlsx` column headers (including trailing Proximate Person columns & '-' row!)
    df_raw = pd.DataFrame({
        "Segment Start Time": ["00:00:01.000", "00:00:04.000", "00:00:07.000", "00:01:30.000", "00:01:34.000"],
        "Segment End Time": ["00:00:03.000", "00:00:06.000", "00:00:08.000", "00:01:33.000", "00:01:36.000"],
        "Transcription (Confidence)": [
            "Look, the magnet pulls the paperclip! (92.0%)",
            "Yeah!",
            "-",
            "Why did it stick?",
            "Unintelligible mumble. (28.0%)",
        ],
        "Speaker": ["adult", "unknown", "adult", "child", "child"],
        "Proximate Person 1": [None, None, None, None, None],
        "Proximate Person 5": [None, None, None, None, None],
    })
    df_raw.to_excel(folder / "98-99-101-202_010126_sync_spkr2_focal_au.xlsx", index=False)

    loaded = load_raw_transcript_folder(folder)
    # '-' placeholder must be skipped, leaving 4 turns (keeping unknown + low-conf for sequence context!)
    assert len(loaded) == 4
    assert list(loaded["speaker"]) == ["adult", "unknown", "child", "child"]
    assert list(loaded["screened"]) == [True, False, True, False]
    assert loaded["asr_conf"].iloc[0] == pytest.approx(0.92)
    assert loaded["gap_prev_s"].iloc[1] == pytest.approx(1.0)
    assert loaded["gap_prev_s"].iloc[2] == pytest.approx(84.0)  # >30s silence gap!
    assert loaded["cb_c_phys"].iloc[0] >= 2.0


def test_export_zero_hit_status_workbook(tmp_path):
    rec = [{
        "source_file": "98-99-101-202_010126_sync_spkr2_focal_au.xlsx",
        "child_id": "202", "line_no": 2, "seg_start": "0:00:01", "seg_end": "0:00:02",
        "speaker": "adult", "utterance": "Sit down.", "confidence": 0.9,
        "predicted_label": LABEL_NOT, "score": 0.01, "scored_by": "hybrid_v3",
        "cosine_score": 0.01, "llm_score": None, "top_match_in_corpus": None,
        "top_match_score": None, "predicted_subtype": "", "rationale": None, "degraded": False,
    }]
    out = write_human_coded_workbook("Classroom99_010126", rec, tmp_path, mode="standalone", threshold=0.5)
    assert out is not None and out.exists()
    xl = pd.ExcelFile(out, engine="openpyxl")
    assert xl.sheet_names == ["INSTRUCTIONS"]


def test_blinded_ht_audit_contains_no_model_scores_or_strata():
    p = PROJECT_ROOT / "reports" / "v3" / "unflagged_recall_audit_BLIND_coder_sheet.xlsx"
    if not p.exists():
        pytest.skip("Blinded HT audit sheet not yet exported")
    try:
        df = pd.read_excel(p, sheet_name="BLIND_AUDIT", engine="openpyxl")
    except PermissionError:
        pytest.skip("Blinded HT audit sheet is currently open/locked in Excel")
    assert len(df) == 600
    forbidden = {"stratum", "stratum_N", "stratum_n", "ht_weight", "v3_hybrid_score", "v3_stage1_score", "v3_llm_rationale"}
    assert forbidden.isdisjoint(set(df.columns))
    assert {"audit_row_id", "context_before_6", "target_utterance", "context_after_6", "review_SCI_or_blank", "confidence_1_to_5"}.issubset(set(df.columns))


def test_usf_exclusions_traps_and_content_protection():
    # Pure shapes, pure counting, and look-without-content MUST fire strict USF traps
    f_sh = extract_codebook_features("Aaron made a rectangle.", "adult")
    assert f_sh["trap_shape_only"] == 1.0 and f_sh["has_content"] == 0.0

    f_ct = extract_codebook_features("One, two, three, four.", "child")
    assert f_ct["trap_count_only"] == 1.0 and f_ct["has_content"] == 0.0

    f_lk = extract_codebook_features("Look closely! Does yours look like mine?", "adult")
    assert f_lk["trap_look_no_content"] == 1.0
    assert f_lk["p_obs"] == 0.0  # observation cue MUST be gated off when has_content == 0!

    # Genuine science content words MUST NOT be falsely trapped!
    f_pc = extract_codebook_features("Look at this pine cone.", "child")
    assert f_pc["has_content"] == 1.0 and f_pc["trap_shape_only"] == 0.0 and f_pc["trap_look_no_content"] == 0.0

    f_mn = extract_codebook_features("So how many moons do we have?", "adult")
    assert f_mn["has_content"] == 1.0 and f_mn["trap_count_only"] == 0.0

    f_an = extract_codebook_features("See, they still have a giraffe and grasshoppers.", "adult")
    assert f_an["has_content"] == 1.0 and f_an["trap_look_no_content"] == 0.0


def test_usf_pm2_cluster_veto_and_context_extraction():
    rows = [
        ("tx1", "adult", "What is the weather today?", 0.0, 0.90, 1),
        ("tx1", "child", "Cloudy.", 1.5, 0.85, 1),
        ("tx1", "adult", "Yes, dark rain clouds cover the sun.", 1.0, 0.92, 1),
        ("tx1", "unknown", "Whoa!", 1.2, 0.30, -1),
        ("tx1", "adult", "What shape is this? Circle.", 45.0, 0.95, 0),  # >30s pause + shape trap!
        ("tx1", "adult", "Gravity pulls everything down.", 45.0, 0.95, 1),  # isolated standalone!
    ]
    df = pd.DataFrame(rows, columns=["transcript_id", "speaker", "utterance", "gap_prev_s", "asr_conf", "label"])
    feats = pd.DataFrame([extract_codebook_features(u, s) for u, s in zip(df["utterance"], df["speaker"])])
    for c in feats.columns:
        df[f"cb_{c}"] = feats[c]

    # 1. Context ±2 extraction respects >30s pauses & preserves unknown/low-conf chronological lines
    cb2, ca2 = extract_pm2_context_columns(df, pause_cut_s=30.0)
    assert "What is the weather today?" in cb2[1]
    assert "dark rain clouds" in ca2[1]
    assert "unknown (low-conf): Whoa!" in ca2[2]
    assert "[>30s pause]" in cb2[4]
    assert "[>30s pause]" in ca2[4]

    # 2. ±2 cluster rule keeps >=3 cluster unchanged, penalizes isolated by -0.60 logit,
    #    and Stage-2 LLM veto crushes LLM-rejected shape trap!
    cal_p = np.array([0.40, 0.35, 0.50, 0.01, 0.08, 0.40], dtype=np.float64)
    llm_s = np.array([0.90, 0.85, 0.92, 0.02, 0.05, 0.90], dtype=np.float64)
    llm_e = np.array([True, True, True, False, True, True], dtype=bool)
    adj, pm2_cnt = apply_usf_cluster_and_veto_rules(df, cal_p, llm_scores=llm_s, llm_eval=llm_e)
    assert int(pm2_cnt[1]) == 3
    assert adj[1] == pytest.approx(0.35, rel=1e-4)  # inside >=3 cluster -> untouched!
    assert int(pm2_cnt[5]) == 1
    assert adj[5] < 0.30 and adj[5] > 0.20  # isolated -0.60 logit penalty: sigmoid(logit(0.40)-0.60) ≈ 0.268!
    assert adj[4] < 0.005  # LLM-rejected shape trap vetoed (-0.60 - 3.5 = -4.1 logit)!

    # 3. clean_training_labels_for_usf masks dirty shape TP (-2) without mutating df["label"]
    df.loc[4, "label"] = 1
    cleaned = clean_training_labels_for_usf(df)
    assert cleaned[4] == -2
    assert df["label"].iloc[4] == 1  # original evaluation label strictly untouched!


def test_predict_new_classroom_id_and_warm_embed_cache():
    from src.v3.predict_new import _embed_unique_nomic, _extract_classroom_id

    assert _extract_classroom_id("Classroom10_020526") == "10"
    assert _extract_classroom_id("Classroom49_012026_updated") == "49"
    assert _extract_classroom_id("UnseenNewRoom") is None

    # Verify _embed_unique_nomic reuses warm cache and ONLY calls embed_batch for cache misses!
    class _StubClient:
        def __init__(self):
            self.calls = []

        def embed_batch(self, *, model, texts):
            self.calls.append((model, list(texts)))
            return [[1.0] + [0.0] * 767 for _ in texts]

    v_cached = np.zeros(768, dtype=np.float16)
    v_cached[1] = 1.0
    cache = {"[adult] Is it raining?": v_cached}
    df = pd.DataFrame({
        "speaker": ["adult", "adult", "child"],
        "utterance": ["Is it raining?", "Is it raining?", "Cloudy."],
    })
    stub = _StubClient()
    mat, uids = _embed_unique_nomic(df, stub, embed_cache=cache)
    assert len(stub.calls) == 1
    assert stub.calls[0][1] == ["classification: [child] Cloudy."]
    assert len(mat) == 2 and list(uids) == [0, 0, 1]
    assert mat[0, 1] == pytest.approx(1.0, abs=1e-3)
    assert "[child] Cloudy." in cache
