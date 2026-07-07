"""Unit tests for src/positives_mining.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.positives_mining import (
    MINED_ID_PREFIX,
    MINED_SOURCE_TAG,
    apply_not_science_review,
    build_corpus_rows,
    dedup_within,
    export_not_science_review,
    extract_sci_positives,
    filter_fragments,
    is_fragment,
    merge_into_corpus,
    net_new_vs_corpus,
    sheet_to_setting,
    validate_ns_decisions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_transcript_xlsx(path):
    """7-column sheet: col C(2)=speaker, D(3)=utterance, G(6)=content code."""
    def sheet(rows):
        # rows: list of (speaker, utterance, content_code)
        data = []
        for spk, utt, code in rows:
            data.append(["", "", spk, utt, "", "", code])
        return pd.DataFrame(data, columns=[f"col{i}" for i in range(7)])

    with pd.ExcelWriter(path) as xw:
        sheet([
            ("T1:", "So the seeds will grow into tall green plants over time.", "SCI"),
            ("T1:", "Plants right?", "SCI"),                       # fragment -> dropped
            ("T2:", "Line up quietly for lunch now everyone please.", ""),   # non-SCI
            ("C:", "I think the bug is moving toward the bright light.", "SCI"),  # child, dropped
            ("T1:", "What do you predict will happen when we add more water?", "SCI"),
        ]).to_excel(xw, sheet_name="01-2_19LG", index=False)
        sheet([
            ("T3:", "The ramp makes the toy car roll faster down the slope.", "SCI"),
            ("T3:", "Clean up your centers before we move on okay.", "MATH"),
        ]).to_excel(xw, sheet_name="8-15C", index=False)
        # Meta sheet should be skipped
        sheet([("T1:", "ignored science content here please", "SCI")]).to_excel(
            xw, sheet_name="TOC", index=False)


@pytest.fixture
def transcript_path(tmp_path):
    p = tmp_path / "Coding Transcripts.xlsx"
    _write_transcript_xlsx(p)
    return p


@pytest.fixture
def fake_corpus():
    return pd.DataFrame([
        {
            "utt_id": "utt_0000",
            "utterance": "Existing curated science utterance about magnets.",
            "label": "SCIENCE_TALK", "setting": "Large Group", "source": "curated",
            "tier2_cues": [], "tier3_cues": [], "was_sci_coded": 1,
            "transcript_ref": None, "topic": None, "citation": None,
            "subtype": ["observation"], "subtype_source": "rule",
            "subtype_confidence": 1.0, "subtype_prompt_version": None,
        },
        {
            # duplicate of a mined row -> should be excluded as net-new
            "utt_id": "utt_0001",
            "utterance": "The ramp makes the toy car roll faster down the slope.",
            "label": "SCIENCE_TALK", "setting": "Centers", "source": "curated",
            "tier2_cues": [], "tier3_cues": [], "was_sci_coded": 1,
            "transcript_ref": None, "topic": None, "citation": None,
            "subtype": ["observation"], "subtype_source": "rule",
            "subtype_confidence": 1.0, "subtype_prompt_version": None,
        },
    ])


@pytest.fixture
def fake_seed():
    return pd.DataFrame([
        {"term": "plant", "variants": ["plants"], "category": "life_science", "tier": "TIER3"},
        {"term": "predict", "variants": [], "category": "prediction", "tier": "TIER2"},
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestSheetToSetting:
    def test_suffix_mapping(self):
        assert sheet_to_setting("01-2_19LG") == "Large Group"
        assert sheet_to_setting("04-04SG") == "Small Group"
        assert sheet_to_setting("8-15C") == "Centers"
        assert sheet_to_setting("01-03C2") == "Centers"

    def test_unknown_suffix(self):
        assert sheet_to_setting("weird-sheet-99") == "Unknown"


class TestFragmentFilter:
    def test_short_is_fragment(self):
        assert is_fragment("Plants right?")
        assert is_fragment("Plants, trees,")

    def test_full_sentence_kept(self):
        assert not is_fragment("So the seeds will grow into tall green plants over time.")

    def test_filter_drops_fragments(self):
        df = pd.DataFrame({"utterance": [
            "Plants right?",
            "So the seeds will grow into tall green plants over time.",
        ]})
        out = filter_fragments(df, verbose=False)
        assert len(out) == 1


class TestDedupAndNetNew:
    def test_dedup_within_collapses_case_ws_dupes(self):
        df = pd.DataFrame({"utterance": [
            "The ramp makes cars roll.",
            "the  ramp   makes cars roll.",  # same after normalize+casefold
            "A different sentence entirely here.",
        ]})
        out = dedup_within(df, verbose=False)
        assert len(out) == 2

    def test_net_new_excludes_corpus_rows(self, fake_corpus):
        df = pd.DataFrame({"utterance": [
            "The ramp makes the toy car roll faster down the slope.",  # in corpus
            "A brand new science observation about clouds.",
        ]})
        out = net_new_vs_corpus(df, fake_corpus, verbose=False)
        assert len(out) == 1
        assert "clouds" in out.iloc[0]["utterance"]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_keeps_only_teacher_sci(self, transcript_path):
        df = extract_sci_positives(transcript_path, verbose=False)
        texts = set(df["utterance"])
        # teacher + SCI rows present
        assert any("seeds will grow" in t for t in texts)
        assert any("predict will happen" in t for t in texts)
        assert any("ramp makes the toy car" in t for t in texts)
        # child SCI excluded
        assert not any("bug is moving" in t for t in texts)
        # non-SCI / MATH excluded
        assert not any("Line up quietly" in t for t in texts)
        assert not any("Clean up your centers" in t for t in texts)

    def test_meta_sheet_skipped(self, transcript_path):
        df = extract_sci_positives(transcript_path, verbose=False)
        assert not any("ignored science content" in t for t in df["utterance"])

    def test_setting_from_sheet(self, transcript_path):
        df = extract_sci_positives(transcript_path, verbose=False)
        lg = df[df["sheet"] == "01-2_19LG"]
        assert (lg["setting"] == "Large Group").all()
        c = df[df["sheet"] == "8-15C"]
        assert (c["setting"] == "Centers").all()


# ---------------------------------------------------------------------------
# build_corpus_rows
# ---------------------------------------------------------------------------

class TestBuildCorpusRows:
    def test_id_scheme_and_provenance(self, fake_corpus):
        df = pd.DataFrame([
            {"utterance": "A new science sentence about shadows.", "setting": "Centers",
             "sheet": "8-15C", "excel_row": 5},
        ])
        rows = build_corpus_rows(df, fake_corpus)
        r = rows.iloc[0]
        assert r["utt_id"].startswith(MINED_ID_PREFIX)
        assert r["source"] == MINED_SOURCE_TAG
        assert r["label"] == "SCIENCE_TALK"
        assert r["was_sci_coded"] == 1
        assert r["transcript_ref"] == "8-15C!R5"
        assert r["tier2_cues"] == [] and r["tier3_cues"] == []

    def test_idempotent_index_continues(self, fake_corpus):
        # Simulate a prior mining run already present in corpus
        prior = fake_corpus.copy()
        prior.loc[len(prior)] = {**fake_corpus.iloc[0].to_dict(),
                                 "utt_id": f"{MINED_ID_PREFIX}0007"}
        df = pd.DataFrame([{"utterance": "Another new science sentence here.",
                            "setting": "Centers", "sheet": "8-15C", "excel_row": 9}])
        rows = build_corpus_rows(df, prior)
        assert rows.iloc[0]["utt_id"] == f"{MINED_ID_PREFIX}0008"


# ---------------------------------------------------------------------------
# merge_into_corpus (end-to-end)
# ---------------------------------------------------------------------------

class TestMerge:
    def _setup(self, tmp_path, fake_corpus, fake_seed, transcript_path):
        proc = tmp_path / "processed"
        proc.mkdir()
        fake_corpus.to_parquet(proc / "corpus.parquet", index=False)
        fake_seed.to_parquet(proc / "seed_words.parquet", index=False)
        return proc

    def test_dry_run_writes_nothing(self, tmp_path, fake_corpus, fake_seed, transcript_path):
        proc = self._setup(tmp_path, fake_corpus, fake_seed, transcript_path)
        before = pd.read_parquet(proc / "corpus.parquet")
        stats = merge_into_corpus(proc, xlsx_path=transcript_path, dry_run=True, verbose=False)
        after = pd.read_parquet(proc / "corpus.parquet")
        assert stats["dry_run"] is True
        assert len(before) == len(after)
        assert stats["net_new_positives"] >= 1

    def test_merge_appends_and_backs_up(self, tmp_path, fake_corpus, fake_seed, transcript_path):
        proc = self._setup(tmp_path, fake_corpus, fake_seed, transcript_path)
        n_before = len(pd.read_parquet(proc / "corpus.parquet"))
        stats = merge_into_corpus(proc, xlsx_path=transcript_path, verbose=False)
        merged = pd.read_parquet(proc / "corpus.parquet")

        assert len(merged) == n_before + stats["net_new_positives"]
        assert merged["utt_id"].is_unique
        # schema preserved
        assert list(merged.columns) == list(fake_corpus.columns)
        # mined rows tagged + every row has >=1 subtype
        mined = merged[merged["source"] == MINED_SOURCE_TAG]
        assert len(mined) == stats["net_new_positives"]
        assert merged["subtype"].apply(lambda s: len(s) > 0).all()
        # backup written
        backups = list(proc.glob("corpus.pre_mining.*.parquet"))
        assert len(backups) == 1
        assert len(pd.read_parquet(backups[0])) == n_before

    def test_merge_excludes_existing_corpus_dupe(self, tmp_path, fake_corpus, fake_seed, transcript_path):
        proc = self._setup(tmp_path, fake_corpus, fake_seed, transcript_path)
        merge_into_corpus(proc, xlsx_path=transcript_path, verbose=False)
        merged = pd.read_parquet(proc / "corpus.parquet")
        # the "ramp" sentence exists in corpus already -> not duplicated
        ramp = merged[merged["utterance"].str.contains("ramp makes the toy car", na=False)]
        assert len(ramp) == 1

    def test_idempotent_second_run_adds_nothing(self, tmp_path, fake_corpus, fake_seed, transcript_path):
        proc = self._setup(tmp_path, fake_corpus, fake_seed, transcript_path)
        merge_into_corpus(proc, xlsx_path=transcript_path, verbose=False)
        n_after_first = len(pd.read_parquet(proc / "corpus.parquet"))
        stats2 = merge_into_corpus(proc, xlsx_path=transcript_path, verbose=False)
        n_after_second = len(pd.read_parquet(proc / "corpus.parquet"))
        assert stats2["net_new_positives"] == 0
        assert n_after_first == n_after_second


# ---------------------------------------------------------------------------
# not_science_shape review workflow
# ---------------------------------------------------------------------------

@pytest.fixture
def ns_corpus():
    base = {
        "label": "SCIENCE_TALK", "setting": "Centers", "source": "transcript_sci_mined",
        "tier2_cues": [], "tier3_cues": [], "was_sci_coded": 1, "topic": None,
        "citation": None, "subtype_source": "llm", "subtype_confidence": 0.0,
        "subtype_prompt_version": "subtype_v1",
    }
    return pd.DataFrame([
        {"utt_id": "utt_0000", "utterance": "Real science about magnets and force.",
         "transcript_ref": None, "subtype": ["observation"], **base},
        {"utt_id": "utt_sci_0001", "utterance": "Now that yogurt's in the garbage.",
         "transcript_ref": "8-13LG!R5", "subtype": ["not_science_shape"], **base},
        {"utt_id": "utt_sci_0002", "utterance": "So then the flower turns into a lemon.",
         "transcript_ref": "01-2_19LG!R9", "subtype": ["not_science_shape"], **base},
        {"utt_id": "utt_sci_0003", "utterance": "Line up for the bathroom now please.",
         "transcript_ref": "8-13LG!R7", "subtype": ["not_science_shape"], **base},
    ])


@pytest.fixture
def ns_pairs():
    return pd.DataFrame([
        {"pair_id": "p0", "anchor_id": "utt_sci_0001", "variant_text": "v", "register": "INFORMAL"},  # dropped
        {"pair_id": "p1", "anchor_id": "utt_sci_0002", "variant_text": "v", "register": "INFORMAL"},  # edited
        {"pair_id": "p2", "anchor_id": "utt_sci_0003", "variant_text": "v", "register": "INFORMAL"},  # kept
        {"pair_id": "p3", "anchor_id": "utt_0000", "variant_text": "v", "register": "INFORMAL"},      # untouched
    ])


class TestNotScienceReview:
    def test_export_selects_only_flagged(self, tmp_path, ns_corpus):
        ns_corpus.to_parquet(tmp_path / "corpus.parquet", index=False)
        path = export_not_science_review(tmp_path, verbose=False)
        sheet = pd.read_excel(path)
        assert len(sheet) == 3
        assert set(sheet["utt_id"]) == {"utt_sci_0001", "utt_sci_0002", "utt_sci_0003"}
        for col in ("decision", "corrected_text", "notes"):
            assert col in sheet.columns

    def test_validate_rejects_bad_decision(self):
        df = pd.DataFrame({"utt_id": ["a"], "decision": ["maybe"], "corrected_text": [None]})
        with pytest.raises(ValueError, match="Invalid decision"):
            validate_ns_decisions(df)

    def test_validate_requires_corrected_text_for_edit(self):
        df = pd.DataFrame({"utt_id": ["a"], "decision": ["edit"], "corrected_text": [None]})
        with pytest.raises(ValueError, match="missing corrected_text"):
            validate_ns_decisions(df)

    def test_apply_drop_edit_keep_and_prune(self, tmp_path, ns_corpus, ns_pairs):
        ns_corpus.to_parquet(tmp_path / "corpus.parquet", index=False)
        ns_pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
        template = pd.DataFrame({
            "utt_id": ["utt_sci_0001", "utt_sci_0002", "utt_sci_0003"],
            "decision": ["drop", "edit", None],  # None -> keep
            "corrected_text": [None, "The flower turns into a lemon as the fruit grows.", None],
            "notes": [None, None, None],
        })
        tpath = tmp_path / "ns.parquet"
        template.to_parquet(tpath, index=False)

        stats = apply_not_science_review(tpath, tmp_path, verbose=False)
        assert stats["dropped"] == 1 and stats["edited"] == 1 and stats["kept"] == 1

        corpus = pd.read_parquet(tmp_path / "corpus.parquet")
        assert "utt_sci_0001" not in set(corpus["utt_id"])           # dropped
        edited = corpus.loc[corpus["utt_id"] == "utt_sci_0002", "utterance"].iloc[0]
        assert "fruit grows" in edited                                # edited
        assert "utt_sci_0003" in set(corpus["utt_id"])               # kept

        pairs = pd.read_parquet(tmp_path / "pairs.parquet")
        # pairs for dropped + edited anchors pruned; kept + untouched pairs stay
        assert set(pairs["anchor_id"]) == {"utt_0000", "utt_sci_0003"}
        assert stats["pruned_pairs"] == 2

    def test_apply_backs_up_corpus(self, tmp_path, ns_corpus):
        ns_corpus.to_parquet(tmp_path / "corpus.parquet", index=False)
        template = pd.DataFrame({
            "utt_id": ["utt_sci_0001"], "decision": ["drop"],
            "corrected_text": [None], "notes": [None],
        })
        tpath = tmp_path / "ns.parquet"
        template.to_parquet(tpath, index=False)
        apply_not_science_review(tpath, tmp_path, verbose=False)
        backups = list(tmp_path.glob("corpus.pre_nsreview.*.parquet"))
        assert len(backups) == 1
