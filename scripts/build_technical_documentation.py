"""Generate the project technical documentation as a .docx file.

Run from the project root:
    python scripts/build_technical_documentation.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "Classroom_Science_Talk_Retrieval_Technical_Documentation.docx"

NAVY = RGBColor(0x1A, 0x2B, 0x3C)
BLUE = RGBColor(0x2E, 0x5C, 0x8A)
GREY = RGBColor(0x55, 0x55, 0x55)


def _set_run_font(run, *, name="Calibri", size=11, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    r = run._element.get_or_add_rPr()
    rFonts = r.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        r.append(rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:eastAsia"), name)
    rFonts.set(qn("w:cs"), name)


def _shade_cell(cell, hex_color: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    shd.set(qn("w:val"), "clear")
    tcPr.append(shd)


def _set_cell_text(cell, text, *, bold=False, size=9, color=None, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    if align is not None:
        p.alignment = align
    run = p.add_run(str(text))
    _set_run_font(run, size=size, bold=bold, color=color)


def _add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.autofit = True
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        _set_cell_text(cell, h, bold=True, size=9, color=RGBColor(0xFF, 0xFF, 0xFF))
        _shade_cell(cell, "2E5C8A")
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            _set_cell_text(cell, val, size=9, color=NAVY)
            if i % 2 == 1:
                _shade_cell(cell, "E8F1FB")
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Inches(w)
    doc.add_paragraph()
    return table


def _heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = NAVY if level == 1 else BLUE
        run.font.name = "Calibri"
    h.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    h.paragraph_format.space_after = Pt(4)
    return h


def _p(doc, text, *, italic=False, bold=False, size=11, space_after=8):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(space_after)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.line_spacing = 1.12
    run = para.add_run(text)
    _set_run_font(run, size=size, bold=bold, italic=italic, color=NAVY)
    return para


def _bullet(doc, text, *, level=0):
    para = doc.add_paragraph(style="List Bullet")
    para.clear()
    para.paragraph_format.left_indent = Inches(0.25 + 0.2 * level)
    para.paragraph_format.space_after = Pt(2)
    run = para.add_run(text)
    _set_run_font(run, size=11, color=NAVY)
    return para


def _code(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after = Pt(8)
    para.paragraph_format.left_indent = Inches(0.2)
    run = para.add_run(text)
    _set_run_font(run, name="Consolas", size=9, color=NAVY)
    return para


def _caption(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(10)
    run = para.add_run(text)
    _set_run_font(run, size=9, italic=True, color=GREY)
    return para


def build_document(output_path: Path = OUTPUT_PATH) -> Path:
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = NAVY
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.12

    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)

        footer = section.footer
        footer.is_linked_to_previous = False
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = fp.add_run(
            "Anita Zucker Center  ·  Classroom Science-Talk Retrieval  ·  Technical Documentation"
        )
        _set_run_font(run, size=8, italic=True, color=GREY)

    # ------------------------------------------------------------------ cover
    for _ in range(3):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("CLASSROOM SCIENCE-TALK RETRIEVAL")
    _set_run_font(r, size=22, bold=True, color=NAVY)

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = st.add_run("Technical Documentation")
    _set_run_font(r, size=16, color=BLUE)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run(
        "A modular retrieve-and-rerank NLP pipeline for finding novel\n"
        "science-practice talk in early-childhood classroom transcripts"
    )
    _set_run_font(r, size=12, italic=True, color=GREY)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_before = Pt(24)
    r = meta.add_run(
        "Anita Zucker Center\n"
        "University of Florida\n\n"
        f"Document date: {date.today():%B %d, %Y}\n"
        "Codebase: Science talk retrieval (Python 3.12)\n"
        "Status: research system; not licensed for redistribution"
    )
    _set_run_font(r, size=11, color=NAVY)

    doc.add_page_break()

    # ------------------------------------------------------------------ 1
    _heading(doc, "1. Purpose and scope", 1)
    _p(
        doc,
        "This document describes the classroom science-talk retrieval system: a "
        "modular natural-language pipeline that locates rare moments of science-practice "
        "talk (observing, predicting, causal reasoning, appealing to evidence, and naming "
        "science content) inside large volumes of everyday early-childhood classroom "
        "conversation. The system is built around University of Florida–hosted large "
        "language models and a locally fine-tuned dense retriever. It produces "
        "reviewer-ready workbooks so expert coders remain the final authority.",
    )
    _p(
        doc,
        "The repository covers two linked bodies of work. Year 1 (Y1) constructs and "
        "evaluates a training corpus and a two-stage detector. Year 2 (Y2) scores "
        "Technology Observation Audio transcripts at classroom-session scale, then "
        "routes flagged utterances to human review. A separate aggregation pass turns "
        "the human-checked analytic sample into per-transcript and overall statistics.",
    )
    _p(doc, "In-scope", bold=True, space_after=4)
    _bullet(doc, "Y1 labeled seed corpus, negative mining, register-variant augmentation, and splits.")
    _bullet(doc, "Track A bi-encoder retrieval (nomic-embed-text-v1.5) and Track B LLM re-ranking (gpt-oss-120b).")
    _bullet(doc, "Confidence routing, human-review tooling, evaluation, and threshold tuning.")
    _bullet(doc, "Y2 deployment scoring of classroom transcripts and the human-checked analytic sample.")
    _p(doc, "Out of scope of this repository (described only as upstream context)", bold=True, space_after=4)
    _bullet(doc, "On-device audio capture (LENA-style recorders and BLE beacons).")
    _bullet(doc, "Automatic speech recognition (Whisper-large-v3) and speaker diarization, which produce the Y2 transcript workbooks this pipeline consumes.")
    _bullet(doc, "Raw transcripts, trained weights, embedding caches, and scored outputs, which are git-ignored and must be supplied locally.")

    # ------------------------------------------------------------------ 2
    _heading(doc, "2. Problem definition", 1)
    _p(
        doc,
        "Science-practice talk is a low-frequency feature of preschool classroom audio. "
        "A typical observation produces hours of speech whose majority is logistics, "
        "redirection, social talk, and classroom management. Hand-coding every utterance "
        "is slow and expensive. The detector’s job is not to replace coders; it is to "
        "hand them a pre-filtered, ranked shortlist of candidate science utterances, "
        "each tagged with a predicted practice subtype, a score, a nearest corpus match, "
        "and (when the LLM stage ran) a short rationale.",
    )
    _p(
        doc,
        "Two constraints shaped the design. First, the expert-labeled seed is small "
        "(on the order of 200 utterances), so the system stretches those labels with "
        "register paraphrases and a large negative pool rather than training a large "
        "classifier from scratch. Second, false positives are costly because every "
        "wrongly flagged utterance consumes reviewer time. Routing, gates, and a high "
        "LLM decision threshold are therefore conservative by design.",
    )

    _heading(doc, "2.1 Operational definition of science talk", 2)
    _p(
        doc,
        "Science talk is inquiry about the world (notice / try / test / predict / explain) "
        "and/or science content terms (plants, forces, materials, structure). It is not "
        "routines, classroom management, social-emotional language, literacy, or off-task "
        "talk. The closed subtype taxonomy is:",
    )
    _add_table(
        doc,
        ["Subtype", "What it covers"],
        [
            ["observation", "Noticing or describing what is seen, heard, or felt (sensory data)."],
            ["prediction", "Anticipating what will happen (guess, what-if, hypothesis)."],
            ["causal_reasoning", "Explaining why/how; cause and effect."],
            ["evidence", "Appealing to data, measurement, a count, or a test result."],
            ["content", "Naming a science topic (plants, force, materials, structure)."],
            ["not_science_shape", "Sentinel: social, management, literacy, or off-task talk. Never combined with a practice label."],
        ],
        col_widths=[1.8, 5.0],
    )
    _p(
        doc,
        "An utterance may carry more than one practice/content subtype. "
        "not_science_shape is reachable only via the LLM path, not via cue rules or the keyword scan.",
    )

    # ------------------------------------------------------------------ 3
    _heading(doc, "3. System architecture", 1)
    _p(
        doc,
        "Runtime inference is a two-stage retrieve-and-rerank design. A cheap dense "
        "retriever scores every utterance against a bank of known science-talk anchors. "
        "A prompted large language model then re-ranks only the top-k matches (or, at "
        "Y2 deployment, only the highest-cosine utterances that fit the dollar budget).",
    )
    _code(
        doc,
        "Classroom transcripts (all kept utterances)\n"
        "        |\n"
        "        v\n"
        "Track A  bi-encoder retrieval   nomic-embed-text-v1.5 (fine-tuned, local)\n"
        "        |  cosine vs L2-normalized anchor bank  ->  top-k candidates\n"
        "        v\n"
        "Track B  zero-shot pair re-ranker   gpt-oss-120b (UF HiPerGator)\n"
        "        |  score in [0, 1] + suggested label + rationale\n"
        "        v\n"
        "Reviewer-ready workbooks / evaluation reports\n"
        "        ^\n"
        "Human review + calibration  (feeds routing thresholds and training gates)",
    )
    _caption(doc, "Figure 1. Retrieve-and-rerank inference path (Tracks A and B).")

    _heading(doc, "3.1 Why two stages", 2)
    _p(
        doc,
        "The bi-encoder encodes the query and each corpus document independently, so the "
        "anchor bank can be embedded once and searched with a matrix-vector product. That "
        "is fast enough to score every screened Y2 utterance locally at no API cost. The "
        "trade-off is that it never sees the query and a candidate together. Hard "
        "look-alikes (“I wonder where your shoes are”) can still rank highly. The "
        "re-ranker reads the pair jointly against a rubric of operational definitions, "
        "three in-context examples, and a strict JSON schema. It is slower and billed, "
        "so it is applied only to a shortlist.",
    )

    _heading(doc, "3.2 Models and hosting", 2)
    _add_table(
        doc,
        ["Role", "Model", "Where it runs", "Env / config"],
        [
            ["Track A embeddings / fine-tune", "nomic-ai/nomic-embed-text-v1.5 (768-d)", "Local sentence-transformers (fine-tune); UF embedding API for the frozen baseline pass", "LLM_MODEL_TRACKA; config/biencoder.json"],
            ["Track B pair judge", "gpt-oss-120b", "UF HiPerGator chat-completions API", "LLM_MODEL_TRACKB; config/reranker.json"],
            ["Augmentation / subtypes / negatives / gate", "llama-3.3-70b-instruct", "UF chat-completions API", "LLM_MODEL_AUGMENT"],
            ["Upstream ASR (context)", "Whisper-large-v3", "Outside this repo", "Produces Y2 xlsx transcripts"],
        ],
        col_widths=[1.7, 1.8, 1.9, 1.5],
    )
    _p(
        doc,
        "API endpoints default to https://api.ai.it.ufl.edu/v1/chat/completions and "
        ".../v1/embeddings. Multiple keys (LLM_API_KEY, LLM_API_KEY_2, …) rotate automatically "
        "when a key’s quota or credential is spent. Completions and embeddings are cached "
        "on disk under cache/ keyed by SHA-256 of (prompt_version, model, endpoint, params). "
        "Request timeouts are (connect=10 s, read=180 s).",
    )

    _heading(doc, "3.3 End-to-end capture context", 2)
    _p(
        doc,
        "The broader study pipeline (see scripts/architecture_diagram.mmd) is: "
        "(1) capture with LENA-style recorders and BLE beacons; (2) ASR with Whisper-large-v3; "
        "(3) normalization, segmentation, diarization, and cleanup into utterance-level rows "
        "with timestamps, speaker, and an ASR confidence suffix of the form (NN%); "
        "(4) this science-talk detector. Stages 1–3 are not implemented in this repository; "
        "stage 4 consumes their xlsx transcripts.",
    )

    # ------------------------------------------------------------------ 4
    _heading(doc, "4. Datasets", 1)
    _p(
        doc,
        "The project uses three related datasets. They must not be conflated: the Y1 seed "
        "is the labeled training corpus; the Y2 raw transcripts are the deployment corpus; "
        "the human-checked analytic sample is a reviewed subset of Y2 used for reporting.",
    )

    _heading(doc, "4.1 Year 1 labeled seed corpus (training)", 2)
    _p(
        doc,
        "Source workbook: data/2026-04-17_science_talk_examples.xlsx, with sheets "
        "“Example utterances”, “Seed words”, and “Category definitions”. Step 1 "
        "(data_loader_1) normalizes this into corpus.parquet, seed_words.parquet, and "
        "category_defs.parquet. The loader asserts at least 195 rows after de-duplication, "
        "at least 188 SCIENCE_TALK rows, and at least 5 NOT_SCIENCE_TALK rows.",
    )
    _add_table(
        doc,
        ["Quantity", "Count / value", "Notes"],
        [
            ["Curated labeled utterances (after cleanup)", "196", "Documented in the results deck and positives_mining.py (~196)."],
            ["SCIENCE_TALK (positives)", "191", "Expert-coded science-practice examples."],
            ["NOT_SCIENCE_TALK in the seed", "5", "Tiny expert-labeled negative set; most negatives are mined later."],
            ["Register paraphrases (pairs.parquet)", "516", "LLM rewrites of anchors into LARGE_GROUP / SMALL_GROUP / INFORMAL."],
            ["Non-science pool (negatives.parquet)", "7,815", "Transcript-clean + LLM hard negatives + seed-term nonscience."],
            ["Approximate training rows after expansion", "~8,500", "Stratified 80 / 10 / 10 train / val / test."],
            ["content subtype (multi-label)", "171", "Dominates the seed; an utterance can have multiple subtypes."],
            ["observation", "68", "Second-largest practice/content class."],
            ["causal_reasoning", "8", "Rare practice; model is thinnest here."],
            ["prediction", "7", "Rare practice."],
            ["evidence", "4", "Rarest practice."],
        ],
        col_widths=[2.6, 1.4, 2.9],
    )
    _p(
        doc,
        "Each corpus row has utt_id (utt_0000…), utterance, label, setting, source, "
        "tier2_cues and tier3_cues (list-typed), was_sci_coded (1 if Notes mention SCI-coded), "
        "transcript_ref, topic, and citation. Settings are mapped onto a closed vocabulary: "
        "Large Group, Small Group, Centers, Transition, Conversation, Blocks/Engineering, Unknown. "
        "A “ / ” suffix on Setting is split into setting plus topic (for example "
        "“Large Group / Animal Observation”). Text is NFKC-normalized with collapsed whitespace; "
        "duplicate utterances are dropped.",
    )
    _p(
        doc,
        "Classroom setting maps onto three teacher registers used in augmentation: "
        "LARGE_GROUP (circle/carpet, more formal), SMALL_GROUP (2–5 children), and "
        "INFORMAL (centers, blocks/engineering, side-by-side play). Unknown-setting anchors "
        "fan out to all three registers.",
    )
    _p(
        doc,
        "Optional expansion of real positives: src/positives_mining.py harvests teacher "
        "rows whose content code contains “SCI” from data/transcripts/Coding Transcripts.xlsx "
        "(the mirror of negative mining, which keeps non-SCI teacher rows). Fragments shorter "
        "than 5 words or 20 characters are dropped. New rows are tagged source=transcript_sci_mined "
        "with utt_id prefix utt_sci_. Downstream pairs, cosines, routing, and splits become "
        "stale after a merge and must be rebuilt (Steps 2–7).",
    )

    _heading(doc, "4.2 Year 2 deployment corpus (unlabeled transcripts)", 2)
    _p(
        doc,
        "Root: data/transcripts/Y2_Technology Observation Audio_Transcripts/<ClassroomID_MMDDYY>/. "
        "Each session folder holds one or more per-child xlsx files produced by ASR. "
        "The deployment module is written against approximately 85 classroom-session folders "
        "and on the order of 1.3 million utterances. Full LLM re-ranking of that volume is "
        "intentionally not attempted (on the order of $2,600 at the configured per-call cost); "
        "cosine covers every kept utterance and the LLM budget is spent on the highest-scoring "
        "candidates. Classroom49_012026 has duplicate folders (_old / _updated / plain); "
        "config/deploy.json keeps only the folder ending in _updated.",
    )
    _add_table(
        doc,
        ["Ingest filter", "Rule"],
        [
            ["Speaker keep-list", "adult and child only (other diarization labels dropped)."],
            ["ASR confidence floor", "0.40. Trailing “(NN%)” is parsed from the Transcription (Confidence) cell. Rows with no parsable confidence are kept."],
            ["Empty / dash text", "Dropped."],
            ["Child ID", "Fourth hyphenated token of the filename stem (for example 65-74-943-170 → 170); otherwise “base”."],
        ],
        col_widths=[2.2, 4.7],
    )

    _heading(doc, "4.3 Human-checked Y2 analytic sample", 2)
    _p(
        doc,
        "After deployment, reviewers fill a review column on one focal-child sheet per "
        "classroom-observation workbook (Human coded/Classroom<id>_<MMDDYY>_ebh.xlsx). "
        "Those workbooks contain only utterances the model labeled SCIENCE_TALK. "
        "Data Aggregation/aggregate_human_coded.py joins each reviewed sheet to the matching "
        "raw transcript to recover denominators (screened utterance counts and audio hours). "
        "The figures below are computed from Data Aggregation/output/sampled_transcripts.xlsx "
        "and are the reporting sample, not the full Y2 corpus.",
    )
    _add_table(
        doc,
        ["Measure", "Value"],
        [
            ["Classrooms represented", "16 (IDs 10, 12, 14, 21, 37, 38, 46, 49, 63, 67, 73, 74, 83, 84, 85, 92)"],
            ["Classroom-observations (unique classroom × date)", "83"],
            ["Focal-child transcripts (unit of analysis)", "83"],
            ["Unique child IDs in the sample", "26"],
            ["Observation window", "October 2025 through April 2026 (filenames coded MMDDYY)"],
            ["Human-coded workbooks on disk", "83 Classroom*_ebh.xlsx files plus nameset.txt"],
            ["Total audio hours (recording span)", "289.23 h  (mean 3.48, SD 0.60, min 2.20, max 4.88)"],
            ["Talk-time hours (speech-segment sum, reference)", "130.95 h  (mean 1.58, min 0.26, max 3.06)"],
            ["Raw transcript rows (reference)", "299,917"],
            ["Screened utterances (pipeline denominator)", "174,979"],
            ["  — adult", "96,747  (mean 1,166 per transcript)"],
            ["  — child", "78,232  (mean 943 per transcript)"],
            ["LLM-flagged science-talk candidates", "8,004  (4.58% of screened; mean 96.4 per transcript, range 12–357)"],
            ["  — adult flagged", "4,566"],
            ["  — child flagged", "3,438"],
            ["Human-verified science talk", "1,984  (24.79% of flagged; 1.13% of screened)"],
            ["  — adult verified", "1,402"],
            ["  — child verified", "582"],
            ["False positives (flagged, marked NOT_SCIENCE_TALK)", "5,985  (74.78% of flagged)"],
            ["Flagged but left unreviewed", "35  (0.44% of flagged)"],
            ["Verified science talk per audio hour", "mean 7.06  (SD 6.70, min 0, max 33.23)"],
            ["Share of all utterances that are verified science talk", "mean 1.35% per transcript (range 0–6.75%)"],
        ],
        col_widths=[3.4, 3.5],
    )
    _caption(
        doc,
        "Table. Human-checked analytic sample. “Screened utterances” are adult/child speech "
        "at or above the 0.40 ASR-confidence floor — the set the model actually evaluated. "
        "Audio hours use recording span (last segment end minus first segment start).",
    )
    _p(
        doc,
        "These totals confirm the design assumption that verified science talk is rare. "
        "The LLM first pass is deliberately broad (about 1 in 22 screened utterances is "
        "flagged). Human review then verifies about one in four flags, so verified science "
        "talk is about 1.1% of screened speech, or roughly seven utterances per recorded hour "
        "on average, with large classroom-to-classroom variation (several transcripts have "
        "zero verified science talk; the maximum is 33 per hour).",
    )
    _p(
        doc,
        "Speaker splits should be interpreted with caution: adult vs. child comes from "
        "automatic diarization and is not perfectly accurate. Percentages can also be "
        "read per speaker: 1,402 of 96,747 adult screened utterances were verified "
        "(1.45%); 582 of 78,232 child screened utterances were verified (0.74%). "
        "Adult flags verify at a higher rate (mean 31.4% of adult flags) than child flags "
        "(mean 20.7%).",
    )

    _heading(doc, "4.4 Processed artifacts (git-ignored)", 2)
    _add_table(
        doc,
        ["Path", "Produced by", "Contents"],
        [
            ["data/processed/corpus.parquet", "Steps 1–2 (+ optional positives mining)", "Normalized labeled utterances + subtypes."],
            ["data/processed/seed_words.parquet", "Step 1", "Seed terms, tiers, categories, variants."],
            ["data/processed/category_defs.parquet", "Step 1", "Operational include/exclude definitions injected into the re-ranker prompt."],
            ["data/processed/negatives.parquet", "Step 3", "Three-source NOT_SCIENCE_TALK pool + gate scores."],
            ["data/processed/pairs.parquet", "Step 4", "Anchor–variant paraphrases + preservation checks."],
            ["data/processed/splits.parquet", "Step 7", "train/val/test assignment by utt_id / neg_id / pair_id."],
            ["data/processed/hard_informal_slice.parquet", "Step 7", "Held-out informal-register families, never used in training."],
            ["data/processed/review_decisions.parquet", "Step 6", "Sidecar of accept/reject/edit; survives corpus rebuilds."],
            ["data/processed/corpus_embeddings.npy", "Step 8", "L2-normalized positive-anchor embeddings."],
            ["models/biencoder/", "Step 8", "Fine-tuned sentence-transformers checkpoint (offline load)."],
            ["cache/", "llm_client_0", "Hashed completion and embedding responses."],
            ["reports/", "Steps 9–12", "Eval reports, metrics JSON, Y2 scored workbooks, run_manifest.json."],
        ],
        col_widths=[2.6, 1.6, 2.7],
    )

    # ------------------------------------------------------------------ 5
    _heading(doc, "5. Pipeline design", 1)
    _p(
        doc,
        "The codebase is a numbered 0–12 pipeline. Steps 1–7 (corpus preparation) are "
        "orchestrated by src/pipeline.py. Human review is interactive and is not a numbered "
        "orchestrator step. Steps 8–12 are run as standalone modules. Each step has a "
        "JSON config under config/, prompts under prompts/, and pytest coverage under tests/.",
    )
    _add_table(
        doc,
        ["Step", "Module", "Purpose"],
        [
            ["0", "src/llm_client_0.py", "Shared LLM/embedding client: on-disk cache, retries, timeouts."],
            ["1", "src/data_loader_1.py", "Load and normalize the labeled xlsx into parquet."],
            ["2", "src/subtypes_2.py", "Assign practice subtypes (rule, keyword, LLM fallback)."],
            ["3", "src/negatives_3.py", "Mine three kinds of NOT_SCIENCE_TALK negatives + LLM gate."],
            ["4", "src/augment_4.py", "Register-variant paraphrases of positives."],
            ["5a", "src/embeddings_baseline_5.py", "Frozen nomic embeddings; baseline_cosine on pairs and negatives."],
            ["5b", "src/confidence_5.py", "Confidence score and auto / spot / review routing."],
            ["6", "src/review_6.py", "Export / apply / status for human accept-reject-edit."],
            ["7", "src/splits.py", "Stratified 80/10/10 splits + hard-informal held-out slice."],
            ["—", "src/positives_mining.py", "Optional: harvest SCI-coded teacher rows into the corpus."],
            ["8", "src/biencoder_8.py", "Fine-tune Track A; write corpus embeddings."],
            ["9", "src/reranker_9.py", "Calibrate the zero-shot Track B pair judge."],
            ["10", "src/query_10.py", "classify(phrase): retrieve + rerank."],
            ["11", "src/evaluate_11.py", "Threshold tuning, ablation, test metrics."],
            ["12", "src/deploy_y2_12.py", "Score Y2 transcripts; resumable, dollar-capped."],
        ],
        col_widths=[0.7, 2.3, 3.9],
    )

    _heading(doc, "5.1 Step 0 — LLM client", 2)
    _p(
        doc,
        "cached_request hashes prompt_version, model, endpoint, and params and stores "
        "JSON under cache/<sha256>.json (atomic write via a .tmp replace). Hits return "
        "the stored raw_response without a network call. connect_to_llm is a smoke-test "
        "helper. This client is the only network path for Steps 2–5 and 9–12.",
    )

    _heading(doc, "5.2 Step 1 — Data ingestion", 2)
    _p(
        doc,
        "clean_corpus parses cue lists on | , ; , strips topic suffixes from Setting, "
        "extracts transcript_ref from Notes via a (…#N) pattern, flags was_sci_coded, "
        "assigns sequential utt_id values, and de-duplicates on normalized utterance text. "
        "validate_corpus is a hard gate on row count, label floors, unique IDs, and setting "
        "vocabulary. Default CLI: python src/data_loader_1.py or python src/pipeline.py --steps 1.",
    )

    _heading(doc, "5.3 Step 2 — Subtype labeling", 2)
    _p(
        doc,
        "Three stages, in order, so that cue-rich Y1 rows almost never hit the LLM:",
    )
    _bullet(doc, "Rule mapping from explicit Tier 2 / Tier 3 cues through CATEGORY_TO_SUBTYPES and TERM_OVERRIDES (predict, guess, what if, hypothesis, …). Any non-empty Tier 3 cue list adds content.")
    _bullet(doc, "Additive keyword scan of utterance text against seed-word terms and variants (word-boundary, longest-token first).")
    _bullet(doc, "LLM zero-shot fallback (llama-3.3-70b-instruct, prompt subtype_v3) only when both prior stages are empty. On the Y1 corpus this path fires zero times because every row has Tier 3 cues; --use-llm-step2 is intended for cue-less corpora such as Y2.")
    _p(
        doc,
        "Every row must receive at least one subtype. Distribution warnings fire if a "
        "practice class is empty or if any class exceeds 70% of rows. Source is recorded "
        "as rule, keyword, rule+keyword, or llm, with confidence 1.0 for deterministic paths.",
    )

    _heading(doc, "5.4 Step 3 — Negative mining", 2)
    _p(
        doc,
        "The gating step for a usable classifier. The negative pool must be at least "
        "3× the positive count and must represent three source types:",
    )
    _bullet(doc, "transcript_clean — teacher utterances from Coding Transcripts.xlsx whose content code is not SCI. Sub-tagged behavior_disapproving (column E = Y) versus other_teacher_talk. These rows are hand-coded in origin and bypass confidence routing.")
    _bullet(doc, "llm_hard_negative — llama-3.3-70b-instruct rewrites of a positive that keep syntax and length but move the topic to management, SEL, literacy, or transitions (prompt neg_hard_v1; default 2 per positive).")
    _bullet(doc, "seed_word_nonscience — utterances that use a science seed term in a clearly non-scientific sense (prompt neg_seed_v1; default 2 per seed term).")
    _p(
        doc,
        "An LLM gate (neg_gate_v1) scores generated negatives from 0.0 (clearly science) "
        "to 1.0 (clearly not science). Default is to gate-score only LLM-generated rows "
        "(~700 calls on Y1). --gate-score-all-rows would score every negative (~7,800 calls). "
        "sample_for_review(n=50) supports the ≥90% true-negative hand-check DoD. "
        "Optional --use-llm-negatives-subtype labels negative subtypes (~1,300 cached calls).",
    )

    _heading(doc, "5.5 Step 4 — Register-variant augmentation", 2)
    _p(
        doc,
        "Each science-talk anchor is paraphrased into registers other than its source "
        "(Unknown anchors get all three). Variants must (1) preserve scientific meaning, "
        "(2) preserve listed cue verbs/nouns in ≥80% of variants (substring check; "
        "vacuous pass if the anchor is cue-less), (3) differ in surface form from the "
        "anchor, and (4) record model_id and prompt_version (aug_var_v1). Output is "
        "pairs.parquet with pair_id, anchor_id, variant_text, register, subtype, "
        "preservation_pct, preservation_check_passed, differs_from_anchor, and llm_self_score. "
        "Default is one variant per target register.",
    )

    _heading(doc, "5.6 Step 5a — Frozen baseline embeddings", 2)
    _p(
        doc,
        "Off-the-shelf nomic-embed-text-v1.5 (prompt_version emb_baseline_v1, 768 dimensions) "
        "embeds unique texts through the UF embedding API. pairs.parquet gains baseline_cosine "
        "between anchor and variant; negatives.parquet gains baseline_cosine between the "
        "anchor positive and the negative (NaN when there is no anchor, as with transcript_clean "
        "and seed_word_nonscience). This encoder is frozen and independent of Step 8 so that "
        "confidence routing cannot loop the fine-tuned model’s own opinion back into training "
        "selection. Failed embeds become a zero vector, which the band score treats as extreme/bad.",
    )

    _heading(doc, "5.7 Step 5b — Confidence scoring and routing", 2)
    _p(
        doc,
        "Each generated pair and each LLM-generated negative receives a confidence in [0, 1] "
        "as a weighted geometric mean of three signals (a zero in any signal strongly penalizes "
        "the score, with a 1e-3 floor so a single zero is not a hard kill):",
    )
    _bullet(doc, "LLM self-report: llm_self_score on pairs; llm_gate_score on negatives (missing → 0.5).")
    _bullet(doc, "Cosine band score: similarity is good only inside a band. Too high means the paraphrase did not change, or the negative looks like a positive; too low means meaning drift or an easy negative.")
    _bullet(doc, "Structural checks: pairs use differs_from_anchor (structural_mode=differs_only after the Hadley audit, because reviewers accepted cue-less paraphrases that still preserved meaning). Negatives use structural_check_passed.")
    _p(
        doc,
        "Current weights (config/confidence.json, version confidence_v2_hadley_audit): "
        "LLM 0.5, cosine 0.8, structural 0.8. Pair cosine band [0.55, 0.92] with falloff 0.35; "
        "negative band [0.25, 0.90]. Routing thresholds: confidence ≥ 0.70 → auto (use in training); "
        "≥ 0.15 → spot (≈12% random spot-check); else review (full manual review). "
        "These cut-offs were retuned after a 50-row Hadley audit (v1 had 42% agreement; "
        "the retune targeted ≥80% while keeping llm_gate=0 rows, confidence ≈ 0.13, in review). "
        "A stratified 50-row routing_audit_template.parquet is emitted for that audit.",
    )

    _heading(doc, "5.8 Step 6 — Human review", 2)
    _p(
        doc,
        "python src/review_6.py --export writes review_queue_6.xlsx of all routing=review rows "
        "plus a ~12% sample of routing=spot, excluding IDs already decided. Reviewers fill "
        "decision ∈ {accept, reject, edit}, reject_reason from a closed list, and corrected_text "
        "when editing. python src/review_6.py --apply merges into review_decisions.parquet "
        "(source of truth; survives pipeline re-runs) and mirrors columns onto pairs/negatives. "
        "Re-export is resumable; re-apply is idempotent (newer decision wins). "
        "filter_verified() is the downstream gate: rejected rows are dropped and edits applied "
        "before splits or training. Codified reject reasons include register_mismatch, "
        "meaning_drift, anchor_copy, not_actually_negative, implausible_classroom, nonsense, "
        "duplicate, pii, and other.",
    )
    _p(
        doc,
        "scripts/build_review_bundle.py standardizes four reviewer workbooks (pairs queue, "
        "routing audit, negatives-quality audit, not-science review) with a shared Review + "
        "INSTRUCTIONS layout.",
    )

    _heading(doc, "5.9 Step 7 — Splits and the hard-informal slice", 2)
    _p(
        doc,
        "Default ratios are 0.80 / 0.10 / 0.10 with seed 17. Stratification keeps subtypes "
        "in every split for positives and source types for negatives. Variant pairs are "
        "grouped by anchor_id so an anchor and its paraphrases never cross the train/val/test "
        "cut (no leakage). Before splitting, filter_verified is applied.",
    )
    _p(
        doc,
        "A separate hard_informal_slice.parquet holds whole anchor families that never enter "
        "training. It includes (1) every positive whose setting is Centers, Blocks/Engineering, "
        "Transition, or Conversation, and (2) up to 30 additional randomly sampled families "
        "that have at least one INFORMAL-register variant (most seed anchors are setting=Unknown "
        "and only become informal via augmentation). Step 11 tunes the decision threshold "
        "subject to hard-informal recall ≥ 0.80 so informal science is not tuned away.",
    )

    _heading(doc, "5.10 Step 8 — Bi-encoder fine-tuning (Track A)", 2)
    _p(
        doc,
        "Warm-start nomic-ai/nomic-embed-text-v1.5 via sentence-transformers (local; you cannot "
        "back-propagate through the UF embedding API). Training triples are (variant query, "
        "anchor document, optional hard-negative document) from TRAIN-split pairs only, using "
        "task prefixes search_query: and search_document:. Loss is MultipleNegativesRankingLoss "
        "(in-batch negatives, plus the explicit hard negative when present).",
    )
    _add_table(
        doc,
        ["Hyperparameter", "Value"],
        [
            ["Epochs", "3"],
            ["Batch size", "16"],
            ["Learning rate", "2e-5"],
            ["Warmup ratio", "0.10"],
            ["Max sequence length", "128"],
            ["Seed", "17"],
            ["AMP", "false"],
            ["Primary eval metric", "MRR@10 (distractor task)"],
            ["k values", "1, 5, 10"],
        ],
        col_widths=[2.4, 4.5],
    )
    _p(
        doc,
        "Two retrieval evaluations are reported. The DoD gate is the distractor task: "
        "query = held-out val variant; corpus = anchor bank plus sampled negatives. "
        "Fine-tuned MRR@10 and recall@10 must strictly improve over the frozen baseline, "
        "or the baseline must already sit at ceiling. The anchor_bank-only task is a "
        "register-invariance diagnostic. After a pass, L2-normalized embeddings of every "
        "verified-positive utterance are written to corpus_embeddings.npy with a utt_id sidecar, "
        "and the checkpoint is saved to models/biencoder/ for offline load (trust_remote_code=True, "
        "device=cpu at query time).",
    )

    _heading(doc, "5.11 Step 9 — LLM re-ranker (Track B)", 2)
    _p(
        doc,
        "A non-fine-tuned pair judge. prompts/reranker_v1.txt (and a paraphrased twin "
        "reranker_v1b.txt for stability testing) inject operational definitions from "
        "category_defs.parquet, the subtype taxonomy, scoring rules, three in-context "
        "examples (easy positive, hard informal positive, hard negative), and a strict "
        "schema {\"score\": float, \"rationale\": \"≤25 words\"}. Decoding is deterministic "
        "(temperature 0.0, max_tokens 800 — gpt-oss-120b emits reasoning_content before the "
        "answer, so a short budget truncates). Parse failures retry up to 3 times and fail "
        "loudly with the row id. Empty completions purge the poisoned cache entry so the "
        "retry hits the network.",
    )
    _p(
        doc,
        "Calibration gates: AUROC ≥ 0.85 on a 100-pair gold audit and Spearman ρ ≥ 0.90 "
        "between the two prompt phrasings (n=50 smoke / 80 full). Retrieval lift and "
        "hard-informal lift are diagnostics until Y2 distractors exist, because the Y1 "
        "dev set is near ceiling after Step 8. A CallLedger counts cache hits versus new "
        "calls against a config cap (400 smoke / 5,000 full).",
    )

    _heading(doc, "5.12 Step 10 — Query-time classify()", 2)
    _p(
        doc,
        "QueryPipeline loads the fine-tuned encoder and the anchor-bank embeddings once, "
        "then classify(phrase) does:",
    )
    _bullet(doc, "Embed the phrase with the query prefix; cosine-search the L2-normalized bank; take top_k (default 20).")
    _bullet(doc, "Score all k (phrase, candidate) pairs with gpt-oss-120b in a thread pool (max_workers=8) under a 60-second per-phrase wall-clock cap.")
    _bullet(doc, "Aggregate by max LLM score (a phrase is SCIENCE_TALK if it strongly matches any science anchor). Label SCIENCE_TALK iff score ≥ threshold (placeholder 0.5 until Step 11 tunes it).")
    _bullet(doc, "On total LLM failure, fall back to cosine-only ranking with degraded=true and degraded_cosine_threshold=0.5 — deployment never silently changes behavior.")
    _p(
        doc,
        "A second classify of the same phrase must produce identical scores and ordering "
        "(temperature 0, pinned prompt_version, stable sort). A re-eval against a warm cache "
        "must make zero new calls.",
    )

    _heading(doc, "5.13 Step 11 — Evaluation and threshold tuning", 2)
    _p(
        doc,
        "The binary threshold is tuned on dev only, never test, by maximizing F1 subject to "
        "hard-informal-slice recall ≥ 0.80 on a grid from 0.05 to 0.95 step 0.05. Three "
        "ablation systems are scored on F1, MRR, and NDCG@10, overall and on the hard-informal slice:",
    )
    _bullet(doc, "(a) Frozen nomic-embed-text-v1.5 cosine — local, free.")
    _bullet(doc, "(b) Fine-tuned bi-encoder cosine — local, free.")
    _bullet(doc, "(c) Fine-tuned bi-encoder + gpt-oss-120b re-rank — the deployed system.")
    _p(
        doc,
        "Outputs: reports/eval_report.md, reports/eval_metrics.json, reports/test_predictions.parquet "
        "(including LLM rationale). Full-mode query caps in config/eval.json are 63 pos + 63 neg "
        "dev, 65 pos + 65 neg test, and 166 hard-informal slice items (20,000 max new LLM calls). "
        "The Definition of Done is on mechanics (tuned on dev, floor enforced, ablation written, "
        "predictions written), not on a saturated absolute F1.",
    )

    _heading(doc, "5.14 Step 12 — Y2 deployment scoring", 2)
    _p(
        doc,
        "Folder-by-folder, memory-bounded, and resumable. For each classroom-session:",
    )
    _bullet(doc, "Ingest every child xlsx with the speaker and confidence filters.")
    _bullet(doc, "Cosine-score every unique utterance against the anchor bank (local, free). Attach cosine_score, top_match_in_corpus, predicted_subtype.")
    _bullet(doc, "Re-rank highest-cosine utterances with classify() until the dollar cap. Default dollar_cap is $15 at cost_per_call $0.0001 (150,000 new pair calls). Unused budget from quiet folders rolls forward. Cached calls are free. After the cap, remaining utterances keep cosine only.")
    _bullet(doc, "Assign predicted_label: LLM rows use llm_label_threshold 0.90; cosine-only rows use cosine_label_threshold 0.40. Both are provisional Y1-tuned values.")
    _bullet(doc, "Write reports/y2_scores/<folder>.xlsx. With science_only=true (default), each child sheet contains only SCIENCE_TALK rows. A review dropdown (SCIENCE_TALK / NOT_SCIENCE_TALK) and an agreement tally sit on every sheet, plus an INSTRUCTIONS sheet.")
    _p(
        doc,
        "Finished folders are skipped on re-run unless --overwrite. A run_manifest.json is "
        "checkpointed after every folder. CLI examples: python src/deploy_y2_12.py --folders "
        "Classroom74_110525  or  python src/deploy_y2_12.py --all --dollar-cap 25.",
    )

    _heading(doc, "5.15 Orchestration and typical commands", 2)
    _code(
        doc,
        "python src/pipeline.py                      # Steps 1–7, LLM stubs\n"
        "python src/pipeline.py --steps 3 4 --use-llm\n"
        "python src/biencoder_8.py                    # Step 8 train + eval + embed\n"
        "python src/reranker_9.py --smoke             # Step 9 calibration\n"
        "python src/evaluate_11.py --full             # Step 11\n"
        "python src/deploy_y2_12.py --all --dollar-cap 25\n"
        "python src/review_6.py --export | --apply FILE --reviewer DH | --status\n"
        "python Data Aggregation/aggregate_human_coded.py\n"
        "pytest -q",
    )

    # ------------------------------------------------------------------ 6
    _heading(doc, "6. Evaluation results (Y1)", 1)
    _p(
        doc,
        "The following figures are the documented Step 11 ablation (results deck). "
        "Y1 negatives are mostly management/social talk and are trivially separable from "
        "science talk, so absolute F1 is a saturated diagnostic. The hard-informal slice "
        "is the number intended to move on messier Y2 distractors.",
    )
    _add_table(
        doc,
        ["System", "Threshold", "Test F1", "Test recall", "Hard-slice recall"],
        [
            ["(a) Frozen cosine", "0.65", "0.952", "1.000", "1.000"],
            ["(b) Fine-tuned cosine", "0.40", "1.000", "1.000", "1.000"],
            ["(c) Fine-tuned + re-rank (deployed)", "0.90", "0.947", "0.900", "0.867"],
        ],
        col_widths=[2.6, 1.2, 1.1, 1.2, 1.6],
    )
    _p(
        doc,
        "Deployed system on held-out test: precision 1.00, recall 0.90, F1 0.95. "
        "Confusion: TP 9, FP 0, FN 1, TN 10 (zero false positives on this small test draw). "
        "DoD (mechanics): PASS. Chosen operating threshold for the deployed re-ranker: 0.90.",
    )

    _heading(doc, "6.1 Human routing audit", 2)
    _p(
        doc,
        "A 50-row stratified routing audit (47 rows scored) compared model routing to "
        "expert judgment:",
    )
    _add_table(
        doc,
        ["Bucket", "Rows", "Agreement with reviewer"],
        [
            ["Auto (use directly)", "23", "100%"],
            ["Spot (spot-check)", "12", "58%"],
            ["Review (human eyes)", "12", "58%"],
            ["Overall", "47", "78.7%"],
        ],
        col_widths=[2.4, 1.4, 3.1],
    )
    _p(
        doc,
        "The auto-accept pile matched the reviewer in every scored case. All disagreements "
        "were in spot/review, which already receive human attention. By kind: paraphrases "
        "100%, non-science examples 70%. After threshold retune (auto_min 0.70 / spot_min 0.15) "
        "the config records 82% agreement on the same audit while still routing llm_gate=0 "
        "rows into review.",
    )

    # ------------------------------------------------------------------ 7
    _heading(doc, "7. Human-in-the-loop and quality control", 1)
    _p(
        doc,
        "Generated training rows are never blindly trusted. Three independent signals "
        "must agree before auto-accept; uncertain rows are over-routed to people because "
        "a false positive costs reviewer time. Review decisions live in a sidecar so a "
        "corpus rebuild does not erase work. Deployment workbooks are themselves a review "
        "surface: predicted_label is a hint, score and rationale are the evidence, and "
        "the review column is the human gold used by the aggregation script.",
    )
    _p(
        doc,
        "Aggregation definitions used in reporting: screened utterances are the "
        "adult/child rows at or above the 0.40 ASR floor (the LLM’s actual denominator); "
        "hours (span) is last segment end minus first segment start; hours (talk time) is "
        "the summed duration of speech segments (reference only). Flagged-but-blank rows "
        "are reported as not human-checked and are excluded from verified counts.",
    )

    # ------------------------------------------------------------------ 8
    _heading(doc, "8. Software architecture and repository layout", 1)
    _code(
        doc,
        "config/          JSON configs per step (biencoder, reranker, query, eval,\n"
        "                 confidence, deploy)\n"
        "prompts/         reranker_v1.txt and paraphrased reranker_v1b.txt\n"
        "src/             pipeline modules (steps 0–12) + positives_mining.py + pipeline.py\n"
        "scripts/         writeup, Hadley deck, review bundle, SCI-positive audit,\n"
        "                 human-coded aggregation (Data Aggregation/)\n"
        "tests/           pytest suite (one module per step, plus conftest.py)\n"
        "Human coded/     reviewer-filled Y2 workbooks (analytic sample)\n"
        "data/            git-ignored raw and processed transcripts and parquet\n"
        "models/ cache/ reports/   git-ignored weights, LLM cache, scored outputs",
    )
    _p(
        doc,
        "Pinned runtime (requirements.txt, validated on Python 3.12): pandas 2.3.1, "
        "numpy 1.26.4, pyarrow ≥14, openpyxl ≥3.1, requests, python-dotenv, torch 2.5.1, "
        "transformers 4.49.0, sentence-transformers ≥3.0, einops ≥0.8, datasets ≥2.19,<3, "
        "scikit-learn ≥1.5, pytest ≥8. Secrets live in a root .env that is never committed. "
        "Exploratory notebooks and internal function-reference markdown are excluded from "
        "the public tree (.gitignore lists notebooks/, DOCUMENTATION.md, FUNCTION_REFERENCE.md).",
    )

    _heading(doc, "8.1 Testing", 2)
    _p(
        doc,
        "The suite under tests/ covers schema and DoD assertions for the data loader, "
        "subtype assignment, negative mining, augmentation, baseline embeddings, confidence "
        "routing, review apply/export, splits (including leakage of anchor families), "
        "positives mining, bi-encoder triple construction, re-ranker parsing and metrics, "
        "query-time classify plumbing, evaluation helpers, the LLM client cache key, and "
        "Y2 ingest/workbook writers. Tests that need the labeled xlsx skip when that file "
        "is absent. Offline --stub paths exist on Steps 9–12 so CI can exercise plumbing "
        "with zero network calls.",
    )

    _heading(doc, "8.2 Configuration reference", 2)
    _add_table(
        doc,
        ["File", "Controls"],
        [
            ["config/biencoder.json", "Base model, prefixes, train/eval hyperparameters, artifact paths."],
            ["config/reranker.json", "Prompt files, temperature, max_tokens, AUROC/Spearman gates, smoke/full budgets."],
            ["config/query.json", "top_k=20, max aggregation, score_threshold, degraded cosine threshold, workers, wall-clock cap."],
            ["config/eval.json", "NDCG k=10, threshold grid, hard-informal recall floor 0.80, smoke/full sample sizes."],
            ["config/confidence.json", "Cosine bands, geometric-mean weights, auto_min/spot_min, audit n=50 seed=42."],
            ["config/deploy.json", "Transcripts root, speaker keep-list, confidence_min=0.40, top_k=10, dollar_cap, label thresholds, science_only, Classroom49 suffix."],
        ],
        col_widths=[2.2, 4.7],
    )

    # ------------------------------------------------------------------ 9
    _heading(doc, "9. Limitations and caveats", 1)
    _bullet(doc, "The Y1 seed is small and skewed: 196 labeled utterances, 191 vs. 5 science/non-science, with content and observation dominating and prediction / causal_reasoning / evidence in the single digits. The model is strongest where labels exist and thin on the rare practices.")
    _bullet(doc, "Most non-science training data is synthetic or transcript-mined, not expert-labeled. Hard negatives are the intended remedy but remain a distribution shift relative to Y2 classroom speech.")
    _bullet(doc, "Y1 evaluation F1 is flattered by easy negatives. Cosine-only systems scoring 1.0 on Y1 is not evidence that they outperform the re-ranker in the wild. Hard-informal recall (0.867 for the deployed system) is the realistic figure.")
    _bullet(doc, "Y1 decision thresholds (LLM 0.90, cosine 0.40) are provisional when applied to Y2 and should be retuned after more reviewed Y2 labels exist.")
    _bullet(doc, "ASR and diarization error propagate. Adult/child splits are approximate. Rows below 40% ASR confidence are excluded from the denominator.")
    _bullet(doc, "The re-ranker is budget-capped at deployment; many Y2 utterances are cosine-only. Labels from cosine and from the LLM are not interchangeable.")
    _bullet(doc, "Speaker-attributed rates in the analytic sample should be treated as indicative. Thirty-five flagged rows were never human-checked and are excluded from verified totals.")
    _bullet(doc, "Research code; no license is granted for redistribution unless stated by the authors. Raw transcripts and credentials must stay out of version control.")

    # ------------------------------------------------------------------ 10
    _heading(doc, "10. Glossary", 1)
    _add_table(
        doc,
        ["Term", "Meaning in this project"],
        [
            ["Anchor", "A labeled SCIENCE_TALK utterance in the corpus; the retrieval document."],
            ["Variant / paraphrase", "Register-preserving rewrite of an anchor (pairs.parquet)."],
            ["Hard negative", "A non-science utterance that mimics science syntax or seed terms."],
            ["Anchor bank", "The set of embedded verified positives searched at query time."],
            ["Screened utterance", "Adult/child ASR row at or above 40% confidence; the scoring denominator."],
            ["Flagged", "Model predicted SCIENCE_TALK (LLM and/or cosine threshold)."],
            ["Verified", "Human review = SCIENCE_TALK on a flagged row."],
            ["Degraded", "classify() ran cosine-only because the LLM endpoint failed."],
            ["Hard-informal slice", "Held-out informal-register families used as a recall floor, not as training."],
            ["DoD", "Definition of Done: mechanical checks a step must pass, distinct from saturated accuracy."],
        ],
        col_widths=[2.0, 4.9],
    )

    # ------------------------------------------------------------------ 11
    _heading(doc, "11. Document history", 1)
    _p(
        doc,
        "This technical documentation was generated from the Science talk retrieval "
        "codebase (README, src/, config/, prompts/, tests/, Data Aggregation/, and "
        "Human coded/) on "
        f"{date.today():%B %d, %Y}. Dataset sizes for the Y1 seed follow the "
        "documented corpus (196 / 516 / 7,815) and loader assertions. Dataset sizes for "
        "the Y2 analytic sample are computed from Data Aggregation/output/sampled_transcripts.xlsx "
        "(83 transcripts). The full Y2 raw corpus (~85 folders, ~1.3M utterances) is not "
        "checked into git and is cited from src/deploy_y2_12.py. Evaluation numbers follow "
        "the Step 11 results recorded for the Hadley review.",
    )

    doc.save(output_path)
    return output_path


if __name__ == "__main__":
    path = build_document()
    print(f"Wrote {path}")
