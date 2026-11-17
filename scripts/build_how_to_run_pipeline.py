"""Generate Getting Started/HOW_TO_RUN_PIPELINE.docx.

Run from the project root:
    python scripts/build_how_to_run_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "Getting Started" / "HOW_TO_RUN_PIPELINE.docx"

NAVY = RGBColor(0x1A, 0x2B, 0x3C)
BLUE = RGBColor(0x2E, 0x5C, 0x8A)
GREY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


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
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
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
        _set_cell_text(cell, h, bold=True, size=9, color=WHITE)
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


def _rich(doc, parts, *, space_after=8):
    """parts: list of (text, kwargs) where kwargs go to _set_run_font."""
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(space_after)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.line_spacing = 1.12
    for text, kwargs in parts:
        run = para.add_run(text)
        kwargs = {"size": 11, "color": NAVY, **kwargs}
        _set_run_font(run, **kwargs)
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
    para.paragraph_format.left_indent = Inches(0.15)
    run = para.add_run(text)
    _set_run_font(run, name="Consolas", size=9, color=NAVY)
    return para


def _note(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after = Pt(8)
    run = para.add_run("Note. ")
    _set_run_font(run, size=10, bold=True, italic=True, color=BLUE)
    run2 = para.add_run(text)
    _set_run_font(run2, size=10, italic=True, color=GREY)
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

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = title.add_run("Science Talk Retrieval")
    _set_run_font(run, size=22, bold=True, color=NAVY)
    title.paragraph_format.space_after = Pt(2)

    sub = doc.add_paragraph()
    run = sub.add_run("How to run the pipeline, step by step")
    _set_run_font(run, size=14, italic=True, color=BLUE)
    sub.paragraph_format.space_after = Pt(12)

    _p(
        doc,
        "This is the command runbook. It tells you exactly which commands to type, "
        "in which order, and what each command writes. For first-time setup "
        "(clone, virtual environment, .env keys, and where to put data files), "
        "use GETTING_STARTED.docx in this same folder. Come back here once that "
        "setup is done.",
    )
    _p(
        doc,
        "Every command below is run from the project root — the folder that "
        "contains README.md and src/. The virtual environment must be active.",
    )

    _heading(doc, "1. Open the project and activate the environment", 1)
    _p(doc, "Windows (PowerShell):")
    _code(
        doc,
        'cd "Science talk retrieval"\n'
        ".\\.venv\\Scripts\\Activate.ps1",
    )
    _p(doc, "macOS / Linux:")
    _code(doc, "cd \"Science talk retrieval\"\nsource .venv/bin/activate")
    _p(
        doc,
        "Confirm the prompt shows (.venv). If packages are not installed yet:",
    )
    _code(doc, "pip install -r requirements.txt")

    _heading(doc, "2. Confirm inputs are in place", 1)
    _p(
        doc,
        "The repository does not include data, review sheets, or API keys. "
        "Copies of the Year-1 workbooks and review samples live under "
        "Getting Started/Data/ in this folder. Copy them to the paths the "
        "code expects, or pass other paths on the command line.",
    )
    _add_table(
        doc,
        ["What", "Put it here", "Needed for"],
        [
            [
                "Labeled seed workbook",
                "data/2026-04-17_science_talk_examples.xlsx",
                "Step 1",
            ],
            [
                "Hand-coded Y1 transcripts",
                "data/transcripts/Coding Transcripts.xlsx",
                "Step 3 and positives_mining.py",
            ],
            [
                "Filled review sheets (if you already have them)",
                "data/processed/review_samples/",
                "Apply commands in section 5",
            ],
            [
                "Y2 ASR session folders",
                "data/transcripts/Y2_Technology Observation Audio_Transcripts/<ClassroomID_MMDDYY>/",
                "Step 12 only",
            ],
            [
                ".env with UF LLM keys",
                "project root (same folder as README.md)",
                "Real LLM / embedding runs",
            ],
        ],
        col_widths=[2.2, 2.8, 1.6],
    )
    _note(
        doc,
        "Do not run --export over a filled review sheet. That overwrites human "
        "work with a blank template. If the filled files are already in "
        "review_samples/, skip every --export and use the apply commands instead.",
    )

    _heading(doc, "3. Optional: check that code and API work", 1)
    _p(doc, "Tests (no data, no API required):")
    _code(doc, "pytest -q")
    _p(doc, "API smoke test (needs .env):")
    _code(doc, "python src/llm_client_0.py")
    _p(
        doc,
        "A JSON completion and a JSON embedding should print. If this fails, "
        "fix keys and URLs in .env before running Steps 3–5 or 9–12 with real models.",
    )

    _heading(doc, "4. How the steps are organized", 1)
    _p(
        doc,
        "pipeline.py runs only Steps 1–7 (corpus prep). Human review is a "
        "separate set of commands. Steps 8–12 are separate modules. You never "
        "have to run all of 1–7 in one shot; --steps lets you run one or more.",
    )
    _add_table(
        doc,
        ["Step", "Command", "What it does", "Writes"],
        [
            ["0", "python src/llm_client_0.py", "API smoke test", "nothing lasting"],
            [
                "1",
                "python src/pipeline.py --steps 1",
                "Load and normalize the seed workbook",
                "data/processed/corpus.parquet, seed_words.parquet, category_defs.parquet",
            ],
            [
                "2",
                "python src/pipeline.py --steps 2",
                "Assign science-practice sub-types",
                "updates corpus.parquet",
            ],
            [
                "3",
                "python src/pipeline.py --steps 3 --use-llm",
                "Mine NOT_SCIENCE_TALK negatives",
                "negatives.parquet + negatives_review_3.xlsx template",
            ],
            [
                "3b",
                "python src/negatives_3.py --apply-negatives-review ...",
                "Apply the filled negatives review",
                "updates negatives.parquet",
            ],
            [
                "3c",
                "python src/positives_mining.py --use-llm-subtypes",
                "Harvest real SCI positives from coded transcripts",
                "appends to corpus.parquet",
            ],
            [
                "3d",
                "python src/positives_mining.py --apply-not-science ...",
                "Apply the filled not-science review of mined positives",
                "updates corpus.parquet",
            ],
            [
                "4",
                "python src/pipeline.py --steps 4 --use-llm",
                "Register-variant paraphrases of positives",
                "pairs.parquet",
            ],
            [
                "5",
                "python src/pipeline.py --steps 5",
                "Frozen baseline embeddings",
                "baseline_cosine on pairs and negatives",
            ],
            [
                "6a",
                "python src/pipeline.py --steps 6",
                "Confidence scores and auto/spot/review routing",
                "routing columns + routing_audit_template_5.xlsx",
            ],
            [
                "6b",
                "python src/review_6.py --apply ...",
                "Apply filled paraphrase / negative queue",
                "review_decisions.parquet; updates pairs and negatives",
            ],
            [
                "6c",
                "python src/confidence_5.py --score-audit-only --score-audit ...",
                "Score the 50-row routing audit (quality gate only)",
                "routing_audit_scored_5.xlsx, routing_audit_report.json",
            ],
            [
                "7",
                "python src/pipeline.py --steps 7",
                "Stratified train/val/test splits (uses verified rows only)",
                "splits.parquet, hard_informal_slice.parquet",
            ],
            [
                "8",
                "python src/biencoder_8.py",
                "Fine-tune nomic-embed-text-v1.5 and embed the corpus",
                "models/biencoder/, corpus_embeddings.npy, reports/biencoder_eval.json",
            ],
            [
                "9",
                "python src/reranker_9.py --smoke",
                "Calibrate the zero-shot LLM re-ranker",
                "reranker calibration reports under reports/",
            ],
            [
                "10",
                "python src/query_10.py --smoke",
                "Query-time retrieve + rerank (or classify one phrase)",
                "query eval under reports/",
            ],
            [
                "11",
                "python src/evaluate_11.py --smoke",
                "Threshold tuning, ablation, test report",
                "reports/eval_report.md, eval_metrics.json, test_predictions.parquet",
            ],
            [
                "12",
                "python src/deploy_y2_12.py --all --dollar-cap 25",
                "Score Year-2 classroom transcripts",
                "reports/y2_scores/<ClassroomFolder>.xlsx",
            ],
        ],
        col_widths=[0.6, 2.4, 2.0, 2.0],
    )
    _p(
        doc,
        "--use-llm turns on the real Llama 3.3-70b model for Steps 3 and 4. "
        "Step 2 uses a separate flag, --use-llm-step2. On the Year-1 seed that "
        "flag almost never fires, because every row already has cues. Without "
        "--use-llm, Steps 3 and 4 use a deterministic stub (fine for plumbing, "
        "not for a real training set).",
    )

    _heading(doc, "5. Copy-paste run: filled review sheets already exist", 1)
    _p(
        doc,
        "Use this path when negatives_review_3.xlsx, not_science_review.xlsx, "
        "review_queue_6.xlsx, and routing_audit_template_5.xlsx are already "
        "filled and sitting in data/processed/review_samples/. Do not export "
        "new templates.",
    )
    _code(
        doc,
        "# 1. Seed corpus\n"
        "python src/pipeline.py --steps 1 2 --use-llm\n"
        "\n"
        "# 2. Negatives, then apply the filled Step 3 sheet\n"
        "python src/pipeline.py --steps 3 --use-llm\n"
        "python src/negatives_3.py --apply-negatives-review data/processed/review_samples/negatives_review_3.xlsx\n"
        "\n"
        "# 3. Extra SCI positives, then apply the filled not-science sheet\n"
        "python src/positives_mining.py --use-llm-subtypes\n"
        "python src/positives_mining.py --apply-not-science data/processed/review_samples/not_science_review.xlsx\n"
        "\n"
        "# 4. Subtypes, augment, embed, route (rebuild after corpus grew)\n"
        "python src/pipeline.py --steps 2 4 5 6 --use-llm\n"
        "\n"
        "# 5. Apply Step 6 queue, score the routing audit, rebuild splits\n"
        "python src/review_6.py --apply data/processed/review_samples/review_queue_6.xlsx --reviewer \"YourName\"\n"
        "python src/confidence_5.py --score-audit-only --score-audit data/processed/review_samples/routing_audit_template_5.xlsx\n"
        "python src/pipeline.py --steps 7\n"
        "\n"
        "# 6. Train, calibrate, evaluate\n"
        "python src/biencoder_8.py\n"
        "python src/reranker_9.py --smoke\n"
        "python src/query_10.py --smoke\n"
        "python src/evaluate_11.py --smoke",
    )
    _note(
        doc,
        "If you re-run Step 3 or Step 4 from scratch, row IDs can change and "
        "an old filled sheet may no longer match. Apply reviews onto the "
        "processed tables that were current when those sheets were exported, "
        "or keep the existing data/processed/ parquets and only apply.",
    )

    _heading(doc, "6. First-time run: no filled review sheets yet", 1)
    _p(
        doc,
        "If reviewers still need blank templates, run the pipeline until each "
        "template is written, send the xlsx out, wait for it to come back "
        "filled, then apply. Never export again on top of the filled file.",
    )

    _heading(doc, "Step 1 — Load the seed workbook", 2)
    _p(
        doc,
        "Reads data/2026-04-17_science_talk_examples.xlsx (three sheets: "
        "example utterances, seed words, category definitions) and writes "
        "normalized parquet tables.",
    )
    _code(doc, "python src/pipeline.py --steps 1")
    _p(doc, "Other seed file or output folder:")
    _code(
        doc,
        "python src/pipeline.py --steps 1 --xlsx data/your_seed.xlsx --out-dir data/processed",
    )
    _p(
        doc,
        "Success: data/processed/corpus.parquet, seed_words.parquet, and "
        "category_defs.parquet exist.",
    )

    _heading(doc, "Step 2 — Sub-type labels", 2)
    _p(
        doc,
        "Adds subtype, subtype_source, subtype_confidence, and "
        "subtype_prompt_version to corpus.parquet. Rule + keyword first; "
        "LLM only for cue-less rows.",
    )
    _code(doc, "python src/pipeline.py --steps 2")
    _p(doc, "Turn on the real LLM fallback (usually a no-op on Year-1):")
    _code(doc, "python src/pipeline.py --steps 2 --use-llm-step2")

    _heading(doc, "Step 3 — Mine negatives, then review them", 2)
    _p(
        doc,
        "Builds NOT_SCIENCE_TALK rows from (1) non-SCI teacher talk in "
        "Coding Transcripts.xlsx, (2) LLM hard negatives, and (3) seed-word "
        "used in a non-science sense. Also writes a review template.",
    )
    _code(doc, "python src/pipeline.py --steps 3 --use-llm")
    _p(doc, "Useful flags:")
    _bullet(doc, "--no-gate-scoring — skip the LLM gate on generated negatives (~700 calls).")
    _bullet(doc, "--use-llm-negatives-subtype — LLM-label subtypes of negatives (~1,300 cached calls).")
    _bullet(doc, "--transcript-xlsx path.xlsx — coded-transcript workbook if it is not in the default place.")
    _bullet(doc, "--max-transcript-negatives N — cap transcript_clean rows.")
    _p(doc, "A reviewer fills decision / label columns on the template. Then apply:")
    _code(
        doc,
        "python src/negatives_3.py --apply-negatives-review data/processed/review_samples/negatives_review_3.xlsx",
    )
    _p(
        doc,
        "This marks human_verified / human_label on the negative pool and drops "
        "POSSIBLE_LEAK rows. Gate: at least 90% of reviewed rows should be true "
        "negatives (NOT_SCIENCE_TALK). Then continue with mining extra positives.",
    )

    _heading(doc, "Extra positives — mine SCI rows from coded transcripts", 2)
    _p(
        doc,
        "Mirrors Step 3: keeps teacher rows whose content code is SCI, filters "
        "fragments, and appends net-new rows to corpus.parquet with utt_sci_ ids. "
        "This is not inside pipeline.py.",
    )
    _code(
        doc,
        "python src/positives_mining.py --dry-run\n"
        "python src/positives_mining.py --use-llm-subtypes",
    )
    _p(
        doc,
        "Export a not-science review queue if you need a new blank sheet "
        "(skip if not_science_review.xlsx is already filled):",
    )
    _code(doc, "python src/positives_mining.py --export-not-science")
    _p(doc, "Apply the filled sheet. --rebuild reruns Steps 2, 4, 5, 6, and 7 with real LLM calls:")
    _code(
        doc,
        "python src/positives_mining.py --apply-not-science data/processed/review_samples/not_science_review.xlsx --rebuild",
    )
    _note(
        doc,
        "If you still need to apply review_queue_6.xlsx after --rebuild, run "
        "Step 7 again after that apply. Merging SCI rows makes pairs, embeddings, "
        "routing, and splits stale; that is why Steps 2 and 4–6 are rebuilt.",
    )

    _heading(doc, "Step 4 — Register-variant augmentation", 2)
    _p(
        doc,
        "For each positive anchor, generates meaning-preserving paraphrases in "
        "other teacher registers (large group, small group, informal/centers). "
        "Writes pairs.parquet.",
    )
    _code(doc, "python src/pipeline.py --steps 4 --use-llm")
    _p(doc, "More than one variant per register:")
    _code(doc, "python src/pipeline.py --steps 4 --use-llm --n-variants-per-register 2")

    _heading(doc, "Step 5 — Frozen baseline embeddings", 2)
    _p(
        doc,
        "Embeds anchors, variants, and hard negatives with off-the-shelf "
        "nomic-embed-text-v1.5 (via the UF embedding API, cached). Adds "
        "baseline_cosine. This is a neutral sanity signal, not the fine-tuned model.",
    )
    _code(doc, "python src/pipeline.py --steps 5")
    _p(
        doc,
        "Success: pairs.parquet and negatives.parquet have a baseline_cosine column. "
        "Re-runs are cheap because embeddings are cached under cache/.",
    )

    _heading(doc, "Step 6 — Confidence, routing, and human review", 2)
    _p(
        doc,
        "Combines LLM self-score, baseline cosine band, and structural checks "
        "into a confidence in [0, 1]. Routes each generated pair / LLM negative to "
        "auto, spot, or review. Writes a 50-row routing audit template.",
    )
    _code(doc, "python src/pipeline.py --steps 6")
    _p(
        doc,
        "Export a blank review queue only if you do not already have a filled "
        "review_queue_6.xlsx:",
    )
    _code(doc, "python src/review_6.py --export")
    _p(
        doc,
        "The reviewer fills decision (accept | reject | edit), reject_reason "
        "when rejecting, and corrected_text when editing. Then apply and check status:",
    )
    _code(
        doc,
        "python src/review_6.py --apply data/processed/review_samples/review_queue_6.xlsx --reviewer \"YourName\"\n"
        "python src/review_6.py --status",
    )
    _p(
        doc,
        "Score the filled routing audit. This is a quality check only "
        "(target: at least 80% agreement). It does not change training rows:",
    )
    _code(
        doc,
        "python src/confidence_5.py --score-audit-only --score-audit data/processed/review_samples/routing_audit_template_5.xlsx",
    )
    _p(
        doc,
        "Success: data/processed/review_decisions.parquet exists; "
        "--status looks complete; routing_audit_scored_5.xlsx and "
        "routing_audit_report.json are written.",
    )

    _heading(doc, "Step 7 — Train / val / test splits", 2)
    _p(
        doc,
        "Builds splits.parquet. filter_verified drops rejected review rows and "
        "applies edits, so Step 8 never trains on rejected paraphrases. Also "
        "writes hard_informal_slice.parquet (held-out informal-register families).",
    )
    _code(doc, "python src/pipeline.py --steps 7")
    _p(doc, "Always re-run Step 7 after applying review_queue_6.xlsx.")

    _heading(doc, "Step 8 — Fine-tune the bi-encoder", 2)
    _p(
        doc,
        "Needs Steps 1–7 artifacts and applied Step 6 decisions. Downloads "
        "nomic-ai/nomic-embed-text-v1.5 from Hugging Face the first time "
        "(internet required). Later loads the saved model from disk.",
    )
    _code(doc, "python src/biencoder_8.py")
    _p(doc, "Shorter training run, or evaluate a saved model without retraining:")
    _code(
        doc,
        "python src/biencoder_8.py --epochs 1\n"
        "python src/biencoder_8.py --eval-only",
    )
    _p(
        doc,
        "Success: models/biencoder/ exists, data/processed/corpus_embeddings.npy "
        "exists, and reports/biencoder_eval.json shows the frozen baseline vs "
        "fine-tuned retrieval numbers.",
    )

    _heading(doc, "Step 9 — Calibrate the LLM re-ranker", 2)
    _p(
        doc,
        "Zero-shot gpt-oss-120b pair judge. --smoke is the usual real run "
        "(capped cost). --full is expensive. --stub is offline plumbing only.",
    )
    _code(
        doc,
        "python src/reranker_9.py --smoke\n"
        "python src/reranker_9.py --full\n"
        "python src/reranker_9.py --stub",
    )

    _heading(doc, "Step 10 — Query-time classify", 2)
    _p(
        doc,
        "Needs the Step 8 model and corpus embeddings. Retrieve with the "
        "bi-encoder, then re-rank the top-k with the Step 9 LLM.",
    )
    _code(
        doc,
        "python src/query_10.py --smoke\n"
        "python src/query_10.py --full\n"
        "python src/query_10.py --stub\n"
        "python src/query_10.py --phrase \"what do you think will happen to the ice?\"",
    )
    _p(
        doc,
        "If this (or Step 12) says to run the bi-encoder first, models/biencoder/ "
        "or corpus_embeddings.npy is missing.",
    )

    _heading(doc, "Step 11 — Evaluate and tune the threshold", 2)
    _p(
        doc,
        "Tunes the SCIENCE_TALK threshold on dev (never on test), writes the "
        "ablation table (frozen cosine vs fine-tuned cosine vs fine-tuned + rerank), "
        "and dumps per-row test predictions for error analysis.",
    )
    _code(
        doc,
        "python src/evaluate_11.py --smoke\n"
        "python src/evaluate_11.py --full\n"
        "python src/evaluate_11.py --stub",
    )
    _p(
        doc,
        "Success: reports/eval_report.md, reports/eval_metrics.json, and "
        "reports/test_predictions.parquet.",
    )

    _heading(doc, "Step 12 — Score Year-2 transcripts", 2)
    _p(
        doc,
        "Needs a trained Step 8 model and Y2 folders under "
        "data/transcripts/Y2_Technology Observation Audio_Transcripts/. "
        "Each session folder should contain per-child .xlsx files with a column "
        "named Transcription (Confidence). Cosine scores every kept utterance; "
        "the LLM re-ranks only the highest-cosine rows under a dollar cap.",
    )
    _code(
        doc,
        "python src/deploy_y2_12.py --folders Classroom74_110525\n"
        "python src/deploy_y2_12.py --all --dollar-cap 25\n"
        "python src/deploy_y2_12.py --folders Classroom74_110525 --stub",
    )
    _p(
        doc,
        "Success: reports/y2_scores/<ClassroomFolder>.xlsx. After a human fills "
        "the review column, save the returned workbook as "
        "Human coded/Classroom<id>_<MMDDYY>_ebh.xlsx.",
    )
    _p(doc, "Aggregate the reviewed Year-2 analytic sample:")
    _code(doc, 'python "Data Aggregation/aggregate_human_coded.py"')
    _p(
        doc,
        "That writes sampled_transcripts.xlsx and summary Word files under "
        "Data Aggregation/output/.",
    )

    _heading(doc, "7. One-shot corpus prep (Steps 1–7, no review apply)", 1)
    _p(
        doc,
        "If you only want to generate processed tables and blank review "
        "templates, this runs the orchestrator end to end:",
    )
    _code(doc, "python src/pipeline.py --use-llm")
    _p(doc, "Equivalent explicit step list:")
    _code(doc, "python src/pipeline.py --steps 1 2 3 4 5 6 7 --use-llm")
    _p(
        doc,
        "That does not apply filled review sheets and does not run Steps 8–12. "
        "Human review is always a separate command (section 5 or 6).",
    )

    _heading(doc, "8. What success looks like", 1)
    _add_table(
        doc,
        ["Goal", "You should see"],
        [
            ["Tests", "pytest -q passes"],
            ["API", "python src/llm_client_0.py prints two JSON blobs"],
            ["Step 1", "data/processed/corpus.parquet"],
            ["Steps 1–7", "pairs.parquet, negatives.parquet, splits.parquet"],
            ["Step 3 review applied", "negatives.parquet has human_verified / human_label on reviewed rows"],
            ["Step 6 review applied", "review_decisions.parquet; python src/review_6.py --status looks complete"],
            ["Routing audit scored", "routing_audit_scored_5.xlsx and routing_audit_report.json"],
            ["Step 8", "models/biencoder/ and reports/biencoder_eval.json"],
            ["Step 11", "reports/eval_report.md"],
            ["Step 12", "reports/y2_scores/*.xlsx"],
            ["Y2 reviewed sample", "Human coded/Classroom*_ebh.xlsx; aggregation under Data Aggregation/output/"],
        ],
        col_widths=[2.4, 4.2],
    )

    _heading(doc, "9. Common problems while running", 1)
    _bullet(doc, "Missing seed xlsx — copy Getting Started/Data/2026-04-17_science_talk_examples.xlsx to data/ first.")
    _bullet(doc, "Missing API key — create .env. Without it, real LLM and embedding steps fail.")
    _bullet(doc, "Filled review sheet not applied — copying the xlsx into review_samples/ is not enough. Run the matching --apply or --score-audit command.")
    _bullet(doc, "--export overwrote a filled sheet — restore the filled file from backup. Export only when you need a new blank template.")
    _bullet(doc, "Apply says IDs are missing — the filled sheet was built from an older parquet. Apply it to that older data/processed/, or do not regenerate Steps 3/4 before applying.")
    _bullet(doc, "Step 10 or 12 says run biencoder first — you need models/biencoder/ and data/processed/corpus_embeddings.npy.")
    _bullet(doc, "Windows activate fails — use .\\.venv\\Scripts\\Activate.ps1 from the project root.")
    _bullet(doc, "Hung LLM call — the client times out after 180 seconds and caches good replies in cache/.")
    _bullet(doc, "Re-running pipeline.py --steps 3 or 4 after reviews were applied — row IDs may change. Prefer applying onto the matching processed tables.")

    _heading(doc, "10. Related documents in this folder", 1)
    _bullet(doc, "GETTING_STARTED.docx — clone, install, .env, and where every input file goes.")
    _bullet(doc, "src_code_walkthrough.docx — what each source file does.")
    _bullet(doc, "This file (HOW_TO_RUN_PIPELINE.docx) — the exact commands, in order.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    return output_path


if __name__ == "__main__":
    path = build_document()
    print(f"Wrote {path}")
