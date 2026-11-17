"""Generate the crossroads research report as a .docx.

Run from the project root:
    python scripts/build_crossroads_report.py
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
OUTPUT_PATH = PROJECT_ROOT / "Science_Talk_Crossroads_Research_Report.docx"

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


def _add_table(doc, headers, rows, col_widths=None, highlight_last=False):
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
            _set_cell_text(
                cell, val, size=9,
                bold=bool(highlight_last and i == len(rows) - 1),
                color=NAVY,
            )
            if highlight_last and i == len(rows) - 1:
                _shade_cell(cell, "D9E8F6")
            elif i % 2 == 1:
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
    para.paragraph_format.line_spacing = 1.15
    run = para.add_run(text)
    _set_run_font(run, size=size, bold=bold, italic=italic, color=NAVY)
    return para


def _rich(doc, parts, *, size=11, space_after=8):
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(space_after)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.line_spacing = 1.15
    for text, bold, italic in parts:
        run = para.add_run(text)
        _set_run_font(run, size=size, bold=bold, italic=italic, color=NAVY)
    return para


def _bullet(doc, text, *, bold_lead=None):
    para = doc.add_paragraph(style="List Bullet")
    para.clear()
    para.paragraph_format.left_indent = Inches(0.3)
    para.paragraph_format.space_after = Pt(3)
    if bold_lead:
        run = para.add_run(bold_lead)
        _set_run_font(run, size=11, bold=True, color=NAVY)
        run = para.add_run(text)
        _set_run_font(run, size=11, color=NAVY)
    else:
        run = para.add_run(text)
        _set_run_font(run, size=11, color=NAVY)
    return para


def _caption(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(10)
    run = para.add_run(text)
    _set_run_font(run, size=9, italic=True, color=GREY)
    return para


def _hanging(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.5)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    _set_run_font(run, size=10, color=NAVY)
    return p


def build_document(output_path: Path = OUTPUT_PATH) -> Path:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = NAVY
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = 1.15

    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)

        footer = section.footer
        footer.is_linked_to_previous = False
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = fp.add_run(
            "Anita Zucker Center  ·  Science-Talk Retrieval  ·  Crossroads report  ·  "
            f"{date.today():%B %Y}"
        )
        _set_run_font(run, size=8, italic=True, color=GREY)

    # ------------------------------------------------------------------ cover
    for _ in range(2):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("CLASSROOM SCIENCE-TALK RETRIEVAL")
    _set_run_font(r, size=22, bold=True, color=NAVY)

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = st.add_run("At a Crossroads")
    _set_run_font(r, size=18, color=BLUE)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run(
        "Evidence review, related literature, and a recommended path\n"
        "for finding science-practice talk in early-childhood transcripts"
    )
    _set_run_font(r, size=12, italic=True, color=GREY)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_before = Pt(28)
    r = meta.add_run(
        "Anita Zucker Center\n"
        "University of Florida\n\n"
        f"Date: {date.today():%B %d, %Y}\n"
        "Sources: Y2 human-checked analytic sample (83 transcripts),\n"
        "error analysis of 7,969 reviewed flags, pipeline configs and prompts,\n"
        "and current educational-NLP / science-talk literature (2019-2025)"
    )
    _set_run_font(r, size=11, color=NAVY)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.paragraph_format.space_before = Pt(24)
    r = note.add_run(
        "This is a decision memo, not a methods paper. It is meant to settle "
        "whether the retrieve-and-rerank design should be patched, or whether "
        "the Y2 labels now justify a different modeling strategy."
    )
    _set_run_font(r, size=11, italic=True, color=GREY)

    doc.add_page_break()

    # ------------------------------------------------------------------ 1
    _heading(doc, "1. The decision in one page", 1)
    _p(
        doc,
        "The retrieve-and-rerank pipeline was the right system for a 196-utterance "
        "seed. It is the wrong primary system now that reviewers have labeled 7,969 "
        "Y2 flags (1,984 confirmed science talk; 5,985 rejects). Year-1 test F1 of "
        "0.95 was a saturated diagnostic on easy negatives. Year-2 precision of 24.9% "
        "is the real operating number, and it is driven by three design choices that "
        "the error analysis and the literature both flag.",
    )
    _bullet(
        doc,
        " Cosine-only rows are promoted to SCIENCE_TALK at a 0.40 similarity floor. "
        "Those rows are 9.6% precise and account for 35% of all false positives. "
        "Standard retrieve-and-rerank practice never treats first-stage similarity as "
        "a final label.",
        bold_lead="Cosine is not a classifier.",
    )
    _bullet(
        doc,
        " The LLM is asked whether a candidate matches a known science-talk "
        "anchor, not whether the utterance itself is science practice. Short deictic "
        "and naming talk shares a surface form with observation and content, so the "
        "pair judge scores it high. Classroom-discourse work instead classifies the "
        "utterance (with neighboring turns) as yes/no against a rubric.",
        bold_lead="The reranker is solving the wrong task.",
    )
    _bullet(
        doc,
        " The seed is 171/196 content, with prediction, causal reasoning, and "
        "evidence in the single digits. The model therefore retrieves object names, "
        "shapes, and colors. Early-childhood science-talk coding has long separated "
        "content labels from inquiry practices; this pipeline currently blends them.",
        bold_lead="The construct is overloaded.",
    )
    _p(
        doc,
        "Two cheap deploy changes (conservative lexical gates; stop writing cosine-only "
        "rows as SCIENCE_TALK) would move precision from 24.9% to about 35% while "
        "keeping 87% of confirmed flags. That is a reviewer-load win, not a modeling "
        "win. The modeling win is to treat the Y2 reviewed set as training data and "
        "move to context-aware utterance classification, which is the literature-standard "
        "approach for classroom discourse (TalkMoves, CLASS utterance tagging, IQA "
        "binary counting). Retrieval should remain as a high-recall first-stage filter "
        "over the 1.3 million raw utterances, not as the decision rule.",
    )
    _p(
        doc,
        "The one measurement that still blocks a claim of net scientific gain is recall: "
        "no unflagged utterances have been sampled. Until 20-30 screened adult and child "
        "rows per classroom are coded, 35% precision is a shorter review sheet, not proof "
        "that more science talk is being recovered.",
        italic=True,
    )

    # ------------------------------------------------------------------ 2
    _heading(doc, "2. What the current system actually does", 1)
    _p(
        doc,
        "Runtime inference is a two-stage retrieve-and-rerank design, hosted on "
        "University of Florida models. Every kept adult or child utterance (ASR "
        "confidence at or above 0.40) is compared to an embedded bank of labeled "
        "science-talk anchors. A large language model then re-ranks only the highest-"
        "cosine candidates that fit a dollar budget.",
    )

    _heading(doc, "2.1 The two tracks", 2)
    _add_table(
        doc,
        ["Track", "Model", "Job at deployment", "Decision rule"],
        [
            [
                "A  bi-encoder",
                "Fine-tuned nomic-embed-text-v1.5 (768-d, local)",
                "Cosine vs L2-normalized anchor bank; scores every utterance at no API cost",
                "SCIENCE_TALK if cosine >= 0.40 and the LLM did not run",
            ],
            [
                "B  pair judge",
                "Zero-shot gpt-oss-120b (UF HiPerGator)",
                "Scores (query, top-k anchor) pairs; k=10 at deploy; max aggregation",
                "SCIENCE_TALK if max pair score >= 0.90",
            ],
            [
                "Augment / subtypes",
                "llama-3.3-70b-instruct",
                "Register paraphrases, hard negatives, cue-less subtype fallback",
                "Training data only",
            ],
        ],
        col_widths=[1.4, 2.0, 2.2, 1.8],
    )
    _caption(
        doc,
        "Table 1. Deployed architecture. Thresholds are from config/deploy.json "
        "(cosine_label_threshold=0.40, llm_label_threshold=0.90) and were tuned on "
        "the saturated Year-1 set.",
    )
    _p(
        doc,
        "The Track B prompt (prompts/reranker_v1.txt) is a pair judge: it asks whether "
        "Phrase A (the classroom utterance) and Phrase B (a retrieved anchor) express "
        "the same scientific practice. The in-context examples are an easy observation "
        "pair, an informal causal pair, and a management hard negative. Aggregation is "
        "max over the top-k anchors, so one lucky match flags the row. That is a "
        "retrieval scoring rule, not a classification rule.",
    )

    _heading(doc, "2.2 How the training corpus was built", 2)
    _p(
        doc,
        "Year 1 starts from data/2026-04-17_science_talk_examples.xlsx (196 labeled "
        "utterances after cleanup: 191 SCIENCE_TALK, 5 NOT_SCIENCE_TALK). Subtypes are "
        "assigned by cue rules, then a keyword scan, then an LLM fallback that never "
        "fires on this corpus because every row has Tier-3 cues. Negatives are mined "
        "three ways (transcript-clean teacher talk without a SCI code; LLM rewrites "
        "that keep syntax but move the topic to management/SEL/literacy; seed-term "
        "nonscience uses). Anchors are paraphrased into LARGE_GROUP, SMALL_GROUP, and "
        "INFORMAL registers. The bi-encoder is trained with MultipleNegativesRankingLoss "
        "on (variant, anchor, hard-negative) triples. A confidence router (auto / spot / "
        "review) gates generated rows into training after a 50-row Hadley audit.",
    )
    _add_table(
        doc,
        ["Training artifact", "Count", "Implication"],
        [
            ["Labeled seed utterances", "196", "Too small for a full classifier in 2025; retrieval was justified then."],
            ["SCIENCE_TALK / NOT in the seed", "191 / 5", "Almost no expert negatives in the seed."],
            ["content subtype (multi-label)", "171", "Anchor bank is mostly science names, not practices."],
            ["observation / causal / prediction / evidence", "68 / 8 / 7 / 4", "The practices the project cares about are the thinnest classes."],
            ["Register paraphrases", "516", "Surface-form diversity, still the same 191 ideas."],
            ["Mined negatives", "7,815", "Mostly easy management/social talk, not Y2 look-alikes."],
        ],
        col_widths=[2.6, 1.6, 3.2],
    )

    _heading(doc, "2.3 Why this design was chosen", 2)
    _p(
        doc,
        "Two constraints shaped it, and both were real. The labeled seed was small, so "
        "training a large classifier from scratch was not an option. False positives are "
        "costly because every wrong flag consumes reviewer time. Dense retrieval "
        "(Karpukhin et al., 2020; Reimers & Gurevych, 2019) plus a second-stage judge "
        "is the standard answer to both constraints in open-domain search. Nomic Embed "
        "v1.5 is a competent 768-d encoder with task prefixes. gpt-oss-120b is a "
        "reasoning model that can emit a short rationale. Human review sheets keep "
        "experts as the final authority. None of that is a mistake. The mistake is "
        "continuing to treat Year-1 operating points as Year-2 truth after 7,969 "
        "human labels exist.",
    )

    # ------------------------------------------------------------------ 3
    _heading(doc, "3. What the data actually say", 1)
    _p(
        doc,
        "There are three datasets. They must not be mixed. Year 1 is the training seed. "
        "Year 2 raw transcripts are the deployment corpus (~85 classroom-session "
        "folders, on the order of 1.3 million utterances). The human-checked analytic "
        "sample is a reviewed subset of Year-2 flags from 16 classrooms and 83 "
        "classroom-observations (October 2025 through April 2026).",
    )

    _heading(doc, "3.1 Year-1 evaluation is saturated and not the operating number", 2)
    _add_table(
        doc,
        ["System", "Threshold", "Test F1", "Test recall", "Hard-informal recall"],
        [
            ["(a) Frozen nomic cosine", "0.65", "0.952", "1.000", "1.000"],
            ["(b) Fine-tuned cosine", "0.40", "1.000", "1.000", "1.000"],
            ["(c) Fine-tuned + LLM rerank (deployed)", "0.90", "0.947", "0.900", "0.867"],
        ],
        col_widths=[2.8, 1.1, 1.1, 1.2, 1.7],
    )
    _caption(
        doc,
        "Table 2. Step 11 ablation on the Year-1 held-out set. Confusion for the "
        "deployed system: TP 9, FP 0, FN 1, TN 10. The technical documentation "
        "already flags this as a saturated diagnostic.",
    )
    _p(
        doc,
        "Fine-tuned cosine scoring a perfect 1.0, and the LLM reranker scoring lower "
        "F1 than cosine, is the opposite of what a retrieve-and-rerank paper would "
        "want to see. It means Year-1 negatives (mostly management and social talk) "
        "are linearly separable in embedding space. The hard-informal slice recall of "
        "0.867 is the only Year-1 number that predicted Year-2 trouble. Thresholds "
        "taken from this table (LLM 0.90, cosine 0.40) were then written into "
        "config/deploy.json and applied to classroom speech that looks nothing like "
        "the seed.",
    )

    _heading(doc, "3.2 Year-2 analytic sample: science talk is rare; flags are not", 2)
    _add_table(
        doc,
        ["Measure", "Value"],
        [
            ["Classrooms / observations / unique children", "16 / 83 / 26"],
            ["Audio hours (recording span)", "289.23 h  (mean 3.48, SD 0.60)"],
            ["Screened utterances (adult/child, ASR >= 0.40)", "174,979"],
            ["  adult / child screened", "96,747 / 78,232"],
            ["Model SCIENCE_TALK flags", "8,004  (4.58% of screened)"],
            ["Human-reviewed flags", "7,969  (35 left blank)"],
            ["Verified science talk (true positives)", "1,984  (24.9% of reviewed flags)"],
            ["Rejected flags (false positives)", "5,985"],
            ["Verified as share of screened speech", "1.13% overall; mean 1.35% per transcript (0-6.75%)"],
            ["Verified science talk per audio hour", "mean 7.06  (SD 6.70, min 0, max 33.23)"],
            ["Adult vs child verification rate of flags", "mean 31.4% adult / 20.7% child"],
            ["Adult vs child verified rate of screened", "1.45% adult / 0.74% child"],
            ["Precision of flags, per transcript", "mean 27.4%, SD 20.6, range 0-77.6%"],
            ["Flags per transcript", "mean 96.4, range 12-357"],
        ],
        col_widths=[3.6, 3.8],
    )
    _caption(
        doc,
        "Table 3. Human-checked analytic sample from Data Aggregation/output/"
        "sampled_transcripts.xlsx and error_analysis/output/analysis_summary.json. "
        "Adult/child is automatic diarization, not gold.",
    )
    _p(
        doc,
        "Three facts follow. First, verified science talk really is rare: about one "
        "utterance in 88, or seven per recorded hour, with classrooms that have none. "
        "That matches independent preschool inquiry coding in which causal talk is "
        "on the order of 2% of children's informational statements and well under 1% "
        "of teachers' (Kurkul, Castelino, Aguiar, & Corriveau, 2021). Second, the "
        "first pass is deliberately broad: about 1 in 22 screened utterances is "
        "flagged, and reviewers confirm about 1 in 4 flags. Third, precision is not "
        "stable across rooms. A mean of 27% with a standard deviation of 21 points "
        "and a range from 0% to 78% means a single global threshold is a poor fit. "
        "Classroom 21 alone contributed 119 of the 400 coded false-positive sample "
        "because it produced so many flags.",
    )

    _heading(doc, "3.3 What cannot be measured yet", 2)
    _p(
        doc,
        "Reviewer sheets contain only model SCIENCE_TALK rows. Precision of the flags "
        "is identified. Recall of science talk in the full transcript is not. False "
        "negatives, subtype calibration on Y2, and the cost of dropping cosine-only "
        "true positives (224 confirmed flags) are all unknown. Any recommendation "
        "that trades precision for recall is provisional until an unflagged sample "
        "exists. That is a standard limitation of positive-only evaluation in retrieval "
        "and in positive-unlabeled learning: you cannot tune a rare-event detector "
        "from the items it already showed you.",
    )

    # ------------------------------------------------------------------ 4
    _heading(doc, "4. Error analysis: what is actually going wrong", 1)
    _p(
        doc,
        "The August 2026 error analysis (error_analysis/REPORT.md) coded 400 stratified "
        "false positives (100 each of adult-LLM, adult-cosine, child-LLM, child-cosine) "
        "and swept cosine thresholds and lexical gates across all 7,969 reviewed flags. "
        "The sample over-represents cosine (50% of the 400; 29% of all flags) so percents "
        "describe error types, not a census of every reject.",
    )

    _heading(doc, "4.1 Three error types, plus a definition problem", 2)
    _add_table(
        doc,
        ["Label", "n of 400", "%", "Adult", "Child", "Cosine", "LLM"],
        [
            ["deictic_attention  (Look. / I saw that.)", "86", "21.5", "27", "59", "34", "52"],
            ["classroom_management", "70", "17.5", "38", "32", "53", "17"],
            ["object_naming", "64", "16.0", "32", "32", "28", "36"],
            ["borderline_science  (reviewer no, reasonably yes)", "44", "11.0", "21", "23", "13", "31"],
            ["shape_color", "38", "9.5", "28", "10", "9", "29"],
            ["counting_math", "37", "9.3", "25", "12", "22", "15"],
            ["other", "26", "6.5", "14", "12", "12", "14"],
            ["social_affect", "15", "3.8", "4", "11", "12", "3"],
            ["play_materials", "12", "3.0", "7", "5", "10", "2"],
            ["literacy", "8", "2.0", "4", "4", "7", "1"],
        ],
        col_widths=[2.6, 0.8, 0.7, 0.8, 0.8, 0.8, 0.7],
    )
    _caption(doc, "Table 4. False-positive taxonomy, 400 stratified rejects.")
    _p(
        doc,
        "Deictic attention, classroom management, and object naming are 55% of the "
        "coded sample. Child false positives are deictic (59 of 200 child rejects vs "
        "27 of 200 adult). Cosine false positives are management and materials (53 of "
        "200 cosine vs 17 of 200 LLM). LLM false positives are deictic, names, and "
        "shapes: the reranker is not confused by sit down; it is confused by Look., "
        "Circle., and one-word object labels that share a register with science "
        "observation. About 1 in 9 rejects is borderline science (How does that work?, "
        "It's getting bigger now., What's going to happen now?). Some of the error "
        "is label noise or a thin operational definition, not a broken retriever. "
        "Do not dump all 5,985 rejects into the negative set.",
    )

    _heading(doc, "4.2 Cosine-only flags should not be SCIENCE_TALK", 2)
    _add_table(
        doc,
        ["Path", "Reviewed flags", "TP", "FP", "Precision"],
        [
            ["Cosine fallback (no LLM)", "2,327", "224", "2,103", "9.6%"],
            ["LLM rerank", "5,642", "1,760", "3,882", "31.2%"],
            ["Combined (current deploy)", "7,969", "1,984", "5,985", "24.9%"],
        ],
        col_widths=[2.4, 1.5, 1.0, 1.1, 1.2],
        highlight_last=True,
    )
    _p(
        doc,
        "Cosine is 29% of reviewed flags and 35% of all false positives. Every cosine-"
        "only flag sits in 0.40-0.48. Raising the floor inside that band does not find "
        "a clean cut: 0.42-0.44 is worse (6.4%) than 0.40-0.42 (11.1%). Dropping cosine-"
        "only SCIENCE_TALK labels moves precision from 24.9% to 31.2% and costs 224 "
        "confirmed flags (11.3% of all confirmed science). Those 224 are longer than "
        "typical false positives (mean 6.8 words, median 5); 43 would be kept by the "
        "science-name hold-out (Is it sunny outside?, They sleep all winter long., "
        "Because lemons are actually sour.). There is no useful operating point except "
        "keep 0.40 or stop promoting cosine-only rows.",
    )

    _heading(doc, "4.3 Cheap gates work on look / count / shape; a length gate does not", 2)
    _add_table(
        doc,
        ["Gate", "Flags removed", "TP lost", "FP removed", "Precision of removed", "Remaining precision"],
        [
            ["Exact look-family", "600", "31", "569", "5.2%", "26.5%"],
            ["Exact count / how many", "158", "1", "157", "0.6%", "25.4%"],
            ["Exact shape or color", "148", "2", "146", "1.4%", "25.3%"],
            ["Conservative (look U count U shape)", "906", "34", "872", "3.8%", "27.6%"],
            ["1-2 words (lexicon hold-out on)", "1,423", "100", "1,323", "7.0%", "28.8%"],
            ["Drop cosine only", "2,327", "224", "2,103", "9.6%", "31.2%"],
            ["Drop cosine U conservative gate", "3,038", "257", "2,781", "8.5%", "35.0%"],
            ["Drop cosine U all gates incl. short", "3,614", "346", "3,268", "9.6%", "37.6%"],
        ],
        col_widths=[2.3, 1.1, 0.8, 1.0, 1.2, 1.3],
        highlight_last=False,
    )
    _caption(doc, "Table 5. Simulated policies on the 7,969 reviewed flags. Remaining precision is of the review sheet, not of the full transcript.")
    _p(
        doc,
        "Count-only is almost free (1 true positive lost). The 128-word science-name "
        "lexicon saved 144 short confirmed names (Iguana., Caterpillar., Jellyfish.) "
        "from a naive length gate; 100 confirmed short names still slip through "
        "(What's hibernation?, A bumblebee., Non-living thing!, chasing prey.). Do not "
        "ship a 1-2 word gate until those 100 are coded into the hold-out. Combined, "
        "dropping cosine-only labels and applying the conservative gate takes precision "
        "to 35.0% and still keeps 1,727 of 1,984 confirmed flags (87%).",
    )

    # ------------------------------------------------------------------ 5
    _heading(doc, "5. Related literature: what other approaches actually do", 1)
    _p(
        doc,
        "The relevant literature is not one field. Science-talk coding in early childhood "
        "tells us what the construct is. Classroom-discourse NLP tells us how people "
        "now label utterances at scale. Information retrieval tells us how retrieve-and-"
        "rerank is supposed to be used. LLM-as-judge work tells us how to prompt a "
        "second stage so it classifies rather than matches. The sections below take "
        "each in turn and map it onto this pipeline.",
    )

    _heading(doc, "5.1 Early-childhood science talk is a practice, in context", 2)
    _p(
        doc,
        "Museum and classroom microanalysis has treated science talk as inquiry about "
        "the world, not as the presence of a science word. Haden, Jant, Hoffman, Marcus, "
        "Geddes, and Gaskins (2014) showed that supporting family STEM conversations in "
        "a children's museum changes what children later remember. Booth, Shavlik, and "
        "Haden (2020) found that parents' causal talk, and especially invitations for "
        "the child to explain, predict children's causal stance and scientific literacy. "
        "Kurkul et al. (2021) coded preschool teacher-child inquiry at the utterance "
        "for delivery (question vs statement) and content (causal vs fact-based/"
        "procedural vs scaffolding), and reported that causal statements were rare: "
        "about 2% of children's informational statements and 0.37% of teachers'. "
        "NGSS-aligned preschool instruments (SCIIENCE, SciEPOP) code science and "
        "engineering practices as episode-level engagement (asking questions, planning "
        "investigations, analyzing data, explaining), not as isolated content nouns "
        "(Kaderavek and colleagues; NGSS Lead States, 2013).",
    )
    _p(
        doc,
        "Two implications are direct. First, a content-heavy seed (plants, animals, "
        "shapes, materials) will retrieve naming. Naming can be science content, but "
        "it is not, by itself, observing, predicting, or explaining. The 16% object-"
        "naming and 9.5% shape/color false positives are the predictable result of "
        "an anchor bank that is 171/196 content. Second, Look. is only observation "
        "when it is about a phenomenon. The ECE literature codes deixis and attention "
        "gets as scaffolding or non-informational unless they sit inside an inquiry "
        "exchange. An isolated-utterance retriever cannot make that cut.",
    )

    _heading(doc, "5.2 Classroom-discourse NLP classifies utterances with neighbors, not against a seed bank", 2)
    _p(
        doc,
        "The systems that currently work on real classroom transcripts do not retrieve "
        "against a few hundred gold examples. They fine-tune a transformer to classify "
        "each utterance, almost always with adjacent turns as context.",
    )
    _bullet(
        doc,
        " Suresh, Jacobs, Lai, and Sumner (2021) and Suresh et al. (2022) fine-tune "
        "RoBERTa on teacher and student talk moves. Adding previous and subsequent "
        "utterances (+/-7) improved F1 by 2.6 points over a single prior turn, to "
        "about 79% F1. The input is the dialogue window, not a nearest neighbor in a "
        "seed bank. A 2023 comparison found fine-tuned BERT beat zero-shot ChatGPT on "
        "student talk moves, though ChatGPT produced readable rationales (Xu and "
        "colleagues, EDM 2023).",
        bold_lead="TalkMoves. ",
    )
    _bullet(
        doc,
        " Jensen, Dale, Donnelly, Stone, Kelly, Godley, and D'Mello (2021) fine-tuned "
        "BERT on 16,977 automatically transcribed, expert-coded classroom utterances. "
        "At 25% of that data (~4,200 utterances) rare categories were near chance; "
        "performance rose through 8,500-17,000. That is the scale of a mature discourse "
        "classifier. This project now has 1,984 verified positives and 5,985 reviewed "
        "negatives: smaller than TalkMoves, but no longer a 196-row few-shot problem.",
        bold_lead="BERT on 17k classroom utterances. ",
    )
    _bullet(
        doc,
        " Whitehill and LoCasale-Crouch (2024) is the closest ECE analogue: Whisper "
        "transcripts of toddler and pre-K classrooms, then Llama 2 and/or bag-of-words "
        "to tag utterances for CLASS Instructional Support, aggregated to a 15-minute "
        "score (Pearson R up to 0.48, approaching human IRR of 0.55). Utterance-level "
        "accuracy still lagged humans. Hybrid LLM plus lexical features beat either "
        "alone. That independently confirms that Look./management lexicons plus an LLM "
        "is a better ECE recipe than embedding similarity alone.",
        bold_lead="CLASS Instructional Support in pre-K. ",
    )
    _bullet(
        doc,
        " Beigman Klebanov, Ilagan, and Mikeska (BEA 2024) combined a fine-tuned "
        "language model with turn-taking structure to score elementary science "
        "discussions (Mystery Powder). Content-only or structure-only models lost; "
        "the hybrid won. This project's flags have speaker labels and timestamps "
        "and currently use neither as features.",
        bold_lead="Science discussion plus interaction structure. ",
    )
    _bullet(
        doc,
        " Demszky and colleagues' uptake detector and M-Powering Teachers (2023) "
        "show that a narrowly defined dialogic move, detected with a fine-tuned "
        "classifier, can be accurate enough to change teacher behavior in RCTs. "
        "Wang and Demszky (BEA 2023) also showed that zero-shot ChatGPT is a weak "
        "classroom observer: suggestions were relevant but 82% pointed at places "
        "where the teacher was already doing the thing. Zero-shot pair judging is "
        "in that family of methods.",
        bold_lead="Uptake and the limits of zero-shot coaching. ",
    )
    _p(
        doc,
        "The pattern is consistent. Successful classroom NLP (1) defines a closed "
        "move set, (2) labels thousands of in-domain utterances, (3) classifies with "
        "dialogue context, and (4) uses the LLM either as a feature or as a few-shot "
        "binary counter, not as a nearest-neighbor judge against a tiny seed.",
    )

    _heading(doc, "5.3 How the LLM stage is supposed to work in 2024-2025", 2)
    _p(
        doc,
        "Tran, Litman, Pierce, Correnti, and Matsumura have the most directly useful "
        "LLM results for this project. For Instructional Quality Assessment scoring, "
        "binary counting (is this turn an instance of dimension X? yes/no) with four "
        "previous turns of history and 10 few-shot examples beat scoring the whole "
        "transcript at once, and beat zero-shot (EDM 2024). In 2025 they showed that "
        "naive retrieval of in-context examples can be worse than a fixed set, because "
        "the retrieved set lacks positives and hard negatives. Re-ranking the retrieved "
        "examples and forcing a label ratio (positive / easy negative / hard negative) "
        "improved every tested dimension. Hard negatives were chosen from educational "
        "knowledge (the near-miss category), not from embedding distance alone.",
    )
    _p(
        doc,
        "That paper is almost a diagnosis of Track B. This pipeline retrieves the "
        "nearest science anchors (all positives) and asks whether the pair matches. "
        "It never shows the model a Look. that is not science, a Circle. that is a "
        "shape lesson, or a Chicken? that is naming. The three in-context examples "
        "in the prompt are frozen and do not include the error types that dominate "
        "Year 2. LLM-as-judge work more generally (Zheng et al., 2023; recent "
        "overconfidence papers) finds that judges are overconfident at the top of "
        "the score range. Here the LLM threshold is 0.90 and most accepted LLM rows "
        "sit at 0.90, so the score is not ranking among candidates; it is a near-"
        "binary stamp. A yes/no science-talk decision with retrieved hard-negative "
        "few-shots is the method the 2024-2025 classroom-LLM literature supports.",
    )

    _heading(doc, "5.4 Retrieve-and-rerank, used as search people use it", 2)
    _p(
        doc,
        "Dense passage retrieval (Karpukhin et al., 2020) and Sentence-BERT "
        "(Reimers & Gurevych, 2019) are the right citations for Track A. The rest of "
        "that literature, however, does not promote first-stage cosine to a document "
        "label. The standard stack is: cheap first-stage retrieval for recall; a "
        "cross-encoder reranker (MiniLM / BGE / DeBERTa trained on MS MARCO or a "
        "domain set) for precision on the top 50-100; optionally an LLM on the last "
        "handful of ambiguous items (Nogueira & Cho, 2019; Formal, Piwowarski, and "
        "colleagues, 2024, comparing cross-encoders with GPT-4 listwise rerankers). "
        "Cross-encoders remain competitive with LLM rerankers out of domain and are "
        "orders of magnitude cheaper. LLM rerankers help on ambiguous queries that "
        "need reasoning, which is exactly the borderline_science slice, not Look. "
        "or Two.",
    )
    _p(
        doc,
        "Hard-negative mining is equally specific. In-batch negatives (this project's "
        "MultipleNegativesRankingLoss) are easy; ANCE (Xiong et al., 2021) mines "
        "global hard negatives from the current index; RocketQA (Qu et al., 2021) "
        "denoises those hard negatives with a cross-encoder so unlabeled positives "
        "are not trained as negatives. The Year-2 cosine false positives are exactly "
        "the global hard negatives ANCE would mine. The borderline_science 11% is "
        "exactly the false-negative contamination RocketQA would drop. Training "
        "another round of nomic on LLM-generated management paraphrases will not "
        "move Year-2 precision, because the LLM path already handles management "
        "(17 of 200 LLM false positives). The remaining errors are deictic, naming, "
        "and shape: the items the current negative miner was not asked to produce.",
    )
    _p(
        doc,
        "Educational tagging has already been framed as retrieval. Hwang, Lee, and "
        "colleagues (Findings of ACL 2023) treat multi-label educational classification "
        "as dense retrieval over a large tag set and add a cross-encoding augmentation "
        "(CEAA) so the bi-encoder learns pair interactions without a second model at "
        "inference. That is a possible upgrade to Track A if the project stays in "
        "retrieval, but it still needs in-domain tags, not 191 Year-1 sentences.",
    )

    _heading(doc, "5.5 Few-shot classification is now on the table", 2)
    _p(
        doc,
        "SetFit (Tunstall et al., 2022) fine-tunes a sentence transformer contrastively "
        "and then trains a classification head, with no prompts, and is competitive "
        "with much larger models at 8-16 examples per class. Hugging Face's own "
        "guidance is that SetFit shines below a few hundred labels per class; above "
        "that, ordinary transformer fine-tuning usually wins. This project now has "
        "1,984 positives. That is past SetFit's few-shot sweet spot and into ordinary "
        "fine-tuning, with the caveat that the labels are not a random sample of "
        "classroom speech: they are a sample of model flags. A classifier trained "
        "only on flags will re-learn the current retriever's biases unless unflagged "
        "negatives are added.",
    )
    _p(
        doc,
        "A 2025 EMNLP comparison of specialised small models versus general LLMs on "
        "text classification found that fine-tuning typically overtakes zero-shot "
        "prompting somewhere between 10 and 500 labeled samples, and overtakes "
        "few-shot in-context learning somewhere between 100 and 2,000, with binary "
        "tasks needing more than multiclass. On that curve, a context-aware classifier "
        "trained on the Year-2 review set is expected to beat the current zero-shot "
        "pair judge. That is the empirical reason to change course, not a fashion "
        "preference for BERT.",
    )

    # ------------------------------------------------------------------ 6
    _heading(doc, "6. Diagnosis: five structural mismatches", 1)
    _p(
        doc,
        "The literature and the error tables converge. The system is not randomly "
        "inaccurate. It is accurate at the task it was trained to do, and that task "
        "is only a cousin of Year-2 science-practice coding.",
    )

    _heading(doc, "6.1 Isolated utterances cannot distinguish deixis from observation", 2)
    _p(
        doc,
        "Look at that. is observation-shaped and attention-shaped. TalkMoves needed "
        "+/-7 turns to tell talk moves apart; Tran et al. needed four previous turns "
        "for IQA binary counting; Whitehill still lagged humans at the utterance "
        "level in pre-K. This pipeline embeds one utterance and compares it to one "
        "anchor. Child false positives being 59/200 deictic is what that architecture "
        "must produce. Adding context is the highest-leverage modeling change that "
        "does not require a new construct.",
    )

    _heading(doc, "6.2 Pair matching is not science-talk classification", 2)
    _p(
        doc,
        "The Track B instruction is: score whether both phrases express the same "
        "scientific practice. A classroom Look. retrieved against Look closely -- "
        "tell me what's happening to the ice cube (the prompt's own easy-positive "
        "example) is a high-scoring pair. Max aggregation then stamps SCIENCE_TALK. "
        "The LLM false-positive profile (deictic, names, shapes) is the pair-match "
        "task succeeding. Switching the prompt to a binary, rubric-grounded, "
        "context-conditioned decision, with few-shot Look./naming/shape rejects, "
        "is what Tran et al. (2024, 2025) found actually works for classroom coding.",
    )

    _heading(doc, "6.3 Content-dominated anchors retrieve naming", 2)
    _p(
        doc,
        "If 171 of 191 positives are content, nearest-neighbor search returns animals, "
        "shapes, materials, and weather words. Some of those are true science content "
        "(Iguana., Jellyfish., hibernation). Many are not (The box!, Chicken?, "
        "What color?). The operational definition in the technical documentation "
        "already says science talk is inquiry and/or science content terms, and is "
        "not routines, management, SEL, or literacy. The and/or is doing too much "
        "work. Until content-only naming is either (a) a separate code with its own "
        "precision target or (b) required to sit inside an inquiry frame, reviewers "
        "and the model will keep fighting over Chicken?.",
    )

    _heading(doc, "6.4 Year-1 negatives taught the wrong hard cases", 2)
    _p(
        doc,
        "Synthetic hard negatives that rewrite a science sentence into line up at the "
        "door train the model to reject management. The LLM path already does that "
        "(17 management errors in 200 LLM rejects). Cosine still promotes management "
        "because the 0.40 floor is low and the first stage never sees a query-document "
        "pair. The Year-2 hard cases are deictic observation, object naming, shape/"
        "color lessons, and borderline heat/living/what-happened talk. Those are "
        "sitting in Human coded/*.xlsx and have not been used as training signal, "
        "except insofar as the error-analysis memo warned not to use all of them.",
    )

    _heading(doc, "6.5 The first stage is being asked to decide", 2)
    _p(
        doc,
        "Full LLM rerank of ~1.3 million utterances was correctly judged too expensive "
        "(on the order of $2,600 at the configured per-call cost). The workaround, "
        "writing cosine-only SCIENCE_TALK at 0.40, converted a recall filter into a "
        "precision disaster: 9.6% precise, 2,103 extra false positives. Information-"
        "retrieval practice is: keep cosine as a ranking feature; spend the LLM (or a "
        "local cross-encoder) on the shortlist; never emit a first-stage score as the "
        "class label. A local MiniLM/BGE cross-encoder can score tens of thousands of "
        "pairs on CPU; it is the missing middle stage between free cosine and billed "
        "gpt-oss-120b.",
    )

    # ------------------------------------------------------------------ 7
    _heading(doc, "7. Option space", 1)
    _p(
        doc,
        "There are four honest options. Patching thresholds without changing the task "
        "is not one of them; the cosine sweep already showed there is no clean cut "
        "inside 0.40-0.48.",
    )

    _add_table(
        doc,
        ["Option", "What it is", "Fits the data?", "Fits the literature?", "Cost / risk"],
        [
            [
                "A. Patch only",
                "Conservative look/count/shape gates; stop cosine-only labels; park cosine on a side sheet",
                "Yes for precision (24.9% to ~35%). Recall unknown.",
                "Yes as IR hygiene. Does not fix pair-judge or isolation.",
                "Days of engineering. 224 flagged TPs leave the analytic sheet.",
            ],
            [
                "B. Stay retrieval, fix the stages",
                "Keep Track A; add context; replace pair-judge with binary few-shot LLM; add local cross-encoder; ANCE-style Y2 hard negatives (drop borderline)",
                "Yes: attacks deictic/naming/shape and cosine promotion.",
                "Yes: DPR + cross-encoder + LLM-on-ambiguous-tail; Tran ICL ratios.",
                "Weeks. Still limited by a 191-idea anchor bank.",
            ],
            [
                "C. Switch to in-domain classification",
                "Fine-tune RoBERTa/DeBERTa or a classification head on nomic using Y2 TP + cleaned FP + unflagged negatives, with +/-k context",
                "Yes: 1,984 / 5,985 is now a classification set. Must add unflagged negatives or it relearns the retriever.",
                "Yes: TalkMoves, Jensen et al., Whitehill hybrid. Standard 2021-2025 approach.",
                "Weeks after the recall sample. Needs classroom-grouped splits (Classroom 21 will leak otherwise).",
            ],
            [
                "D. Dual-code and dual-model",
                "Split science-practice vs science-content. Retrieve/classify them separately. Report both.",
                "Yes: matches the FP taxonomy and the seed skew.",
                "Yes: NGSS practices vs content; Kurkul causal vs fact-based.",
                "Requires a codebook revision and some recoding of the 1,984 TPs.",
            ],
        ],
        col_widths=[1.3, 1.8, 1.5, 1.6, 1.5],
    )
    _caption(doc, "Table 6. Strategic options. A is a deploy patch. B and C are modeling strategies. D is a construct decision that B and C both need.")
    _p(
        doc,
        "The recommended combination is A now, D as a codebook decision this month, "
        "a recall sample in parallel, then C as the primary scorer with B's first "
        "stage kept as a high-recall filter. That is a hybrid, not a rewrite: cosine "
        "still scans 1.3 million rows for free; a local classifier or cross-encoder "
        "scores the shortlist; gpt-oss-120b is reserved for borderline items and for "
        "rationales on the review sheet. Retrieval does not disappear. It stops being "
        "the decision.",
    )

    # ------------------------------------------------------------------ 8
    _heading(doc, "8. Recommended path", 1)

    _heading(doc, "8.1 Ship this week (Option A)", 2)
    _bullet(doc, "Before Track B and before writing the reviewer sheet, drop exact look-family, exact count/how-many, and exact shape/color unless the utterance is in the science-name hold-out. Expected effect on this sample: -872 FPs, -34 TPs, precision 24.9% to 27.6%. Count-only is almost free.")
    _bullet(doc, "Do not promote cosine-only rows to SCIENCE_TALK at 0.40. Require LLM (or a new local scorer) confirmation, or park them on a separate cosine-only, not for the analytic sample sheet. Expected effect: 24.9% to 31.2%. Combined with the conservative gate: 35.0%, keeping 1,727 of 1,984 confirmed flags.")
    _bullet(doc, "Do not ship the 1-2 word gate yet. Code the 100 short confirmed names it would still delete, grow the lexicon, then revisit.")
    _bullet(doc, "Do not train on all 5,985 FPs as hard negatives. Drop borderline_science (~11% of the coded sample). Prefer cosine-path management, look, count, and shape rejects.")

    _heading(doc, "8.2 Settle the construct (Option D, this month)", 2)
    _p(
        doc,
        "Decide, in writing, whether the analytic target is (1) science-practice talk "
        "(observe, predict, explain, evidence), (2) science-content naming, or (3) "
        "both, reported separately. The current and/or definition produces a model "
        "that is rewarded for retrieving Iguana. and punished for retrieving Chicken? "
        "with no structural difference between those strings. A dual code also lets "
        "the 11% borderline slice be adjudicated against a rule (heat, living/dead, "
        "what happened, science word in an inquiry frame) instead of reviewer taste. "
        "Double-code a few hundred TPs and the 44 borderline examples; compute kappa; "
        "then freeze the codebook before the next training round.",
    )

    _heading(doc, "8.3 Measure recall before claiming a net gain", 2)
    _p(
        doc,
        "Draw, per classroom-observation, a stratified unflagged sample: 15-20 adult "
        "and 15-20 child screened rows that the model did not label SCIENCE_TALK, "
        "spread across cosine bands (below 0.30, 0.30-0.40) and across the session. "
        "Code them with the same review column. That sample identifies false negatives "
        "and lets you estimate recall with a Horvitz-Thompson-style correction "
        "(flags are sampled at rate 1; unflagged at a known rate). Until that exists, "
        "dropping 224 cosine true positives is a precision win of unknown recall cost. "
        "Twenty to thirty rows per transcript is what the error-analysis memo already "
        "asked for; 83 transcripts times ~40 rows is about 3,300 extra codes, which is "
        "less work than the 7,969 already done and is the only way to know whether "
        "the detector is missing the rare practices the seed barely contains.",
    )

    _heading(doc, "8.4 Rebuild the scorer on Year-2 labels (Options C + B)", 2)
    _p(
        doc,
        "Once unflagged negatives and a frozen codebook exist, train a context-aware "
        "utterance classifier. This is the TalkMoves / Jensen / Whitehill pattern, "
        "adapted to a rare binary event.",
    )
    _bullet(doc, "Input: target utterance plus previous and next 2-4 turns, with speaker tags (adult/child). Truncate each turn. This is the Suresh et al. (2022) representation at a smaller window, which is appropriate for short preschool turns.")
    _bullet(doc, "Labels: SCIENCE_PRACTICE and/or SCIENCE_CONTENT per the codebook decision; drop or separately model borderline until adjudicated. Split by classroom, not by utterance, so Classroom 21 cannot dominate both train and test.")
    _bullet(doc, "Model, in increasing ambition: (1) a logistic or SetFit head on the existing nomic encoder, as a cheap baseline; (2) fine-tuned RoBERTa or DeBERTa-v3 on the windowed input, the TalkMoves recipe; (3) keep nomic as a first-stage filter so the classifier is only run on a high-recall shortlist if 1.3 million rows are still too many.")
    _bullet(doc, "Replace Track B's pair prompt with a binary science-talk prompt that includes four prior turns and a retrieved few-shot set with a forced mix of confirmed practices, easy management, and hard deictic/naming/shape negatives (Tran et al., 2025). Use gpt-oss-120b on the ambiguous tail and for the rationale column reviewers already like.")
    _bullet(doc, "Insert a local cross-encoder (for example a MiniLM or BGE reranker, optionally fine-tuned on Y2 pairs) between cosine and the LLM so that cosine-only SCIENCE_TALK never happens again. Formal et al. (2024) found traditional cross-encoders remain competitive with LLM rerankers out of domain at a fraction of the cost.")
    _bullet(doc, "If retrieval is retained as first stage, mine hard negatives from Y2 cosine false positives (ANCE), and denoise with the borderline drop (RocketQA). Stop generating management paraphrases as the main negative diet.")

    _heading(doc, "8.5 What better looks like, numerically", 2)
    _p(
        doc,
        "Do not target Year-1 F1. Target Year-2 operating points that match the "
        "project's use case: a reviewer-ready shortlist.",
    )
    _add_table(
        doc,
        ["Goal", "Why this number", "How you will know"],
        [
            [
                "Precision of the analytic sheet >= 45-50%",
                "Cuts reviewer load roughly in half vs 24.9% without starving the sample. 35% is the no-retrain floor from gates + dropping cosine.",
                "Same human-coded workbook pipeline on a held-out classroom group.",
            ],
            [
                "Estimated recall of verified-style science practice >= 70% on the unflagged sample",
                "The hard-informal floor was 0.80 on Year 1; 0.70 is more honest once content-only naming is split out.",
                "Horvitz-Thompson estimate from flagged + unflagged codes.",
            ],
            [
                "Child deictic precision materially better than now",
                "59/200 child FPs are Look.-family. Context plus a binary prompt should move this first.",
                "Re-run the 400-style taxonomy on a new FP sample.",
            ],
            [
                "Subtype calibration on practices, not just content",
                "Prediction / causal / evidence were n=7 / 8 / 4 in the seed. If those remain invisible, the system is a content-name detector.",
                "Per-subtype precision among verified Y2 rows after recoding.",
            ],
        ],
        col_widths=[2.2, 2.8, 2.4],
    )

    # ------------------------------------------------------------------ 9
    _heading(doc, "9. What not to do", 1)
    _bullet(doc, "Do not nibble cosine from 0.40 to 0.42 or 0.45. The bin table is non-monotonic. The literature and the sweep agree: first-stage scores are not labels.")
    _bullet(doc, "Do not treat Year-1 test F1 (0.95, zero false positives on 20 items) as evidence that the reranker works. The technical documentation already says this. Year 2 is the test set.")
    _bullet(doc, "Do not fine-tune the bi-encoder again on the same Year-1 triples plus more LLM management paraphrases. That retrains the skill the LLM already has.")
    _bullet(doc, "Do not use all 5,985 rejects as negatives. Eleven percent are plausible science; putting them in the negative class will teach the model to miss heat, living/dead, and what-happened talk, which is the thin practice tail you care about.")
    _bullet(doc, "Do not ship a blanket short-utterance gate. One-word science names are real true positives in this corpus (Observe!, Carnivore., Jellyfish.).")
    _bullet(doc, "Do not evaluate the next model on randomly split utterances. Split by classroom. Precision already varies 0-78% across transcripts; Classroom 21 will otherwise inflate both train accuracy and test despair.")
    _bullet(doc, "Do not spend the next quarter only on prompt wording for the pair judge. The pair-judge objective is the bug. Few-shot binary classification with hard negatives is the 2025 method.")
    _bullet(doc, "Do not wait for a perfect ASR. Whitehill, Jensen, and TalkMoves all trained on automatic transcripts with nontrivial word error. The 0.40 confidence floor is already conservative. Diarization error is a caveat on adult/child rates, not a reason to freeze the detector.")

    # ------------------------------------------------------------------ 10
    _heading(doc, "10. Suggested 90-day sequence", 1)
    _add_table(
        doc,
        ["Window", "Work", "Done when"],
        [
            [
                "Days 1-10",
                "Deploy conservative gates. Stop cosine-only SCIENCE_TALK on the analytic sheet. Write the dual-code (practice vs content) rule in one page.",
                "A new review workbook template exists. Precision on a replay of the 7,969 flags matches Table 5 (~35% if both patches apply).",
            ],
            [
                "Days 1-30 (parallel)",
                "Unflagged recall sample: ~20 adult + ~20 child per transcript, or a power-calculated subset of classrooms if 3,300 rows is too many. Double-code borderline + a practice/content subsample.",
                "A recall estimate with confidence intervals. Kappa on the construct split.",
            ],
            [
                "Days 15-45",
                "Build a Y2 training set: verified TPs, cleaned FPs, unflagged negatives, classroom-grouped splits, windowed context. Train SetFit/nomic-head and a RoBERTa window classifier. Ablate context window and speaker tags.",
                "An ablation table on held-out classrooms: current deploy vs gates-only vs classifier vs classifier+LLM-tail.",
            ],
            [
                "Days 45-70",
                "Replace Track B with binary few-shot + label-ratio examples drawn from the error taxonomy. Add a local cross-encoder middle stage. Keep gpt-oss-120b for rationales and the ambiguous band.",
                "LLM FPs of type Look./Circle./Chicken? drop on a new 200-row FP sample.",
            ],
            [
                "Days 70-90",
                "Rescore a held-out set of classrooms. Re-export review sheets. Decide whether content-only naming belongs on the analytic sheet.",
                "A Year-2 eval report that quotes precision, estimated recall, and per-error-type rates, not Year-1 F1.",
            ],
        ],
        col_widths=[1.3, 3.2, 2.9],
    )

    _p(
        doc,
        "If resources force a single modeling bet after the deploy patch, bet on a "
        "context-aware classifier trained on Year-2 labels (Option C), not on another "
        "round of nomic contrastive training against Year-1 anchors. The seed got the "
        "project off the ground. The 7,969 reviewed flags are the dataset the field "
        "would now use.",
    )

    # ------------------------------------------------------------------ 11
    _heading(doc, "11. Limits of this memo", 1)
    _bullet(doc, "I did not re-run models. Numbers are from analysis_summary.json, sampled_transcripts.xlsx, error_analysis/REPORT.md, config/*.json, and the technical documentation generated from this repo.")
    _bullet(doc, "I did not have access to git-ignored weights, the full 1.3 million-utterance corpus, or reports/eval_metrics.json beyond what the documentation records.")
    _bullet(doc, "Literature was searched in September 2026 across ACL/BEA/EDM/LAK, ECE science-talk journals, and arXiv. It is representative of the current methods, not an exhaustive annotated bibliography.")
    _bullet(doc, "Adult/child splits inherit diarization error. Classroom 21's weight in the FP sample means high-precision rooms are under-described.")
    _bullet(doc, "Recall remains unidentified. Every precision gain in Table 5 is a statement about the review sheet, not about science-talk recovery in the wild.")

    # ------------------------------------------------------------------ 12
    _heading(doc, "12. Bottom line", 1)
    _p(
        doc,
        "The project is not failing because retrieve-and-rerank is an unserious idea. "
        "It is stuck because a design that was correct for 196 expert examples is still "
        "being used as the decision rule after reviewers have labeled nearly 8,000 "
        "in-domain flags, because the decision rule is pair-similarity rather than "
        "science-talk classification, because first-stage cosine is emitted as a label, "
        "and because the seed taught content names while Year-2 errors are deixis, "
        "naming, and shape in isolated preschool turns.",
    )
    _p(
        doc,
        "Related work in 2021-2025 would now do three things this pipeline does not: "
        "classify the utterance with neighboring turns; train on in-domain labels at "
        "the thousand-example scale; and use the LLM as a binary, few-shot, hard-"
        "negative-aware judge (or not at all, except for rationales). The cheap "
        "patches in the error-analysis memo are correct and should ship. They are not "
        "the crossroads. The crossroads is whether Year 2 becomes training data. It "
        "should.",
    )

    # ------------------------------------------------------------------ references
    _heading(doc, "References", 1)
    _p(
        doc,
        "Sources used in the argument above. Project-internal figures are cited from "
        "the repository paths named in the text.",
        italic=True,
    )

    refs = [
        "Beigman Klebanov, B., Ilagan, M. J., & Mikeska, J. N. (2024). Automated evaluation of teacher encouragement of student-to-student interactions in a simulated classroom discussion. In Proceedings of the 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024).",
        "Booth, A. E., Shavlik, M., & Haden, C. A. (2020). Parents' causal talk: Links to children's causal stance and emerging scientific literacy. Developmental Psychology, 56(11), 2055-2064. https://doi.org/10.1037/dev0001108",
        "Demszky, D., Liu, J., Hill, H. C., Jurafsky, D., & Piech, C. (2023). Can automated feedback improve teachers' uptake of student ideas? Evidence from a randomized controlled trial in a large-scale online course. Educational Evaluation and Policy Analysis. https://doi.org/10.3102/01623737231169270",
        "Formal, T., Piwowarski, B., Clinchant, S., and colleagues (2024). A thorough comparison of cross-encoders and LLMs for reranking SPLADE. arXiv:2403.10407.",
        "Gilkerson, J., Richards, J. A., Warren, S. F., Montgomery, J. K., Greenwood, C. R., Kimbrough Oller, D., Hansen, J. H. L., & Paul, T. D. (2017). Mapping the early language environment using all-day recordings and automated analysis. American Journal of Speech-Language Pathology, 26(2), 248-265.",
        "Haden, C. A., Jant, E. A., Hoffman, P. C., Marcus, M., Geddes, J. R., & Gaskins, S. (2014). Supporting family conversations and children's STEM learning in a children's museum. Early Childhood Research Quarterly, 29(3), 333-344.",
        "Hwang, S., and colleagues (2023). Cross encoding as augmentation: Towards effective educational text classification. In Findings of the Association for Computational Linguistics: ACL 2023.",
        "Jensen, E., Dale, M., Donnelly, P. J., Stone, C., Kelly, S., Godley, A., & D'Mello, S. K. (2021). A deep transfer learning approach to modeling teacher discourse in the classroom. In LAK21: 11th International Learning Analytics and Knowledge Conference. https://doi.org/10.1145/3448139.3448168",
        "Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain question answering. In Proceedings of EMNLP 2020 (pp. 6769-6781).",
        "Kurkul, K. E., Castelino, S., Aguiar, N., & Corriveau, K. H. (2021). 'How will you construct a pathway system?': Microanalysis of teacher-child scientific conversations. Journal of Childhood, Education & Society, 2(3). https://doi.org/10.37291/2717638X.202123117",
        "NGSS Lead States. (2013). Next Generation Science Standards: For states, by states. National Academies Press. Appendix F, Science and Engineering Practices.",
        "Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. arXiv:1901.04085.",
        "Qu, Y., Ding, Y., Liu, J., Liu, K., Ren, R., Zhao, W. X., Dong, D., Wu, H., & Wang, H. (2021). RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. In Proceedings of NAACL 2021.",
        "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of EMNLP-IJCNLP 2019 (pp. 3982-3992).",
        "Suresh, A., Jacobs, J., Lai, V., & Sumner, T. (2021). Using transformers to provide teachers with personalized feedback on their classroom discourse: The TalkMoves application. arXiv:2105.07949.",
        "Suresh, A., Jacobs, J., Harty, C., Perkoff, M., Martin, J. H., & Sumner, T. (2022). Fine-tuning transformers with additional context to classify discursive moves in mathematics classrooms. In Proceedings of the 17th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2022).",
        "Tran, N., Litman, D., Pierce, B., Correnti, R., & Matsumura, L. C. (2024). Analyzing large language models for classroom discussion assessment. In Proceedings of the 17th International Conference on Educational Data Mining.",
        "Tran, N., Litman, D., Pierce, B., Correnti, R., & Matsumura, L. C. (2025). Improving in-context learning example retrieval for classroom discussion assessment with re-ranking and label ratio regulation. In Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025).",
        "Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M., & Pereg, O. (2022). Efficient few-shot learning without prompts. arXiv:2209.11055.",
        "Wang, R., & Demszky, D. (2023). Is ChatGPT a good teacher coach? Measuring zero-shot performance for scoring and providing actionable insights on classroom instruction. In Proceedings of BEA 2023.",
        "Whitehill, J., & LoCasale-Crouch, J. (2024). Automated evaluation of classroom instructional support with LLMs and BoWs: Connecting global predictions to specific feedback. arXiv:2310.01132.",
        "Xiong, L., Xiong, C., Li, Y., Tang, K., Liu, J., Bennett, P., Ahmed, J., & Overwijk, A. (2021). Approximate nearest neighbor negative contrastive learning for dense text retrieval (ANCE). In Proceedings of ICLR 2021.",
        "Xu, and colleagues (2023). Can ChatGPT detect student talk moves in classroom discourse? A preliminary comparison with BERT. In Proceedings of EDM 2023 (poster).",
        "Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems.",
    ]
    for ref in refs:
        _hanging(doc, ref)

    doc.save(output_path)
    return output_path


if __name__ == "__main__":
    path = build_document()
    print(f"Wrote {path}")
