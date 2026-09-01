"""Build a colleague-facing Word copy of error_analysis/REPORT.md.

Run from the project root:
    python error_analysis/build_report_docx.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "Science_Talk_Error_Analysis_Report.docx"

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
            _set_cell_text(cell, val, size=9, bold=bool(highlight_last and i == len(rows) - 1), color=NAVY)
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


def _p(doc, parts, *, size=11, space_after=8):
    """parts is a string or a list of (text, bold, italic) tuples."""
    para = doc.add_paragraph()
    para.paragraph_format.space_after = Pt(space_after)
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.line_spacing = 1.15
    if isinstance(parts, str):
        parts = [(parts, False, False)]
    for text, bold, italic in parts:
        run = para.add_run(text)
        _set_run_font(run, size=size, bold=bold, italic=italic, color=NAVY)
    return para


def _bullet(doc, parts):
    para = doc.add_paragraph(style="List Bullet")
    para.clear()
    para.paragraph_format.left_indent = Inches(0.3)
    para.paragraph_format.space_after = Pt(3)
    if isinstance(parts, str):
        parts = [(parts, False, False)]
    for text, bold, italic in parts:
        run = para.add_run(text)
        _set_run_font(run, size=11, bold=bold, italic=italic, color=NAVY)
    return para


def _caption(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(10)
    run = para.add_run(text)
    _set_run_font(run, size=9, italic=True, color=GREY)
    return para


def _numbered(doc, parts):
    para = doc.add_paragraph(style="List Number")
    para.clear()
    para.paragraph_format.left_indent = Inches(0.35)
    para.paragraph_format.space_after = Pt(6)
    if isinstance(parts, str):
        parts = [(parts, False, False)]
    for text, bold, italic in parts:
        run = para.add_run(text)
        _set_run_font(run, size=11, bold=bold, italic=italic, color=NAVY)
    return para


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
        footer = section.footer
        footer.is_linked_to_previous = False
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = fp.add_run(
            "Anita Zucker Center  ·  Classroom Science-Talk Retrieval  ·  Error analysis  ·  31 August 2026"
        )
        _set_run_font(run, size=8, italic=True, color=GREY)

    title = doc.add_heading("Error analysis of classroom science-talk retrieval", level=0)
    for run in title.runs:
        run.font.color.rgb = NAVY
        run.font.name = "Calibri"
    _p(doc, "False-positive taxonomy, cosine fallback, and pre-rerank gates", size=13)
    _p(
        doc,
        [
            ("Memo for colleagues  ·  31 August 2026  ·  Y2 human-coded sample", False, True),
        ],
        size=10,
        space_after=12,
    )

    _heading(doc, "In brief")
    _p(
        doc,
        [
            ("Among utterances the model flagged as science talk, human reviewers agreed ", False, False),
            ("24.9%", True, False),
            (" of the time (1,984 of 7,969 reviewed rows across 83 workbooks and 16 classrooms). "
             "That is precision, not accuracy or recall: the review sheets contain only model "
             "SCIENCE_TALK flags.", False, False),
        ],
    )
    _p(doc, "Three analyses on those reviewed flags:")
    _bullet(doc, [
        ("Taxonomy of 400 stratified false positives. ", True, False),
        ("Deictic attention, classroom management, and object naming are 55% of coded rejects. "
         "About 1 in 9 rejects is borderline science and should not be mined as a hard negative.", False, False),
    ])
    _bullet(doc, [
        ("Cosine fallback. ", True, False),
        ("2,327 cosine-only flags sit in 0.40–0.48 and are only 9.6% precise. Raising the "
         "threshold inside that band does not help. Dropping cosine-only SCIENCE_TALK labels "
         "moves precision from 24.9% to 31.2% and costs 224 confirmed flags.", False, False),
    ])
    _bullet(doc, [
        ("Pre-rerank gates. ", True, False),
        ("An exact look / count / shape gate, with a science-name hold-out, cuts 872 false "
         "positives and only 34 true positives (precision 27.6%). Combined with dropping cosine, "
         "precision reaches 35.0% while keeping 87% of confirmed flags.", False, False),
    ])
    _p(
        doc,
        [
            ("Recommended next change: ship the conservative gate now; decide cosine after a "
             "small sample of unflagged utterances. Do not ship a 1–2 word length gate yet.", False, False),
        ],
    )

    _heading(doc, "Shared setup (all three analyses)")
    _p(doc, [
        ("What the model did. ", True, False),
        ("For each classroom utterance it either said SCIENCE_TALK or not. Reviewer sheets include only the SCIENCE_TALK ones.", False, False),
    ], space_after=4)
    _p(doc, [
        ("What the human did. ", True, False),
        ("For each of those flags, a reviewer marked SCIENCE_TALK (agree) or NOT_SCIENCE_TALK (reject).", False, False),
    ], space_after=4)
    _p(doc, [
        ("What that means. ", True, False),
        ("Measurable: of the flags the model showed, how often was the model right? Not measurable: how much real science talk the model never showed.", False, False),
    ])
    _add_table(
        doc,
        ["Term", "Meaning here"],
        [
            ["Flag", "An utterance the model labeled SCIENCE_TALK"],
            ["True positive (TP)", "Human also said SCIENCE_TALK"],
            ["False positive (FP)", "Human said NOT_SCIENCE_TALK"],
            ["Precision", "TP / (TP + FP). Current value: 24.9%"],
            ["Lost TP among flagged", "A confirmed science flag a new rule would drop. Not a miss in the full transcript."],
            ["LLM path", "Flag scored by the large-language-model reranker (usually score = 0.90)"],
            ["Cosine path", "Flag scored by embedding similarity only. No LLM. Score 0.40–0.48."],
        ],
    )
    _add_table(
        doc,
        ["Done", "Not done"],
        [
            ["Precision of current flags: 24.9% (1,984 / 7,969)", "Recall (science talk the model never flagged)"],
            ["Taxonomy of 400 false positives", "Accuracy over all classroom talk"],
            ["Cosine raise / drop, and look / count / shape / short gates", ""],
        ],
    )

    # ------------------------------------------------------------------ A1
    _heading(doc, "Analysis 1. What kinds of mistakes are the false positives?")
    _p(doc, [
        ("Question. ", True, False),
        ("When a human rejects a flag, what kind of talk was it? Pointing, counting, management, a real science maybe, or something else.", False, False),
    ])
    _heading(doc, "How this was done", level=2)
    _numbered(doc, "Start from all 5,985 human rejects.")
    _numbered(doc, "Draw 400 of them — not purely at random.")
    _numbered(doc, "Split into four equal buckets of 100 so each group is represented: adult+LLM, adult+cosine, child+LLM, child+cosine.")
    _numbered(doc, "Inside each bucket, take utterances from as many classrooms as possible.")
    _numbered(doc, "Assign one label per utterance (codebook below).")
    _numbered(doc, "Auto-label first with simple rules (Look. → deictic, Two. → counting, …).")
    _numbered(doc, "Hand-label the 150 the rules could not place.")
    _p(doc, [
        ("How to read the counts. ", True, False),
        ("Percents are of this 400, not of all 5,985. The sample was forced to 50% cosine; in the full data cosine is only 35% of false positives. The 400 shows what error types exist, not a census of every reject.", False, False),
    ], space_after=4)
    _p(doc, [
        ("Limit. ", True, False),
        ("Classroom 21 is 119 of 400 because it produced many false positives. High-precision rooms are thin in the sample.", False, False),
    ])

    _heading(doc, "Codebook", level=2)
    _add_table(
        doc,
        ["Label", "Definition"],
        [
            ["deictic_attention", "Attention-getting or pointing without science content (“Look.” / “I saw that.”)"],
            ["classroom_management", "Directives, transitions, seating, clean-up, “try / ready / please”"],
            ["object_naming", "Everyday object or animal label with no scientific framing (“Chicken?” / “That’s a box.”)"],
            ["shape_color", "Naming a shape or color, or “what color / what shape,” as the point of the utterance"],
            ["counting_math", "Rote numbers or “how many” with no investigation"],
            ["borderline_science", "Reviewer said no, but the utterance could reasonably be science practice"],
            ["play_materials", "Play-Doh, paper, craft, toy play"],
            ["social_affect", "Possession, feelings, social bids"],
            ["literacy", "Letters, writing, names-as-text, storybook read-aloud"],
            ["other", "Residual after rules and hand review (6.5% of the sample)"],
        ],
    )

    _heading(doc, "Sample results", level=2)
    _add_table(
        doc,
        ["Label", "n", "% of 400", "Adult", "Child", "Cosine", "LLM"],
        [
            ["deictic_attention", "86", "21.5", "27", "59", "34", "52"],
            ["classroom_management", "70", "17.5", "38", "32", "53", "17"],
            ["object_naming", "64", "16.0", "32", "32", "28", "36"],
            ["borderline_science", "44", "11.0", "21", "23", "13", "31"],
            ["shape_color", "38", "9.5", "28", "10", "9", "29"],
            ["counting_math", "37", "9.3", "25", "12", "22", "15"],
            ["other", "26", "6.5", "14", "12", "12", "14"],
            ["social_affect", "15", "3.8", "4", "11", "12", "3"],
            ["play_materials", "12", "3.0", "7", "5", "10", "2"],
            ["literacy", "8", "2.0", "4", "4", "7", "1"],
        ],
    )
    _caption(doc, "Table 1. Stratified sample of 400 human-rejected SCIENCE_TALK flags. Source: Human coded workbooks, coded 31 August 2026.")
    _p(
        doc,
        [
            ("Three error types account for 55% of the coded sample: ", False, False),
            ("deictic attention, classroom management, and object naming.", True, False),
        ],
    )

    _heading(doc, "What the splits show", level=2)
    _bullet(doc, [
        ("Child false positives are deictic. ", True, False),
        ("59 of 200 child rejects are deictic_attention vs. 27 of 200 adult. Child talk is short pointing (“Look!” / “Look at that.”).", False, False),
    ])
    _bullet(doc, [
        ("Cosine false positives are management and materials. ", True, False),
        ("53 of 200 cosine rejects are classroom_management vs. 17 of 200 LLM rejects. Play and literacy also concentrate on the cosine path. The 0.40 cosine floor is promoting transition talk into the review sheet.", False, False),
    ])
    _bullet(doc, [
        ("LLM false positives are deictic, names, and shapes. ", True, False),
        ("The reranker is not confused by “sit down”; it is confused by “Look.”, “Circle.”, and one-word object labels that share a register with science observation.", False, False),
    ])
    _bullet(doc, [
        ("About 1 in 9 rejects is borderline. ", True, False),
        ("44 of 400 (11%) could reasonably have been science. Some of the “error” is label noise or a thin operational definition, not a broken retriever. Do not dump all 5,985 false positives into the negative set without dropping this slice.", False, False),
    ])

    _heading(doc, "Examples from the coded sample", level=2)
    _add_table(
        doc,
        ["Label", "Examples"],
        [
            ["deictic_attention", "Look at that. / I saw that. / What do you see?"],
            ["classroom_management", "Addy, have a seat please. / Carry. Carry. Carry. / Okay, put your wet clothes in there."],
            ["object_naming", "The box! / It’s a robot. / Do you have a Christmas tree?"],
            ["shape_color", "What shape is that? / W, circle. / “shape where all points around the outside are the same distance from its” (a circle definition)"],
            ["counting_math", "Two. / Eight. / One, two, three, four. / One. Two. Three. Four. Five."],
            ["borderline_science", "How does that work? / It’s getting bigger now. / It’s a butterfly. / What’s going to happen now?"],
            ["play_materials", "Play-Doh is full. / I’m going to throw a ball."],
            ["literacy", "Please, um, write your name on that side. / storybook Mother Goat put the rocks…"],
        ],
    )

    # ------------------------------------------------------------------ A2
    _heading(doc, "Analysis 2. Raise the cosine cutoff, or drop cosine-only flags?")
    _p(doc, [
        ("Question. ", True, False),
        ("Some flags never went through the LLM. They were labeled SCIENCE_TALK because embedding similarity (cosine) was at least 0.40. Are those flags any good? If 0.40 is raised, or those flags are dropped, what happens to precision?", False, False),
    ])
    _heading(doc, "How this was done", level=2)
    _numbered(doc, "Split the 7,969 reviewed flags into two groups: LLM vs cosine-only.")
    _numbered(doc, "Compute precision in each group (Table 2).")
    _numbered(doc, "For cosine-only flags only, bin the cosine score (0.40–0.42, 0.42–0.44, …) and compute precision in each bin (Table 3).")
    _numbered(doc, "Simulate a new rule: keep every LLM flag. Keep a cosine flag only if its cosine is at least t. Repeat for t = 0.40, 0.42, …, 0.50 (Table 4).")
    _numbered(doc, "At t = 0.50, no cosine flag survives (all of them are below 0.50). That row is “drop cosine-only SCIENCE_TALK.”")
    _p(doc, [
        ("How to read Table 4. ", True, False),
        ("Remaining flags = what would still appear on the review sheet. Remaining TP = confirmed science that would still be kept. Lost TP among flagged = confirmed science that would be dropped. Still not full-transcript recall.", False, False),
    ], space_after=4)
    _p(doc, [
        ("Result (short). ", True, False),
        ("Cosine-only flags are 9.6% precise. All of them sit in 0.40–0.48. Raising 0.40 a little does not find a clean cut. Dropping them moves overall precision from 24.9% to 31.2% and costs 224 confirmed flags.", False, False),
    ])
    _add_table(
        doc,
        ["Path", "Reviewed flags", "True positives", "False positives", "Precision"],
        [
            ["Cosine fallback", "2,327", "224", "2,103", "9.6%"],
            ["LLM rerank", "5,642", "1,760", "3,882", "31.2%"],
            ["Combined (current)", "7,969", "1,984", "5,985", "24.9%"],
        ],
    )
    _caption(doc, "Table 2. Precision by scoring path among reviewed flags.")
    _p(
        doc,
        [
            ("Cosine is ", False, False),
            ("29% of reviewed flags", True, False),
            (" and ", False, False),
            ("35% of all false positives", True, False),
            (", and it is the main reason overall precision sits at 24.9% instead of 31.2%.", False, False),
        ],
    )

    _heading(doc, "Cosine score bins", level=2)
    _add_table(
        doc,
        ["Cosine bin", "n", "Precision"],
        [
            ["0.40–0.42", "1,192", "11.1%"],
            ["0.42–0.44", "689", "6.4%"],
            ["0.44–0.46", "370", "10.0%"],
            ["0.46–0.48", "68", "14.7%"],
            ["0.48–0.50", "8", "12.5%"],
        ],
    )
    _caption(doc, "Table 3. Reviewed cosine-only flags. Raising the threshold inside 0.40–0.48 does not find a clean band; 0.42–0.44 is worse than 0.40–0.42.")
    _p(
        doc,
        "There is no useful operating point except “keep the floor at 0.40” or “do not promote cosine-only rows to SCIENCE_TALK.”",
    )

    _heading(doc, "Threshold sweep", level=2)
    _p(doc, "LLM flags are always kept. Cosine flags are kept only if cosine is at least t.")
    _add_table(
        doc,
        ["Cosine minimum", "Remaining flags", "Remaining TP", "Lost TP among flagged", "Precision"],
        [
            ["0.40 (current)", "7,969", "1,984", "0", "24.9%"],
            ["0.42", "6,777", "1,852", "132", "27.3%"],
            ["0.44", "6,088", "1,808", "176", "29.7%"],
            ["0.45", "5,856", "1,786", "198", "30.5%"],
            ["0.46", "5,718", "1,771", "213", "31.0%"],
            ["0.48", "5,650", "1,761", "223", "31.2%"],
            ["0.50 (drop all cosine)", "5,642", "1,760", "224", "31.2%"],
        ],
        highlight_last=True,
    )
    _caption(doc, "Table 4. Precision if cosine-only SCIENCE_TALK labels require cosine ≥ t.")
    _p(doc, "Dropping cosine-only SCIENCE_TALK labels:")
    _bullet(doc, [
        ("Precision ", False, False),
        ("24.9% → 31.2%", True, False),
        (".", False, False),
    ])
    _bullet(doc, "Removes 2,103 false positives and 224 true positives among flagged (11.3% of all confirmed science flags).")
    _bullet(doc, "Those 224 cosine true positives are longer than typical false positives (mean 6.8 words, median 5). Forty-three of them are science-content names a hold-out lexicon would keep. Examples that would be lost: “Is it sunny outside?”, “They sleep all winter long.”, “Because lemons are actually sour.”, “A mouse.”")
    _p(
        doc,
        [
            ("Recommendation: ", True, False),
            ("do not nibble the threshold. Either (a) stop writing cosine-only rows as SCIENCE_TALK "
             "(require LLM confirmation, even if those rows wait on the budget cap), or (b) keep "
             "them in the workbook but on a separate “cosine-only, not for the analytic sample” sheet "
             "so reviewers are not counting 9.6%-precise rows as model science talk.", False, False),
        ],
    )

    # ------------------------------------------------------------------ A3
    _heading(doc, "Analysis 3. Drop obvious junk before the LLM, without dropping real science names?")
    _p(doc, [
        ("Question. ", True, False),
        ("A lot of false positives are “Look.”, “Two.”, “Circle.” Can those be dropped with simple rules before the expensive reranker — without also dropping real one-word science (“Iguana.”, “Jellyfish.”)?", False, False),
    ])
    _heading(doc, "How this was done", level=2)
    _numbered(doc, "Four simple “drop this utterance” rules (a gate): exact look-family; exact count / “how many”; exact shape or color; short (1 or 2 words).")
    _numbered(doc, "A hold-out list of science-content names (128 words: iguana, habitat, pollen, jellyfish, …). If the utterance is basically one of those names, it is not gated.")
    _numbered(doc, "Each gate is applied to all 7,969 reviewed flags.")
    _numbered(doc, "For each gate, count how many flags it would remove, how many of those were TP vs FP, and precision of what remains.")
    _numbered(doc, "Gates are then stacked with the cosine decision from Analysis 2 (Table 6).")
    _p(doc, [
        ("How to read Table 5. ", True, False),
        ("Precision of removed = if this gate deletes a row, how often was that row actually science? Low is good (junk is being deleted). Remaining precision = precision of the sheet after the gate.", False, False),
    ])
    _p(doc, [
        ("Hold-out check. ", True, False),
        ("The 1–2 word gate without the hold-out deletes 244 true positives. With it, 100. The list saved 144 short confirmed names. It is not finished: 100 confirmed names still slip through (“What’s hibernation?”, “A bumblebee.”).", False, False),
    ])

    _heading(doc, "Per-gate effect on the 7,969 reviewed flags", level=2)
    _add_table(
        doc,
        ["Gate", "Flags removed", "TP removed", "FP removed", "Precision of removed", "Remaining precision"],
        [
            ["Exact look-family", "600", "31", "569", "5.2%", "26.5%"],
            ["Exact count / “how many”", "158", "1", "157", "0.6%", "25.4%"],
            ["Exact shape or color", "148", "2", "146", "1.4%", "25.3%"],
            ["Conservative (look ∪ count ∪ shape)", "906", "34", "872", "3.8%", "27.6%"],
            ["1–2 words (hold-out on)", "1,423", "100", "1,323", "7.0%", "28.8%"],
            ["All proposed gates", "1,703", "126", "1,577", "7.4%", "29.7%"],
        ],
    )
    _caption(doc, "Table 5. Each gate is applied alone, except the conservative and “all proposed” rows. Science-name hold-out is on.")
    _p(
        doc,
        [
            ("The same 1–2 word rule ", False, False),
            ("without", True, True),
            (" the hold-out removes 244 true positives instead of 100. The lexicon saved ", False, False),
            ("144 short confirmed science names", True, False),
            (" (Iguana., Caterpillar., Jellyfish., Cloudy?, It’s hot., An elephant., …). "
             "That hold-out is not optional if you gate on length.", False, False),
        ],
    )
    _p(
        doc,
        "A blanket short gate still costs 100 confirmed true positives the lexicon missed "
        "(What’s hibernation?, A flower., A bumblebee., Non-living thing!, chasing prey., Square.). "
        "Expand the lexicon before shipping a length gate — or do not ship a length gate yet.",
    )

    _heading(doc, "Combined with the cosine decision", level=2)
    _add_table(
        doc,
        ["Policy", "Remaining flags", "Remaining TP", "Lost TP among flagged", "Precision"],
        [
            ["Current deploy", "7,969", "1,984", "0", "24.9%"],
            ["Conservative gate only", "7,063", "1,950", "34", "27.6%"],
            ["Drop cosine only", "5,642", "1,760", "224", "31.2%"],
            ["Drop cosine ∪ conservative gate", "4,931", "1,727", "257", "35.0%"],
            ["Drop cosine ∪ all gates (incl. short)", "4,355", "1,638", "346", "37.6%"],
        ],
    )
    _caption(doc, "Table 6. Stacked policies on the reviewed-flag sample. The recommended combination is drop-cosine plus the conservative gate (35.0%).")

    # ------------------------------------------------------------------ recs
    _heading(doc, "Recommended next change")
    _p(
        doc,
        [
            ("Ship the conservative gate now. Decide cosine separately after a recall sample.", True, False),
        ],
    )
    _numbered(doc, [
        ("Before the LLM reranker (and before writing the reviewer sheet): ", True, False),
        ("drop exact look-family, exact count/how-many, and exact shape/color, unless the "
         "utterance is in the science-name hold-out. Expected effect on this sample: −872 false "
         "positives, −34 true positives, precision 24.9% → 27.6%. Count-only is almost free (1 true positive lost).", False, False),
    ])
    _numbered(doc, [
        ("Do not promote cosine-only rows to SCIENCE_TALK at 0.40. ", True, False),
        ("Either require LLM confirmation or park them off the analytic sheet. Expected effect: "
         "24.9% → 31.2%, at the cost of 224 flagged true positives whose recall impact is unknown.", False, False),
    ])
    _numbered(doc, [
        ("Together, those two moves take precision to 35.0% ", True, False),
        ("and still keep 1,727 of 1,984 confirmed flags (87%).", False, False),
    ])
    _numbered(doc, [
        ("Do not ship the 1–2 word gate yet. ", True, False),
        ("It is the largest remaining false-positive sink, but it still deletes 100 confirmed "
         "science names after the current lexicon. Code those 100, grow the hold-out, then revisit.", False, False),
    ])
    _numbered(doc, [
        ("Do not train on all 5,985 false positives as hard negatives. ", True, False),
        ("Drop the borderline_science slice (~11% of the coded sample) and prefer cosine-path "
         "management / look / count / shape rejects.", False, False),
    ])
    _p(
        doc,
        [
            ("The missing measurement is still a ", False, False),
            ("sample of unflagged utterances", True, False),
            (" (20–30 screened adult/child rows per classroom). Until that exists, 35% precision "
             "is a reviewer-load win, not a proven net gain in science-talk recovery.", False, False),
        ],
    )

    _heading(doc, "Appendix. How this was produced")
    _p(
        doc,
        "Analyses live on git branch error-analysis/fp-taxonomy-and-gates. Reproduce with "
        "python error_analysis/run_analyses.py (requires the Human coded workbooks). "
        "The coded 400-row sheet and machine-readable totals are in error_analysis/output/.",
    )
    _add_table(
        doc,
        ["File", "Role"],
        [
            ["error_analysis/REPORT.md", "Source memo"],
            ["error_analysis/codebook.py", "Taxonomy rules, gates, science-name lexicon"],
            ["error_analysis/hand_codes.py", "Hand labels for the 150 residual sample rows"],
            ["error_analysis/output/fp_sample_400_coded.csv", "Coded sample of 400"],
            ["error_analysis/output/cosine_threshold_sweep.csv", "Analysis 2 sweep"],
            ["error_analysis/output/gate_sweep.csv", "Analysis 3 sweep"],
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    return output_path


if __name__ == "__main__":
    path = build_document()
    print(f"wrote {path}")
