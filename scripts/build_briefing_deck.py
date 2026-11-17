"""Pictorial briefing: current design, Year 2 results, and error analysis.

About 9 minutes. Speaker notes are the spoken script for each slide.

    python scripts/build_briefing_deck.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Science_Talk_Briefing.pptx"

W = Inches(13.333)
H = Inches(7.5)

NAVY = RGBColor(0x1B, 0x28, 0x38)
INK = RGBColor(0x1B, 0x28, 0x38)
TEAL = RGBColor(0x1F, 0x6A, 0x64)
TEAL_DEEP = RGBColor(0x16, 0x4E, 0x4A)
TEAL_SOFT = RGBColor(0xE4, 0xF2, 0xF0)
CORAL = RGBColor(0xC4, 0x54, 0x3A)
CORAL_SOFT = RGBColor(0xF8, 0xEB, 0xE6)
GOLD = RGBColor(0xC4, 0x89, 0x2A)
GOLD_SOFT = RGBColor(0xF8, 0xF1, 0xE0)
BLUE = RGBColor(0x2C, 0x5F, 0x8A)
BLUE_SOFT = RGBColor(0xE6, 0xF0, 0xF8)
PAPER = RGBColor(0xF6, 0xF3, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SLATE = RGBColor(0x5C, 0x67, 0x70)
SAND = RGBColor(0xEB, 0xE4, 0xD8)
CARD = RGBColor(0xFF, 0xFC, 0xF8)


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def _shape(slide, kind, l, t, w, h, fill, line=None):
    sh = slide.shapes.add_shape(kind, l, t, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.25)
    # Drop the default shadow-less look; tighten rounded corners when present.
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sh.adjustments[0] = 0.12
        except Exception:
            pass
    return sh


def _rect(slide, l, t, w, h, fill):
    return _shape(slide, MSO_SHAPE.RECTANGLE, l, t, w, h, fill)


def _round(slide, l, t, w, h, fill, line=None):
    return _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h, fill, line)


def _oval(slide, l, t, w, h, fill):
    return _shape(slide, MSO_SHAPE.OVAL, l, t, w, h, fill)


def _no_margin(tf):
    tf.word_wrap = True
    tf.auto_size = None
    body = tf._txBody
    pr = body.find(qn("a:bodyPr"))
    if pr is not None:
        pr.set("lIns", str(int(Emu(Inches(0.08)))))
        pr.set("rIns", str(int(Emu(Inches(0.08)))))
        pr.set("tIns", str(int(Emu(Inches(0.04)))))
        pr.set("bIns", str(int(Emu(Inches(0.04)))))


def _tb(slide, l, t, w, h, lines, *, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT):
    """lines: list of (text, size, bold, color, italic, space_before, align_or_None)."""
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.paragraphs[0].alignment = align
    try:
        tf._txBody.bodyPr.set("anchor", {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}[anchor])
    except Exception:
        pass
    for i, item in enumerate(lines):
        text, size, bold, color = item[0], item[1], item[2], item[3]
        italic = item[4] if len(item) > 4 else False
        before = item[5] if len(item) > 5 else 0
        al = item[6] if len(item) > 6 and item[6] is not None else align
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = al
        p.space_before = Pt(before)
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return box


def _notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def _footer(slide, n, total=10):
    _tb(
        slide, Inches(0.48), Inches(7.12), Inches(8.5), Inches(0.28),
        [("Anita Zucker Center  ·  Science-talk retrieval", 11, False, SLATE, False, 0)],
    )
    _tb(
        slide, Inches(10.4), Inches(7.12), Inches(2.45), Inches(0.28),
        [(f"{n}  /  {total}", 11, False, SLATE, False, 0, PP_ALIGN.RIGHT)],
        align=PP_ALIGN.RIGHT,
    )


def _kicker(slide, text):
    _rect(slide, Inches(0), Inches(0), W, Inches(0.08), TEAL)
    _tb(
        slide, Inches(0.48), Inches(0.22), Inches(12.3), Inches(0.28),
        [(text.upper(), 12, True, TEAL, False, 0)],
    )


def _h1(slide, text, top=0.46):
    _tb(
        slide, Inches(0.48), Inches(top), Inches(12.3), Inches(0.62),
        [(text, 30, True, INK, False, 0)],
    )


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def s01(prs):
    s = _blank(prs)
    _bg(s, NAVY)
    _rect(s, Inches(0), Inches(0), Inches(0.18), H, TEAL)
    _rect(s, Inches(0.18), Inches(2.15), Inches(4.2), Inches(0.07), GOLD)

    _tb(s, Inches(0.7), Inches(1.35), Inches(11), Inches(0.36),
        [("ANITA ZUCKER CENTER", 14, True, GOLD, False, 0)])
    _tb(s, Inches(0.7), Inches(2.4), Inches(12), Inches(2.1), [
        ("Finding science talk", 48, True, WHITE, False, 0),
        ("in classroom audio", 48, True, WHITE, False, 0),
    ])
    _tb(s, Inches(0.7), Inches(4.7), Inches(10.5), Inches(1.1), [
        ("The pipeline, the Year 2 results,", 20, False, RGBColor(0xD5, 0xDE, 0xE6), False, 0),
        ("and the error analysis.", 20, False, RGBColor(0xD5, 0xDE, 0xE6), False, 2),
    ])
    _tb(s, Inches(0.7), Inches(6.45), Inches(11), Inches(0.4),
        [("Pipeline  ·  Results  ·  Error analysis", 14, False, RGBColor(0x9A, 0xAA, 0xB8), False, 0)])
    _notes(s, (
        "About 25 seconds.\n\n"
        "This is a nine-minute walk through three things: how the science-talk finder "
        "then the Year 2 results, then the error analysis. I will stay with the pictures "
        "and the numbers on the slides."
    ))


def s03(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Current pipeline")
    _h1(s, "Built once on the seed, then run on every classroom", top=0.40)

    _tb(s, Inches(0.42), Inches(1.12), Inches(3.2), Inches(0.28),
        [("1   BUILD", 13, True, GOLD, False, 0)])
    build = [
        ("Seed", "196 labeled lines"),
        ("Subtypes", "rules, then keywords"),
        ("Negatives", "transcript, LLM, seed words"),
        ("Paraphrases", "three speaking registers"),
        ("Human gate", "accept, reject, or edit"),
        ("Fine-tune", "nomic, then Year-1 cutoffs"),
    ]
    for i, (title, body) in enumerate(build):
        x = 0.42 + i * 2.14
        _round(s, Inches(x), Inches(1.42), Inches(1.92), Inches(1.15), GOLD_SOFT)
        _tb(s, Inches(x + 0.06), Inches(1.5), Inches(1.8), Inches(0.95), [
            (title, 14, True, INK, False, 0, PP_ALIGN.CENTER),
            (body, 11, False, SLATE, False, 2, PP_ALIGN.CENTER),
        ], align=PP_ALIGN.CENTER)
        if i < 5:
            _shape(s, MSO_SHAPE.RIGHT_ARROW, Inches(x + 1.94), Inches(1.86), Inches(0.18), Inches(0.16), GOLD)

    _tb(s, Inches(0.42), Inches(2.78), Inches(4.2), Inches(0.28),
        [("2   SCORE A TRANSCRIPT", 13, True, TEAL, False, 0)])
    score = [
        (TEAL, "Ingest", "Adult and child.\nASR confidence ≥ 0.40."),
        (BLUE, "Track A", "Cosine vs the full\nanchor bank. Every line."),
        (GOLD, "Track B", "Judge the top 10 anchors.\nScore = the maximum.\nStops at $15."),
        (CORAL, "Label", "Judge ≥ 0.90,\nor cosine ≥ 0.40\nif the judge never ran."),
        (NAVY, "Workbook", "Science-talk rows only.\nReviewer marks yes or no."),
    ]
    for i, (color, title, body) in enumerate(score):
        x = 0.38 + i * 2.58
        _round(s, Inches(x), Inches(3.12), Inches(2.32), Inches(2.35), WHITE)
        _rect(s, Inches(x), Inches(3.12), Inches(2.32), Inches(0.1), color)
        _tb(s, Inches(x + 0.08), Inches(3.3), Inches(2.16), Inches(0.4),
            [(title, 16, True, INK, False, 0, PP_ALIGN.CENTER)], align=PP_ALIGN.CENTER)
        _tb(s, Inches(x + 0.08), Inches(3.76), Inches(2.16), Inches(1.5),
            [(body, 12, False, SLATE, False, 0, PP_ALIGN.CENTER)], align=PP_ALIGN.CENTER)
        if i < 4:
            _shape(s, MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.34), Inches(4.08), Inches(0.22), Inches(0.18), TEAL)

    _round(s, Inches(0.42), Inches(5.65), Inches(12.5), Inches(1.22), TEAL_SOFT)
    _tb(s, Inches(0.62), Inches(5.78), Inches(12.1), Inches(0.95), [
        ("The Year 2 review does not retrain the model.", 16, True, TEAL_DEEP, False, 0),
        ("Track A scores every line. Track B sees only the closest lines that still fit the budget. A line the judge never sees is still labeled science talk when cosine is at least 0.40.", 13, False, INK, False, 4),
    ])
    _footer(s, 2)
    _notes(s, (
        "About 70 seconds.\n\n"
        "The pipeline is two phases, and they should not be collapsed into one. "
        "Phase one is built once, from the Year 1 seed. We load 196 labeled lines, "
        "assign science subtypes, mine negatives three ways, paraphrase each anchor "
        "into other speaking registers, and pass the generated rows through a human "
        "gate — accept, reject, or edit — before anything is trusted for training. "
        "Then we fine-tune the nomic bi-encoder and choose the cutoffs on the Year 1 "
        "held-out set. llama-3.3-70b writes the paraphrases and the hard negatives. "
        "It is not the scorer. Phase two runs on every Year 2 classroom. We keep adult "
        "and child lines with recognizer confidence of at least 40 percent. Track A, "
        "the fine-tuned bi-encoder, cosine-scores every line against the full anchor "
        "bank, locally. Track B, gpt-oss-120b, then judges only the highest-cosine "
        "lines, comparing each one to its top 10 anchors and taking the maximum score, "
        "until the 15-dollar cap. The label rule is explicit. If the judge ran, science "
        "talk requires 0.90. If it never ran, science talk requires cosine 0.40. The "
        "workbook keeps only predicted science talk, and a reviewer marks yes or no. "
        "That review is the analytic sample. It does not flow back into training."
    ))


def s03b(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Current architecture")
    _h1(s, "Two models at deployment, one router in training")

    cols = [
        (TEAL, "Track A  ·  retrieval", [
            ("nomic-embed-text-v1.5", 16, True, INK, False, 0),
            ("768 dimensions, fine-tuned locally.", 13, False, SLATE, False, 6),
            ("Cosine vs an L2-normalized bank of known science lines. Every utterance, no API cost.", 13, False, INK, False, 8),
            ("If the judge never runs: SCIENCE TALK when cosine ≥ 0.40.", 13, True, TEAL_DEEP, False, 8),
            ("Trained from 196 seed lines → 516 paraphrases and 7,815 negatives.", 13, False, SLATE, False, 8),
        ]),
        (BLUE, "Track B  ·  pair judge", [
            ("gpt-oss-120b", 16, True, INK, False, 0),
            ("Zero-shot. Temperature 0.", 13, False, SLATE, False, 6),
            ("Asks whether this line and a retrieved anchor are the same science practice. Not a yes/no classifier.", 13, False, INK, False, 8),
            ("Deploy: top 10 anchors. Score = the maximum. SCIENCE TALK at ≥ 0.90.", 13, True, BLUE, False, 8),
            ("Stops at a $15 cap (~150,000 new calls). Cache hits are free.", 13, False, SLATE, False, 8),
        ]),
        (GOLD, "Router  ·  training only", [
            ("Confidence, then a person", 16, True, INK, False, 0),
            ("Three signals: LLM self-score, cosine band, structural check.", 13, False, SLATE, False, 6),
            ("Weights 0.5 / 0.8 / 0.8. Frozen baseline encoder, so the fine-tuned model does not grade its own rows.", 13, False, INK, False, 8),
            ("≥ 0.70 auto-accept.  ≥ 0.15 spot-check.  Below that, full review.", 13, True, RGBColor(0x8A, 0x62, 0x12), False, 8),
            ("Cutoffs retuned after a 50-row expert audit.", 13, False, SLATE, False, 8),
        ]),
    ]
    for i, (color, title, lines) in enumerate(cols):
        x = 0.38 + i * 4.3
        _round(s, Inches(x), Inches(1.48), Inches(4.1), Inches(5.35), WHITE)
        _rect(s, Inches(x), Inches(1.48), Inches(4.1), Inches(0.58), color)
        _tb(s, Inches(x + 0.16), Inches(1.58), Inches(3.8), Inches(0.42),
            [(title, 15, True, WHITE, False, 0)])
        _tb(s, Inches(x + 0.18), Inches(2.22), Inches(3.75), Inches(4.35), lines)
    _footer(s, 3)
    _notes(s, (
        "About 70 seconds.\n\n"
        "Track A is a fine-tuned nomic embedder, 768 dimensions, running locally. It "
        "compares every kept line to a bank of known science utterances and keeps the "
        "best cosine. It was trained contrastively on paraphrases and hard negatives "
        "grown from 196 expert seed lines: 516 register paraphrases and 7,815 negatives. "
        "Track B is gpt-oss-120b, temperature zero. The prompt is a pair judge: does "
        "this classroom line express the same scientific practice as this anchor? At "
        "deployment it scores the top 10 anchors and the label uses the maximum, so "
        "one strong match is enough. Science talk requires 0.90. The lab query path "
        "uses top 20; deployment uses 10 to stretch the budget. The default cap is "
        "15 dollars, about 150,000 new calls, and cached calls are free. llama-3.3-70b "
        "is not in the deployed scorer. It only writes training paraphrases, hard "
        "negatives, and subtype fallbacks. The confidence router is also training-only. "
        "It mixes the LLM’s self-score, a cosine band, and a structural check, with "
        "weights 0.5, 0.8, and 0.8. At or above 0.70 the row is auto-accepted. Between "
        "0.15 and 0.70 it is spot-checked. Below that a person reviews it. Those "
        "cutoffs were retuned after the 50-row audit on the next-but-one slide."
    ))


def s04(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Current design")
    _h1(s, "Two doors into the same science-talk label")

    # Left door
    _round(s, Inches(0.42), Inches(1.55), Inches(6.05), Inches(4.15), WHITE)
    _rect(s, Inches(0.42), Inches(1.55), Inches(6.05), Inches(0.7), TEAL)
    _tb(s, Inches(0.62), Inches(1.68), Inches(5.7), Inches(0.48),
        [("Door A   ·   the judge ran", 20, True, WHITE, False, 0)])
    _tb(s, Inches(0.7), Inches(2.45), Inches(5.5), Inches(2.9), [
        ("Score at least 0.90", 28, True, TEAL, False, 0),
        ("5,642 reviewed flags", 18, False, INK, False, 12),
        ("Reviewers agreed  31%", 18, True, INK, False, 4),
        ("gpt-oss-120b · max of 10 pair scores · temperature 0", 14, False, SLATE, False, 14),
    ])

    # Right door
    _round(s, Inches(6.85), Inches(1.55), Inches(6.05), Inches(4.15), WHITE)
    _rect(s, Inches(6.85), Inches(1.55), Inches(6.05), Inches(0.7), CORAL)
    _tb(s, Inches(7.05), Inches(1.68), Inches(5.7), Inches(0.48),
        [("Door B   ·   the judge never ran", 20, True, WHITE, False, 0)])
    _tb(s, Inches(7.12), Inches(2.45), Inches(5.5), Inches(2.9), [
        ("Similarity at least 0.40", 28, True, CORAL, False, 0),
        ("2,327 reviewed flags", 18, False, INK, False, 12),
        ("Reviewers agreed  10%", 18, True, INK, False, 4),
        ("nomic bi-encoder only, after the dollar cap. Similarity makes the final call.", 14, False, SLATE, False, 14),
    ])

    _tb(s, Inches(0.48), Inches(5.9), Inches(12.3), Inches(1.0), [
        ("Both doors print SCIENCE TALK on the review sheet. Door B is 29% of the flags and 35% of the mistakes.", 16, True, INK, False, 0),
    ])
    _footer(s, 4)
    _notes(s, (
        "About 55 seconds.\n\n"
        "This is the design choice that drives the results. If the language model ran, "
        "the line is called science talk only when its score is at least 0.90. That path "
        "produced 5,642 reviewed flags, and reviewers agreed 31 percent of the time. If "
        "the language model never ran, because the budget ran out, the line is still "
        "called science talk when similarity is at least 0.40. That path produced 2,327 "
        "flags, and reviewers agreed only about 10 percent of the time. Both paths print "
        "the same label. The sheet does record which door was used, in a column called "
        "scored by. Similarity was built as a first filter. At deployment it is also "
        "allowed to make the final call. Those cosine-only flags are 29 percent of the "
        "reviewed sheet and 35 percent of all the false positives. Door A is gpt-oss-120b "
        "taking the maximum of ten pair scores. Door B is the nomic bi-encoder alone, "
        "after the dollar cap has stopped new judge calls."
    ))


def s04b(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Human review")
    _h1(s, "Six reviews, six different jobs", top=0.40)

    cards = [
        (TEAL, "1", "Seed labels", "196 lines",
         "The only gold the retriever trained on.", "191 science  ·  5 not"),
        (TEAL, "2", "Negative check", "50 lines",
         "Are mined negatives really not science?", "49 clear  ·  1 possible leak"),
        (BLUE, "3", "Positive cleanup", "20 lines",
         "Drop mined “science” that is not.", "18 dropped  ·  2 kept"),
        (BLUE, "4", "Routing audit", "47 of 50",
         "Can auto-accept be trusted?", "Auto pile matched the expert"),
        (GOLD, "5", "Paraphrase check", "75 lines",
         "Safety check on auto-routed variants.", "74 accepted  ·  1 edited"),
        (CORAL, "6", "Year 2 flags", "7,969 lines",
         "Is each deployed flag real science?", "1,984 yes  ·  5,985 no"),
    ]
    for i, (color, num, title, n, purpose, outcome) in enumerate(cards):
        col, row = i % 3, i // 3
        x = 0.38 + col * 4.3
        y = 1.38 + row * 2.55
        _round(s, Inches(x), Inches(y), Inches(4.12), Inches(2.4), WHITE)
        _oval(s, Inches(x + 0.16), Inches(y + 0.16), Inches(0.46), Inches(0.46), color)
        _tb(s, Inches(x + 0.16), Inches(y + 0.22), Inches(0.46), Inches(0.36),
            [(num, 14, True, WHITE, False, 0, PP_ALIGN.CENTER)], align=PP_ALIGN.CENTER)
        _tb(s, Inches(x + 0.72), Inches(y + 0.14), Inches(3.2), Inches(0.36),
            [(title, 16, True, INK, False, 0)])
        _tb(s, Inches(x + 0.72), Inches(y + 0.48), Inches(3.2), Inches(0.32),
            [(n, 13, True, color, False, 0)])
        _tb(s, Inches(x + 0.18), Inches(y + 1.05), Inches(3.75), Inches(1.15), [
            (purpose, 13, False, SLATE, False, 0),
            (outcome, 14, True, INK, False, 4),
        ])
    _tb(s, Inches(0.42), Inches(6.55), Inches(12.4), Inches(0.48),
        [("A seventh pass coded 400 rejects (150 by hand) to name the mistakes. It diagnosed the model. It did not retrain it.", 13, False, SLATE, False, 0)])
    _footer(s, 5)
    _notes(s, (
        "About 75 seconds.\n\n"
        "Human review was not one step. It happened six times, and each pass had a "
        "different job. First, experts labeled the Year 1 seed: 196 lines, 191 science "
        "talk and 5 not. That is the anchor bank. Second, a 50-line check of mined "
        "negatives — transcript talk, model rewrites, and seed words used in a "
        "non-science sense. Forty-nine were truly not science; one was a possible leak. "
        "The bar for that check was 90 percent true negatives. Third, 20 mined "
        "positives that looked suspicious: 18 were dropped and 2 kept, so the positive "
        "bank did not absorb management talk. Fourth, a 50-row routing audit, 47 of "
        "them scored. The question was whether the confidence router could auto-accept "
        "generated rows. The auto pile, 23 rows, matched the expert every time. Spot "
        "and full-review piles agreed only 58 percent of the time, which is acceptable "
        "because those rows already go to a person. Overall agreement was 79 percent. "
        "We then retuned the cutoffs — auto at 0.70, spot at 0.15 — and the same audit "
        "scored 82 percent. Fifth, 75 paraphrases the router had marked auto were "
        "still read: 74 accepted, 1 edited. Sixth, and by far the largest, reviewers "
        "marked 7,969 Year 2 flags across 83 sessions. That pass is the analytic "
        "sample. It measures precision. It was not fed back into training. Thirty-five "
        "flags were left blank. A later coding of 400 rejects, 150 of them by hand "
        "after rules placed the rest, is how we named the error types. That coding "
        "did not change the model."
    ))


def s05(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Results")
    _h1(s, "The lab test looked solved. The classrooms did not.")

    _round(s, Inches(0.42), Inches(1.55), Inches(6.05), Inches(4.35), WHITE)
    _tb(s, Inches(0.7), Inches(1.75), Inches(5.5), Inches(0.35),
        [("YEAR 1  ·  HELD-OUT SEED", 13, True, BLUE, False, 0)])
    _tb(s, Inches(0.7), Inches(2.15), Inches(5.5), Inches(1.15),
        [("95%", 60, True, BLUE, False, 0)])
    _tb(s, Inches(0.7), Inches(3.4), Inches(5.5), Inches(2.2), [
        ("F1 on a tiny, easy test set.", 18, True, INK, False, 0),
        ("9 real  ·  0 wrong  ·  1 missed", 16, False, INK, False, 8),
        ("Fine-tuned similarity scored a perfect 1.0. The cutoffs 0.40 and 0.90 were taken from this test.", 14, False, SLATE, False, 10),
    ])

    _round(s, Inches(6.85), Inches(1.55), Inches(6.05), Inches(4.35), WHITE)
    _tb(s, Inches(7.12), Inches(1.75), Inches(5.5), Inches(0.35),
        [("YEAR 2  ·  83 CLASSROOM SESSIONS", 13, True, CORAL, False, 0)])
    _tb(s, Inches(7.12), Inches(2.15), Inches(5.5), Inches(1.15),
        [("25%", 60, True, CORAL, False, 0)])
    _tb(s, Inches(7.12), Inches(3.4), Inches(5.5), Inches(2.2), [
        ("of flags a reviewer confirmed.", 18, True, INK, False, 0),
        ("1,984 yes  ·  5,985 no", 16, False, INK, False, 8),
        ("This is the operating number. Preschool talk does not look like the seed.", 14, False, SLATE, False, 10),
    ])

    _tb(s, Inches(0.48), Inches(6.1), Inches(12.3), Inches(0.85),
        [("Do not quote 95% as how the system does in classrooms.", 16, True, INK, False, 0)])
    _footer(s, 6)
    _notes(s, (
        "About 50 seconds.\n\n"
        "Before classrooms, we tested on the Year 1 seed. That test looked almost solved. "
        "The deployed system scored about 0.95 F1: nine true positives, zero false "
        "positives, and one miss, on a tiny held-out set. Fine-tuned similarity scored a "
        "perfect 1.0, which tells us the Year 1 negatives — mostly management and social "
        "talk — were easy to separate. The cutoffs we still use, 0.40 for similarity and "
        "0.90 for the language model, were taken from that test and written into the "
        "deployment config. Year 2 is a different problem. Of the flags reviewers marked, "
        "one in four was real science talk: 1,984 yes and 5,985 no. That 25 percent is "
        "the operating number. The lab score is not."
    ))


def s06(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Results")
    _h1(s, "What 83 reviewed sessions actually contain")

    rows = [
        (TEAL, "174,979", "lines screened", "adult and child, recognizer sure enough"),
        (GOLD, "8,004", "called science", "about 1 in 22 lines  ·  7,969 were reviewed"),
        (CORAL, "1,984", "reviewer said yes", "24.9% of reviewed flags  ·  about 1% of all speech"),
    ]
    for i, (color, num, label, sub) in enumerate(rows):
        y = 1.5 + i * 1.45
        width = 7.4 - i * 1.35
        _round(s, Inches(0.48), Inches(y), Inches(width), Inches(1.28), color)
        _tb(s, Inches(0.7), Inches(y + 0.16), Inches(width - 0.4), Inches(0.58),
            [(num, 28, True, WHITE, False, 0)])
        _tb(s, Inches(0.7), Inches(y + 0.7), Inches(width - 0.4), Inches(0.42),
            [(f"{label}   ·   {sub}", 13, False, WHITE, False, 0)])

    # Side facts
    facts = [
        ("16", "classrooms"),
        ("289", "audio hours"),
        ("26", "children sampled"),
        ("31% / 21%", "adult / child flags kept"),
    ]
    for i, (n, lab) in enumerate(facts):
        y = 1.5 + i * 1.22
        _round(s, Inches(8.55), Inches(y), Inches(4.3), Inches(1.08), WHITE)
        _tb(s, Inches(8.75), Inches(y + 0.08), Inches(3.95), Inches(0.5),
            [(n, 22, True, INK, False, 0)])
        _tb(s, Inches(8.75), Inches(y + 0.55), Inches(3.95), Inches(0.4),
            [(lab, 13, False, SLATE, False, 0)])
    _footer(s, 7)
    _notes(s, (
        "About 60 seconds.\n\n"
        "These are the 83 reviewed sessions: 16 classrooms, about 289 hours of audio, "
        "26 children, from October 2025 through April 2026. The pipeline screened "
        "174,979 adult and child lines. It flagged 8,004 of them as science talk, about "
        "one in twenty-two. Reviewers marked 7,969 of those flags — 35 were left blank. "
        "They confirmed 1,984 and rejected 5,985. So confirmed science talk is about "
        "1 percent of everything that was said, and about a quarter of what the model "
        "showed. On average across transcripts, adult flags were confirmed more often "
        "than child flags, roughly 31 percent versus 21 percent. Precision also swings "
        "by room, from 0 to about 78 percent, so one global cutoff is a rough fit. "
        "What we cannot see on this slide is recall: the review sheets contain only "
        "lines the model already flagged."
    ))


def s07(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Error analysis")
    _h1(s, "Most wrong flags are ordinary classroom talk", top=0.42)

    _tb(s, Inches(0.48), Inches(1.12), Inches(12.3), Inches(0.36),
        [("400 rejected flags, labeled by hand. Half adult, half child; half each door.", 14, False, SLATE, False, 0)])

    bars = [
        ("Look.  /  pointing", 86, "21%", "“Look at that.”", TEAL),
        ("Sit down  /  transitions", 70, "18%", "“Have a seat please.”", TEAL),
        ("Naming a thing", 64, "16%", "“The box!”", TEAL),
        ("Maybe science", 44, "11%", "“How does that work?”", GOLD),
        ("Shape or color", 38, "10%", "“What shape is that?”", CORAL),
        ("Counting", 37, "9%", "“One, two, three.”", CORAL),
    ]
    max_n = 86
    max_w = 6.3
    for i, (name, n, pct, ex, color) in enumerate(bars):
        y = 1.55 + i * 0.82
        _tb(s, Inches(0.42), Inches(y), Inches(2.55), Inches(0.55),
            [(name, 14, True, INK, False, 0)])
        bw = max(0.35, max_w * (n / max_n))
        _round(s, Inches(3.05), Inches(y + 0.05), Inches(bw), Inches(0.42), color)
        _tb(s, Inches(3.15 + bw), Inches(y), Inches(1.3), Inches(0.5),
            [(pct, 16, True, INK, False, 0)])
        _tb(s, Inches(10.15), Inches(y), Inches(2.8), Inches(0.55),
            [(ex, 13, False, SLATE, True, 0)])

    _footer(s, 8)
    _notes(s, (
        "About 70 seconds.\n\n"
        "We took 400 of the rejects and gave each one label. The sample was balanced on "
        "purpose: 100 adult language-model, 100 adult similarity, 100 child language-model, "
        "100 child similarity, spread across classrooms. It shows what kinds of mistakes "
        "exist. It is not a census of all 5,985 rejects, and Classroom 21 is over-represented "
        "because it produced so many flags. Three ordinary types are 55 percent of this "
        "sample: pointing and “look,” classroom management, and naming an everyday object "
        "with no science frame. About one in nine is borderline — a reviewer said no, but "
        "a reasonable person might have said science. Do not dump every reject into training "
        "without setting that slice aside. Children account for most of the “look” mistakes: "
        "59 of 200 child rejects, versus 27 of 200 adult. The similarity path accounts for "
        "most of the management mistakes: 53 of 200, versus 17 of 200 on the language-model "
        "path. The judge is not confused by “have a seat.” It is confused by short lines "
        "that share a surface with observation."
    ))


def s08(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Error analysis")
    _h1(s, "Raising the similarity cutoff does not find a clean cut")
    _tb(s, Inches(0.48), Inches(1.15), Inches(7.4), Inches(0.55),
        [("Bar height is how often that band was real science. The two taller bars are only 68 and 8 lines.", 13, False, SLATE, False, 0)])

    bins = [
        ("0.40–0.42", 11.1, "1,192 lines"),
        ("0.42–0.44", 6.4, "689"),
        ("0.44–0.46", 10.0, "370"),
        ("0.46–0.48", 14.7, "68"),
        ("0.48–0.50", 12.5, "8"),
    ]
    base_y = 5.55
    max_h = 3.15
    for i, (lab, prec, nlab) in enumerate(bins):
        x = 0.55 + i * 1.45
        bh = max_h * (prec / 20.0)  # scale so 20% would be full; all bars stay short
        top = base_y - bh
        _round(s, Inches(x), Inches(top), Inches(1.15), Inches(bh), CORAL if prec < 12 else GOLD)
        _tb(s, Inches(x - 0.05), Inches(top - 0.38), Inches(1.3), Inches(0.36),
            [(f"{prec:.0f}%", 14, True, INK, False, 0, PP_ALIGN.CENTER)], align=PP_ALIGN.CENTER)
        _tb(s, Inches(x - 0.15), Inches(5.62), Inches(1.5), Inches(0.55),
            [(lab, 11, True, INK, False, 0, PP_ALIGN.CENTER)], align=PP_ALIGN.CENTER)
        _tb(s, Inches(x - 0.1), Inches(6.05), Inches(1.45), Inches(0.35),
            [(nlab, 11, False, SLATE, False, 0, PP_ALIGN.CENTER)], align=PP_ALIGN.CENTER)

    _round(s, Inches(8.05), Inches(1.6), Inches(4.8), Inches(4.55), WHITE)
    _tb(s, Inches(8.25), Inches(1.8), Inches(4.45), Inches(4.1), [
        ("If we drop Door B", 14, True, CORAL, False, 0),
        ("25%  →  31%", 32, True, INK, False, 8),
        ("Precision of the review sheet.", 14, False, SLATE, False, 4),
        ("Cost: 224 confirmed science lines leave the sheet.", 15, True, INK, False, 16),
        ("“They sleep all winter long.”", 14, False, TEAL_DEEP, True, 12),
        ("“Because lemons are actually sour.”", 14, False, TEAL_DEEP, True, 4),
    ])
    _footer(s, 9)
    _notes(s, (
        "About 55 seconds.\n\n"
        "Could we just raise 0.40? These bars are the precision of similarity-only flags "
        "inside each score band. Every band sits between about 6 and 15 percent. The "
        "0.42 to 0.44 band is worse than the band below it. There is no clean cutoff "
        "hiding in there. Almost every cosine-only flag lives between 0.40 and 0.48. "
        "The useful choice is binary: keep the floor, or stop promoting similarity-only "
        "lines to science talk. Dropping them lifts precision from 24.9 percent to "
        "31.2 percent. It also removes 224 lines a reviewer had confirmed — 11 percent "
        "of all confirmed flags. Some of those are real science: “They sleep all winter "
        "long,” “Because lemons are actually sour.” We do not yet know how many true "
        "science lines were never flagged at all, so this is a precision gain with an "
        "unknown recall cost."
    ))


def s09(prs):
    s = _blank(prs)
    _bg(s, PAPER)
    _kicker(s, "Error analysis")
    _h1(s, "Three exact phrases are safe to drop first")

    gates = [
        ("Look.", "600 flags", "31 real science lost", "“Look at that.”"),
        ("Count.", "158 flags", "1 real science lost", "“Two.”  “How many?”"),
        ("Shape.", "148 flags", "2 real science lost", "“Circle.”  “What color?”"),
    ]
    for i, (title, flags, lost, ex) in enumerate(gates):
        x = 0.42 + i * 4.25
        _round(s, Inches(x), Inches(1.5), Inches(4.05), Inches(2.35), WHITE)
        _oval(s, Inches(x + 0.2), Inches(1.7), Inches(0.7), Inches(0.7), TEAL)
        _tb(s, Inches(x + 0.2), Inches(1.82), Inches(0.7), Inches(0.48),
            [(str(i + 1), 18, True, WHITE, False, 0, PP_ALIGN.CENTER)], align=PP_ALIGN.CENTER)
        _tb(s, Inches(x + 1.05), Inches(1.78), Inches(2.8), Inches(0.5),
            [(title, 22, True, INK, False, 0)])
        _tb(s, Inches(x + 0.25), Inches(2.55), Inches(3.6), Inches(1.1), [
            (flags, 16, True, TEAL, False, 0),
            (lost, 14, False, CORAL, False, 2),
            (ex, 13, False, SLATE, True, 4),
        ])

    _round(s, Inches(0.42), Inches(4.08), Inches(8.15), Inches(2.75), TEAL)
    _tb(s, Inches(0.68), Inches(4.25), Inches(7.7), Inches(2.4), [
        ("Together, then drop Door B", 16, True, RGBColor(0xC9, 0xE8, 0xE3), False, 0),
        ("25%   →   28%   →   35%", 32, True, WHITE, False, 6),
        ("Still keep 1,727 of 1,984 confirmed lines  ·  87%", 15, False, WHITE, False, 8),
        ("872 wrong flags leave with the phrase filter alone. Only 34 real ones go with them.", 13, False, RGBColor(0xE4, 0xF2, 0xF0), False, 6),
    ])

    _round(s, Inches(8.75), Inches(4.08), Inches(4.1), Inches(2.75), GOLD_SOFT)
    _tb(s, Inches(8.95), Inches(4.25), Inches(3.75), Inches(2.4), [
        ("Not yet", 14, True, GOLD, False, 0),
        ("Do not drop every 1–2 word line.", 16, True, INK, False, 6),
        ("“Iguana.” and “Jellyfish.” are real. A name list still misses about 100 short science lines.", 13, False, INK, False, 8),
    ])
    _footer(s, 10)
    _notes(s, (
        "About 60 seconds.\n\n"
        "A cheaper filter can run before the language model, on exact phrases, and it "
        "spares a hold-out list of science names. Drop the look family: 600 flags, only "
        "31 of them real science. Drop exact counting and “how many”: 158 flags, one "
        "real science line. Drop exact shapes and colors: 148 flags, two real science "
        "lines. Together that removes 872 wrong flags and 34 real ones, and precision "
        "moves from 24.9 percent to 27.6 percent. Stack that with dropping similarity-only "
        "labels, and precision reaches 35 percent, while we still keep 1,727 of 1,984 "
        "confirmed flags — 87 percent. Do not yet drop every one- or two-word line. "
        "Without a name list, a short-line rule deletes 244 confirmed science lines. "
        "With the current list of 128 names, it still deletes 100, including “What’s "
        "hibernation?” and “A bumblebee.” Grow that list before any length rule. And "
        "about 11 percent of the coded rejects were borderline science — do not train "
        "on all 5,985 rejects as if they were clear negatives."
    ))


def build() -> Path:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    prs.core_properties.title = "Finding science talk in classroom audio"
    prs.core_properties.subject = "Design, Year 2 results, and error analysis"
    prs.core_properties.category = "Briefing"
    for fn in (s01, s03, s03b, s04, s04b, s05, s06, s07, s08, s09):
        fn(prs)
    # Notes page size is independent; ensure every slide has notes.
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(path)
