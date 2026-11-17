"""Build a plain-language PowerPoint of the three error analyses.

Run from the project root:
    python error_analysis/build_error_analysis_deck.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
OUTPUT_PATH = HERE / "Science_Talk_Error_Analysis.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

NAVY = RGBColor(0x1A, 0x2B, 0x3C)
BLUE = RGBColor(0x2E, 0x5C, 0x8A)
LIGHT_BLUE = RGBColor(0xE8, 0xF1, 0xFB)
AMBER = RGBColor(0xB8, 0x86, 0x2A)
GREY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
RED = RGBColor(0xB5, 0x3A, 0x2E)


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    return box, tf


def _set_run(run, *, size, bold=False, color=NAVY, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Calibri"


def _rect(slide, left, top, width, height, fill):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def _title_bar(slide, text, *, subtitle=None):
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(0.28), BLUE)
    box_h = Inches(1.22) if subtitle else Inches(1.0)
    _, tf = _textbox(slide, Inches(0.55), Inches(0.40), Inches(12.2), box_h)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    _set_run(r, size=26, bold=True, color=NAVY)
    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        _set_run(r2, size=14, italic=True, color=GREY)


def _footer(slide, page: str):
    _, tf = _textbox(slide, Inches(0.55), Inches(7.12), Inches(12.2), Inches(0.3))
    r = tf.paragraphs[0].add_run()
    r.text = f"Anita Zucker Center  ·  Classroom science-talk retrieval  ·  {page}"
    _set_run(r, size=10, italic=True, color=GREY)


def _bullets(slide, items, *, left=Inches(0.6), top=Inches(1.6),
             width=Inches(12.1), height=Inches(5.2), size=18, gap=8):
    _, tf = _textbox(slide, left, top, width, height)
    first = True
    for text, level, bold, color in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(gap)
        mark = "" if not text else ("•  " if level == 0 else "    ·  ")
        r = p.add_run()
        r.text = f"{mark}{text}"
        _set_run(
            r,
            size=size - (2 if level else 0),
            bold=bold,
            color=color or (NAVY if level == 0 else GREY),
        )


def _table(slide, rows, *, left, top, width, height, header=True, font=13, col_widths=None):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width, height)
    table = shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Pt(6)
            cell.margin_right = Pt(6)
            cell.text_frame.word_wrap = True
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
            run = para.add_run()
            run.text = str(val)
            is_header = header and r_idx == 0
            _set_run(run, size=font, bold=is_header, color=WHITE if is_header else NAVY)
            cell.fill.solid()
            if is_header:
                cell.fill.fore_color.rgb = BLUE
            else:
                cell.fill.fore_color.rgb = LIGHT_BLUE if r_idx % 2 else WHITE
    return table


def _card(slide, left, top, width, height, value, label, *, value_color=NAVY):
    _rect(slide, left, top, width, height, LIGHT_BLUE)
    _, tf = _textbox(slide, left + Inches(0.12), top + Inches(0.18), width - Inches(0.24), Inches(0.7))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = value
    _set_run(r, size=28, bold=True, color=value_color)
    _, tf2 = _textbox(slide, left + Inches(0.12), top + Inches(0.88), width - Inches(0.24), height - Inches(1.05))
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = label
    _set_run(r2, size=13, color=GREY)


def _note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_title(prs):
    s = _blank(prs)
    _bg(s, NAVY)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.55), SLIDE_W, Inches(0.08))
    band.fill.solid()
    band.fill.fore_color.rgb = AMBER
    band.line.fill.background()

    _, tf = _textbox(s, Inches(0.9), Inches(2.75), Inches(11.5), Inches(2.0))
    r = tf.paragraphs[0].add_run()
    r.text = "When the science-talk detector is wrong"
    _set_run(r, size=36, bold=True, color=WHITE)
    p = tf.add_paragraph()
    r = p.add_run()
    r.text = "Three error analyses of human-reviewed classroom transcripts"
    _set_run(r, size=20, italic=True, color=RGBColor(0xBF, 0xD4, 0xEA))

    _, tf2 = _textbox(s, Inches(0.9), Inches(5.7), Inches(11.5), Inches(0.8))
    r = tf2.paragraphs[0].add_run()
    r.text = "Anita Zucker Center  ·  31 August 2026  ·  83 reviewed classroom sessions"
    _set_run(r, size=16, color=RGBColor(0x9F, 0xB8, 0xD0))
    _note(s, "Open with the human review, not the model. Reviewers checked utterances the detector already marked as possible science talk.")


def slide_one_number(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "The one number to remember")
    _card(s, Inches(0.6), Inches(1.7), Inches(3.9), Inches(2.3), "1 in 4", "flags the detector showed were real science talk", value_color=RED)
    _card(s, Inches(4.7), Inches(1.7), Inches(3.9), Inches(2.3), "1,984", "human-confirmed science utterances", value_color=GREEN)
    _card(s, Inches(8.8), Inches(1.7), Inches(3.9), Inches(2.3), "5,985", "flags a human said were not science", value_color=RED)
    _bullets(s, [
        ("A flag is an utterance the detector marked as possible science talk.", 0, False, None),
        ("Reviewers then said yes or no. They only saw the flagged lines, not the rest of the transcript.", 0, False, None),
        ("So 24.9% is how often a flag was right. It is not how much science talk was missed.", 0, True, BLUE),
    ], top=Inches(4.3), size=18, height=2.4, gap=10)
    _footer(s, "2")
    _note(s, "Stay on this slide until the difference between flags and missed science is clear.")


def slide_how_review_works(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "How a review works")
    steps = [
        ("1", "Classroom talk", "Every spoken line from a session."),
        ("2", "Detector flags some lines", "Only those lines go to a reviewer."),
        ("3", "A person says yes or no", "Yes = science talk. No = a false alarm."),
    ]
    for i, (num, title, body) in enumerate(steps):
        left = Inches(0.6 + i * 4.2)
        _rect(s, left, Inches(1.85), Inches(3.9), Inches(3.4), LIGHT_BLUE)
        _, tf = _textbox(s, left + Inches(0.2), Inches(2.05), Inches(3.5), Inches(0.7))
        r = tf.paragraphs[0].add_run()
        r.text = num
        _set_run(r, size=28, bold=True, color=BLUE)
        _, tf2 = _textbox(s, left + Inches(0.2), Inches(2.75), Inches(3.5), Inches(0.7))
        r = tf2.paragraphs[0].add_run()
        r.text = title
        _set_run(r, size=18, bold=True, color=NAVY)
        _, tf3 = _textbox(s, left + Inches(0.2), Inches(3.5), Inches(3.5), Inches(1.4))
        r = tf3.paragraphs[0].add_run()
        r.text = body
        _set_run(r, size=16, color=GREY)
    _footer(s, "3")
    _note(s, "This is the whole study sample: flagged lines that a person then judged.")


def slide_can_cannot(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "What these analyses can and cannot say")
    _rect(s, Inches(0.6), Inches(1.7), Inches(5.9), Inches(4.6), LIGHT_BLUE)
    _rect(s, Inches(6.8), Inches(1.7), Inches(5.9), Inches(4.6), RGBColor(0xF7, 0xEE, 0xE6))
    _, tf = _textbox(s, Inches(0.85), Inches(1.9), Inches(5.4), Inches(0.5))
    r = tf.paragraphs[0].add_run()
    r.text = "Can say"
    _set_run(r, size=20, bold=True, color=GREEN)
    _bullets(s, [
        ("Of the lines the detector showed, how often a person agreed.", 0, False, None),
        ("What the false alarms usually look like.", 0, False, None),
        ("What happens if a simple filter is added, or a shortcut path is turned off.", 0, False, None),
    ], left=Inches(0.85), top=Inches(2.5), width=Inches(5.4), height=Inches(3.5), size=16, gap=12)

    _, tf2 = _textbox(s, Inches(7.05), Inches(1.9), Inches(5.4), Inches(0.5))
    r = tf2.paragraphs[0].add_run()
    r.text = "Cannot say"
    _set_run(r, size=20, bold=True, color=RED)
    _bullets(s, [
        ("How much real science talk never got flagged.", 0, False, None),
        ("Overall accuracy on every line in the classroom.", 0, False, None),
        ("Whether a filter that looks good on flags would hide science that was never shown.", 0, False, None),
    ], left=Inches(7.05), top=Inches(2.5), width=Inches(5.4), height=Inches(3.5), size=16, gap=12)
    _footer(s, "4")


def slide_three_questions(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Three questions")
    rows = [
        ("1", "What kinds of mistakes are the false alarms?", "Look at 400 rejected flags and label the type of talk."),
        ("2", "Is the fast shortcut any good?", "Some flags skip the careful second check. Raise its bar, or turn it off?"),
        ("3", "Can obvious junk be dropped with simple rules?", "Filter Look., Two., Circle. before the careful check, without dropping Iguana."),
    ]
    for i, (num, q, how) in enumerate(rows):
        top = Inches(1.65 + i * 1.65)
        _rect(s, Inches(0.6), top, Inches(12.1), Inches(1.5), LIGHT_BLUE)
        _, tf = _textbox(s, Inches(0.8), top + Inches(0.18), Inches(1.0), Inches(1.1))
        r = tf.paragraphs[0].add_run()
        r.text = num
        _set_run(r, size=32, bold=True, color=BLUE)
        _, tf2 = _textbox(s, Inches(1.9), top + Inches(0.22), Inches(10.5), Inches(0.55))
        r = tf2.paragraphs[0].add_run()
        r.text = q
        _set_run(r, size=20, bold=True, color=NAVY)
        _, tf3 = _textbox(s, Inches(1.9), top + Inches(0.8), Inches(10.5), Inches(0.5))
        r = tf3.paragraphs[0].add_run()
        r.text = how
        _set_run(r, size=15, color=GREY)
    _footer(s, "5")


def slide_a1_method(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Analysis 1. What kinds of mistakes?", subtitle="How this was done")
    _bullets(s, [
        ("Start with all 5,985 flags a person rejected.", 0, False, None),
        ("Draw 400 of them, not purely at random.", 0, False, None),
        ("Take 100 from each mix of speaker and scoring path: adult or child, careful check or fast shortcut.", 0, False, None),
        ("Give each line one label (pointing, counting, classroom management, and so on).", 0, False, None),
        ("Simple rules first. Then a person labeled the 150 the rules could not place.", 0, False, None),
        ("Percents are of this 400, not of every reject. The 400 shows what error types exist.", 0, True, BLUE),
    ], top=Inches(1.65), size=18, gap=10)
    _footer(s, "6")


def slide_a1_types(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Analysis 1. The kinds of false alarms")
    _table(
        s,
        [
            ["Kind of talk", "Share of the 400", "Everyday example"],
            ["Pointing / getting attention", "21.5%", "Look at that."],
            ["Classroom management", "17.5%", "Have a seat please."],
            ["Naming an everyday object", "16.0%", "It's a robot."],
            ["Maybe science, reviewer said no", "11.0%", "How does that work?"],
            ["Shape or color", "9.5%", "What shape is that?"],
            ["Counting or how many", "9.3%", "One, two, three, four."],
            ["Play, feelings, writing, other", "15.3%", "Play-Doh is full."],
        ],
        left=Inches(0.55),
        top=Inches(1.55),
        width=Inches(12.2),
        height=Inches(5.15),
        font=15,
    )
    _footer(s, "7")
    _note(s, "Three types (pointing, management, object naming) are 55% of the 400.")


def slide_a1_findings(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Analysis 1. What stood out")
    _bullets(s, [
        ("Children's false alarms are mostly pointing. Look! and Look at that.", 0, False, None),
        ("The fast shortcut's false alarms are mostly management and materials. Sit down. Play-Doh. Paper.", 0, False, None),
        ("The careful check is fooled by Look., Circle., and one-word names that sound like observing.", 0, False, None),
        ("About 1 in 9 rejects could reasonably have been science. Do not treat every reject as a clean wrong answer.", 0, True, AMBER),
    ], top=Inches(1.75), size=20, gap=14)
    _footer(s, "8")


def slide_a2_two_paths(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(
        s,
        "Analysis 2. Two ways a line becomes a flag",
        subtitle="A flag is a spoken line the detector already marked as possible science talk. This analysis asks: of those flags, which scoring path was used, and how often was it right?",
    )
    _rect(s, Inches(0.6), Inches(1.85), Inches(5.9), Inches(4.55), LIGHT_BLUE)
    _rect(s, Inches(6.8), Inches(1.85), Inches(5.9), Inches(4.55), RGBColor(0xF7, 0xEE, 0xE6))
    _, tf = _textbox(s, Inches(0.85), Inches(2.05), Inches(5.4), Inches(0.9))
    r = tf.paragraphs[0].add_run()
    r.text = "Careful second check"
    _set_run(r, size=22, bold=True, color=BLUE)
    _bullets(s, [
        ("A large language model reads the line and scores it.", 0, False, None),
        ("Most of these flags get a score of 0.90.", 0, False, None),
        ("5,642 reviewed flags.", 0, False, None),
        ("31.2% were real science talk.", 0, True, GREEN),
    ], left=Inches(0.85), top=Inches(3.05), width=Inches(5.4), height=Inches(3.1), size=16, gap=8)

    _, tf2 = _textbox(s, Inches(7.05), Inches(2.05), Inches(5.4), Inches(0.9))
    r = tf2.paragraphs[0].add_run()
    r.text = "Fast shortcut"
    _set_run(r, size=22, bold=True, color=AMBER)
    _bullets(s, [
        ("No second check. Just how similar the line is to known science examples.", 0, False, None),
        ("Every one of these scores sits between 0.40 and 0.48.", 0, False, None),
        ("2,327 reviewed flags.", 0, False, None),
        ("9.6% were real science talk.", 0, True, RED),
    ], left=Inches(7.05), top=Inches(3.05), width=Inches(5.4), height=Inches(3.1), size=16, gap=8)
    _footer(s, "9")
    _note(s, "A flag is already on the review sheet. This slide only splits those flags by how they got there.")


def slide_a2_method(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Analysis 2. Raise the bar, or turn the shortcut off?", subtitle="How this was done")
    _bullets(s, [
        ("Split the 7,969 reviewed flags into careful-check vs fast-shortcut.", 0, False, None),
        ("Measure how often each group was right.", 0, False, None),
        ("For the shortcut only, look at thin score bands (0.40 to 0.42, 0.42 to 0.44, and so on).", 0, False, None),
        ("Then pretend a new rule: keep every careful-check flag. Keep a shortcut flag only if its score is at least t.", 0, False, None),
        ("Repeat for several values of t. At 0.50, every shortcut flag is gone.", 0, False, None),
    ], top=Inches(1.7), size=18, gap=10)
    _footer(s, "10")


def slide_a2_sweep(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(
        s,
        "Analysis 2. Raising the shortcut bar a little does not help",
        subtitle="Each row is a pretend new rule for shortcut flags only. Careful-check flags stay. Then: how many flags would still be shown, and how often would they be real science?",
    )
    _table(
        s,
        [
            [
                "Pretend new rule (shortcut flags only)",
                "Flags still shown",
                "Real science still shown",
                "Real science no longer shown",
                "Of those still shown, how many were science",
            ],
            [
                "Leave the current rule: show a shortcut line if its score is 0.40 or higher",
                "7,969",
                "1,984",
                "0",
                "24.9%",
            ],
            [
                "Raise the bar: show a shortcut line only if its score is 0.42 or higher",
                "6,777",
                "1,852",
                "132",
                "27.3%",
            ],
            [
                "Raise the bar further: show a shortcut line only if its score is 0.45 or higher",
                "5,856",
                "1,786",
                "198",
                "30.5%",
            ],
            [
                "Stop showing shortcut lines as science talk (none of them score 0.50 or higher)",
                "5,642",
                "1,760",
                "224",
                "31.2%",
            ],
        ],
        left=Inches(0.35),
        top=Inches(1.68),
        width=Inches(12.6),
        height=Inches(3.9),
        font=12,
        col_widths=[Inches(5.35), Inches(1.7), Inches(1.85), Inches(1.95), Inches(1.75)],
    )
    _bullets(s, [
        ("There is no clean cutoff inside 0.40 to 0.48. The 0.42 to 0.44 band is worse than 0.40 to 0.42.", 0, False, None),
        ("Turning the shortcut off moves the share that were science from 24.9% to 31.2%. It also drops 224 confirmed science lines.", 0, True, BLUE),
    ], top=Inches(5.7), size=15, height=1.2, gap=6)
    _footer(s, "11")


def slide_a3_method(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Analysis 3. Simple filters before the careful check", subtitle="How this was done")
    _bullets(s, [
        ("A filter here is a short rule that drops a line before the expensive second check.", 0, False, None),
        ("Four rules: exact Look. / Look at that.; exact counting; exact shape or color; 1 or 2 words.", 0, False, None),
        ("Hold out science names so Iguana., Jellyfish., Habitat., and It's hot. are not dropped.", 0, False, None),
        ("Apply each rule to all 7,969 reviewed flags. Count junk removed and real science lost.", 0, False, None),
        ("Then combine the filters with turning the shortcut off.", 0, False, None),
    ], top=Inches(1.7), size=18, gap=10)
    _footer(s, "12")


def slide_a3_gates(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Analysis 3. What each filter would do")
    _table(
        s,
        [
            ["Filter", "Lines dropped", "Real science lost", "False alarms cut", "Share remaining that were science"],
            ["Look. / Look at that.", "600", "31", "569", "26.5%"],
            ["Counting / how many", "158", "1", "157", "25.4%"],
            ["Shape or color", "148", "2", "146", "25.3%"],
            ["Those three together", "906", "34", "872", "27.6%"],
            ["1 or 2 words (names kept)", "1,423", "100", "1,323", "28.8%"],
        ],
        left=Inches(0.4),
        top=Inches(1.5),
        width=Inches(12.5),
        height=Inches(4.35),
        font=14,
    )
    _bullets(s, [
        ("Counting is almost free: 157 false alarms gone, 1 real science line lost.", 0, True, GREEN),
    ], top=Inches(6.0), size=16, height=0.7, gap=4)
    _footer(s, "13")


def slide_a3_holdout(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Analysis 3. Keep real science names")
    _bullets(s, [
        ("A 1 or 2 word filter without a name list would drop 244 confirmed science lines.", 0, False, None),
        ("With the name list, that falls to 100. The list saved 144 short names.", 0, True, GREEN),
        ("Saved examples: Iguana.  Caterpillar.  Jellyfish.  Cloudy?  It's hot.  An elephant.", 0, False, None),
        ("Still slipping through: What's hibernation?  A bumblebee.  Non-living thing!", 0, False, None),
        ("Do not ship a short-word filter until that leftover list is coded and added.", 0, True, AMBER),
    ], top=Inches(1.75), size=18, gap=12)
    _footer(s, "14")


def slide_combined(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Putting the two changes together")
    _table(
        s,
        [
            ["If this were the rule", "Share that were science", "Confirmed science kept"],
            ["Leave everything as it is now", "24.9%", "1,984 of 1,984"],
            ["Add the Look / count / shape filter only", "27.6%", "1,950 of 1,984"],
            ["Turn the fast shortcut off only", "31.2%", "1,760 of 1,984"],
            ["Do both", "35.0%", "1,727 of 1,984 (87%)"],
        ],
        left=Inches(0.55),
        top=Inches(1.6),
        width=Inches(12.2),
        height=Inches(3.7),
        font=16,
    )
    _bullets(s, [
        ("Doing both still keeps 87% of the confirmed science flags and cuts a large pile of false alarms.", 0, True, BLUE),
        ("That is less work for reviewers. It is not yet proof that less real science is missed in the full transcript.", 0, False, None),
    ], top=Inches(5.5), size=16, height=1.3, gap=8)
    _footer(s, "15")


def slide_next(prs):
    s = _blank(prs)
    _bg(s)
    _title_bar(s, "Recommended next change")
    _bullets(s, [
        ("Ship the Look / count / shape filter now. Expected: 872 fewer false alarms, 34 confirmed science lines dropped.", 0, True, GREEN),
        ("Treat the fast shortcut as a separate decision. Do not nibble 0.40 up to 0.42.", 0, False, None),
        ("Do not ship a 1 or 2 word filter yet. Grow the science-name list first.", 0, False, None),
        ("Do not train on all 5,985 rejects. Skip the maybe-science slice. Prefer shortcut management, Look, count, and shape.", 0, False, None),
        ("Still needed: 20 to 30 unflagged lines per classroom, coded by the same reviewers, to see what the detector never showed.", 0, True, AMBER),
    ], top=Inches(1.7), size=18, gap=12)
    _footer(s, "16")


def build(output_path: Path = OUTPUT_PATH) -> Path:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    slide_title(prs)
    slide_one_number(prs)
    slide_how_review_works(prs)
    slide_can_cannot(prs)
    slide_three_questions(prs)
    slide_a1_method(prs)
    slide_a1_types(prs)
    slide_a1_findings(prs)
    slide_a2_two_paths(prs)
    slide_a2_method(prs)
    slide_a2_sweep(prs)
    slide_a3_method(prs)
    slide_a3_gates(prs)
    slide_a3_holdout(prs)
    slide_combined(prs)
    slide_next(prs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)
    return output_path


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
