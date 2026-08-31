# Error analysis report: FP taxonomy, cosine fallback, and pre-rerank gates

**Branch:** `error-analysis/fp-taxonomy-and-gates`
**Date:** 31 August 2026
**Sample:** 83 human-coded Y2 workbooks, 16 classrooms, 7,969 reviewed model `SCIENCE_TALK` flags (1,984 confirmed, 5,985 rejected). Precision of the current deploy is **24.9%**.

This memo does the first three analyses proposed after that review. It does **not** estimate recall. Unflagged utterances were never shown to reviewers, so every “lost TP” figure below is a TP *among already-flagged rows*, not a TP in the full transcript.

Reproduce with `python error_analysis/run_analyses.py` (requires `Human coded/*.xlsx`). Artifacts land in `error_analysis/output/`.

---

## What we measured, and what we did not

| Quantity | Status |
|---|---|
| Precision of current SCIENCE_TALK flags | 24.9% (1,984 / 7,969) |
| False-positive taxonomy on a stratified 400 | Done (150 hand-coded residuals) |
| Effect of raising / dropping the cosine fallback | Done, on flagged rows only |
| Effect of look / count / shape / short gates + science-name hold-out | Done, on flagged rows only |
| Recall / missed science talk in unflagged speech | **Not measured** — needs a new unflagged sample |

Sheets are `science_only=true`: they contain only model `SCIENCE_TALK`. Humans then marked `SCIENCE_TALK` or `NOT_SCIENCE_TALK`. That is a precision sample, not a confusion matrix over all talk.

---

## Analysis 1 — False-positive taxonomy (n = 400)

### Design

- Population: 5,985 human-rejected flags.
- Sample: **400**, seed `20260831`.
- Stratification: **100 per speaker × scoring-path cell** (adult/child × llm/cosine), classrooms spread inside each cell.
- Coding: rule-based codebook first (`error_analysis/codebook.py`), then **hand codes for all 150 residuals** that the rules left as `other` (`error_analysis/hand_codes.py`).
- Labels are mutually exclusive. First matching rule wins; hand codes override for the residual set.

The sample is balanced on speaker and scoring path, so it **oversamples cosine** (50% of the sample vs 35% of all FPs) and **low-precision rooms** (Classroom 21 alone is 119 of 400). Sample percentages are for coverage of error types, not population prevalence. Rule-based labels on all 5,985 FPs are reported beside the sample as a lower-bound check on the cheap buckets.

### Codebook

| Label | Definition |
|---|---|
| `deictic_attention` | Attention-getting or pointing without science content (`Look.`, `I saw that.`) |
| `classroom_management` | Directives, transitions, seating, clean-up, “try / ready / please” |
| `object_naming` | Everyday object or animal label with no scientific framing (`Chicken?`, `That's a box.`) |
| `shape_color` | Naming a shape or color, or “what color / what shape”, as the point of the utterance |
| `counting_math` | Rote numbers or “how many” with no investigation |
| `borderline_science` | Reviewer said no, but the utterance could reasonably be science practice (heat, living/dead, “what happened”, a science word in context) |
| `play_materials` | Play-Doh, paper, craft, toy play |
| `social_affect` | Possession, feelings, social bids |
| `literacy` | Letters, writing, names-as-text, storybook read-aloud |
| `other` | Residual after rules + hand review (6.5% of the sample) |

### Sample results

| Label | n | % of 400 | Adult | Child | Cosine | LLM |
|---|---:|---:|---:|---:|---:|---:|
| deictic_attention | 86 | 21.5 | 27 | 59 | 34 | 52 |
| classroom_management | 70 | 17.5 | 38 | 32 | 53 | 17 |
| object_naming | 64 | 16.0 | 32 | 32 | 28 | 36 |
| borderline_science | 44 | 11.0 | 21 | 23 | 13 | 31 |
| shape_color | 38 | 9.5 | 28 | 10 | 9 | 29 |
| counting_math | 37 | 9.3 | 25 | 12 | 22 | 15 |
| other | 26 | 6.5 | 14 | 12 | 12 | 14 |
| social_affect | 15 | 3.8 | 4 | 11 | 12 | 3 |
| play_materials | 12 | 3.0 | 7 | 5 | 10 | 2 |
| literacy | 8 | 2.0 | 4 | 4 | 7 | 1 |

**Three error types account for 55% of the coded sample:** deictic attention, classroom management, and object naming.

### What the splits show

- **Child FPs are deictic.** 59 of 200 child rejects are `deictic_attention` vs 27 of 200 adult. Child talk is short pointing (`Look!`, `Look at that.`).
- **Cosine FPs are management and materials.** 53 of 200 cosine rejects are `classroom_management` vs 17 of 200 LLM rejects. Play and literacy also concentrate on the cosine path. The 0.40 cosine floor is promoting transition talk into the review sheet.
- **LLM FPs are deictic, names, and shapes.** The reranker is not confused by “sit down”; it is confused by `Look.`, `Circle.`, and one-word object labels that share a register with science observation.
- **About 1 in 9 rejects is borderline.** 44 of 400 (11%) could reasonably have been science. Some of the “error” is label noise or a thin operational definition, not a broken retriever. Do not dump all 5,985 FPs into the negative set without dropping this slice.

### Examples (from the coded sample)

| Label | Examples |
|---|---|
| deictic_attention | `Look at that.` / `I saw that.` / `What do you see?` |
| classroom_management | `Addy, have a seat please.` / `Carry. Carry. Carry.` / `Okay, put your wet clothes in there.` |
| object_naming | `The box!` / `It's a robot.` / `Do you have a Christmas tree?` |
| shape_color | `What shape is that?` / `W, circle.` / `shape where all points around the outside are the same distance from its` (a circle definition) |
| counting_math | `Two.` / `Eight.` / `One, two, three, four.` / `One. Two. Three. Four. Five.` |
| borderline_science | `How does that work?` / `It's getting bigger now.` / `It's a butterfly.` / `What's going to happen now?` |
| play_materials | `Play-Doh is full.` / `I'm going to throw a ball.` |
| literacy | `Please, um, write your name on that side.` / storybook `Mother Goat put the rocks…` |

The coded sheet is `error_analysis/output/fp_sample_400_coded.csv`.

---

## Analysis 2 — Cosine fallback: raise it, or drop it

Every cosine-only flag in this sample has cosine **0.40–0.48**. None would survive a 0.50 threshold. The LLM path never uses that band: its scores collapse at 0.90–0.93.

| Path | Reviewed flags | TP | FP | Precision |
|---|---:|---:|---:|---:|
| Cosine fallback | 2,327 | 224 | 2,103 | **9.6%** |
| LLM rerank | 5,642 | 1,760 | 3,882 | **31.2%** |
| Combined (current) | 7,969 | 1,984 | 5,985 | **24.9%** |

Cosine is **29% of reviewed flags** and **35% of all false positives**, and it is the main reason overall precision sits at 24.9% instead of 31.2%.

### Cosine score bins (reviewed cosine flags only)

| Cosine bin | n | Precision |
|---|---:|---:|
| 0.40–0.42 | 1,192 | 11.1% |
| 0.42–0.44 | 689 | 6.4% |
| 0.44–0.46 | 370 | 10.0% |
| 0.46–0.48 | 68 | 14.7% |
| 0.48–0.50 | 8 | 12.5% |

Raising the threshold inside 0.40–0.48 does **not** find a clean band. The 0.42–0.44 slice is *worse* than 0.40–0.42. There is no useful operating point except “keep the floor at 0.40” or “do not promote cosine-only rows to SCIENCE_TALK”.

### Threshold sweep (LLM flags kept; cosine flags kept only if cosine ≥ t)

| Cosine min | Remaining flags | Remaining TP | Lost TP among flagged | Precision |
|---:|---:|---:|---:|---:|
| 0.40 (current) | 7,969 | 1,984 | 0 | 24.9% |
| 0.42 | 6,777 | 1,852 | 132 | 27.3% |
| 0.44 | 6,088 | 1,808 | 176 | 29.7% |
| 0.45 | 5,856 | 1,786 | 198 | 30.5% |
| 0.46 | 5,718 | 1,771 | 213 | 31.0% |
| 0.48 | 5,650 | 1,761 | 223 | 31.2% |
| **0.50 (drop all cosine)** | **5,642** | **1,760** | **224** | **31.2%** |

Dropping cosine-only SCIENCE_TALK labels:

- Precision **24.9% → 31.2%**
- Removes 2,103 FPs and 224 TPs among flagged (11.3% of all confirmed science flags)
- Those 224 cosine TPs are longer than typical FPs (mean 6.8 words, median 5). 43 of them are science-content names the hold-out lexicon would keep. Examples that *would* be lost: `Is it sunny outside?`, `They sleep all winter long.`, `Because lemons are actually sour.`, `A mouse.`

**Recommendation:** do not nibble the threshold. Either (a) stop writing cosine-only rows as `SCIENCE_TALK` (require LLM confirmation, even if that means they wait on the budget cap), or (b) keep them in the workbook but on a separate “cosine-only, not for the analytic sample” sheet so reviewers are not counting 9.6%-precise rows as model science talk.

---

## Analysis 3 — Gates before Track B, with a science-name hold-out

Gates fire only when the utterance is **not** a science-content name (`Iguana.`, `Jellyfish.`, `Habitat.`, `It's hot.`, `A kangaroo.`, …). The lexicon is in `codebook.py` (128 tokens, built from TP-enriched words plus confirmed short TPs).

### Per-gate effect on the 7,969 reviewed flags

| Gate | Flags removed | TP removed | FP removed | Precision of removed | Remaining precision |
|---|---:|---:|---:|---:|---:|
| Exact look-family | 600 | 31 | 569 | 5.2% | 26.5% |
| Exact count / “how many” | 158 | 1 | 157 | **0.6%** | 25.4% |
| Exact shape or color | 148 | 2 | 146 | 1.4% | 25.3% |
| **Conservative (look ∪ count ∪ shape)** | **906** | **34** | **872** | **3.8%** | **27.6%** |
| 1–2 words (hold-out on) | 1,423 | 100 | 1,323 | 7.0% | 28.8% |
| All proposed gates | 1,703 | 126 | 1,577 | 7.4% | 29.7% |

The same 1–2 word rule **without** the hold-out removes 244 TPs instead of 100. The lexicon saved **144 short confirmed science names** (`Iguana.`, `Caterpillar.`, `Jellyfish.`, `Cloudy?`, `It's hot.`, `An elephant.`, …). That hold-out is not optional if you gate on length.

A blanket short gate still costs 100 confirmed TPs that the lexicon missed (`What's hibernation?`, `A flower.`, `A bumblebee.`, `Non-living thing!`, `chasing prey.`, `Square.`). Expand the lexicon before shipping a length gate — or do not ship a length gate yet.

### Combined with the cosine decision

| Policy | Remaining flags | Remaining TP | Lost TP among flagged | Precision |
|---|---:|---:|---:|---:|
| Current deploy | 7,969 | 1,984 | 0 | 24.9% |
| Conservative gate only | 7,063 | 1,950 | 34 | 27.6% |
| Drop cosine only | 5,642 | 1,760 | 224 | 31.2% |
| **Drop cosine ∪ conservative gate** | **4,931** | **1,727** | **257** | **35.0%** |
| Drop cosine ∪ all gates (incl. short) | 4,355 | 1,638 | 346 | 37.6% |

---

## Recommended next change

**Ship the conservative gate now. Decide cosine separately after a recall sample.**

1. **Before Track B (and before writing the reviewer sheet):** drop exact look-family, exact count/how-many, and exact shape/color, unless the utterance is in the science-name hold-out. Expected effect on this sample: **−872 FPs, −34 TPs, precision 24.9% → 27.6%**. Count-only is almost free (1 TP lost).
2. **Do not promote cosine-only rows to `SCIENCE_TALK` at 0.40.** Either require LLM confirmation or park them off the analytic sheet. Expected effect: **24.9% → 31.2%**, at the cost of 224 flagged TPs whose recall impact is unknown.
3. **Together, those two moves take precision to 35.0%** and still keep 1,727 of 1,984 confirmed flags (87%).
4. **Do not ship the 1–2 word gate yet.** It is the largest remaining FP sink, but it still deletes 100 confirmed science names after the current lexicon. Code those 100, grow the hold-out, then revisit.
5. **Do not train on all 5,985 FPs as hard negatives.** Drop the `borderline_science` slice (~11% of the coded sample) and prefer cosine-path management / look / count / shape rejects.

The missing measurement is still a **sample of unflagged utterances** (20–30 screened adult/child rows per classroom). Until that exists, 35% precision is a reviewer-load win, not a proven net gain in science-talk recovery.

---

## Files

| Path | Role |
|---|---|
| `error_analysis/codebook.py` | Taxonomy rules, gates, science-name lexicon |
| `error_analysis/hand_codes.py` | Hand labels for the 150 residual sample rows |
| `error_analysis/load_reviews.py` | Reader for `Human coded/*.xlsx` |
| `error_analysis/run_analyses.py` | Stratified sample + all three analyses |
| `error_analysis/output/fp_sample_400_coded.csv` | Coded 400 |
| `error_analysis/output/fp_label_counts.csv` | Sample vs rule-based population counts |
| `error_analysis/output/cosine_threshold_sweep.csv` | Analysis 2 |
| `error_analysis/output/gate_sweep.csv` | Analysis 3 |
| `error_analysis/output/analysis_summary.json` | Machine-readable totals |
