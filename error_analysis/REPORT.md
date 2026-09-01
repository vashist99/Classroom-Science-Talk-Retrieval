# Error analysis report: FP taxonomy, cosine fallback, and pre-rerank gates

**Branch:** `error-analysis/fp-taxonomy-and-gates`
**Date:** 31 August 2026
**Sample:** 83 human-coded Y2 workbooks, 16 classrooms, 7,969 reviewed model `SCIENCE_TALK` flags (1,984 confirmed, 5,985 rejected). Precision of the current deploy is **24.9%**.

This memo presents three analyses on those reviewed flags. Each section starts with the **question**, then **how this was done**, then the tables.

---

## Shared setup (all three analyses)

**What the model did.** For each classroom utterance it either said SCIENCE_TALK or not. Reviewer sheets include **only** the SCIENCE_TALK ones.

**What the human did.** For each of those flags, a reviewer marked SCIENCE_TALK (agree) or NOT_SCIENCE_TALK (reject).

**What that means.** Measurable: *of the flags the model showed, how often was the model right?* Not measurable: *how much real science talk the model never showed.*

| Term | Meaning here |
|---|---|
| Flag | An utterance the model labeled SCIENCE_TALK |
| True positive (TP) | Human also said SCIENCE_TALK |
| False positive (FP) | Human said NOT_SCIENCE_TALK |
| Precision | TP / (TP + FP). Current value: **24.9%** |
| Lost TP among flagged | A confirmed science flag a new rule would drop. Not a miss in the full transcript. |
| LLM path | Flag scored by the large-language-model reranker (usually score = 0.90) |
| Cosine path | Flag scored by embedding similarity only. No LLM. Score 0.40–0.48. |

| Done | Not done |
|---|---|
| Precision of current flags: 24.9% (1,984 / 7,969) | Recall (science talk the model never flagged) |
| Taxonomy of 400 FPs | Accuracy over all classroom talk |
| Cosine raise / drop | |
| Look / count / shape / short gates | |

---

## Analysis 1 — What kinds of mistakes are the false positives?

**Question.** When a human rejects a flag, *what kind of talk was it?* (pointing, counting, management, a real science maybe, …)

**How this was done**

1. Start from all **5,985** human rejects.
2. Draw **400** of them, not purely at random.
3. Split into four equal buckets of 100, so each group is represented:
   - adult + LLM
   - adult + cosine
   - child + LLM
   - child + cosine
4. Inside each bucket, take utterances from as many classrooms as possible.
5. Assign **one** label per utterance (codebook below).
6. Auto-label first with simple rules (`Look.` → deictic, `Two.` → counting, …).
7. Hand-label the **150** the rules could not place.

**How to read the counts.** Percents are *of this 400*, not of all 5,985. The sample was forced to 50% cosine; in the full data cosine is only 35% of FPs. The 400 shows *what error types exist*, not a census of every reject.

**Limit.** Classroom 21 is 119 of 400 because it produced many FPs. High-precision rooms are thin in the sample.

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

## Analysis 2 — Raise the cosine cutoff, or drop cosine-only flags?

**Question.** Some flags never went through the LLM. They were labeled SCIENCE_TALK because embedding similarity (cosine) was at least 0.40. Are those flags any good? If 0.40 is raised, or those flags are dropped, what happens to precision?

**How this was done**

1. Split the 7,969 reviewed flags into two groups: **LLM** vs **cosine-only**.
2. Compute precision in each group (Table 2).
3. For cosine-only flags only, bin the cosine score (0.40–0.42, 0.42–0.44, …) and compute precision in each bin (Table 3).
4. Simulate a new rule: **keep every LLM flag**. Keep a cosine flag only if its cosine is ≥ *t*. Repeat for t = 0.40, 0.42, …, 0.50 (Table 4).
5. At t = 0.50, no cosine flag survives (all of them are below 0.50). That row is “drop cosine-only SCIENCE_TALK.”

**How to read Table 4**

- Remaining flags = what would still appear on the review sheet.
- Remaining TP = confirmed science that would still be kept.
- Lost TP among flagged = confirmed science that would be dropped. Still not full-transcript recall.

**Result (short).** Cosine-only flags are 9.6% precise. All of them sit in 0.40–0.48. Raising 0.40 a little does not find a clean cut. Dropping them moves overall precision from 24.9% to 31.2% and costs 224 confirmed flags.

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

**Recommendation:** do not nibble the threshold. Either (a) stop writing cosine-only rows as `SCIENCE_TALK` (require LLM confirmation, even if those rows wait on the budget cap), or (b) keep them in the workbook but on a separate “cosine-only, not for the analytic sample” sheet so reviewers are not counting 9.6%-precise rows as model science talk.

---

## Analysis 3 — Drop obvious junk *before* the LLM, without dropping real science names?

**Question.** A lot of FPs are `Look.`, `Two.`, `Circle.` Can those be dropped with simple rules *before* the expensive reranker — without also dropping real one-word science (`Iguana.`, `Jellyfish.`)?

**How this was done**

1. Four simple “drop this utterance” rules (a **gate**):
   - exact look-family (`Look.` / `Look at that.`)
   - exact count / “how many”
   - exact shape or color (`Circle.` / `What color?`)
   - short: 1 or 2 words
2. A **hold-out list** of science-content names (128 words: iguana, habitat, pollen, jellyfish, …). If the utterance is basically one of those names, it is **not gated**.
3. Each gate is applied to all 7,969 reviewed flags.
4. For each gate, count:
   - how many flags it would remove
   - how many of those were TP vs FP
   - precision of what remains
5. Gates are then stacked with the cosine decision from Analysis 2 (Table 6).

**How to read Table 5**

- Precision of removed = if this gate deletes a row, how often was that row actually science? Low is good (junk is being deleted).
- Remaining precision = precision of the sheet *after* the gate.

**Hold-out check.** The 1–2 word gate without the hold-out deletes 244 TPs. With it, 100. The list saved 144 short confirmed names. It is not finished: 100 confirmed names still slip through (`What's hibernation?`, `A bumblebee.`).

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
