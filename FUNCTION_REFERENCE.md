# Pipeline Function Reference

> A function-by-function map of the corpus-prep pipeline (`src/`). For the
> conceptual walkthrough of *why* each step exists, see
> [`DOCUMENTATION.md`](DOCUMENTATION.md). This document answers "what does this
> specific function do?".

Modules are listed in pipeline order. Within each module, functions appear in
source order. Private helpers (leading `_`) are included and marked _(helper)_.
Every module exposes a `run(...)` function that the orchestrator
(`pipeline.py`) calls; CLI entry points (`_parse_args`, `__main__`) are
summarized once rather than repeated.

---

## Table of contents

- [Step 0 — `llm_client_0.py` (shared LLM/embedding infrastructure)](#step-0--llm_client_0py)
- [Step 1 — `data_loader_1.py` (ingestion & normalization)](#step-1--data_loader_1py)
- [Step 2 — `subtypes_2.py` (sub-type labeling)](#step-2--subtypes_2py)
- [Step 3 — `negatives_3.py` (negative mining)](#step-3--negatives_3py)
- [Step 4 — `augment_4.py` (register-variant augmentation)](#step-4--augment_4py)
- [Step 5a — `embeddings_baseline_5.py` (frozen-baseline embeddings)](#step-5a--embeddings_baseline_5py)
- [Step 5b — `confidence_5.py` (confidence scoring & routing)](#step-5b--confidence_5py)
- [Step 6 — `review_6.py` (human review tooling)](#step-6--review_6py)
- [Step 7 — `splits.py` (train/val/test splits)](#step-7--splitspy)
- [Orchestrator — `pipeline.py`](#orchestrator--pipelinepy)

---

## Step 0 — `llm_client_0.py`

Shared, cached HTTP client for every LLM completion and embedding call. One
function (`cached_request`) is the single chokepoint all network traffic flows
through.

| Function | What it does |
|---|---|
| `_make_cache_key(prompt_version, model, endpoint, params)` _(helper)_ | Builds the deterministic cache key: SHA-256 hash of the canonical (sorted, separator-tight) JSON of the request. Identical requests always hash to the same key. |
| `_cache_path(cache_key)` _(helper)_ | Maps a cache key to its `cache/<key>.json` path. |
| `_load_from_cache(cache_key)` _(helper)_ | Returns the saved cache entry dict if the file exists, else `None`. |
| `_save_to_cache(cache_key, response)` _(helper)_ | Writes a cache entry atomically (write to `.json.tmp`, then rename) so a crash mid-write never corrupts the cache; cleans up the temp file on error. |
| `cached_request(*, api_key, url, endpoint, model, params, prompt_version)` | The core entry point. On a cache hit returns the stored `raw_response` (no network call). On a miss it POSTs to the endpoint, saves the full response + metadata + timestamp, and returns the raw response. Failed responses are not cached, so transient errors retry next run. |
| `connect_to_llm(api_key, model_type)` | Standalone connectivity smoke-test: reads the completion/embedding URLs and a model name from env (`augment` / `tracka` / `trackb`), fires one trivial "poem about a cat" request to each endpoint, and returns both raw JSON responses. Used only by the module's `__main__` block to verify credentials. |

---

## Step 1 — `data_loader_1.py`

Reads the 3-sheet seed xlsx and writes `corpus.parquet`, `seed_words.parquet`,
`category_defs.parquet`.

| Function | What it does |
|---|---|
| `normalize_text(s)` | NFKC-normalizes a string, collapses internal whitespace, strips ends. Passes NaN through unchanged. |
| `parse_cue_list(s)` | Splits a cue cell on `|`, `,`, or `;` into a list of trimmed strings; NaN/empty → `[]`. |
| `normalize_setting(s)` | Returns `(setting, topic)`: maps the raw setting to the controlled `SETTING_VOCAB` (unknown → `"Unknown"`) and peels off any `" / Topic"` suffix into a separate topic value. |
| `extract_transcript_ref(notes)` | Regex-extracts a `01-2_19LG #61`-style transcript id from free-text notes, else `None`. |
| `add_was_sci_coded(df)` | Adds a 0/1 `was_sci_coded` column flagging rows whose Notes mention `"SCI-coded"` (i.e. previously coded scientific by another project). |
| `clean_corpus(ex_utter_raw, *, verbose)` | The main Step-1 transform on the `Example utterances` sheet: normalize text, drop NaN/duplicate utterances, parse Tier2/Tier3 cue lists, normalize settings + split topic, extract transcript refs, assign `utt_0000…` ids, and rename columns to the analytical schema. |
| `clean_seed_words(seed_raw)` | Renames seed-word columns to snake_case and parses the `variants` cell into a list. |
| `clean_category_defs(cat_raw)` | Renames the category-definitions columns to a stable snake_case schema. |
| `validate_corpus(corpus)` | Hard DoD assertions: ≥195 rows, unique ids, no null utterances, ≥188 `SCIENCE_TALK` + ≥5 `NOT_SCIENCE_TALK`, no unexpected setting values. Fails loudly rather than silently fixing. |
| `load_and_clean(xlsx_path, *, verbose)` | Reads all three sheets and returns the cleaned `(corpus, seed_words, category_defs)` tuple (running `validate_corpus` on the corpus). |
| `write_processed(corpus, seed, cat, out_dir)` | Writes the three parquet artifacts and returns a name→path map. |
| `run(xlsx_path, out_dir, *, verbose)` | End-to-end Step 1: load → clean → validate → write. Returns the path map. |

---

## Step 2 — `subtypes_2.py`

Tags every utterance with one or more "soft practice" sub-types via a
three-stage cascade (rules → keyword scan → LLM zero-shot fallback) and writes
the augmented `corpus.parquet`.

| Function | What it does |
|---|---|
| `llm_subtype_stub(utterance)` | Deterministic fake classifier: always returns `(["observation"], 0.0, "subtype_v0_stub")` so stubbed rows are easy to find/redo later. Default fallback when not using a real LLM. |
| `build_subtype_prompt(utterance)` | Builds the versioned (`subtype_v3`) zero-shot classification prompt over the closed sub-type vocabulary, explicitly offering `not_science_shape` as an honest escape hatch and demanding strict JSON output. |
| `parse_subtype_response(raw_text)` | Defensive parser: extracts JSON (even when prose-wrapped), filters hallucinated labels against `SUBTYPES_ALL`, forces `not_science_shape` to stand alone, clamps confidence to [0,1], and falls back to `(["not_science_shape"], 0.0)` on total failure. |
| `make_real_llm_subtype_classifier(model, *, ...)` | Factory returning a real LLM-backed classifier callable `(utterance) -> SubtypeResult`. Reads API key/URL from env, calls `cached_request` at `temperature=0.0`, and degrades gracefully (parser fallback) on any endpoint error. Drop-in replacement for `llm_subtype_stub`. |
| `build_seed_index(df_seed)` | Builds a lookup mapping every lowercase term **and** variant → its seed-words row. |
| `map_cues_to_subtypes(tier2_cues, tier3_cues, seed_index)` | Stage-1 deterministic rule: turns hand-coded cue lists into sub-types via `TERM_OVERRIDES` and `CATEGORY_TO_SUBTYPES`; any Tier3 cue adds `content`. |
| `scan_text_for_subtypes(text, seed_index)` | Stage-2 keyword scan: word-boundary regex search of the utterance text for known seed terms/variants, mapping matches to sub-types (Tier3 → `content`, overrides, category mapping). |
| `assign_subtypes(df_corpus, df_seed, *, llm_classifier, skip_keyword_scan)` | Orchestrates the cascade per row, recording `subtype`, `subtype_source` (rule / keyword / rule+keyword / llm), `subtype_confidence`, and `subtype_prompt_version`. Non-LLM rows get confidence 1.0; only empty rows hit the LLM. `skip_keyword_scan=True` disables stage 2 (used for negatives). |
| `check_subtype_distribution(df_corpus, *, verbose)` | Hard-asserts every row has ≥1 sub-type; logs soft warnings for empty practice/content classes or any class exceeding 70% dominance; returns a stats dict. |
| `run(processed_dir, *, llm_classifier, verbose)` | End-to-end Step 2: load corpus + seeds, assign sub-types, validate distribution, persist back to `corpus.parquet`. |

---

## Step 3 — `negatives_3.py`

Builds the `NOT_SCIENCE_TALK` pool from three sources (transcript-mined,
LLM hard negatives, LLM seed-word non-science) and writes `negatives.parquet`.

### Prompt builders
| Function | What it does |
|---|---|
| `build_hard_negative_prompt(positive_utterance, subtypes, n)` | Prompt (`neg_hard_v1`) asking the LLM for N hard negatives that mirror a positive's syntax/length but are clearly about non-science topics. |
| `build_seed_nonscience_prompt(term, category, definition, n)` | Prompt (`neg_seed_v1`) asking for N utterances that use a science seed term in a clearly non-scientific sense, with the term's operational definition included so the model knows what to avoid. |
| `build_gate_prompt(text)` | Prompt (`neg_gate_v1`) asking the LLM to score an utterance from 0.0 (definitely science) to 1.0 (definitely not science) — the QC "gate". |

### Response parsers
| Function | What it does |
|---|---|
| `parse_negatives_json(raw_text)` | Best-effort extraction of the `negatives` list from an LLM response (tries whole-string JSON, then scans for embedded `{...}` objects). |
| `parse_gate_score(raw_text)` | Pulls a single 0–1 float out of the gate response and clamps it; returns `None` if none found. |

### LLM callables
| Function | What it does |
|---|---|
| `stub_llm_callable(prompt, prompt_version)` | Deterministic stub: gate prompts return `"0.85"`; generation prompts return canned classroom-management sentences (seeded by the prompt) in the expected JSON shape. Lets the whole step run offline. |
| `make_real_llm_callable(model, *, ...)` | Factory for a real LLM callable `(prompt, prompt_version) -> raw_text` using `cached_request` at `temperature=0.3`; returns `""` on any endpoint error so a single failure can't abort a multi-thousand-row loop. |

### Per-source generators
| Function | What it does |
|---|---|
| `generate_hard_negatives_for_positive(positive_row, *, n, llm_callable)` | Generates N `llm_hard_negative` rows anchored to one positive (carrying `anchor_utt_id`, prompt version, etc.). |
| `generate_seed_nonscience_for_term(seed_row, cat_definitions, *, n, llm_callable)` | Generates N `seed_word_nonscience` rows for one seed term (carrying `anchor_seed_term`, prompt version, etc.). |
| `mine_transcript_negatives(transcript_paths, seed_index, *, max_per_file)` | Mines plain-text transcripts (one utterance per line), keeping only lines that contain **no** seed token/variant; tags each as `transcript_clean` / `other_teacher_talk`. |
| `mine_transcript_xlsx(xlsx_path, seed_index, *, ...)` | Mines a hand-coded transcript workbook: keeps teacher rows (speaker regex), drops rows coded `SCI` (leakage), tags survivors `behavior_disapproving` (col E = `Y`) or `other_teacher_talk`, records `<sheet>!R<row>` provenance, with optional seed-word filtering and a total cap. |

### Structural checks
| Function | What it does |
|---|---|
| `passes_structural_checks(text, *, anchor_seed_term)` | Cheap deterministic gate before any LLM scoring: non-empty, 2–40 words, and (for seed-word negatives) must actually contain the seed term. |

### Sub-type assignment & gate scoring
| Function | What it does |
|---|---|
| `assign_subtypes_to_negatives(df_neg, df_seed, *, llm_classifier, skip_keyword_scan)` | Reuses Step 2's `assign_subtypes` on negatives so they share the positive sub-type vocabulary. Defaults `skip_keyword_scan=True` so negatives route through the LLM's `not_science_shape` escape hatch instead of keyword-matching into a practice label. |
| `gate_score_negatives(df_neg, *, llm_callable, only_llm_generated, verbose)` | Adds an `llm_gate_score` ∈ [0,1] per row. By default scores only LLM-generated rows (cheap — transcript rows are already human-coded); skipped rows get NaN. |

### Orchestration, validation, review
| Function | What it does |
|---|---|
| `build_negative_pool(df_corpus, df_seed, df_cat, *, ...)` | End-to-end pool builder: generate hard negatives + seed-word negatives + (optional) transcript negatives → structural filter → de-dup → cross-leak drop (any negative whose text equals a positive) → sub-type tagging → schema finalization with `neg_id`/provenance/review columns. Returns the negatives DataFrame. |
| `validate_negatives(df_neg, n_positives, *, target_ratio, require_three_sources, verbose)` | Hard + soft DoD checks: unique `neg_id`, non-null text, ≥3× positive count (warn), all three source types present, every practice/content sub-type has negatives. Returns a stats/warnings dict. |
| `sample_for_review(df_neg, n, *, seed, stratify_by_source)` | Returns N rows for hand-checking, stratified across `source_type` by default. |
| `export_negative_review_template(df_neg, processed_dir, *, n, seed, verbose)` | Writes a 50-row stratified hand-review template (`negatives_review_template.parquet`) with blank `review_human_label`/`review_notes` columns. |
| `compute_review_pass_rate(df_neg)` | Of the human-reviewed rows, computes the fraction truly marked `NOT_SCIENCE_TALK` (the ≥90% true-negative DoD gate). |
| `validate_negatives_review(df)` | Validates a filled negatives-quality review sheet (`review_human_label` ∈ NOT_SCIENCE_TALK/POSSIBLE_LEAK/UNCLEAR; blank = skip); returns reviewed rows only. |
| `apply_negatives_review(template_path, processed_dir, *, drop_unclear, verbose)` | Marks `human_verified`/`human_label` on `negatives.parquet`, **drops `POSSIBLE_LEAK`** rows (optionally `UNCLEAR`), backs up first, and reports the true-negative pass rate. CLI: `--apply-negatives-review` / `--drop-unclear`. |
| `run(processed_dir, *, ...)` | CLI/orchestrator entry: pick stub vs real LLM, locate transcripts (auto-detecting the default workbook), build the pool, optionally gate-score, validate, export the review template, and write `negatives.parquet`. |

---

## Step 4 — `augment_4.py`

Generates register-variant paraphrases of every positive anchor and writes
`pairs.parquet` (one row per anchor↔variant pair).

| Function | What it does |
|---|---|
| `_as_list(x)` _(helper)_ | Coerces parquet-roundtripped list-likes (numpy arrays, tuples, NaN, None) back into a plain Python list. |
| `setting_to_register(setting)` | Maps a corpus `setting` value to a register code (`LARGE_GROUP`/`SMALL_GROUP`/`INFORMAL`) or `None` if unknown. |
| `target_registers_for_anchor(source_register)` | Returns the registers to generate variants in: the *other two* for a known source, or all three for an unknown source (satisfying the ≥2-different-registers DoD). |
| `build_register_variant_prompt(anchor_text, target_register, *, ...)` | Builds the paraphrase prompt (`aug_var_v1`): preserve scientific meaning **and** the listed cue words, match the target register's style, differ from the anchor surface form, and report a self-score; demands strict JSON. |
| `parse_variants_json(raw_text)` | Extracts `variants: [{text, self_score}]` from the LLM response (handles whole-string JSON, embedded objects, bare-string items), clamping self-scores to [0,1]. |
| `stub_llm_callable(prompt, prompt_version)` | Deterministic variant generator: prefixes the anchor with a register-flavored opener and appends any missing cues, trivially satisfying cue-preservation and differs-from-anchor checks. |
| `check_cue_preservation(variant_text, tier2_cues, tier3_cues, *, threshold)` | Returns `(preserved_cues, preservation_pct, passed)` — fraction of anchor cues appearing as substrings of the variant; cue-less anchors pass vacuously (pct = 1.0). |
| `differs_from_anchor(variant_text, anchor_text)` | True if the (normalized) variant differs from the anchor and is non-empty — the surface-leak guard. |
| `generate_variants_for_anchor(anchor_row, *, n_per_register, llm_callable, target_registers)` | Generates variants of one anchor across its target registers, attaching cue-preservation results, differs flag, self-score, and provenance to each row. |
| `build_variant_pairs(df_corpus, *, n_per_register, llm_callable, drop_failed_leak_check, verbose)` | End-to-end pair builder over all positives: generate variants, optionally drop anchor-identical leaks, add `model_id`/`created_at`/`pair_id`, finalize the schema, and log register/preservation/coverage stats. |
| `validate_variant_pairs(df_pairs, df_corpus, *, ...)` | Enforces the four DoD checks: each anchor has variants in ≥2 non-source registers, ≥80% cue preservation (warn), zero surface leaks (assert), and every row has `model_id` + `prompt_version`; also reports anchor coverage. |
| `sample_pairs_for_review(df_pairs, n, *, seed, stratify_by_register)` | Pulls N variants for hand QC, stratified across registers. |
| `export_pairs_review_template(df_pairs, processed_dir, *, n, seed, verbose)` | Writes the stratified `pairs_review_template.parquet` with blank `review_human_ok`/`review_notes` columns. |
| `run(processed_dir, *, use_real_llm, n_per_register, verbose)` | CLI/orchestrator entry: pick stub vs real LLM, build pairs, validate, export the review template, and write `pairs.parquet`. |

---

## Step 5a — `embeddings_baseline_5.py`

Attaches a neutral `baseline_cosine` similarity to pairs and anchored
negatives, using the frozen off-the-shelf `nomic-embed-text-v1.5` encoder.

| Function | What it does |
|---|---|
| `_normalize_text_for_embedding(text)` _(helper)_ | Strips a value to a clean string; null/NaN → `""`. |
| `get_embedding_for_text(text, *, model, api_key, url, prompt_version, dim)` | Returns a `dim`-vector embedding for one text via `cached_request`, or a zero vector on empty input or any endpoint failure (so cosine degrades to 0.0 instead of crashing). |
| `embed_texts(texts, *, ...)` | Embeds an iterable of texts, de-duplicating within the batch and reusing per-text results; accepts an injected `embedder` for tests, otherwise builds the real cached embedder. Returns a stacked array aligned to the input order. |
| `cosine_similarity_pairs(a, b)` | Row-aligned pairwise cosine similarity between two equal-shaped matrices; zero-norm rows safely yield 0.0. |
| `add_baseline_cosine_to_pairs(df_pairs, *, model, verbose, embedder)` | Embeds anchor + variant texts and writes `baseline_cosine = cosine(anchor, variant)` onto a pairs frame. |
| `add_baseline_cosine_to_negatives(df_neg, df_corpus, *, model, verbose, embedder)` | For negatives that have an anchor positive (`llm_hard_negative`), writes `baseline_cosine = cosine(anchor_positive, neg_text)`; anchor-less rows get NaN. |
| `run(processed_dir, *, model, verbose, embedder)` | End-to-end Step 5a: load parquets, add `baseline_cosine` to both pairs and negatives in place, and log mean/median cosines. |

---

## Step 5b — `confidence_5.py`

Combines three signals into a `confidence` score and an auto/spot/review
`routing` label, and provides the human-audit scoring tooling.

| Function | What it does |
|---|---|
| `load_config(path)` | Loads the externalized thresholds/weights/band edges from `config/confidence.json` (raises if missing). |
| `band_score(x, lo, hi, falloff)` | Scores values in [0,1]: 1.0 inside `[lo, hi]`, linearly ramping to 0 outside over a `falloff` window; NaN → neutral 0.5. Rewards "healthy middle band" cosines. |
| `weighted_geometric_mean(values, weights, *, floor)` | Per-row weighted geometric mean (floored so a single zero punishes hard without nuking to exactly 0) — encodes "all signals must agree". |
| `_route(conf, auto_min, spot_min)` _(helper)_ | Vectorized threshold application: `auto` ≥ auto_min, else `spot` ≥ spot_min, else `review`. |
| `score_pairs(df_pairs, config)` | Computes the three pair signals (LLM self-score, cosine band, structural per `structural_mode`), combines them into `confidence`, assigns `routing`, and writes all `confidence_*` columns. |
| `score_negatives(df_neg, config)` | Same for negatives (LLM gate score with NaN→0.5, cosine band, `structural_check_passed`), but only LLM-generated rows get a routing label/confidence; transcript rows stay `routing=None`, `confidence=NaN`. |
| `summarize_routing(df)` | Returns bucket counts + percentages over rows with non-null routing. |
| `export_routing_audit_sample(df_pairs, df_neg, processed_dir, *, n, seed, verbose)` | Builds a stratified (by kind × routing) ~50-row audit template with resolved anchor text and blank `human_routing`/`agree`/`notes` columns; writes parquet + an Excel mirror. |
| `_normalize_routing_labels(series)` _(helper)_ | Lowercases/strips routing labels and coerces blanks/`nan`/`<na>` to NA. |
| `compute_audit_agreement(df_audit)` | Given a filled audit, computes model-vs-human agreement: overall proportion, per-bucket and per-kind breakdowns, a confusion matrix, and disagreement count. |
| `refresh_audit_routing(df_audit, df_pairs, df_neg)` | Re-joins a filled audit's `confidence`/`routing` columns from current parquets (on `id`) so it can be re-scored after config retuning without regenerating the sample; preserves row order/human labels. |
| `rescore_processed_routing(processed_dir, *, config_path, verbose)` | Recomputes confidence + routing on pairs/negatives in place (e.g. after editing the config) without touching the audit template. |
| `load_routing_audit(path)` | Loads an audit template from `.xlsx`/`.xls`/`.parquet`. |
| `score_routing_audit(audit_path, *, ...)` | Full audit scoring run: loads the filled audit, optionally refreshes routing from parquets, computes the `agree` column + agreement report vs the config's target, and writes `routing_audit_scored.*`, `routing_audit_report.json`, and `routing_audit_disagreements.xlsx`. |
| `_print_audit_report(report)` _(helper)_ | Pretty console summary of an audit scoring run (agreement, pass/fail, per-bucket/kind, confusion, output paths). |
| `run(processed_dir, *, config_path, audit_n, audit_seed, verbose)` | End-to-end Step 5b: score pairs + negatives, write them back, log routing distributions, and emit the audit template. |

---

## Step 6 — `review_6.py`

Interactive human-review tooling (export queue → human fills decisions →
apply). Not a numbered orchestrated step; run via `--export` / `--apply` /
`--status`.

| Function | What it does |
|---|---|
| `ensure_decision_columns(df)` | Adds the six decision columns (`decision`, `reviewer`, `decided_at`, `corrected_text`, `reject_reason`, `reject_notes`) as NA if missing. |
| `_normalize_str(series)` _(helper)_ | Strips strings and coerces blank/`nan`/`<NA>`/`None` to NA. |
| `load_sidecar(processed_dir)` | Loads the `review_decisions.parquet` sidecar (the source of truth that survives pipeline re-runs); returns an empty typed frame if absent. |
| `save_sidecar(df, processed_dir)` | Writes the decisions sidecar. |
| `build_review_queue(df_pairs, df_neg, *, decided_ids, spot_rate, pair_auto_sample, seed)` | Assembles the review queue: all `routing=review` rows + a ~12% `spot` sample (+ optional `auto`-pair safety sample), excluding ids already decided (resume support) and de-duplicating; adds blank decision columns. |
| `export_review_queue(processed_dir, *, ...)` | Builds and writes the queue template (`review_queue.parquet` + `.xlsx`), logging the breakdown and valid decision/reason vocabularies; returns `None` if empty. |
| `validate_decisions(df)` | Validates a filled template and returns only decided rows; raises `ValueError` listing every problem (bad decision value, reject without a valid reason code, edit without `corrected_text`). |
| `apply_review_decisions(template_path, processed_dir, *, reviewer, verbose)` | Merges a filled template into the sidecar (newest decision per id wins) and mirrors decisions into both parquets; stamps reviewer + UTC `decided_at`; requires a reviewer for every decided row. |
| `_write_decisions_into_parquet(parquet_path, id_col, decisions)` _(helper)_ | Projects sidecar decisions onto one parquet by id, preserving all other rows; returns the count updated. |
| `review_status(processed_dir, *, spot_rate, verbose)` | DoD status report: per kind, how many `review` rows remain pending and whether the `spot` sample target is met; computes overall `dod_met`. |
| `filter_verified(df, *, kind, verbose)` | The downstream gate for Step 7/Step 8: keeps accepted + edited rows (applying `corrected_text`) and undecided rows that are `auto`/None-routed; drops rejected rows and still-pending `review`/`spot` rows. |

---

## Step 7 — `splits.py`

Produces a single leakage-safe, stratified `splits.parquet` covering positives,
negatives, and pairs.

| Function | What it does |
|---|---|
| `_stratify_one(keys, *, ratios, seed)` _(helper)_ | Assigns each row to train/val/test so every unique `keys` value is split by `ratios`; uses a seeded RNG and routes rounding remainder into test. |
| `_primary_subtype(subtypes)` _(helper)_ | Collapses a row's sub-type list to a single string for stratification (lexicographically smallest label; `"unknown"` if empty). |
| `make_splits(df_corpus, df_negatives, df_pairs, *, ratios, seed)` | Builds the unified splits frame: positives stratified by primary sub-type, negatives by `source_type × primary sub-type`, and pairs **co-located with their anchor's split** (no anchor leakage). Validates ratios sum to 1.0. |
| `summarize(df_splits)` | Returns a pretty-printed overall + by-kind split distribution string. |
| `run(processed_dir, *, ratios, seed, verbose)` | End-to-end Step 7: load corpus/negatives/pairs, build splits, and write `splits.parquet`. |

---

## Step 8 — bi-encoder fine-tuning — `biencoder_8.py`

| Function | What it does |
|---|---|
| `load_config(config_path)` | Loads `config/biencoder.json` (base model, prefixes, train/eval/output settings). |
| `set_seeds(seed)` | Seeds `random`, `numpy`, and `torch` for reproducibility. |
| `load_verified_frames(processed_dir)` | Loads corpus/pairs/negatives/splits and applies the Step 6 `filter_verified` gate to pairs + negatives. |
| `verified_positive_corpus(corpus)` | Returns corpus rows with `label == SCIENCE_TALK` (the anchor bank / embedding set). |
| `build_training_triples(frames, *, seed)` | Reconstructs `(variant_query, anchor, hard_negative?)` triples from **TRAIN-split** pairs only; hard negative = a TRAIN `llm_hard_negative` anchored to the same `utt_id`, else `None` (in-batch fallback). No leakage. |
| `triples_to_input_examples(triples, config)` | Wraps triples in `sentence_transformers.InputExample`, applying the query/document task prefixes. |
| `build_eval_data(frames, *, split, max_distractors, seed)` | Builds the val query set, anchor bank, sampled negative distractors, and `pair_id → {anchor_id}` relevance. |
| `_encode(model, texts, prefix, batch_size)` _(helper)_ | Encodes with a task prefix and L2-normalizes; returns `(0, dim)` for empty input. |
| `retrieval_metrics(query_emb, query_ids, doc_emb, doc_ids, relevant, k_values)` | Computes MRR@max(k) and recall@k over normalized embeddings. |
| `evaluate_model(model, eval_data, config)` | Runs both the `distractor` and `anchor_bank` retrieval evals for one model. |
| `load_base_model(config)` | Loads the base `nomic-embed-text-v1.5` SentenceTransformer (CPU, `trust_remote_code`). |
| `train_biencoder(model, examples, config)` | Fine-tunes in place with `MultipleNegativesRankingLoss` (epochs/batch/lr/warmup from config). |
| `embed_corpus(model, frames, config, processed_dir)` | Writes L2-normalized `corpus_embeddings.npy` + `corpus_embeddings_meta.parquet` for every verified positive; returns norm-check stats. |
| `_strict_improvement(base, fine, metric, *, ceiling)` _(helper)_ | Compares one metric; flags `improved`, `saturated` (baseline at ceiling), and `met` (improved or saturated). |
| `run(processed_dir, *, config_path, eval_only, epochs, verbose)` | End-to-end Step 8: build data, eval frozen baseline, fine-tune, eval, embed corpus, write `reports/biencoder_eval.json`, save model. |

---

## Step 9 — LLM pair re-ranker — `reranker_9.py`

| Function | What it does |
|---|---|
| `load_config(config_path)` | Loads `config/reranker.json` (model env, prompt files, temperature/tokens/retries, gate thresholds, smoke/full budgets). |
| `build_operational_definitions(df_cat)` | Formats the operational science-practice definitions from `category_defs.parquet` for injection into the prompt. |
| `build_system_prompt(processed_dir, *, template_path)` | Loads a prompt template and fills `{operational_definitions}`. |
| `build_user_message(query, candidate)` | Formats the per-pair user turn (quoted Phrase A / Phrase B + JSON instruction). |
| `parse_score(text)` | Parses `{"score","rationale"}` JSON or `<score>` tag, clamps to [0,1]; **raises `RerankerParseError`** on garbage (never silent). |
| `make_real_caller(model, *, ..., ledger, verbose)` | Builds a `cached_request`-backed caller; on an empty completion it **purges the poisoned cache entry** so retries re-hit the network. Supports **automatic API-key rotation**: when a key's budget/credential is exhausted it advances to the next key (`LLM_API_KEY`, then `LLM_API_KEY_2`, `LLM_API_KEY_3`, ...; cache is key-agnostic so prior work stays free). |
| `_collect_api_keys(primary_env)` _(helper)_ | Ordered, de-duped key pool from `LLM_API_KEY` (+ optional comma list) then contiguous `LLM_API_KEY_2..N`. |
| `_response_kind(raw)` _(helper)_ | Classifies an endpoint response as `ok` / `quota_auth` (rotate) / `transient` (retry same key). |
| `stub_caller(system, user, prompt_version)` | Deterministic offline caller (lexical-overlap score) for tests/plumbing. |
| `score_pair_full(query, candidate, *, caller, system_prompt, prompt_version, max_retries, row_id)` | Scores one pair, retries on parse failure, raises loudly with the row id after exhausting retries. |
| `score_pair(query, candidate, *, model_id, prompt_version, ...)` | Public convenience: real-endpoint float score in [0,1]. |
| `auroc(labels, scores)` / `spearman(x, y)` | Dependency-free, tie-aware AUROC (Mann-Whitney) and Spearman ρ. |
| `ndcg_at_k(rels, k)` / `rank_metrics(order_ids, relevant, k)` | NDCG@k and precision@1 / MRR@k / NDCG@k for a ranked list. |
| `build_pair_audit(frames, *, n, split, seed)` | Builds gold-labeled audit pairs: `(anchor, its variant)`=1, `(anchor, a negative)`=0. |
| `score_audit(audit, *, caller, ...)` | Scores the audit and returns AUROC + raw labels/scores. |
| `run_stability(frames, *, caller, system_a, system_b, ...)` | Scores the same pairs under two prompt phrasings and returns Spearman ρ. |
| `rerank_diagnostic(frames, *, caller, ..., sample, top_k)` | (Full mode) bi-encoder top-k vs LLM-reranked precision@1/MRR/NDCG@10 — a ceiling-limited diagnostic. |
| `CallLedger` | Counts new-vs-cached calls against the on-disk cache for the cost DoD. |
| `run(processed_dir, *, config_path, mode, use_stub, verbose)` | End-to-end Step 9 calibration: AUROC + stability gate (+ optional retrieval diagnostic), writes `reports/reranker_eval.json` and `reranker_calibration.md`. |

---

## Step 10 — query-time pipeline — `query_10.py`

| Function | What it does |
|---|---|
| `load_config(config_path)` | Loads `config/query.json` (top_k, aggregation, thresholds, workers, wall-clock cap, smoke/full budgets). |
| `QueryPipeline(processed_dir, *, config_path, caller, model, verbose)` | Loads the fine-tuned bi-encoder, anchor-bank embeddings + meta, subtype map, and the Step 9 prompt/caller **once**. |
| `QueryPipeline.retrieve(phrase, top_k)` | Embeds the phrase (query prefix) and cosine-searches the anchor bank → top-k `(utt_id, utterance, bi_score, subtype)`. |
| `QueryPipeline._embed_and_search(phrase, top_k)` _(helper)_ | Retrieval with separate embed/search timings. |
| `QueryPipeline.rerank(phrase, candidates)` | Scores all candidates with the Step 9 re-ranker, in parallel (`max_workers`). |
| `QueryPipeline._score_one(phrase, cand)` _(helper)_ | Scores one pair; classifies failures as `parse` (drop) vs `network` (degrade). |
| `QueryPipeline.classify(phrase, *, top_k)` | Returns `{label, score, degraded, over_budget, ranked_candidates, timing}`; `score=max` over LLM scores, stable deterministic ordering, cosine fallback on outage. |
| `QueryPipeline.reset_ledger()` | Zeroes the call ledger (for the caching DoD re-run). |
| `classify(phrase, *, processed_dir, config_path)` | Module-level convenience over a lazily-built shared pipeline. |
| `build_dev_queries(frames, *, n_pos, n_neg, seed)` | Builds dev positives (variant → gold anchor) and negatives (text) from the val split. |
| `run(processed_dir, *, config_path, mode, use_stub, verbose)` | Dev eval: caching (0 new on re-run), determinism, graceful degradation, latency, lift diagnostic, classification sanity; writes `reports/query_eval.json`. |

---

## Step 11 — evaluation & threshold tuning — `evaluate_11.py`

| Function | What it does |
|---|---|
| `load_config(config_path)` | Loads `config/eval.json` (grid, recall floor, ndcg_k, smoke/full sizes). |
| `classification_metrics(y_true, y_pred)` | Precision / recall / F1 + TP/FP/FN/TN for the positive (science) class. |
| `threshold_grid(cfg)` | Builds the threshold sweep from config. |
| `tune_threshold(dev_scores, dev_labels, slice_scores, grid, floor)` | Max-F1 threshold on dev **subject to** hard-informal recall ≥ floor; falls back if unreachable. |
| `mean_ranking_metrics(orders, k)` | Mean precision@1 / MRR@k / NDCG@k over positive queries (gold anchor). |
| `_primary_subtype(val)` _(helper)_ | Normalizes a list/scalar/NaN subtype into a stable group key. |
| `build_split_queries(frames, split, *, n_pos, n_neg, seed)` | Builds a split's eval queries: pair variants (gold anchor) + negative texts. |
| `build_slice_queries(processed_dir, *, n, seed)` | Builds hard-informal-slice positive queries. |
| `_cosine_system(model, qp, dp, doc_emb, doc_ids, queries, k)` _(helper)_ | Scores queries by max cosine over the anchor bank; ranks positives. |
| `_rerank_system(pipe, queries, k)` _(helper)_ | Scores queries with `classify()`; captures scores, rankings, top match + rationale. |
| `_subtype_breakdown / _errors_and_predictions / _write_report` _(helpers)_ | Per-sub-type recall, top-20 errors + per-row predictions, and the markdown report. |
| `run(processed_dir, *, config_path, mode, use_stub, verbose)` | End-to-end Step 11: 3-system ablation, dev threshold tuning, test metrics, writes `reports/eval_report.md`, `eval_metrics.json`, `test_predictions.parquet`. |

---

## Step 12 — Y2 transcript scoring / deployment — `deploy_y2_12.py`

| Function | What it does |
|---|---|
| `load_config(config_path)` | Loads `config/deploy.json` (filters, `top_k`, `dollar_cap`, `cost_per_call`, thresholds, output dir, Classroom49 rule). |
| `parse_transcription(cell)` | Splits `"text (57.6%)"` → `("text", 0.576)`; no suffix → `(text, None)`. |
| `child_id_from_filename(name)` | Derives focal child id from the filename (`65-74-943-170…` → `170`; suffix-less → `base`). |
| `sheet_name_for(child_id, used)` | Excel-safe, ≤31-char, collision-free sheet name. |
| `ingest_file(path, cfg)` | Reads one transcript xlsx → kept utterance records (drops `-` rows, applies speaker + confidence filters, parses text/confidence, attaches line_no/timestamps/child_id). |
| `_primary_subtype(val)` _(helper)_ | Normalizes a list/scalar/NaN subtype into a stable key. |
| `cosine_score_all(pipe, records, *, batch_size, verbose)` | Stage 1: dedupes, batch-encodes with the fine-tuned bi-encoder, scores vs the anchor bank; attaches `cosine_score`, `top_match_in_corpus`, `predicted_subtype` to every record (in place). |
| `budgeted_rerank(pipe, records, cfg, *, max_rerank, dollar_budget, reset, verbose)` | Stage 2: re-ranks highest-cosine utterances via `classify()`; `CallLedger` halts before the budget (charging new calls only); attaches `llm_score`/`rationale`/`degraded`. `dollar_budget` overrides the cap (per-folder cumulative allowance); `reset=False` keeps the ledger running across folders so the cap is cumulative; `max_rerank` caps the offline `--stub` smoke. Returns per-call and cumulative ledger summary. |
| `assign_label(rec, cfg)` | Predicted label from `llm_score` (if reranked) or `cosine_score`, plus a unified display `score` (in place). |
| `write_classroom_workbook(folder_name, recs, cfg, *, verbose)` | Writes one xlsx per session: one chronological sheet per child + `INSTRUCTIONS`. Each child sheet has an empty `review` column (with a SCIENCE/NOT dropdown) and an agreement tally at the bottom. With `cfg["science_only"]` (default true) sheets keep only `SCIENCE_TALK` rows, no-science child sheets are dropped, and the workbook is skipped (returns `None`) if nothing qualifies; with `science_only=false` it also writes the full transcript plus a `SUMMARY` sheet. |
| `_add_review_tools(ws, *, n_rows)` _(helper)_ | Adds the `review` dropdown and the bottom-of-sheet tally (`Reviewed`, `Agreements with model` = `SUMPRODUCT(review=predicted_label)`, `Agreement rate`) as live Excel formulas. |
| `_build_instructions(science_only)` _(helper)_ | Builds the `INSTRUCTIONS` sheet text, adjusted for science-only vs full-transcript mode and explaining the review column + tally. |
| `run(folders, *, config_path, all_, use_stub, processed_dir, overrides, overwrite, verbose)` | Orchestrates the scorer **folder-by-folder**: ingest → cosine → budgeted rerank → labels → write that folder's workbook, then checkpoint `run_manifest.json`. Memory-bounded (one folder at a time) and **resumable** — finished workbooks are skipped unless `overwrite=True`. A single persistent ledger enforces the dollar cap on **cumulative** spend, with each folder's allowance rolling forward. |
| `_resolve_folders / _print_summary / _parse_args` _(helpers)_ | Folder selection (with Classroom49 dedupe), run summary, and CLI parsing. |

---

## Orchestrator — `pipeline.py`

| Function | What it does |
|---|---|
| `run(*, xlsx_path, out_dir, steps, ...)` | Runs the requested numbered steps (1–7) in order, wiring each step's flags (real-LLM toggles, transcript sources, counts, config/encoder paths, split ratios/seed). Selects stub vs real LLM per step and prints section banners. |
| `_parse_args()` _(helper)_ | Defines the full CLI surface (`--steps`, `--use-llm`, `--use-llm-step2`, transcript options, gate-scoring toggles, encoder/config paths, split ratios, `--quiet`, etc.). |

### CLI entry points (all modules)

Every `src/` module is independently runnable (`python src/<module>.py --help`)
and defines an `argparse`-based `__main__` block plus, for several modules, a
`_parse_args()` helper. These dispatch to the module's `run(...)` (or, for
`confidence_5.py`, to `rescore_processed_routing` / `score_routing_audit`, and
for `review_6.py`, to `export_review_queue` / `apply_review_decisions` /
`review_status`). They are thin wrappers around the documented functions above
and contain no additional pipeline logic.
