# TeLI Science Talk Classifier — V3 Guide

V3 finds **science-practice talk** (observing, predicting, causal reasoning, appealing to evidence, and naming science content) inside Pre-K classroom transcripts and outputs reviewer-ready Excel workbooks aligned with the USF TeLI Coding Manual.

It supports two **engines** and two **output modes**:

- **Engines**
  - `hybrid` *(recommended)* — Local Stage-1 context scanner + Stage-2 dual-LLM judges (`gemma-4-31b-it` + `gpt-oss-20b` via UF NaviGator). Highest accuracy; costs roughly **$0.10–$0.25 per classroom folder**.
  - `local` — Distilled Stage-1 neural/lexical ensemble only (**$0 LLM judge calls**; still uses `nomic-embed-text-v1.5` embeddings, which cost fractions of a cent or $0 if cached).
- **Modes**
  - `screening` — High-recall (~90% recall) workbooks built for human coders to review.
  - `standalone` — High-precision, calibrated predictions for automated counts without human review.
  - `both` *(default)* — Generates both `screening` and `standalone` workbooks in one pass.

---

## 1. One-Time Setup

Run all commands from the **repository root** (`Classroom-Science-Talk-Retrieval/`).

### Install Python dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Add UF NaviGator API credentials

Create a `.env` file in the repository root (already git-ignored — never commit it):

```dotenv
LLM_API_KEY=your-uf-navigator-key
# Optional extra keys for automatic quota rotation:
LLM_API_KEY_2=your-backup-key-2
LLM_API_KEY_3=your-backup-key-3

COMPLETION_URL=https://api.ai.it.ufl.edu/v1/chat/completions
EMBEDDING_URL=https://api.ai.it.ufl.edu/v1/embeddings
```

> **Note:** `EMBEDDING_URL` is required for both `hybrid` and `local` scoring (to compute `nomic-embed-text-v1.5` embeddings for any utterances not already in the local cache). `COMPLETION_URL` is used by `hybrid` scoring and Stage-2 training.

---

## 2. Run Analysis on Classroom Transcripts (Scoring)

Use this workflow when you already have the trained V3 model bundle and want to score one classroom folder or an entire directory of classroom folders.

### Input Files Needed

| Input | Required? | Path / Format |
| --- | --- | --- |
| **Trained V3 model bundle** | Required | `models/v3/deploy_config.json`, `models/v3/portable_models.pkl`, `models/v3/pgkd_bigru.pt` (plus `models/v3/fold_models/` for automatic zero-leakage held-out scoring of Year-2 classrooms). Obtain from the team drive or generate via [Section 3](#3-re-train--reproduce-v3-from-scratch). |
| **Raw classroom `.xlsx` transcripts** | Required | One folder per classroom visit (e.g., `Classroom10_020526/`) containing one or more child `.xlsx` files. Each Excel sheet needs the standard USF columns: `Segment Start Time`, `Segment End Time`, `Transcription (Confidence)`, and `Speaker`. |
| **API credentials** | Required | `.env` in the repository root (see [Section 1](#1-one-time-setup)). |
| **Warm Nomic embedding cache** | Optional (faster) | `data/v3/embeddings/vocab.parquet` and `data/v3/embeddings/nomic-embed-text-v1_5_unique_f16.npy`. If present, previously seen utterances load instantly from disk instead of calling the embedding API. |

### Commands to Run

#### Option A: Score a single classroom folder

```bash
# Recommended: Hybrid engine (Local Stage-1 + Dual LLM judges), both screening & standalone workbooks
python src/v3/predict_new.py score \
  --input-dir "data/transcripts/Y2_Technology Observation Audio_Transcripts/Classroom10_020526" \
  --output-dir "reports/v3/new_predictions" \
  --engine hybrid \
  --mode both \
  --budget 5.0
```

To run with **$0 LLM judge calls** (local distilled model only), switch `--engine hybrid` to `--engine local`:

```bash
python src/v3/predict_new.py score \
  --input-dir "data/transcripts/Y2_Technology Observation Audio_Transcripts/Classroom10_020526" \
  --output-dir "reports/v3/new_predictions" \
  --engine local \
  --mode both
```

#### Option B: Score an entire directory of classroom folders (e.g., all 85 Y2 folders)

`score-batch` processes every classroom subfolder in `--input-root`, saves a checkpoint after every folder, and **automatically resumes** where it left off if interrupted or if a budget cap is reached.

```bash
python src/v3/predict_new.py score-batch \
  --input-root "data/transcripts/Y2_Technology Observation Audio_Transcripts" \
  --output-dir "reports/v3/y2_full_85_classrooms" \
  --engine hybrid \
  --mode both \
  --budget 25.0 \
  --workers 6
```

> **Zero-leakage routing (automatic):** If a folder belongs to one of the 16 Year-2 classrooms used during training (e.g., `Classroom10_*`), `predict_new.py` automatically loads that classroom's held-out fold model (`models/v3/fold_models/`) so the model never predicts on a classroom it trained on. Brand-new classrooms automatically use the full 83-room model (`models/v3/portable_models.pkl` + `pgkd_bigru.pt`).

#### Useful Optional Flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--mode` | `both` | Choose `screening`, `standalone`, or `both`. |
| `--engine` | `hybrid` | Choose `hybrid` (highest accuracy) or `local` ($0 LLM calls). |
| `--budget` | `5.0` (`score`) / `25.0` (`score-batch`) | Hard USD spending cap. The run stops cleanly before exceeding it. |
| `--workers` | `6` (`score-batch`) | Concurrent API worker threads. |
| `--no-secondary` | off | In `hybrid` mode, skip the secondary verifier (`gpt-oss-20b`) and use only `gemma-4-31b-it`. |
| `--full-transcript` | off | Export all screened utterances instead of only predicted `SCIENCE_TALK` rows. |
| `--no-resume` | off (`score-batch`) | Re-score folders from scratch instead of skipping already-completed folders. |

### Output Files Generated

Inside `--output-dir` (e.g., `reports/v3/new_predictions/` or `reports/v3/y2_full_85_classrooms/`):

| Output File | Description |
| --- | --- |
| `screening/<ClassroomFolder>.xlsx` | **Human-review workbook** (high recall, ~90%). One tab per child transcript (`child_<id>`) plus an `INSTRUCTIONS` tab, `review` dropdown validation, and agreement tally formulas at the bottom of each sheet. |
| `standalone/<ClassroomFolder>.xlsx` | **Automated-count workbook** (high precision, max-F1 calibrated cutoff). Same 21-column format. |
| `batch_summary_85_classrooms.csv` *(batch only)* | One row per classroom folder with file count, screened turns, `screening` & `standalone` science-talk counts, calibrated science-talk %, LLM error count, cost (USD), and runtime. |
| `batch_manifest.json` *(batch only)* | Run-level totals (folders completed/failed, total files, total predictions, total USD spent). |
| `batch_progress.jsonl` *(batch only)* | Per-folder checkpoint log used for crash-safe auto-resume. |

#### What's inside the 21-column Excel workbooks

Every workbook uses the exact 21-column schema compatible with `Data Aggregation/aggregate_human_coded.py`:

1. `line_no`, `seg_start`, `seg_end`, `speaker`, `child_id` — exact transcript location and speaker.
2. `context_before_2`, `utterance`, `context_after_2` — the flagged utterance flanked by the 2 chronological turns before and after it (stopping at `>30s` silence gaps with `[>30s pause]`).
3. `confidence`, `predicted_label`, `review` — ASR confidence, model prediction (`SCIENCE_TALK` / `NOT_SCIENCE_TALK`), and blank dropdown column for human coders.
4. `score`, `pm2_sci_count`, `scored_by`, `cosine_score`, `llm_score` — calibrated probability, count of science-leaning turns in the `±2` window (used for the `≥3`-turn cluster rule), engine tag (`hybrid_v3` or `local_v3`), Stage-1 score, and Stage-2 LLM score.
5. `top_match_in_corpus`, `top_match_score`, `predicted_subtype`, `rationale`, `degraded` — legacy compatibility fields, science subtype, and the LLM's short explanation (`hybrid` mode).

---

## 3. Re-Train & Reproduce V3 From Scratch

Only run this section if you need to **rebuild the training datasets, retrain the models, regenerate `models/v3/`, or re-run the full benchmark and `Human coded/` verification**.

### Input Files Needed

Because raw classroom transcripts and human-coded sheets are private and git-ignored, place them in these exact paths relative to the repository root:

| Input Path | Required? | Purpose |
| --- | --- | --- |
| `Human coded/*.xlsx` *(and optional `Human coded.zip`)* | Required | 83 human-reviewed Year-2 workbooks (`Classroom*_ebh.xlsx`). If an Excel file is open locally, the loader automatically falls back to `Human coded.zip`. |
| `data/transcripts/Y2_Technology Observation Audio_Transcripts/` | Required | Raw Year-2 classroom folders containing full child `.xlsx` transcripts. |
| `data/transcripts/Coding Transcripts.xlsx` | Required | 40 Year-1 full classroom transcripts used strictly for unbiased cross-year transfer evaluation. |
| `data/2026-04-17_science_talk_examples.xlsx` | Required | Year-1 seed science-talk examples (`Example utterances` sheet). |
| `reports/human_coded_reviewed_sheet_to_transcript_map.txt` | Required | Maps each reviewed `(workbook, sheet)` pair to its matching raw transcript `.xlsx` filename. |
| `reports/y2_scores_codebook_v2/` | Optional | Legacy v2 predictions used only for v1/v2 vs. v3 benchmark comparison. |
| `error_analysis/output/fp_sample_400_coded.csv` | Optional | Coded false-positive taxonomy lookup for error breakdown analysis. |
| `.env` | Required | UF NaviGator API credentials (see [Section 1](#1-one-time-setup)). |

### Commands to Run (In Order)

```bash
# Step 1 — Build leak-free Y2 + Y1 datasets and 5-fold classroom splits
python src/v3/dataset.py

# Step 2 — Compute and cache Nomic + SFR-Mistral utterance embeddings (~$0.15)
python src/v3/embeddings.py --budget 5.0

# Step 3 — Train Stage-1 lexical (TF-IDF) and contextual embedding models
python src/v3/stage1_lexical_and_embeds.py

# Step 4 — Train Stage-1 sequence models (BiGRU, TCN-Transformer, Time-Aware HMM)
python src/v3/stage1_sequence.py

# Step 5 — Train Stage-1 stacker and ensemble (4C)
python src/v3/stage1_stacker.py

# Step 6 — Run Stage-2 Hybrid LLM cascade + PGKD local distillation + blinded recall audit
python src/v3/hybrid_and_distill.py

# Step 7 — Train and save the portable production bundle (full + all 5 held-out folds)
python src/v3/predict_new.py train-artifacts

# Step 8 — Export verified out-of-fold workbooks for the 83 focal Y2 folders
python src/v3/cli.py --engine hybrid --mode both
python src/v3/cli.py --engine local  --mode both

# Step 9 — Run full verification against Human coded/ + build benchmark tables
python src/v3/verify_against_human_coded.py
python src/v3/build_benchmark_report.py
```

### Output Files Generated

| Step | Output Path | What It Contains |
| --- | --- | --- |
| 1 | `data/v3/y2_utterances.parquet`, `y1_coding_transcripts.parquet`, `y1_seed.parquet`, `dataset_summary.json` | Aligned chronological utterances, pause gaps, codebook features, and 5-fold classroom assignments. |
| 2 | `data/v3/embeddings/vocab.parquet`, `*_unique_f16.npy` | Deduplicated float16 utterance embeddings (also reused as a warm cache during scoring). |
| 3–5 | `reports/v3/stage1_preds/*.npy`, `reports/v3/metrics/*.json` | Out-of-fold Stage-1 predictions and calibrated metrics for every Stage-1 model. |
| 6 | `reports/v3/stage2_llm/*`, `reports/v3/stage1_preds/6A_*`, `6B_*`, `reports/v3/unflagged_recall_audit_BLIND_coder_sheet.xlsx` | Dual-LLM scores/rationales, Hybrid (6A) and PGKD Local (6B) predictions, and the blinded recall audit sheet for human coders (**held-out evaluation only — never train on it**). |
| 7 | `models/v3/deploy_config.json`, `portable_models.pkl`, `pgkd_bigru.pt`, `models/v3/fold_models/*` | Production model bundle + all 5 held-out classroom fold bundles required by `predict_new.py score` and `score-batch`. |
| 8 | `reports/v3/workbooks_hybrid/{screening,standalone}/*.xlsx`, `reports/v3/workbooks_local/{screening,standalone}/*.xlsx`, `classroom_quantification_summary.csv` | Leak-free out-of-fold 21-column review workbooks and per-classroom science-talk rate estimates. |
| 9 | `reports/v3/human_coded_verification_report.xlsx`, `human_coded_verification_summary.json`, `human_coded_side_by_side_all_7969.{xlsx,csv}`, `human_coded_side_by_side_83/*.xlsx`, `benchmark_report.md` | Full row-by-row comparison of v1, v2, and v3 against all 7,969 human-coded rows plus overall benchmark summary. |

---

## 4. Verify Installation (Tests)

Run the V3 test suite from the repository root:

```bash
python -m pytest tests/v3/test_v3_pipeline.py -v
```

> **Windows tip:** Always use `python -m pytest` (or `.\.venv\Scripts\python.exe -m pytest`) rather than `pytest.exe`, which may be blocked by Windows Application Control policies.

---

## 5. Quick Troubleshooting

- **`Missing models/v3/deploy_config.json`**: Copy the `models/v3/` folder from the team drive, or generate it by running `python src/v3/predict_new.py train-artifacts` (after Steps 1–6).
- **`UF API keys or endpoint URL not configured`**: Make sure `.env` exists in the repository root and defines `LLM_API_KEY`, `EMBEDDING_URL`, and `COMPLETION_URL`.
- **Batch run stopped early or hit `--budget`**: Simply re-run the exact same `score-batch` command (raising `--budget` if needed). Already-finished folders are skipped automatically via `batch_progress.jsonl`, and all prior API calls are cached under `cache/v3/` so you are never charged twice.
- **`PermissionError` reading/writing an `.xlsx` file**: Close the workbook in Excel and re-run.
