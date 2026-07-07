"""End-to-end pipeline orchestrator.

Runs the corpus-prep pipeline in order:

    Step 1  data_loader_1            data ingestion / cleaning
    Step 2  subtypes_2                soft-practice sub-type labeling
    Step 3  negatives_3               negative mining (transcript + LLM)
    Step 4  augment_4                 register-variant augmentation
    Step 5  embeddings_baseline_5     frozen-baseline encoder pass
    Step 6  confidence_5              confidence scoring + auto/spot/review routing
    Step 7  splits                    stratified train/val/test splits

Human review (plan's Step 6) is interactive and therefore NOT a numbered
pipeline step -- use `python src/review_6.py --export / --apply / --status`.
Decisions live in data/processed/review_decisions.parquet (sidecar) and are
mirrored onto pairs/negatives parquets; downstream code should gate training
data through `review_6.filter_verified()`.

(Step 8 -- bi-encoder training -- is a separate module and isn't orchestrated
here yet.)

Usage:
    python src/pipeline.py                            # run all steps with defaults
    python src/pipeline.py --steps 1                  # only Step 1
    python src/pipeline.py --steps 1 2                # explicit step list
    python src/pipeline.py --xlsx path.xlsx --out-dir custom/processed
    python src/pipeline.py --steps 2 --use-llm-step2  # Step 2 LLM fallback (no-op on Y1 corpus)
    python src/pipeline.py --steps 3 --use-llm        # Step 3 with real LLM + gate scoring on (~700 gate calls)
    python src/pipeline.py --steps 3 --use-llm --no-gate-scoring  # skip gate
    python src/pipeline.py --steps 4 --use-llm        # Step 4 with real Llama 3.3-70b
    python src/pipeline.py --steps 5                  # frozen-baseline embedding pass (~1k calls, cached)
    python src/pipeline.py --steps 6                  # confidence + routing + audit template
    python src/pipeline.py --steps 5 6                # full Step 5 (embedding + confidence)
    python src/pipeline.py --steps 7                  # rebuild splits.parquet only

The --use-llm flag (singular) controls Steps 3 and 4. Step 2's LLM fallback is
gated by a separate --use-llm-step2 flag because (a) it fires on a different
trigger (cue-less utterances) and (b) on the Y1 corpus it fires zero times so
flipping it on is free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import (
    augment_4,
    confidence_5,
    data_loader_1,
    embeddings_baseline_5,
    negatives_3,
    splits,
    subtypes_2,
)

ALL_STEPS = (1, 2, 3, 4, 5, 6, 7)


def run(
    *,
    xlsx_path: Path = data_loader_1.DEFAULT_XLSX,
    out_dir: Path = data_loader_1.DEFAULT_PROCESSED_DIR,
    steps: tuple[int, ...] = ALL_STEPS,
    use_real_llm: bool = False,
    use_real_llm_step2: bool = False,
    use_real_llm_negatives_subtype: bool = False,
    transcript_dir: Path | None = None,
    transcript_xlsx: Path | None = None,
    max_transcript_negatives: int | None = None,
    n_hard_per_positive: int = 2,
    n_per_seed_term: int = 2,
    seed_term_sample: int | None = None,
    apply_gate_scoring: bool = True,
    gate_score_only_llm_generated: bool = True,
    skip_keyword_scan_for_negatives: bool = True,
    n_variants_per_register: int = 1,
    baseline_encoder_model: str = embeddings_baseline_5.DEFAULT_BASELINE_MODEL,
    confidence_config_path: Path = confidence_5.DEFAULT_CONFIG_PATH,
    audit_n: int | None = None,
    audit_seed: int | None = None,
    split_ratios: tuple[float, float, float] = splits.DEFAULT_RATIOS,
    split_seed: int = 17,
    informal_variant_families: int = splits.DEFAULT_INFORMAL_VARIANT_FAMILIES,
    verbose: bool = True,
) -> None:
    if 1 in steps:
        if verbose:
            print(f"\n=== Step 1: data ingestion ===")
        data_loader_1.run(xlsx_path=xlsx_path, out_dir=out_dir, verbose=verbose)

    if 2 in steps:
        if verbose:
            print(f"\n=== Step 2: sub-type labeling ===")
        if use_real_llm_step2:
            classifier = subtypes_2.make_real_llm_subtype_classifier()
            if verbose:
                print("  Step 2 LLM fallback: real Llama 3.3-70b (cached)")
        else:
            classifier = subtypes_2.llm_subtype_stub
            if verbose:
                print("  Step 2 LLM fallback: deterministic stub (subtype_v0_stub)")
        subtypes_2.run(processed_dir=out_dir, llm_classifier=classifier, verbose=verbose)

    if 3 in steps:
        if verbose:
            print(f"\n=== Step 3: negative mining ===")
        negatives_3.run(
            processed_dir=out_dir,
            use_real_llm=use_real_llm,
            use_real_llm_negatives_subtype=use_real_llm_negatives_subtype,
            transcript_dir=transcript_dir,
            transcript_xlsx=transcript_xlsx,
            max_transcript_negatives=max_transcript_negatives,
            n_hard_per_positive=n_hard_per_positive,
            n_per_seed_term=n_per_seed_term,
            seed_term_sample=seed_term_sample,
            apply_gate_scoring=apply_gate_scoring,
            gate_score_only_llm_generated=gate_score_only_llm_generated,
            skip_keyword_scan_for_negatives=skip_keyword_scan_for_negatives,
            verbose=verbose,
        )

    if 4 in steps:
        if verbose:
            print(f"\n=== Step 4: register-variant augmentation ===")
        augment_4.run(
            processed_dir=out_dir,
            use_real_llm=use_real_llm,
            n_per_register=n_variants_per_register,
            verbose=verbose,
        )

    if 5 in steps:
        if verbose:
            print(f"\n=== Step 5: frozen-baseline embedding pass ===")
        embeddings_baseline_5.run(
            processed_dir=out_dir,
            model=baseline_encoder_model,
            verbose=verbose,
        )

    if 6 in steps:
        if verbose:
            print(f"\n=== Step 6: confidence scoring + routing ===")
        confidence_5.run(
            processed_dir=out_dir,
            config_path=confidence_config_path,
            audit_n=audit_n,
            audit_seed=audit_seed,
            verbose=verbose,
        )

    if 7 in steps:
        if verbose:
            print(f"\n=== Step 7: stratified train/val/test splits ===")
        splits.run(
            processed_dir=out_dir,
            ratios=split_ratios,
            seed=split_seed,
            n_variant_families=informal_variant_families,
            verbose=verbose,
        )

    if verbose:
        print(f"\nPipeline finished. Artifacts in {out_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--xlsx", type=Path, default=data_loader_1.DEFAULT_XLSX,
                        help=f"Source xlsx (default: {data_loader_1.DEFAULT_XLSX})")
    parser.add_argument("--out-dir", type=Path, default=data_loader_1.DEFAULT_PROCESSED_DIR,
                        help=f"Output directory (default: {data_loader_1.DEFAULT_PROCESSED_DIR})")
    parser.add_argument("--steps", type=int, nargs="+", choices=ALL_STEPS, default=list(ALL_STEPS),
                        help="Steps to run (default: all)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Steps 3 & 4: use real Llama 3.3-70b (default: deterministic stub).")
    parser.add_argument("--use-llm-step2", action="store_true",
                        help="Step 2: use real Llama 3.3-70b for the cue-less LLM fallback. "
                             "Fires zero API calls on the Y1 corpus (every row has Tier3 cues); "
                             "wire this on for cue-less corpora (e.g. Y2 transcripts).")
    parser.add_argument("--use-llm-negatives-subtype", action="store_true",
                        help="Step 3: use real Llama 3.3-70b to label sub-types of negatives "
                             "(~1,300 cached calls on Y1). Independent of --use-llm so you can "
                             "pay for one without the other.")
    parser.add_argument("--transcript-dir", type=Path, default=None,
                        help="Step 3: directory of plain-text transcripts to mine for negatives.")
    parser.add_argument("--transcript-xlsx", type=Path, default=None,
                        help="Step 3: coded-transcript workbook (default: auto-detect "
                             "data/transcripts/Coding Transcripts.xlsx).")
    parser.add_argument("--max-transcript-negatives", type=int, default=None,
                        help="Step 3: cap on transcript_clean rows (prevents drowning LLM sources).")
    parser.add_argument("--n-hard-per-positive", type=int, default=2,
                        help="Step 3: number of hard negatives per positive (default: 2).")
    parser.add_argument("--n-per-seed-term", type=int, default=2,
                        help="Step 3: number of nonscience uses per seed term (default: 2).")
    parser.add_argument("--seed-term-sample", type=int, default=None,
                        help="Step 3: cap the number of seed terms used (default: all).")
    parser.add_argument("--no-gate-scoring", action="store_true",
                        help="Step 3: skip LLM gate on LLM-generated negatives (~700 calls). "
                             "By default gate scoring runs.")
    parser.add_argument("--gate-score-all-rows", action="store_true",
                        help="With gate scoring on: score every row, not just LLM-generated "
                             "(~7800 calls, ~100min). Default: only LLM-generated.")
    parser.add_argument("--keep-keyword-scan-for-negatives", action="store_true",
                        help="Step 3: restore v1 behavior where seed-word matches in negative "
                             "TEXT bypass the LLM. By default we route all negatives through "
                             "the LLM for the not_science_shape escape hatch.")
    parser.add_argument("--n-variants-per-register", type=int, default=1,
                        help="Step 4: variants per target register per anchor (default: 1).")
    parser.add_argument("--baseline-encoder-model", type=str,
                        default=embeddings_baseline_5.DEFAULT_BASELINE_MODEL,
                        help=f"Step 5: frozen baseline encoder model name "
                             f"(default {embeddings_baseline_5.DEFAULT_BASELINE_MODEL}).")
    parser.add_argument("--confidence-config", type=Path,
                        default=confidence_5.DEFAULT_CONFIG_PATH,
                        help=f"Step 6: confidence + routing config json "
                             f"(default {confidence_5.DEFAULT_CONFIG_PATH}).")
    parser.add_argument("--audit-n", type=int, default=None,
                        help="Step 6: # rows in the human-routing audit template "
                             "(default: read from config, usually 50).")
    parser.add_argument("--audit-seed", type=int, default=None,
                        help="Step 6: RNG seed for the audit sample (default: from config).")
    parser.add_argument("--train", type=float, default=splits.DEFAULT_RATIOS[0],
                        help=f"Step 7: train fraction (default {splits.DEFAULT_RATIOS[0]}).")
    parser.add_argument("--val", type=float, default=splits.DEFAULT_RATIOS[1],
                        help=f"Step 7: val fraction (default {splits.DEFAULT_RATIOS[1]}).")
    parser.add_argument("--test", type=float, default=splits.DEFAULT_RATIOS[2],
                        help=f"Step 7: test fraction (default {splits.DEFAULT_RATIOS[2]}).")
    parser.add_argument("--informal-variant-families", type=int,
                        default=splits.DEFAULT_INFORMAL_VARIANT_FAMILIES,
                        help="Step 7: extra anchor families held out in the "
                             "hard-informal slice (default "
                             f"{splits.DEFAULT_INFORMAL_VARIANT_FAMILIES}).")
    parser.add_argument("--split-seed", type=int, default=17,
                        help="Step 7: RNG seed for stratified shuffling (default 17).")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        xlsx_path=args.xlsx,
        out_dir=args.out_dir,
        steps=tuple(args.steps),
        use_real_llm=args.use_llm,
        use_real_llm_step2=args.use_llm_step2,
        use_real_llm_negatives_subtype=args.use_llm_negatives_subtype,
        transcript_dir=args.transcript_dir,
        transcript_xlsx=args.transcript_xlsx,
        max_transcript_negatives=args.max_transcript_negatives,
        n_hard_per_positive=args.n_hard_per_positive,
        n_per_seed_term=args.n_per_seed_term,
        seed_term_sample=args.seed_term_sample,
        apply_gate_scoring=not args.no_gate_scoring,
        gate_score_only_llm_generated=not args.gate_score_all_rows,
        skip_keyword_scan_for_negatives=not args.keep_keyword_scan_for_negatives,
        n_variants_per_register=args.n_variants_per_register,
        baseline_encoder_model=args.baseline_encoder_model,
        confidence_config_path=args.confidence_config,
        audit_n=args.audit_n,
        audit_seed=args.audit_seed,
        split_ratios=(args.train, args.val, args.test),
        split_seed=args.split_seed,
        informal_variant_families=args.informal_variant_families,
        verbose=not args.quiet,
    )
