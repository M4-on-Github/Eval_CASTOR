"""
CASTOR Evaluator — Pipeline 5 (LLM-as-a-Judge Panel)

Sequential runner for interactive use on a pleiades GPU node. Runs all three
judge models one at a time (using vLLM, inside the Apptainer container), then
aggregates into a consensus file.

For parallel cluster submission (recommended), use containers/submit_judges.sh.

Usage (inside the container on a GPU node):
    python3 pipelines/eval_judge_panel.py --run answers_baseline [--limit 10]

    # With explicit model-dir (defaults to /data/$USER/<model_dir>):
    python3 pipelines/eval_judge_panel.py --run answers_baseline \\
        --model-dir-base /data/myuser
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

EVAL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.judge_panel.run_judge import run as run_judge, _MODEL_CONFIG
from pipelines.judge_panel.aggregate import aggregate_run, JUDGE_MODELS
from shared.metrics import panel_score_summary

GT_PATH    = EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv"
RESULTS_IN = EVAL_ROOT.parent / "results" / "castor_results"
OUT_BASE   = EVAL_ROOT / "results" / "p5_judge"


def main():
    ap = argparse.ArgumentParser(
        description="Pipeline 5: run all three judges sequentially + aggregate."
    )
    ap.add_argument("--run",            required=True, help="Inference run name (no .jsonl)")
    ap.add_argument("--model-dir-base", default=f"/data/{os.environ.get('USER', 'user')}",
                    help="Base directory containing model weight subdirs "
                         "(default: /data/$USER)")
    ap.add_argument("--limit",          type=int, default=None,
                    help="Process only first N records (smoke test)")
    args = ap.parse_args()

    input_path = RESULTS_IN / f"{args.run}.jsonl"
    if not input_path.exists():
        print(f"ERROR: inference file not found: {input_path}")
        sys.exit(1)

    out_dir = OUT_BASE / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPipeline 5 — LLM-as-a-Judge Panel (sequential)")
    print(f"  Run           : {args.run}")
    print(f"  Input         : {input_path}")
    print(f"  Model-dir base: {args.model_dir_base}")
    print(f"  Output        : {out_dir}")
    if args.limit:
        print(f"  Limit         : {args.limit} records (smoke test)")

    # ── Phase 1: run each judge model sequentially ────────────────────────────
    for model in JUDGE_MODELS:
        cfg       = _MODEL_CONFIG[model]
        model_dir = f"{args.model_dir_base}/{cfg['dir']}"
        tp_size   = cfg["tp"]

        print(f"\n{'─'*60}")
        print(f"  Judge     : {model}")
        print(f"  Model dir : {model_dir}  (tp={tp_size})")
        print(f"{'─'*60}")

        run_judge(
            input_path=input_path,
            gt_path=GT_PATH,
            out_dir=out_dir,
            model=model,
            model_dir=model_dir,
            tp_size=tp_size,
            limit=args.limit,
        )

    # ── Phase 2: aggregate ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Aggregating ...")
    print(f"{'─'*60}")
    consensus_path, flagged_path = aggregate_run(args.run, out_dir)

    # ── Phase 3: append to summary CSV ───────────────────────────────────────
    summary_csv = OUT_BASE / "eval_summary_judge.csv"
    row = panel_score_summary(consensus_path, args.run)

    existing_runs = set()
    if summary_csv.exists():
        try:
            existing_runs = set(pd.read_csv(summary_csv)["run"].tolist())
        except Exception:
            pass

    if row["run"] not in existing_runs:
        new_df = pd.DataFrame([row])
        if summary_csv.exists():
            new_df.to_csv(summary_csv, mode="a", header=False, index=False)
        else:
            new_df.to_csv(summary_csv, index=False)
        print(f"\n  Appended row to {summary_csv.relative_to(EVAL_ROOT)}")
    else:
        print(f"\n  Run '{args.run}' already in {summary_csv.name} — skipping append.")

    # ── Summary ───────────────────────────────────────────────────────────────
    SEP = "=" * 72
    print(f"\n{SEP}")
    print(f"  PIPELINE 5 COMPLETE")
    print(SEP)
    print(f"  Consensus  -> {consensus_path.relative_to(EVAL_ROOT)}")
    print(f"  Flagged    -> {flagged_path.relative_to(EVAL_ROOT)}")
    print(f"  Summary CSV-> {summary_csv.relative_to(EVAL_ROOT)}")
    for k, v in row.items():
        if k != "run":
            print(f"    {k:<30}: {v}")


if __name__ == "__main__":
    main()
