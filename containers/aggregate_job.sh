#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CASTOR Judge Panel — aggregation SLURM batch script.
#
# Runs after all three judge jobs complete (--dependency=afterok:J1:J2:J3).
# Merges the three per-model JONLs into a consensus file and appends a
# summary row to eval_summary_judge.csv.
#
# No GPU required — runs on any pleiades node.
#
# Do NOT call directly — submitted by submit_judges.sh with the appropriate
# --dependency flag.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -J castor_aggregate
# NOTE: --output / --error are set by submit_judges.sh

set -e

RUN_NAME="${1:?Usage: aggregate_job.sh RUN_NAME}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"

echo "==========================================="
echo " CASTOR Judge Panel — Aggregation"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Run name : $RUN_NAME"
echo "==========================================="

JUDGE_DIR="$DATA_DIR/castor_results/p5_judge/$RUN_NAME"
if [ ! -d "$JUDGE_DIR" ]; then
    echo "ERROR: judge output directory not found: $JUDGE_DIR" >&2
    exit 1
fi

# ── Aggregation ───────────────────────────────────────────────────────────────
# Runs on the host Python (no container needed — pure pandas/json only).
# Requires: python3 + pandas + scikit-learn in the host environment.
# On pleiades, use the system Python or activate the user conda env first.
python3 "$REPO/pipelines/judge_panel/aggregate.py" \
    --run "$RUN_NAME" \
    --dir "$JUDGE_DIR"

# ── Summary CSV ───────────────────────────────────────────────────────────────
CONSENSUS_FILE="$JUDGE_DIR/${RUN_NAME}_consensus.jsonl"
if [ ! -f "$CONSENSUS_FILE" ]; then
    echo "ERROR: consensus file not found: $CONSENSUS_FILE" >&2
    exit 1
fi

python3 - <<'PYEOF'
import sys, os
from pathlib import Path

repo      = Path(os.environ["SLURM_SUBMIT_DIR"]) if "SLURM_SUBMIT_DIR" in os.environ \
            else Path(__file__).parent.parent
sys.path.insert(0, str(repo))

import pandas as pd
from shared.metrics import panel_score_summary

data_dir     = Path(f"/data/{os.environ['USER']}")
run_name     = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN_NAME", "")
judge_dir    = data_dir / "castor_results" / "p5_judge" / run_name
consensus_p  = judge_dir / f"{run_name}_consensus.jsonl"
summary_csv  = judge_dir.parent / "eval_summary_judge.csv"

row = panel_score_summary(consensus_p, run_name)

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
    print(f"Appended summary row to {summary_csv}")
else:
    print(f"Run '{run_name}' already in {summary_csv.name} — skipped.")

for k, v in row.items():
    if k != "run":
        print(f"  {k:<35}: {v}")
PYEOF

echo "==========================================="
echo " Aggregation complete : $(date)"
echo " Output dir           : $JUDGE_DIR"
echo "==========================================="
