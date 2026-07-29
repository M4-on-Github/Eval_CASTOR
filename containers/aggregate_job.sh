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
# Do NOT call directly — submitted by judge_panel_submit.sh with the appropriate
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

JUDGE_DIR="$REPO/results/p5_judge/$RUN_NAME"
if [ ! -d "$JUDGE_DIR" ]; then
    echo "ERROR: judge output directory not found: $JUDGE_DIR" >&2
    exit 1
fi

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: container SIF not found: $SIF — run build_judge_container.sh first" >&2
    exit 1
fi

# ── Aggregation ───────────────────────────────────────────────────────────────
# aggregate.py uses only stdlib (json, statistics) — runs on bare host Python.
python3 "$REPO/pipelines/judge_panel/aggregate.py" \
    --run "$RUN_NAME" \
    --dir "$JUDGE_DIR"

# ── Summary CSV ───────────────────────────────────────────────────────────────
CONSENSUS_FILE="$JUDGE_DIR/${RUN_NAME}_consensus.jsonl"
if [ ! -f "$CONSENSUS_FILE" ]; then
    echo "ERROR: consensus file not found: $CONSENSUS_FILE" >&2
    exit 1
fi

# pandas is only available inside the container (not on the bare cluster Python).
# Export shell vars so the Python here-doc can reach them via os.environ.
# (Single-quoted heredoc does not expand shell variables, so we pass via env.)
apptainer exec \
    --containall \
    --bind "$DATA_DIR:$DATA_DIR" \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    --env REPO="$REPO" \
    --env RUN_NAME="$RUN_NAME" \
    --env USER="$USER" \
    "$SIF" \
    python3 - <<'PYEOF'
import sys, os
from pathlib import Path

repo     = Path(os.environ["REPO"])
run_name = os.environ["RUN_NAME"]
sys.path.insert(0, str(repo))

import pandas as pd
from shared.metrics import panel_score_summary

judge_dir   = repo / "results" / "p5_judge" / run_name
consensus_p = judge_dir / f"{run_name}_consensus.jsonl"
summary_csv = judge_dir.parent / "eval_summary_judge.csv"

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
