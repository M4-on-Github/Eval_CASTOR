#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 8 — Plan Coherence Analysis, aggregation SLURM job.
#
# Runs after all 5 judge jobs complete
#   (--dependency=afterok:J1:J2:J3:J4:J5 set by submit_coherence.sh).
# CPU-only — no GPU required.
#
# Do NOT call directly — submitted by submit_coherence.sh.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH -J p8_coherence_agg
# NOTE: --output / --error are set by submit_coherence.sh

set -euo pipefail

RUN_NAME="${1:?Usage: coherence_aggregate_job.sh RUN_NAME}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"

echo "==========================================="
echo " CASTOR P8 — Plan Coherence Aggregation"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Run name : $RUN_NAME"
echo "==========================================="

AGG_DIR="$REPO/results/p8_plan_coherence/$RUN_NAME"
if [ ! -d "$AGG_DIR" ]; then
    echo "ERROR: per-model judge output directory not found: $AGG_DIR" >&2
    echo "       Was at least one coherence_judge_job.sh task successful?" >&2
    exit 1
fi

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found — run: bash containers/build_judge_container.sh --model salvage_embed" >&2
    exit 1
fi

echo " Aggregating from: $AGG_DIR"
echo "==========================================="

apptainer exec \
    --containall \
    --pwd "$REPO" \
    --bind /tmp:/tmp \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env PYTHONUNBUFFERED=1 \
    "$SIF" \
    python3 "$REPO/pipelines/plan_coherence/aggregate_coherence.py" \
        --run "$RUN_NAME" \
        --dir "$REPO/results/p8_plan_coherence"

EXIT_CODE=$?
echo "==========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " P8 aggregation complete : $(date)"
    ls -lh "$AGG_DIR"/ 2>/dev/null || true
    echo ""
    echo " Cumulative summary:"
    echo "   $REPO/results/p8_plan_coherence/eval_summary_coherence.csv"
else
    echo " P8 aggregation FAILED (exit $EXIT_CODE) : $(date)" >&2
fi
echo "==========================================="
exit $EXIT_CODE
