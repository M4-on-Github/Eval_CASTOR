#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 9 (plan adequacy) — Stage 2 SLURM batch script: execute + aggregate
# + report.
#
# CPU-only (no GPU needed -- run_executor.py/aggregate.py/report.py are pure
# Python + the executor's deterministic rule engine, no model calls).
# Reuses castor_judge.sif for a consistent Python environment (pandas etc.
# already present there) rather than depending on the login/compute node's
# bare Python.
#
# Runs as one task of a SLURM job array (--array=0-N-1, one task per run,
# element-wise dependent on the matching Stage 1 array task via
# --dependency=aftercorr) -- the run name is looked up by SLURM_ARRAY_TASK_ID
# from the manifest file path passed as $1, same manifest Stage 1 used.
# Requires SLURM_ARRAY_TASK_ID to be set.
#
# Do NOT call directly with sbatch -- use submit_plan_adequacy.sh instead.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH -J p9_stage2
# NOTE: --array / --output / --error are set by submit_plan_adequacy.sh

set -euo pipefail

MANIFEST="${1:?Usage: plan_adequacy_stage2_job.sh MANIFEST_FILE (run as a SLURM array task)}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set -- this script must run as a SLURM array task, see submit_plan_adequacy.sh}"
RUN_NAME="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [ -z "$RUN_NAME" ]; then
    echo "ERROR: no run name at manifest line $((SLURM_ARRAY_TASK_ID + 1)) in $MANIFEST" >&2
    exit 1
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"

echo "==========================================="
echo " CASTOR P9 Stage 2 -- Execute, Aggregate, Report"
echo " Run name : $RUN_NAME"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo "==========================================="

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found -- run: bash containers/build_judge_container.sh --model salvage_embed" >&2
    exit 1
fi

GT="$REPO/human_ground_truth_label/human_gt.csv"
OUT_DIR="$REPO/results/p9_plan_adequacy"
RECORDS="$OUT_DIR/calibration/records_${RUN_NAME}.jsonl"
# calibrate.py --dump-records output, if any exists for this run name -- see
# report.py's --records flag (Part 1c). Optional; report.py degrades to
# "no calibration records available" when missing, it does not error.

apptainer exec --containall \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env PYTHONUNBUFFERED=1 \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    python3 "$REPO/pipelines/plan_adequacy/aggregate.py" \
        --run "$RUN_NAME" \
        --dir "$OUT_DIR" \
        --gt  "$GT"

REPORT_ARGS=(--run "$RUN_NAME" --dir "$OUT_DIR")
if [ -f "$RECORDS" ]; then
    REPORT_ARGS+=(--records "$RECORDS")
fi

apptainer exec --containall \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env PYTHONUNBUFFERED=1 \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    python3 "$REPO/pipelines/plan_adequacy/report.py" "${REPORT_ARGS[@]}"

EXIT_CODE=$?
echo "==========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " P9 Stage 2 complete ($RUN_NAME) : $(date)"
    echo " Output dir : $OUT_DIR/$RUN_NAME/"
else
    echo " P9 Stage 2 FAILED (exit $EXIT_CODE) : $(date)" >&2
fi
echo "==========================================="
exit $EXIT_CODE
