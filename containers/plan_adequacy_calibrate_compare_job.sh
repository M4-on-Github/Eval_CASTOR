#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 9 — Calibration comparison job. CPU only, runs after all
# per-model calibration jobs succeed (see submit_plan_adequacy_calibrate.sh's
# --dependency=afterok). Reads calibration_*.json, writes comparison.json.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -J p9_calibrate_compare

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"
SIF="$DATA_DIR/castor_judge.sif"
OUT_DIR="$REPO/results/p9_plan_adequacy/calibration"

echo "==========================================="
echo " CASTOR P9 — Calibration Comparison"
echo " Job ID   : $SLURM_JOB_ID"
echo " Started  : $(date)"
echo "==========================================="

apptainer exec \
    --containall \
    --pwd "$REPO" \
    --bind "$REPO:$REPO" \
    --env PYTHONUNBUFFERED=1 \
    "$SIF" \
    python3 "$REPO/pipelines/plan_adequacy/aggregate_calibration.py" --dir "$OUT_DIR"
