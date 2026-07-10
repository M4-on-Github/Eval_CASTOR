#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 6 (salvage plan analysis) — Stage 2+3+4 SLURM batch script.
#
# Stage 2 (normalize.py, embedding + clustering) and Stage 3+4
# (eval_salvage_plan.py, contingency table + stats + report) need
# pandas/scipy/scikit-learn/sentence-transformers, which the bare cluster
# Python doesn't have. Reuses the existing castor_judge.sif (Judge Panel
# container) rather than building a new one -- it already has these packages
# (see containers/container_judge.def) and no GPU is needed for this
# CPU-only work. Stage 2 embeds phrases with a small local sentence-
# transformers model (--backend local) since there is no Ollama on the
# cluster.
#
# Runs as one task of a SLURM job array (--array=0-N-1, one task per run,
# element-wise dependent on the matching Stage 1 array task via
# --dependency=aftercorr) -- the run name is looked up by SLURM_ARRAY_TASK_ID
# from the manifest file path passed as $1, same manifest Stage 1 used.
# Requires SLURM_ARRAY_TASK_ID to be set.
#
# Do NOT call directly with sbatch -- use submit_salvage.sh instead.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -J salvage_stage234
# NOTE: --array / --output / --error are set by submit_salvage.sh

set -e

MANIFEST="${1:?Usage: salvage_stage234_job.sh MANIFEST_FILE THRESHOLD (run as a SLURM array task)}"
THRESHOLD="${2:?Usage: salvage_stage234_job.sh MANIFEST_FILE THRESHOLD (run as a SLURM array task)}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set -- this script must run as a SLURM array task, see submit_salvage.sh}"
RUN_NAME="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [ -z "$RUN_NAME" ]; then
    echo "ERROR: no run name at manifest line $((SLURM_ARRAY_TASK_ID + 1)) in $MANIFEST" >&2
    exit 1
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"

echo "==========================================="
echo " Pipeline 6 Stage 2+3+4 -- Cluster, Stats, Report"
echo " Job ID    : $SLURM_JOB_ID"
echo " Node      : $(hostname)"
echo " Started   : $(date)"
echo " Run name  : $RUN_NAME"
echo " Threshold : $THRESHOLD"
echo "==========================================="

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: container SIF not found: $SIF" >&2
    echo "       Run: bash containers/build_judge_container.sh" >&2
    exit 1
fi

EMBED_MODEL_DIR="$DATA_DIR/all-minilm-l6-v2"
if [ ! -d "$EMBED_MODEL_DIR" ]; then
    echo "ERROR: embedding model not found: $EMBED_MODEL_DIR" >&2
    echo "       Run: bash containers/build_judge_container.sh --model salvage_embed" >&2
    exit 1
fi

apptainer exec --containall \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env CASTOR_SALVAGE_RESULTS_DIR="${CASTOR_SALVAGE_RESULTS_DIR:-$REPO/p6_plans_to_judge}" \
    --env CASTOR_SALVAGE_EMBED_MODEL="$EMBED_MODEL_DIR" \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    python3 "$REPO/pipelines/salvage_analysis/normalize.py" \
        --run "$RUN_NAME" \
        --threshold "$THRESHOLD" \
        --backend local

apptainer exec --containall \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env CASTOR_SALVAGE_RESULTS_DIR="${CASTOR_SALVAGE_RESULTS_DIR:-$REPO/p6_plans_to_judge}" \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    python3 "$REPO/pipelines/eval_salvage_plan.py" \
        --run "$RUN_NAME"

echo "==========================================="
echo " Stage 2+3+4 complete : $(date)"
echo " Output dir           : $REPO/results/p6_salvage_plan/"
echo "==========================================="
