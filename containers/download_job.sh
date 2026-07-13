#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Model download SLURM job.
#
# Called by judge_panel_submit.sh when weights are not yet present locally.
# Runs huggingface-cli download inside the castor_judge.sif container.
# No GPU required — any pleiades node with network access will do.
#
# Args:
#   $1  HF_REPO   — HuggingFace repo ID (e.g. "casperhansen/deepseek-r1-distill-qwen-32b-awq")
#   $2  DST_DIR   — local destination directory for the weights
#
# NOTE: if this job fails with a network error, your cluster's compute nodes
# may not have internet access. In that case run the download on the head node:
#   hf download $HF_REPO --local-dir $DST_DIR
# then re-run judge_panel_submit.sh (it will skip the download step automatically).
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH -J dl_model
# NOTE: --output / --error set by judge_panel_submit.sh

set -euo pipefail

HF_REPO="${1:?Usage: download_job.sh HF_REPO DST_DIR}"
DST_DIR="${2:?Usage: download_job.sh HF_REPO DST_DIR}"

DATA_DIR="/data/$USER"
SIF="$DATA_DIR/castor_judge.sif"

echo "==========================================="
echo " Model Download Job"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Repo     : $HF_REPO"
echo " Dest     : $DST_DIR"
echo "==========================================="

# Already downloaded check
if [ -d "$DST_DIR" ] && [ "$(ls -A "$DST_DIR" 2>/dev/null)" ]; then
    echo "Destination already exists and is non-empty — skipping download."
    echo "  $DST_DIR"
    exit 0
fi

mkdir -p "$DST_DIR"

export HF_HOME="$DATA_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

# Pass HF token if set in environment (for gated models)
HF_TOKEN_ARG=""
if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
    HF_TOKEN_ARG="--token $HUGGINGFACE_TOKEN"
fi

echo "[$(date)] Starting download inside container ..."

apptainer exec --containall \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$HF_HOME" \
    --env HF_HUB_DISABLE_PROGRESS_BARS=0 \
    --env HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
    --bind /tmp:/tmp \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    hf download "$HF_REPO" \
        --local-dir "$DST_DIR" \
        ${HF_TOKEN_ARG}

echo "[$(date)] Download complete."
echo "  Destination: $DST_DIR"
du -sh "$DST_DIR" 2>/dev/null || true
echo "==========================================="
