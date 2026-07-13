#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# AutoAWQ quantization SLURM job.
#
# Called by judge_panel_submit.sh when FP16 weights exist but AWQ dir is missing.
# Runs inside the castor_judge.sif container on a single RTX 6000 Ada GPU.
#
# Args:
#   $1  SRC_DIR  — path to FP16 model weights
#   $2  DST_DIR  — path where AWQ-quantized weights will be saved
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --constraint=RTX6000ADA
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH -J quant_model
# NOTE: --gpus / --mem / --output / --error are set by judge_panel_submit.sh

set -euo pipefail

SRC_DIR="${1:?Usage: quantize_job.sh SRC_DIR DST_DIR}"
DST_DIR="${2:?Usage: quantize_job.sh SRC_DIR DST_DIR}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"
SIF="$DATA_DIR/castor_judge.sif"

echo "==========================================="
echo " AutoAWQ Quantization Job"
echo " Job ID  : $SLURM_JOB_ID"
echo " Node    : $(hostname)"
echo " Started : $(date)"
echo " Source  : $SRC_DIR"
echo " Output  : $DST_DIR"
echo "==========================================="

if [ -d "$DST_DIR" ] && [ "$(ls -A "$DST_DIR" 2>/dev/null)" ]; then
    echo "AWQ directory already exists and is non-empty — skipping quantization."
    echo "  $DST_DIR"
    exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HOME="$DATA_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

QUANT_SCRIPT="$REPO/pipelines/judge_panel/quantize_model.py"
if [ ! -f "$QUANT_SCRIPT" ]; then
    echo "ERROR: quantize_model.py not found at $QUANT_SCRIPT" >&2
    exit 1
fi

echo "[$(date)] Running AutoAWQ quantization inside container ..."
apptainer exec --containall --nv \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env HF_HOME="$HF_HOME" \
    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1 \
    --bind /tmp:/tmp \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    bash -c "pip install autoawq --quiet 2>/dev/null || true && python3 '$QUANT_SCRIPT' --src '$SRC_DIR' --dst '$DST_DIR'"

echo "[$(date)] Quantization complete."
echo "  Output: $DST_DIR"
echo "==========================================="
