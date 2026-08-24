#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 9 — Extraction-model calibration, single-model SLURM batch script.
#
# Do NOT call directly with sbatch -- use submit_plan_adequacy_calibrate.sh
# instead, which fans this out across the candidate model list and submits
# a comparison job after.
#
# Reuses castor_judge.sif (already built for P7/P8) and the P8 judge_panel
# model weights already downloaded under /data/$USER -- no new container,
# no new download, for glm4_32b/llama_3_3_70b. See design plan section 4g.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --constraint=RTX6000ADA
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH -J p9_calibrate
# NOTE: --mem / --output / --error are set by submit_plan_adequacy_calibrate.sh

set -euo pipefail

MODEL="${1:?Usage: plan_adequacy_calibrate_job.sh MODEL_KEY}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"

echo "==========================================="
echo " CASTOR P9 — Extraction Model Calibration"
echo " Model    : $MODEL"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo "==========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── Resolve model directory (same keys as P8's judge, plus phi4_14b) ────────
case "$MODEL" in
    glm4_32b)       MODEL_DIR="$DATA_DIR/glm-4-32b-0414-gptq" ;;
    llama_3_3_70b)  MODEL_DIR="$DATA_DIR/llama-3.3-70b-instruct-w4a16" ;;
    phi4_14b)       MODEL_DIR="$DATA_DIR/phi-4-w4a16" ;;
    *) echo "ERROR: unknown model key '$MODEL' (expected glm4_32b, llama_3_3_70b, or phi4_14b)" >&2; exit 1 ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model weights not found: $MODEL_DIR" >&2
    echo "       Download with: sbatch containers/download_job.sh <HF_REPO> $MODEL_DIR" >&2
    exit 1
fi

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found -- run: bash containers/build_judge_container.sh --model salvage_embed" >&2
    exit 1
fi

GOLD="$REPO/pipelines/plan_adequacy/calibration/gold_tool_calls.jsonl"
if [ ! -f "$GOLD" ]; then
    echo "ERROR: gold set not found: $GOLD" >&2
    echo "       Build it with: python3 pipelines/plan_adequacy/calibration/build_gold_combined.py" >&2
    exit 1
fi

OUT_DIR="$REPO/results/p9_plan_adequacy/calibration"
mkdir -p "$OUT_DIR"

echo " Model dir : $MODEL_DIR"
echo " Gold set  : $GOLD"
echo " Output    : $OUT_DIR/calibration_${MODEL}.json"
echo "==========================================="

apptainer exec \
    --containall \
    --nv \
    --pwd "$REPO" \
    --bind /tmp:/tmp \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env PYTHONUNBUFFERED=1 \
    --env HF_HOME="$DATA_DIR/.cache/huggingface" \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1 \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$SIF" \
    python3 "$REPO/pipelines/plan_adequacy/calibrate.py" \
        --model     "$MODEL" \
        --model-dir "$MODEL_DIR" \
        --gold      "$GOLD" \
        --out       "$OUT_DIR"

EXIT_CODE=$?
echo "==========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " P9 calibration complete ($MODEL) : $(date)"
    ls -lh "$OUT_DIR/calibration_${MODEL}.json" 2>/dev/null || true
else
    echo " P9 calibration FAILED (exit $EXIT_CODE) : $(date)" >&2
fi
echo "==========================================="
exit $EXIT_CODE
