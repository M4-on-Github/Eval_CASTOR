#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 9 (plan adequacy) — Stage 1 SLURM batch script: extraction.
#
# Runs extract.py's guided-JSON extraction over one plan-folder run,
# writing tool_calls.jsonl. Reuses castor_judge.sif and the same model
# weights directories as P9 calibration / P8's judge panel -- see
# plan_adequacy_calibrate_job.sh's header for why no new container/download
# is needed for glm4_32b/llama_3_3_70b/phi4_14b.
#
# Runs as one task of a SLURM job array (--array=0-N-1, one task per run) --
# the run name isn't a plain CLI arg, it's looked up by SLURM_ARRAY_TASK_ID
# from the manifest file path passed as $1 (one run name per line, written
# by submit_plan_adequacy.sh). Requires SLURM_ARRAY_TASK_ID to be set.
#
# Do NOT call directly with sbatch -- use submit_plan_adequacy.sh instead.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --constraint=RTX6000ADA
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH -J p9_stage1
# NOTE: --array / --output / --error are set by submit_plan_adequacy.sh

set -euo pipefail

MANIFEST="${1:?Usage: plan_adequacy_stage1_job.sh MANIFEST_FILE MODEL_KEY (run as a SLURM array task)}"
MODEL="${2:?Usage: plan_adequacy_stage1_job.sh MANIFEST_FILE MODEL_KEY (run as a SLURM array task)}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set -- this script must run as a SLURM array task, see submit_plan_adequacy.sh}"
RUN_NAME="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [ -z "$RUN_NAME" ]; then
    echo "ERROR: no run name at manifest line $((SLURM_ARRAY_TASK_ID + 1)) in $MANIFEST" >&2
    exit 1
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"

echo "==========================================="
echo " CASTOR P9 Stage 1 -- Tool-Call Extraction"
echo " Model    : $MODEL"
echo " Run name : $RUN_NAME"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo "==========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── Resolve model directory -- same keys as plan_adequacy_calibrate_job.sh ──
case "$MODEL" in
    glm4_32b)       MODEL_DIR="$DATA_DIR/glm-4-32b-0414-gptq" ;;
    llama_3_3_70b)  MODEL_DIR="$DATA_DIR/llama-3.3-70b-instruct-w4a16" ;;
    phi4_14b)       MODEL_DIR="$DATA_DIR/phi-4-w4a16" ;;
    *) echo "ERROR: unknown model key '$MODEL' (expected glm4_32b, llama_3_3_70b, or phi4_14b)" >&2; exit 1 ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model weights not found: $MODEL_DIR" >&2
    exit 1
fi

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found -- run: bash containers/build_judge_container.sh --model salvage_embed" >&2
    exit 1
fi

INPUT="$REPO/pipelines/plan_adequacy/inbox/${RUN_NAME}.jsonl"
if [ ! -f "$INPUT" ]; then
    echo "ERROR: input plan JSONL not found: $INPUT" >&2
    exit 1
fi

GT="$REPO/human_ground_truth_label/human_gt.csv"
OUT_DIR="$REPO/results/p9_plan_adequacy"
mkdir -p "$OUT_DIR"

echo " Model dir : $MODEL_DIR"
echo " Input     : $INPUT"
echo " Output    : $OUT_DIR/$RUN_NAME/tool_calls.jsonl"
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
    python3 "$REPO/pipelines/plan_adequacy/extract.py" \
        --model     "$MODEL" \
        --model-dir "$MODEL_DIR" \
        --input     "$INPUT" \
        --out       "$OUT_DIR" \
        --gt        "$GT"

EXIT_CODE=$?
echo "==========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " P9 Stage 1 complete ($RUN_NAME) : $(date)"
else
    echo " P9 Stage 1 FAILED (exit $EXIT_CODE) : $(date)" >&2
fi
echo "==========================================="
exit $EXIT_CODE
