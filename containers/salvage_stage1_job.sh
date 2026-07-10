#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 6 (salvage plan analysis) — Stage 1 SLURM batch script.
#
# Element extraction is an LLM task. There is no Ollama on the cluster, so
# this loads qwen25_72b directly via vLLM inside castor_judge.sif (same
# container/model already built for the Judge Panel) and runs one batch
# generation pass over all pending records. qwen25_72b (not deepseek_r1) is
# the default: it's not a reasoning model, so guided JSON decoding applies
# cleanly and structurally rules out malformed-JSON output -- deepseek_r1
# hit repeated reasoning-related failures (rule recitation, repetition
# loops) that guided decoding can't safely fix without suppressing its
# <think> block entirely. See extract.py's _VLLM_MODEL_CONFIG comment.
#
# Runs as one task of a SLURM job array (--array=0-N-1, one task per run) --
# the run name isn't a plain CLI arg, it's looked up by SLURM_ARRAY_TASK_ID
# from the manifest file path passed as $1 (one run name per line, written
# by submit_salvage.sh). Requires SLURM_ARRAY_TASK_ID to be set.
#
# Do NOT call directly with sbatch -- use submit_salvage.sh instead.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --constraint=RTX6000ADA
#SBATCH --gpus=1
#SBATCH --mem=52G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH -J salvage_stage1
# NOTE: --array / --output / --error are set by submit_salvage.sh

set -e

MANIFEST="${1:?Usage: salvage_stage1_job.sh MANIFEST_FILE (run as a SLURM array task)}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set -- this script must run as a SLURM array task, see submit_salvage.sh}"
RUN_NAME="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [ -z "$RUN_NAME" ]; then
    echo "ERROR: no run name at manifest line $((SLURM_ARRAY_TASK_ID + 1)) in $MANIFEST" >&2
    exit 1
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"
MODEL_DIR="$DATA_DIR/qwen25-72b-instruct-awq"

echo "==========================================="
echo " Pipeline 6 Stage 1 -- Element Extraction (qwen25_72b / vLLM, guided JSON decoding)"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Run name : $RUN_NAME"
echo "==========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: container SIF not found: $SIF" >&2
    echo "       Run: bash containers/build_judge_container.sh" >&2
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model weights not found: $MODEL_DIR" >&2
    echo "       Run: bash containers/build_judge_container.sh --model qwen25_72b" >&2
    exit 1
fi

export HF_HOME="$DATA_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

CASTOR_SALVAGE_RESULTS_DIR="${CASTOR_SALVAGE_RESULTS_DIR:-$REPO/p6_plans_to_judge}"
echo " Results dir (input search): $CASTOR_SALVAGE_RESULTS_DIR"
echo " (If no <run>.jsonl full-answer file is found there, separated-into-parts"
echo "  shards for $RUN_NAME are auto-combined and cached in results/p6_salvage_plan/.)"

apptainer exec --containall --nv \
    --pwd "$REPO" \
    --env USER="$USER" \
    --env HOME="$HOME" \
    --env HF_HOME="$HF_HOME" \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1 \
    --env CASTOR_SALVAGE_RESULTS_DIR="$CASTOR_SALVAGE_RESULTS_DIR" \
    --bind /tmp:/tmp \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    python3 "$REPO/pipelines/salvage_analysis/extract.py" \
        --run "$RUN_NAME" \
        --backend vllm \
        --model-dir "$MODEL_DIR" \
        --model-key qwen25_72b \
        --tp 1

echo "==========================================="
echo " Stage 1 complete : $(date)"
echo "==========================================="
