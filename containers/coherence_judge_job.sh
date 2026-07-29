#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 8 — Plan Coherence Analysis, single-model SLURM batch script.
#
# Do NOT call directly with sbatch — use submit_coherence.sh instead.
# submit_coherence.sh sets --gpus and --mem correctly per model.
#
# Resources set by submit_coherence.sh:
#   deepseek_r1_32b  : --mem=52G
#   glm4_32b         : --mem=52G
#   llama_3_3_70b    : --mem=52G
#   phi4_14b         : --mem=20G
#   gemma4_31b       : --mem=40G  (uses gemma4_judge.sif, vLLM 0.24)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --constraint=RTX6000ADA
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH -J p8_coherence
# NOTE: --output / --error / --mem are set by submit_coherence.sh

set -euo pipefail

MODEL="${1:?Usage: coherence_judge_job.sh MODEL_KEY RUN_NAME}"
RUN_NAME="${2:?Usage: coherence_judge_job.sh MODEL_KEY RUN_NAME}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"

echo "==========================================="
echo " CASTOR P8 — Plan Coherence Judge"
echo " Model    : $MODEL"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Run name : $RUN_NAME"
echo "==========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── Resolve model directory ───────────────────────────────────────────────────
case "$MODEL" in
    deepseek_r1_32b)  MODEL_DIR="$DATA_DIR/deepseek-r1-distill-qwen-32b-awq" ;;
    glm4_32b)         MODEL_DIR="$DATA_DIR/glm-4-32b-0414-gptq" ;;
    llama_3_3_70b)    MODEL_DIR="$DATA_DIR/llama-3.3-70b-instruct-w4a16" ;;
    phi4_14b)         MODEL_DIR="$DATA_DIR/phi-4-w4a16" ;;
    gemma4_31b)        MODEL_DIR="$DATA_DIR/gemma4-31b-it-w4a16" ;;
    *) echo "ERROR: unknown model key '$MODEL'" >&2; exit 1 ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model weights not found: $MODEL_DIR" >&2
    echo "       Download with: sbatch containers/download_job.sh <HF_REPO> $MODEL_DIR" >&2
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_JSONL="$REPO/p8_to_check/${RUN_NAME}.jsonl"
if [ ! -f "$INPUT_JSONL" ]; then
    echo "ERROR: inference JSONL not found: $INPUT_JSONL" >&2
    echo "       Drop the JSONL into p8_to_check/ first." >&2
    exit 1
fi

# gemma4_31b requires vLLM 0.24 — use a separate SIF built from container_gemma4_judge.def
if [ "$MODEL" = "gemma4_31b" ]; then
    SIF="$DATA_DIR/gemma4_judge.sif"
    if [ ! -f "$SIF" ]; then
        echo "ERROR: $SIF not found — build with: sbatch containers/build_gemma4_judge_container.sh" >&2
        exit 1
    fi
else
    SIF="$DATA_DIR/castor_judge.sif"
    if [ ! -f "$SIF" ]; then
        echo "ERROR: $SIF not found — run: bash containers/build_judge_container.sh --model salvage_embed" >&2
        exit 1
    fi
fi

OUT_DIR="$REPO/results/p8_plan_coherence"
GT_CSV="$REPO/human_ground_truth_label/human_gt.csv"
mkdir -p "$OUT_DIR"

echo " Model dir : $MODEL_DIR"
echo " Input     : $INPUT_JSONL"
echo " Output    : $OUT_DIR/$RUN_NAME/"
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
    python3 "$REPO/pipelines/plan_coherence/run_coherence_judge.py" \
        --model     "$MODEL" \
        --model-dir "$MODEL_DIR" \
        --input     "$INPUT_JSONL" \
        --out       "$OUT_DIR" \
        --gt        "$GT_CSV" \
        ${LIMIT:+--limit "$LIMIT"}

EXIT_CODE=$?
echo "==========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " P8 judge complete ($MODEL) : $(date)"
    ls -lh "$OUT_DIR/$RUN_NAME/${RUN_NAME}_${MODEL}.csv" 2>/dev/null || true
else
    echo " P8 judge FAILED (exit $EXIT_CODE) : $(date)" >&2
fi
echo "==========================================="
exit $EXIT_CODE
