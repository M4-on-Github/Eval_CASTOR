#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 7 — Assertion Coverage Analysis, SLURM array task.
#
# Runs check_assertions.py for ONE run (selected from the manifest by
# SLURM_ARRAY_TASK_ID) using Selene 8B AWQ inside castor_judge.sif.
#
# One LLM call per (image, assertion) pair — ~3000 calls total per run,
# ~5-10 min with vLLM prefix caching. Resume-safe: already-processed images
# are skipped on re-submission.
#
# Do NOT call directly with sbatch — use submit_assertion_coverage.sh instead.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --gpus=1
#SBATCH --constraint=RTX6000ADA
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH -J p7_coverage
# NOTE: --array / --output / --error are set by submit_assertion_coverage.sh

set -euo pipefail

MANIFEST="${1:?Usage: assertion_coverage_job.sh MANIFEST_FILE (run as a SLURM array task)}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set — run via submit_assertion_coverage.sh}"

RUN_NAME="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")"
if [ -z "$RUN_NAME" ]; then
    echo "ERROR: no run name at manifest line $((SLURM_ARRAY_TASK_ID + 1)) in $MANIFEST" >&2
    exit 1
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"

echo "==========================================="
echo " CASTOR Pipeline 7 — Assertion Coverage"
echo " Job ID   : $SLURM_JOB_ID (task $SLURM_ARRAY_TASK_ID)"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Run name : $RUN_NAME"
echo "==========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── Inputs ───────────────────────────────────────────────────────────────────
INPUT_JSONL="$DATA_DIR/castor_results/${RUN_NAME}.jsonl"
if [ ! -f "$INPUT_JSONL" ]; then
    echo "ERROR: inference JSONL not found: $INPUT_JSONL" >&2
    exit 1
fi

MODEL_AWQ="$DATA_DIR/selene-1-mini-llama-3.1-8b-awq"
MODEL_FP16="$DATA_DIR/selene-1-mini-llama-3.1-8b-fp16"

if [ -d "$MODEL_AWQ" ]; then
    MODEL_DIR="$MODEL_AWQ"
elif [ -d "$MODEL_FP16" ]; then
    MODEL_DIR="$MODEL_FP16"
    echo "WARNING: AWQ weights not found, falling back to FP16 (uses more VRAM)"
else
    echo "ERROR: Selene weights not found at $MODEL_AWQ or $MODEL_FP16" >&2
    exit 1
fi

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found — run: bash containers/build_judge_container.sh --model salvage_embed" >&2
    exit 1
fi

OUT_DIR="$REPO/results/p7_assertion_coverage"
GT_CSV="$REPO/human_ground_truth_label/human_gt.csv"

echo " Input    : $INPUT_JSONL"
echo " Model    : $MODEL_DIR"
echo " Output   : $OUT_DIR/$RUN_NAME/"
echo " GT CSV   : $GT_CSV"
echo "==========================================="

mkdir -p "$OUT_DIR"

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
    python3 "$REPO/pipelines/assertion_coverage/check_assertions.py" \
        --input   "$INPUT_JSONL" \
        --model-dir "$MODEL_DIR" \
        --gt      "$GT_CSV" \
        --out     "$OUT_DIR"

EXIT_CODE=$?

echo "==========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo " P7 complete : $(date)"
    echo " Outputs:"
    ls -lh "$OUT_DIR/$RUN_NAME/"* 2>/dev/null || true
else
    echo " P7 FAILED (exit $EXIT_CODE) : $(date)" >&2
fi
echo "==========================================="

exit $EXIT_CODE
