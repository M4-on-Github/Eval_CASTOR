#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CASTOR Judge Panel — single-model SLURM batch script.
#
# Do NOT call directly with sbatch — use submit_judges.sh instead.
# submit_judges.sh sets --gpus and --mem correctly per model, then passes
# the MODEL and RUN_NAME as positional arguments.
#
# Resources allocated by submit_judges.sh:
#   qwen25_72b  : --gpus=1 --mem=52G
#   deepseek_r1 : --gpus=1 --mem=52G
#   gptoss_120b : --gpus=2 --mem=104G
#
# All three jobs share this same script; the GPU count is set at submission.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --constraint=RTX6000ADA
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -J castor_judge
# NOTE: --output / --error / --gpus / --mem are set by submit_judges.sh

set -e

MODEL="${1:?Usage: submit_judge_job.sh MODEL RUN_NAME}"
RUN_NAME="${2:?Usage: submit_judge_job.sh MODEL RUN_NAME}"

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
mkdir -p "$LOGS_DIR"

JOB_START=$SECONDS

echo "==========================================="
echo " CASTOR Judge Panel — $MODEL"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Run name : $RUN_NAME"
echo " Repo     : $REPO"
echo "==========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── Paths ─────────────────────────────────────────────────────────────────────
SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: $SIF not found. Run build_judge_container.sh first." >&2
    exit 1
fi
echo "[$(date)] Container: $SIF"

export HF_HOME="$DATA_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$DATA_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

INPUT_JSONL="$DATA_DIR/castor_results/${RUN_NAME}.jsonl"
OUT_DIR="$DATA_DIR/castor_results/p5_judge/${RUN_NAME}"
GT_CSV="$REPO/human_ground_truth_label/human_gt.csv"
mkdir -p "$OUT_DIR"

if [ ! -f "$INPUT_JSONL" ]; then
    echo "ERROR: inference JSONL not found: $INPUT_JSONL" >&2
    exit 1
fi

# ── Model config ──────────────────────────────────────────────────────────────
case "$MODEL" in
    qwen25_72b)
        MODEL_DIR="$DATA_DIR/qwen25-72b-instruct"
        TP_SIZE=1
        ;;
    deepseek_r1)
        MODEL_DIR="$DATA_DIR/deepseek-r1-distill-llama-70b"
        TP_SIZE=1
        ;;
    gptoss_120b)
        MODEL_DIR="$DATA_DIR/gpt-oss-120b"
        TP_SIZE=2
        ;;
    *)
        echo "ERROR: unknown model '$MODEL'" >&2
        exit 1
        ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: model weights not found: $MODEL_DIR" >&2
    echo "       Run: bash containers/build_judge_container.sh --model $MODEL" >&2
    exit 1
fi
echo "[$(date)] Model dir: $MODEL_DIR (tp=$TP_SIZE)"

# ── Run inference inside container ────────────────────────────────────────────
# --containall prevents cluster apptainer.conf from shadowing /opt inside
# the container. We bind back only the two directories we need.
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
    python3 "$REPO/pipelines/judge_panel/run_judge.py" \
        --model     "$MODEL" \
        --model-dir "$MODEL_DIR" \
        --tp        "$TP_SIZE" \
        --input     "$INPUT_JSONL" \
        --out       "$OUT_DIR" \
        --gt        "$GT_CSV" \
        ${LIMIT:+--limit "$LIMIT"}

ELAPSED=$(( SECONDS - JOB_START ))
echo "==========================================="
echo " Finished   : $(date)"
echo " Wall time  : $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo " Output dir : $OUT_DIR"
echo "==========================================="
