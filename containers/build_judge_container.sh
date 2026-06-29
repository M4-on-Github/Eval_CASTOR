#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build the CASTOR Judge Panel Apptainer container and download model weights.
#
# Run once on an interactive SLURM compute node (or as a batch job) before
# submit_judges.sh. Safe to re-run: skips the build if the .def hasn't
# changed, and skips model downloads if the directory already exists.
#
# Usage:
#   bash containers/build_judge_container.sh [--model MODEL] [--force]
#
#   --model MODEL    Download only this model's weights after building.
#                    One of: qwen25_72b | deepseek_r1 | gptoss_120b | all
#                    Default: all
#   --force          Rebuild the SIF even if hash matches.
#
# Run from ~/Eval_CASTOR/:
#   bash containers/build_judge_container.sh
#   bash containers/build_judge_container.sh --model qwen25_72b
#
# Submit as a SLURM job (CPU node, no GPU needed for build):
#   sbatch --partition=pleiades --cpus-per-task=8 --mem=32G --time=2:00:00 \
#          --job-name=build_judge --output=/data/$USER/logs/build_judge_%j.out \
#          containers/build_judge_container.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
mkdir -p "$LOGS_DIR"

SIF="$DATA_DIR/castor_judge.sif"
DEF="$SCRIPT_DIR/container_judge.def"

# ── Parse args ───────────────────────────────────────────────────────────────
MODEL_FILTER="all"
FORCE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_FILTER="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "==========================================="
echo " CASTOR Judge Panel — Container Builder"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " User     : $USER"
echo " SIF      : $SIF"
echo " DEF      : $DEF"
echo " Model(s) : $MODEL_FILTER"
echo "==========================================="

# ── Build container (hash-gated) ─────────────────────────────────────────────
DEF_HASH=$(sha256sum "$DEF" | cut -d' ' -f1)
SIF_HASH_FILE="$SIF.def.sha256"

if $FORCE || [ ! -f "$SIF" ] || [ ! -f "$SIF_HASH_FILE" ] || \
   [ "$DEF_HASH" != "$(cat "$SIF_HASH_FILE" 2>/dev/null)" ]; then
    echo "[$(date)] Building $SIF (def hash: $DEF_HASH) ..."
    apptainer build --fakeroot "$SIF" "$DEF"
    echo "$DEF_HASH" > "$SIF_HASH_FILE"
    echo "[$(date)] Container built successfully."
else
    echo "[$(date)] Container up-to-date (hash: $DEF_HASH) — skipping build."
fi

# ── Model download helper ─────────────────────────────────────────────────────
# Downloads via HuggingFace hub inside the container so the same Python env
# is used as at inference time (no host conda conflicts).
download_model() {
    local HF_REPO="$1"
    local LOCAL_DIR="$DATA_DIR/$2"
    local LABEL="$3"

    if [ -d "$LOCAL_DIR" ] && [ -n "$(ls -A "$LOCAL_DIR" 2>/dev/null)" ]; then
        echo "[$(date)] [$LABEL] Already present at $LOCAL_DIR — skipping download."
        return 0
    fi

    mkdir -p "$LOCAL_DIR"
    echo "[$(date)] [$LABEL] Downloading $HF_REPO → $LOCAL_DIR ..."
    apptainer exec --containall --nv \
        --bind "$DATA_DIR:$DATA_DIR" \
        "$SIF" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_REPO',
    local_dir='$LOCAL_DIR',
    ignore_patterns=['*.pt', 'original/**'],
)
print('Download complete.')
"
    echo "[$(date)] [$LABEL] Done: $LOCAL_DIR"
}

# ── Download models ───────────────────────────────────────────────────────────
export HF_HOME="$DATA_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

case "$MODEL_FILTER" in
    qwen25_72b|all)
        download_model "Qwen/Qwen2.5-72B-Instruct" "qwen25-72b-instruct" "Qwen2.5-72B"
        ;;&
    deepseek_r1|all)
        download_model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
                       "deepseek-r1-distill-llama-70b" "DeepSeek-R1-70B"
        ;;&
    gptoss_120b|all)
        download_model "openai/gpt-oss-120b" "gpt-oss-120b" "GPT-OSS-120B"
        ;;
    *)
        echo "ERROR: unknown --model value '$MODEL_FILTER'"
        echo "       Use: qwen25_72b | deepseek_r1 | gptoss_120b | all"
        exit 1
        ;;
esac

echo "==========================================="
echo " Build complete : $(date)"
echo " SIF            : $SIF"
echo " Model(s)       : $MODEL_FILTER"
echo "==========================================="
