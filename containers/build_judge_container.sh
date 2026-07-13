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
#                    One of: qwen25_72b | deepseek_r1 | salvage_embed | all
#                    Default: all
#                    (qwen25_72b + deepseek_r1 = P6 Stage 1 extraction models;
#                     salvage_embed = sentence-transformers/all-MiniLM-L6-v2,
#                     used by Pipeline 6 Stage 2 clustering -- no Ollama on
#                     the cluster, see pipelines/salvage_analysis/normalize.py)
#                    NOTE: P5 judge models (deepseek_r1_32b, glm4_32b, etc.)
#                    are downloaded by judge_panel_submit.sh, not here.
#   --force          Rebuild the SIF even if hash matches.
#
# Run from ~/Eval_CASTOR/:
#   bash containers/build_judge_container.sh
#   bash containers/build_judge_container.sh --model qwen25_72b
#
# Submit as a SLURM job (CPU node, no GPU needed for build):
#   sbatch --partition=pleiades --cpus-per-task=8 --mem=64G --time=2:00:00 \
#          --job-name=build_judge --output=/data/$USER/logs/build_judge_%j.out \
#          containers/build_judge_container.sh
#
# NOTE: --mem=64G is required for vLLM 0.8.x. mksquashfs loads the full
# uncompressed image into /tmp (RAM-backed on most nodes) to compress it.
# APPTAINER_TMPDIR is set below to redirect that to the NAS instead.
# ─────────────────────────────────────────────────────────────────────────────
set -e

# When sbatch stages this script to /var/spool/slurmd/jobXXX/, BASH_SOURCE[0]
# resolves to the spool copy — but container_judge.def is only in the real repo.
# Use SLURM_SUBMIT_DIR (the directory sbatch was called from) when inside an
# actual SLURM job; fall back to BASH_SOURCE[0] for interactive runs.
# The contamination guard is checking SLURM_JOB_ID: stale SLURM_SUBMIT_DIR in
# an interactive shell never has SLURM_JOB_ID set.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/containers" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
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

# Redirect mksquashfs temp space from RAM-backed /tmp to the NAS so that large
# images (vLLM 0.8.x is ~3× bigger than 0.6.3) don't OOM during squashfs creation.
export APPTAINER_TMPDIR="$DATA_DIR/apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

if $FORCE || [ ! -f "$SIF" ] || [ ! -f "$SIF_HASH_FILE" ] || \
   [ "$DEF_HASH" != "$(cat "$SIF_HASH_FILE" 2>/dev/null)" ]; then
    echo "[$(date)] Building $SIF (def hash: $DEF_HASH) ..."
    apptainer build --force --fakeroot "$SIF" "$DEF"
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
        --env HF_HOME="$HF_HOME" \
        --env HF_TOKEN="${HF_TOKEN:-}" \
        "$SIF" \
        python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_REPO',
    local_dir='$LOCAL_DIR',
    token=os.environ.get('HF_TOKEN') or None,
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
        # AWQ 4-bit quantized — fits on 1× RTX 6000 Ada (48 GB)
        download_model "Qwen/Qwen2.5-72B-Instruct-AWQ" \
                       "qwen25-72b-instruct-awq" "Qwen2.5-72B-AWQ"
        ;;&
    deepseek_r1|all)
        # Community AWQ 4-bit repo (~40 GB, fits on 1× RTX 6000 Ada).
        download_model "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ" \
                       "deepseek-r1-distill-llama-70b-awq" "DeepSeek-R1-70B-AWQ"
        ;;&
    salvage_embed|all)
        # Small embedding model for Pipeline 6 Stage 2 clustering (~90 MB,
        # CPU-only at inference time) -- see normalize.py's --backend local.
        download_model "sentence-transformers/all-MiniLM-L6-v2" \
                       "all-minilm-l6-v2" "Salvage-Embed-MiniLM"
        ;;
    *)
        echo "ERROR: unknown --model value '$MODEL_FILTER'"
        echo "       Use: qwen25_72b | deepseek_r1 | salvage_embed | all"
        exit 1
        ;;
esac

echo "==========================================="
echo " Build complete : $(date)"
echo " SIF            : $SIF"
echo " Model(s)       : $MODEL_FILTER"
echo "==========================================="
