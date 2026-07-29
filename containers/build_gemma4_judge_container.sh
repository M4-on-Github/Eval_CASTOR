#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build gemma4_judge.sif — vLLM 0.24.0 container for the Gemma-4-31B P8 judge.
#
# Run once on the head node; takes ~15–20 min.
#
#   sbatch containers/build_gemma4_judge_container.sh
#
# Output: /data/$USER/gemma4_judge.sif
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH -J build_gemma4_judge
#SBATCH --output=/data/%u/logs/build_gemma4_judge_%j.out
#SBATCH --error=/data/%u/logs/build_gemma4_judge_%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="/data/$USER"
SIF="$DATA_DIR/gemma4_judge.sif"
DEF="$REPO/containers/container_gemma4_judge.def"
HASH_FILE="$SIF.def.sha256"

mkdir -p "$DATA_DIR/logs"

DEF_HASH=$(sha256sum "$DEF" | cut -d' ' -f1)

if [ -f "$SIF" ] && [ -f "$HASH_FILE" ] && [ "$DEF_HASH" = "$(cat "$HASH_FILE")" ]; then
    echo "[$(date)] gemma4_judge.sif is up-to-date (hash: $DEF_HASH) — skipping build."
    exit 0
fi

echo "[$(date)] Building $SIF from $DEF (hash: $DEF_HASH) ..."
apptainer build --fakeroot "$SIF" "$DEF"
echo "$DEF_HASH" > "$HASH_FILE"
echo "[$(date)] Done: $SIF ($(du -sh "$SIF" | cut -f1))"
