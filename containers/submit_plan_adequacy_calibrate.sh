#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 9 (plan adequacy) — calibration bake-off orchestration.
#
# Submits one calibration job per candidate model (glm4_32b, llama_3_3_70b,
# phi4_14b -- design plan section 4g), then a comparison job that waits for
# all of them and picks a winner against the go/no-go thresholds.
#
# Run from ~/BenchyBench/Eval_CASTOR/ on the head node (head1.condo.cs.cmu.edu):
#
#   bash containers/submit_plan_adequacy_calibrate.sh                 # all 3 candidates
#   bash containers/submit_plan_adequacy_calibrate.sh --model glm4_32b  # single model
#
# Prerequisites:
#   - castor_judge.sif already built (bash containers/build_judge_container.sh --model salvage_embed)
#   - glm4_32b / llama_3_3_70b weights already downloaded (reused from P8's
#     judge_panel_submit.sh -- see coherence_judge_job.sh's own prerequisites)
#   - phi4_14b weights downloaded: sbatch containers/download_job.sh \
#       RedHatAI/phi-4-quantized.w4a16 /data/$USER/phi-4-w4a16
#   - The gold set built: python3 pipelines/plan_adequacy/calibration/build_gold_combined.py
#
# Outputs:
#   results/p9_plan_adequacy/calibration/calibration_<model>.json  (per model)
#   results/p9_plan_adequacy/calibration/comparison.json           (winner)
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/p9_calibrate_<model>_<jobid>.out
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
mkdir -p "$LOGS_DIR"

SINGLE_MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) SINGLE_MODEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: container not found: $SIF" >&2
    echo "       Run: bash containers/build_judge_container.sh --model salvage_embed" >&2
    exit 1
fi

GOLD="$REPO/pipelines/plan_adequacy/calibration/gold_tool_calls.jsonl"
if [ ! -f "$GOLD" ]; then
    echo "ERROR: gold set not found: $GOLD" >&2
    echo "       Build it with: python3 pipelines/plan_adequacy/calibration/build_gold_combined.py" >&2
    exit 1
fi

declare -A MODEL_MEM=(
    [glm4_32b]="52G"
    [llama_3_3_70b]="52G"
    [phi4_14b]="20G"
)

if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
else
    MODELS=(glm4_32b llama_3_3_70b phi4_14b)
fi

echo "==========================================="
echo " Pipeline 9 — Extraction Model Calibration"
echo " Models   : ${MODELS[*]}"
echo " SIF      : $SIF"
echo " Gold set : $GOLD"
echo "==========================================="

JOB_IDS=()
for MODEL in "${MODELS[@]}"; do
    MEM="${MODEL_MEM[$MODEL]:-40G}"
    JOB_ID=$(sbatch --parsable \
        -p pleiades \
        --constraint=RTX6000ADA \
        --gpus=1 \
        --mem="$MEM" \
        --output="$LOGS_DIR/p9_calibrate_${MODEL}_%j.out" \
        --error="$LOGS_DIR/p9_calibrate_${MODEL}_%j.err" \
        "$SCRIPT_DIR/plan_adequacy_calibrate_job.sh" "$MODEL")
    echo "  [$MODEL]  mem=$MEM  job=$JOB_ID"
    JOB_IDS+=("$JOB_ID")
done

DEPENDENCY="afterok"
for JID in "${JOB_IDS[@]}"; do
    DEPENDENCY="${DEPENDENCY}:${JID}"
done

COMPARE_JOB=$(sbatch --parsable \
    --dependency="$DEPENDENCY" \
    --output="$LOGS_DIR/p9_calibrate_compare_%j.out" \
    --error="$LOGS_DIR/p9_calibrate_compare_%j.err" \
    "$SCRIPT_DIR/plan_adequacy_calibrate_compare_job.sh")

echo ""
echo "  [comparison] job=$COMPARE_JOB (after ${JOB_IDS[*]})"
echo ""
echo "==========================================="
echo " Monitor: squeue -u \$USER"
echo " Result : tail -f $LOGS_DIR/p9_calibrate_compare_${COMPARE_JOB}.out"
echo "==========================================="
