#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 8 (plan coherence) — orchestration script.
#
# Submits 5 parallel single-model judge jobs then an aggregation job that
# waits for all 5 to succeed before it starts.
#
# Drop the full-answer JSONL(s) you want scored into p8_to_check/ first, then
# run from ~/Eval_CASTOR/ on the head node (head1.condo.cs.cmu.edu):
#
#   bash containers/submit_coherence.sh                      # all JSONLs in p8_to_check/
#   bash containers/submit_coherence.sh --run answers_baseline  # single run
#
# Arguments:
#   --run NAME        Optional. Run name (stem of the JSONL, no extension).
#                     If omitted, all *.jsonl files in p8_to_check/ are submitted.
#   --limit N         Optional. Smoke test: pass --limit N to each judge job.
#                     Set N=5 to score only 5 images per model.
#
# Prerequisites:
#   castor_judge.sif must already exist:
#     bash containers/build_judge_container.sh --model salvage_embed
#
#   New judge model weights must be downloaded to /data/$USER/:
#     sbatch containers/download_job.sh RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16 \
#            /data/$USER/llama-3.3-70b-instruct-w4a16
#     sbatch containers/download_job.sh RedHatAI/phi-4-quantized.w4a16 \
#            /data/$USER/phi-4-w4a16
#     sbatch containers/download_job.sh google/gemma-4-31B-it-qat-w4a16-ct \
#            /data/$USER/gemma4-31b-it-w4a16
#
#   Existing P5 models reused for P8 (no extra download needed):
#     deepseek-r1-distill-qwen-32b-awq  (from judge_panel_submit.sh)
#     glm-4-32b-0414-gptq               (from judge_panel_submit.sh)
#
# Outputs (in repo-local results/, gitignored):
#   results/p8_plan_coherence/<run>/<run>_<model>.csv   (per judge, 5 files)
#   results/p8_plan_coherence/<run>/<run>_per_step.csv  (aggregated per step)
#   results/p8_plan_coherence/<run>/<run>_per_image.csv (aggregated per image)
#   results/p8_plan_coherence/<run>/<run>_summary.csv   (run-level summary)
#   results/p8_plan_coherence/eval_summary_coherence.csv  (cumulative, appended)
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/p8_coherence_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
TO_CHECK_DIR="$REPO/p8_to_check"
mkdir -p "$LOGS_DIR"

RUN_NAME=""
LIMIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)   RUN_NAME="$2"; shift 2 ;;
        --limit) LIMIT="$2";    shift 2 ;;
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

# ── Per-model VRAM / mem settings ─────────────────────────────────────────────
declare -A MODEL_MEM=(
    [deepseek_r1_32b]="52G"
    [glm4_32b]="52G"
    [llama_3_3_70b]="52G"
    [phi4_14b]="20G"
    [gemma4_31b]="40G"
)

MODELS=(deepseek_r1_32b glm4_32b llama_3_3_70b phi4_14b gemma4_31b)

# ── Collect run names ──────────────────────────────────────────────────────────
if [ -n "$RUN_NAME" ]; then
    RUN_NAMES=("$RUN_NAME")
else
    mapfile -t JSONL_FILES < <(find "$TO_CHECK_DIR" -maxdepth 1 -name "*.jsonl" | sort)
    if [ ${#JSONL_FILES[@]} -eq 0 ]; then
        echo "ERROR: no *.jsonl files found in $TO_CHECK_DIR" >&2
        echo "       Drop inference JSONLs there first, or use --run NAME." >&2
        exit 1
    fi
    RUN_NAMES=()
    for f in "${JSONL_FILES[@]}"; do
        RUN_NAMES+=("$(basename "$f" .jsonl)")
    done
fi

echo "==========================================="
echo " Pipeline 8 — Plan Coherence Analysis"
echo " Runs     : ${RUN_NAMES[*]}"
echo " SIF      : $SIF"
[ -n "$LIMIT" ] && echo " Limit    : $LIMIT images (smoke test)"
echo "==========================================="

# ── Submit jobs for each run ───────────────────────────────────────────────────
submit_run() {
    local RN="$1"
    local INPUT_JSONL="$TO_CHECK_DIR/${RN}.jsonl"
    if [ ! -f "$INPUT_JSONL" ]; then
        echo "  SKIP $RN: $INPUT_JSONL not found" >&2
        return
    fi

    echo ""
    echo "  --- $RN ---"
    local JOB_IDS=()
    for MODEL in "${MODELS[@]}"; do
        local MEM="${MODEL_MEM[$MODEL]}"
        local EXPORT_ARGS="ALL"
        [ -n "$LIMIT" ] && EXPORT_ARGS="ALL,LIMIT=$LIMIT"

        local JOB_ID
        JOB_ID=$(sbatch --parsable \
            -p pleiades \
            --constraint=RTX6000ADA \
            --gpus=1 \
            --mem="$MEM" \
            --export="$EXPORT_ARGS" \
            --output="$LOGS_DIR/p8_coherence_${MODEL}_%j.out" \
            --error="$LOGS_DIR/p8_coherence_${MODEL}_%j.err" \
            "$SCRIPT_DIR/coherence_judge_job.sh" "$MODEL" "$RN")

        echo "    [$MODEL]  mem=$MEM  job=$JOB_ID"
        JOB_IDS+=("$JOB_ID")
    done

    local DEPENDENCY="afterok"
    for JID in "${JOB_IDS[@]}"; do
        DEPENDENCY="${DEPENDENCY}:${JID}"
    done

    local AGG_JOB
    AGG_JOB=$(sbatch --parsable \
        --dependency="$DEPENDENCY" \
        --output="$LOGS_DIR/p8_coherence_agg_%j.out" \
        --error="$LOGS_DIR/p8_coherence_agg_%j.err" \
        "$SCRIPT_DIR/coherence_aggregate_job.sh" "$RN")

    echo "    [aggregation] job=$AGG_JOB (after ${JOB_IDS[*]})"
    echo "    tail -f $LOGS_DIR/p8_coherence_agg_${AGG_JOB}.out"
}

for RN in "${RUN_NAMES[@]}"; do
    submit_run "$RN"
done

echo ""
echo "==========================================="
echo " All runs submitted. Monitor with:"
echo "   squeue -u $USER"
echo " Outputs:"
echo "   $REPO/results/p8_plan_coherence/<run>/"
echo "   $REPO/results/p8_plan_coherence/eval_summary_coherence.csv"
echo "==========================================="
