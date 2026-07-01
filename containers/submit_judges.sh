#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CASTOR Judge Panel — orchestration script.
#
# Submits three parallel judge jobs (one per model) and a downstream
# aggregation job that runs only after all three succeed.
#
# Run from ~/Eval_CASTOR/ on the head node (head1.condo.cs.cmu.edu):
#
#   bash containers/submit_judges.sh answers_baseline
#   bash containers/submit_judges.sh answers_degf --limit 20
#
# Arguments:
#   RUN_NAME       Name of the inference JSONL (without .jsonl extension).
#                  The file must exist at /data/$USER/castor_results/$RUN_NAME.jsonl
#   --limit N      (optional) Score only the first N records per model.
#                  Set LIMIT env var inside each job for run_judge.py to pick up.
#
# Prerequisite: build container + download models first:
#   bash containers/build_judge_container.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/castor_judge_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
mkdir -p "$LOGS_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
RUN_NAME="${1:?Usage: submit_judges.sh RUN_NAME [--limit N]}"
shift

LIMIT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        *)       echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
INPUT_JSONL="$DATA_DIR/castor_results/${RUN_NAME}.jsonl"
if [ ! -f "$INPUT_JSONL" ]; then
    echo "ERROR: inference JSONL not found: $INPUT_JSONL" >&2
    exit 1
fi

SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: container SIF not found: $SIF" >&2
    echo "       Run: bash containers/build_judge_container.sh" >&2
    exit 1
fi

echo "==========================================="
echo " CASTOR Judge Panel — Submitting jobs"
echo " Run name  : $RUN_NAME"
echo " Input     : $INPUT_JSONL"
echo " Container : $SIF"
[ -n "$LIMIT" ] && echo " Limit     : $LIMIT records (smoke test)"
echo "==========================================="

# ── Common sbatch options ─────────────────────────────────────────────────────
COMMON_OPTS=(
    -p pleiades
    --constraint=RTX6000ADA
    --parsable
)

# Export LIMIT so submit_judge_job.sh can pick it up via ${LIMIT:+--limit "$LIMIT"}
export LIMIT

# ── Submit three judge jobs in parallel ───────────────────────────────────────
echo "[$(date)] Submitting qwen25_72b ..."
J_QWEN=$(sbatch "${COMMON_OPTS[@]}" \
    --gpus=1 --mem=52G \
    --job-name="judge_qwen" \
    --output="$LOGS_DIR/castor_judge_qwen_%j.out" \
    --error="$LOGS_DIR/castor_judge_qwen_%j.err" \
    "$SCRIPT_DIR/submit_judge_job.sh" qwen25_72b "$RUN_NAME")
echo "  qwen25_72b  -> job $J_QWEN"

echo "[$(date)] Submitting deepseek_r1 ..."
J_DEEP=$(sbatch "${COMMON_OPTS[@]}" \
    --gpus=2 --mem=80G \
    --job-name="judge_deep" \
    --output="$LOGS_DIR/castor_judge_deepseek_%j.out" \
    --error="$LOGS_DIR/castor_judge_deepseek_%j.err" \
    "$SCRIPT_DIR/submit_judge_job.sh" deepseek_r1 "$RUN_NAME")
echo "  deepseek_r1 -> job $J_DEEP"

echo "[$(date)] Submitting gptoss_120b ..."
J_GPTOSS=$(sbatch "${COMMON_OPTS[@]}" \
    --gpus=2 --mem=104G \
    --job-name="judge_gptoss" \
    --output="$LOGS_DIR/castor_judge_gptoss_%j.out" \
    --error="$LOGS_DIR/castor_judge_gptoss_%j.err" \
    "$SCRIPT_DIR/submit_judge_job.sh" gptoss_120b "$RUN_NAME")
echo "  gptoss_120b -> job $J_GPTOSS"

# ── Submit aggregation job (runs only if all three succeed) ───────────────────
DEPENDENCY="afterok:${J_QWEN}:${J_DEEP}:${J_GPTOSS}"
echo "[$(date)] Submitting aggregation (dependency: $DEPENDENCY) ..."
J_AGG=$(sbatch \
    -p pleiades \
    --cpus-per-task=4 --mem=8G \
    --time=1:00:00 \
    --job-name="judge_agg" \
    --output="$LOGS_DIR/castor_judge_agg_%j.out" \
    --error="$LOGS_DIR/castor_judge_agg_%j.err" \
    --dependency="$DEPENDENCY" \
    --parsable \
    "$SCRIPT_DIR/aggregate_job.sh" "$RUN_NAME")
echo "  aggregation -> job $J_AGG"

echo "==========================================="
echo " Jobs submitted:"
echo "   qwen25_72b  : $J_QWEN"
echo "   deepseek_r1 : $J_DEEP"
echo "   gptoss_120b : $J_GPTOSS"
echo "   aggregation : $J_AGG  (starts after all three succeed)"
echo ""
echo " Monitor:"
echo "   squeue -u $USER"
echo "   tail -f $LOGS_DIR/castor_judge_qwen_${J_QWEN}.out"
echo ""
echo " Outputs will be written to:"
echo "   $DATA_DIR/castor_results/p5_judge/$RUN_NAME/"
echo "==========================================="
