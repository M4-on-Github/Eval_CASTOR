#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CASTOR Judge Panel — unified setup and submission script.
#
# Run from ~/Eval_CASTOR/ on the head node. Safe to re-run at any point:
# existing weights are detected and reused, already-scored records are skipped.
#
# For each model the script determines what is needed and submits a
# chain of SLURM jobs automatically:
#
#   READY      weights exist → judge job submitted immediately
#   QUANT      FP16 exists, no AWQ → quant job → judge job (chained)
#   DL+JUDGE   pre-quant AWQ missing → download job → judge job (chained)
#   DL+QUANT   FP16 missing → download job → quant job → judge job (chained)
#
# Usage:
#   bash containers/judge_panel_submit.sh RUN_NAME [OPTIONS]
#
# Options:
#   --limit N        Score only the first N records (smoke test)
#   --dry-run        Print the job plan without submitting anything
#   --no-download    Skip auto-download; print the command instead (use if
#                    compute nodes on your cluster have no internet access)
#
# Environment overrides:
#   SKIP_MODELS      Space-separated model keys to skip entirely
#   HF_TOKEN         HuggingFace token for gated models (usually not needed)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
mkdir -p "$LOGS_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
RUN_NAME="${1:?Usage: judge_panel_submit.sh RUN_NAME [--limit N] [--dry-run] [--no-download]}"
shift

LIMIT=""
DRY_RUN=0
NO_DOWNLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)       LIMIT="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1;  shift ;;
        --no-download) NO_DOWNLOAD=1; shift ;;
        *)             echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
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

# Export so submit_judge_job.sh picks it up via ${LIMIT:+--limit "$LIMIT"}
export LIMIT

SKIP_MODELS="${SKIP_MODELS:-}"
HF_TOKEN="${HF_TOKEN:-}"

# ── Model definitions ─────────────────────────────────────────────────────────
# Five parallel indexed arrays — order must match across all arrays.
#
#   KEYS        key used in run_judge.py _MODEL_CONFIG
#   AWQ_DIRS    directory under /data/$USER/ for quantized (AWQ or GPTQ) weights
#   FP16_DIRS   directory for FP16 source weights ("" = pre-quantized, dl AWQ directly)
#   NEEDS_QUANT 1 = self-quantize with AutoAWQ after download; 0 = already quantized on HF
#   MEM_GB      --mem for the judge job
#   HF_REPOS    HuggingFace repo ID to download

KEYS=(deepseek_r1_32b glm4_32b selene_mini_8b)

AWQ_DIRS=(
    "deepseek-r1-distill-qwen-32b-awq"
    "glm-4-32b-0414-gptq"
    "selene-1-mini-llama-3.1-8b-awq"
)

FP16_DIRS=(
    ""                             # pre-quantized: download AWQ dir directly
    ""                             # pre-quantized: download GPTQ dir directly
    "selene-1-mini-llama-3.1-8b-fp16"
)

NEEDS_QUANT=(0 0 1)

MEM_GB=(52 52 16)

HF_REPOS=(
    "casperhansen/deepseek-r1-distill-qwen-32b-awq"
    "mratsim/GLM-4-32B-0414.w4a16-gptq"
    "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
)

# sbatch options shared across all inference-capable jobs
COMMON_OPTS=(-p pleiades --constraint=RTX6000ADA --parsable)
# Download jobs need no GPU; use pleiades without constraint (any node)
DL_OPTS=(-p pleiades --parsable)

# ── Header ────────────────────────────────────────────────────────────────────
echo "==========================================="
echo " CASTOR Judge Panel"
echo " Run name  : $RUN_NAME"
echo " Input     : $INPUT_JSONL"
[ -n "$LIMIT" ]          && echo " Limit     : $LIMIT records"
[ "$DRY_RUN"  -eq 1 ]   && echo " Mode      : DRY RUN — no jobs submitted"
[ "$NO_DOWNLOAD" -eq 1 ] && echo " Downloads : disabled (--no-download)"
echo "==========================================="
echo ""
printf "%-22s  %-12s  %-5s  %s\n" "MODEL" "STATUS" "VRAM" "JOB CHAIN"
printf "%-22s  %-12s  %-5s  %s\n" "─────────────────────" "────────────" "────" "────────────────────────────────────────────"

# ── Per-model loop ────────────────────────────────────────────────────────────
JUDGE_JOB_IDS=()

_dir_nonempty() { [ -d "$1" ] && [ "$(ls -A "$1" 2>/dev/null)" ]; }

for i in "${!KEYS[@]}"; do
    KEY="${KEYS[$i]}"
    AWQ_PATH="$DATA_DIR/${AWQ_DIRS[$i]}"
    FP16_DIR="${FP16_DIRS[$i]}"
    FP16_PATH="${FP16_DIR:+$DATA_DIR/$FP16_DIR}"
    NQ="${NEEDS_QUANT[$i]}"
    MEM="${MEM_GB[$i]}"
    HF="${HF_REPOS[$i]}"

    # ── Skip override ──────────────────────────────────────────────────────────
    if echo "$SKIP_MODELS" | grep -qw "$KEY"; then
        printf "%-22s  %-12s  %-5s  %s\n" "$KEY" "SKIPPED" "${MEM}G" "SKIP_MODELS env var"
        JUDGE_JOB_IDS+=("")
        continue
    fi

    # ── Determine status ───────────────────────────────────────────────────────
    if _dir_nonempty "$AWQ_PATH"; then
        STATUS="READY"
    elif [ "$NQ" -eq 1 ] && [ -n "$FP16_PATH" ] && _dir_nonempty "$FP16_PATH"; then
        STATUS="QUANT"          # FP16 present, quantization needed
    elif [ "$NQ" -eq 1 ]; then
        STATUS="DL+QUANT"       # need to download FP16, then quantize
    else
        STATUS="DL+JUDGE"       # need to download pre-quantized weights
    fi

    J_DOWNLOAD=""
    J_QUANT=""
    CHAIN_DESC=""

    # ── Submit download job (if weights are missing) ───────────────────────────
    if [[ "$STATUS" == DL* ]]; then
        if [ "$NQ" -eq 1 ]; then
            DL_SRC="$HF"
            DL_DST="$FP16_PATH"
        else
            DL_SRC="$HF"
            DL_DST="$AWQ_PATH"
        fi

        if [ "$NO_DOWNLOAD" -eq 1 ]; then
            # Just print the command; cannot chain jobs without a job ID
            CMD="hf download $DL_SRC --local-dir $DL_DST"
            printf "%-22s  %-12s  %-5s  %s\n" "$KEY" "NO-DL" "${MEM}G" "$CMD"
            JUDGE_JOB_IDS+=("")
            continue
        fi

        if [ "$DRY_RUN" -eq 0 ]; then
            HF_TOKEN_OPT="${HF_TOKEN:+--env HUGGINGFACE_TOKEN=$HF_TOKEN}"
            J_DOWNLOAD=$(sbatch "${DL_OPTS[@]}" \
                --cpus-per-task=4 --mem=8G \
                --time=6:00:00 \
                --job-name="dl_${KEY}" \
                --output="$LOGS_DIR/dl_${KEY}_%j.out" \
                --error="$LOGS_DIR/dl_${KEY}_%j.err" \
                "$SCRIPT_DIR/download_job.sh" "$DL_SRC" "$DL_DST")
            CHAIN_DESC="dl=$J_DOWNLOAD"
        else
            CHAIN_DESC="[DRY] dl"
        fi
    fi

    # ── Submit quantization job (if self-quantization needed) ─────────────────
    if [[ "$STATUS" == "QUANT" || "$STATUS" == "DL+QUANT" ]]; then
        QUANT_DEP_OPTS=()
        [ -n "$J_DOWNLOAD" ] && QUANT_DEP_OPTS=(--dependency="afterok:${J_DOWNLOAD}")

        if [ "$DRY_RUN" -eq 0 ]; then
            J_QUANT=$(sbatch "${COMMON_OPTS[@]}" \
                --gpus=1 --mem=60G \
                --cpus-per-task=8 \
                --time=2:00:00 \
                --job-name="quant_${KEY}" \
                --output="$LOGS_DIR/quant_${KEY}_%j.out" \
                --error="$LOGS_DIR/quant_${KEY}_%j.err" \
                "${QUANT_DEP_OPTS[@]+"${QUANT_DEP_OPTS[@]}"}" \
                "$SCRIPT_DIR/quantize_job.sh" \
                    "${FP16_PATH:-$AWQ_PATH}" "$AWQ_PATH")
            CHAIN_DESC="${CHAIN_DESC:+$CHAIN_DESC → }quant=$J_QUANT"
        else
            CHAIN_DESC="${CHAIN_DESC:+$CHAIN_DESC → }[DRY] quant"
        fi
    fi

    # ── Submit judge job ───────────────────────────────────────────────────────
    JUDGE_DEP_OPTS=()
    if [ -n "$J_QUANT" ]; then
        JUDGE_DEP_OPTS=(--dependency="afterok:${J_QUANT}")
    elif [ -n "$J_DOWNLOAD" ]; then
        JUDGE_DEP_OPTS=(--dependency="afterok:${J_DOWNLOAD}")
    fi

    if [ "$DRY_RUN" -eq 0 ]; then
        J_JUDGE=$(sbatch "${COMMON_OPTS[@]}" \
            --gpus=1 --mem="${MEM}G" \
            --cpus-per-task=8 \
            --time=12:00:00 \
            --job-name="judge_${KEY}" \
            --output="$LOGS_DIR/castor_judge_${KEY}_%j.out" \
            --error="$LOGS_DIR/castor_judge_${KEY}_%j.err" \
            "${JUDGE_DEP_OPTS[@]+"${JUDGE_DEP_OPTS[@]}"}" \
            "$SCRIPT_DIR/submit_judge_job.sh" "$KEY" "$RUN_NAME")
        CHAIN_DESC="${CHAIN_DESC:+$CHAIN_DESC → }judge=$J_JUDGE"
        JUDGE_JOB_IDS+=("$J_JUDGE")
    else
        CHAIN_DESC="${CHAIN_DESC:+$CHAIN_DESC → }[DRY] judge"
        JUDGE_JOB_IDS+=("DRY_${KEY}")
    fi

    printf "%-22s  %-12s  %-5s  %s\n" "$KEY" "$STATUS" "${MEM}G" "$CHAIN_DESC"
done

echo ""

# ── Aggregation: runs after all judge jobs succeed ────────────────────────────
JUDGE_IDS_JOINED=""
for JID in "${JUDGE_JOB_IDS[@]}"; do
    [[ -z "$JID" || "$JID" == DRY_* ]] && continue
    JUDGE_IDS_JOINED="${JUDGE_IDS_JOINED}:${JID}"
done
JUDGE_IDS_JOINED="${JUDGE_IDS_JOINED#:}"

if [ -z "$JUDGE_IDS_JOINED" ]; then
    echo "━━━ No judge jobs could be submitted. ━━━"
    if [ "$NO_DOWNLOAD" -eq 1 ]; then
        echo "Run the download commands shown above, then re-run this script."
    else
        echo "Check the errors above — the download jobs should handle it on next run."
    fi
    exit 0
fi

printf "%-22s  %-12s  %-5s  " "aggregation" "" ""
if [ "$DRY_RUN" -eq 0 ]; then
    J_AGG=$(sbatch \
        -p pleiades \
        --cpus-per-task=4 --mem=8G \
        --time=1:00:00 \
        --job-name="judge_agg" \
        --output="$LOGS_DIR/castor_judge_agg_%j.out" \
        --error="$LOGS_DIR/castor_judge_agg_%j.err" \
        --dependency="afterok:${JUDGE_IDS_JOINED}" \
        --parsable \
        "$SCRIPT_DIR/aggregate_job.sh" "$RUN_NAME")
    echo "job=$J_AGG (after ${JUDGE_IDS_JOINED})"
else
    echo "[DRY] after judge jobs"
fi

echo ""
echo "Monitor : squeue -u $USER"
echo "Logs    : $LOGS_DIR/castor_judge_<KEY>_<JOBID>.out"
echo "Output  : $DATA_DIR/castor_results/p5_judge/$RUN_NAME/"
echo "==========================================="
