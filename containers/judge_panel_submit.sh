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
#   bash containers/judge_panel_submit.sh                   # all JSONLs in p5_to_judge/
#   bash containers/judge_panel_submit.sh --run RUN_NAME    # single run
#
# Options:
#   --run NAME       Run name (stem of the JSONL in p5_to_judge/). If omitted,
#                    all *.jsonl files in p5_to_judge/ are submitted.
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
TO_CHECK_DIR="$REPO/p5_to_judge"
mkdir -p "$LOGS_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
RUN_NAME=""
LIMIT=""
DRY_RUN=0
NO_DOWNLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)         RUN_NAME="$2"; shift 2 ;;
        --limit)       LIMIT="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=1;  shift ;;
        --no-download) NO_DOWNLOAD=1; shift ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)             echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

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

_dir_nonempty() { [ -d "$1" ] && [ "$(ls -A "$1" 2>/dev/null)" ]; }

# ── Per-run submission function ───────────────────────────────────────────────
submit_run() {
    local RN="$1"
    local INPUT_JSONL="$TO_CHECK_DIR/${RN}.jsonl"

    if [ ! -f "$INPUT_JSONL" ]; then
        echo "  SKIP $RN: $INPUT_JSONL not found" >&2
        return
    fi

    echo ""
    echo "==========================================="
    echo " CASTOR Judge Panel"
    echo " Run name  : $RN"
    echo " Input     : $INPUT_JSONL"
    [ -n "$LIMIT" ]          && echo " Limit     : $LIMIT records"
    [ "$DRY_RUN"  -eq 1 ]   && echo " Mode      : DRY RUN — no jobs submitted"
    [ "$NO_DOWNLOAD" -eq 1 ] && echo " Downloads : disabled (--no-download)"
    echo "==========================================="
    printf "%-22s  %-12s  %-5s  %s\n" "MODEL" "STATUS" "VRAM" "JOB CHAIN"
    printf "%-22s  %-12s  %-5s  %s\n" "─────────────────────" "────────────" "────" "────────────────────────────────────────────"

    local JUDGE_JOB_IDS=()

    for i in "${!KEYS[@]}"; do
        local KEY="${KEYS[$i]}"
        local AWQ_PATH="$DATA_DIR/${AWQ_DIRS[$i]}"
        local FP16_DIR="${FP16_DIRS[$i]}"
        local FP16_PATH="${FP16_DIR:+$DATA_DIR/$FP16_DIR}"
        local NQ="${NEEDS_QUANT[$i]}"
        local MEM="${MEM_GB[$i]}"
        local HF="${HF_REPOS[$i]}"

        if echo "$SKIP_MODELS" | grep -qw "$KEY"; then
            printf "%-22s  %-12s  %-5s  %s\n" "$KEY" "SKIPPED" "${MEM}G" "SKIP_MODELS env var"
            JUDGE_JOB_IDS+=("")
            continue
        fi

        local STATUS
        if _dir_nonempty "$AWQ_PATH"; then
            STATUS="READY"
        elif [ "$NQ" -eq 1 ] && [ -n "$FP16_PATH" ] && _dir_nonempty "$FP16_PATH"; then
            STATUS="QUANT"
        elif [ "$NQ" -eq 1 ]; then
            STATUS="DL+QUANT"
        else
            STATUS="DL+JUDGE"
        fi

        local J_DOWNLOAD="" J_QUANT="" CHAIN_DESC=""

        if [[ "$STATUS" == DL* ]]; then
            local DL_SRC DL_DST
            if [ "$NQ" -eq 1 ]; then
                DL_SRC="$HF"; DL_DST="$FP16_PATH"
            else
                DL_SRC="$HF"; DL_DST="$AWQ_PATH"
            fi

            if [ "$NO_DOWNLOAD" -eq 1 ]; then
                printf "%-22s  %-12s  %-5s  %s\n" "$KEY" "NO-DL" "${MEM}G" \
                    "hf download $DL_SRC --local-dir $DL_DST"
                JUDGE_JOB_IDS+=("")
                continue
            fi

            if [ "$DRY_RUN" -eq 0 ]; then
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

        if [[ "$STATUS" == "QUANT" || "$STATUS" == "DL+QUANT" ]]; then
            local QUANT_DEP_OPTS=()
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

        local JUDGE_DEP_OPTS=()
        if [ -n "$J_QUANT" ]; then
            JUDGE_DEP_OPTS=(--dependency="afterok:${J_QUANT}")
        elif [ -n "$J_DOWNLOAD" ]; then
            JUDGE_DEP_OPTS=(--dependency="afterok:${J_DOWNLOAD}")
        fi

        if [ "$DRY_RUN" -eq 0 ]; then
            local J_JUDGE
            J_JUDGE=$(sbatch "${COMMON_OPTS[@]}" \
                --gpus=1 --mem="${MEM}G" \
                --cpus-per-task=8 \
                --time=12:00:00 \
                --job-name="judge_${KEY}" \
                --output="$LOGS_DIR/castor_judge_${KEY}_%j.out" \
                --error="$LOGS_DIR/castor_judge_${KEY}_%j.err" \
                "${JUDGE_DEP_OPTS[@]+"${JUDGE_DEP_OPTS[@]}"}" \
                "$SCRIPT_DIR/submit_judge_job.sh" "$KEY" "$RN")
            CHAIN_DESC="${CHAIN_DESC:+$CHAIN_DESC → }judge=$J_JUDGE"
            JUDGE_JOB_IDS+=("$J_JUDGE")
        else
            CHAIN_DESC="${CHAIN_DESC:+$CHAIN_DESC → }[DRY] judge"
            JUDGE_JOB_IDS+=("DRY_${KEY}")
        fi

        printf "%-22s  %-12s  %-5s  %s\n" "$KEY" "$STATUS" "${MEM}G" "$CHAIN_DESC"
    done

    echo ""

    local JUDGE_IDS_JOINED=""
    for JID in "${JUDGE_JOB_IDS[@]}"; do
        [[ -z "$JID" || "$JID" == DRY_* ]] && continue
        JUDGE_IDS_JOINED="${JUDGE_IDS_JOINED}:${JID}"
    done
    JUDGE_IDS_JOINED="${JUDGE_IDS_JOINED#:}"

    if [ -z "$JUDGE_IDS_JOINED" ]; then
        echo "━━━ No judge jobs submitted for $RN. ━━━"
        [ "$NO_DOWNLOAD" -eq 1 ] && echo "Run the download commands above, then re-run."
        return
    fi

    printf "%-22s  %-12s  %-5s  " "aggregation" "" ""
    if [ "$DRY_RUN" -eq 0 ]; then
        local J_AGG
        J_AGG=$(sbatch \
            -p pleiades \
            --cpus-per-task=4 --mem=8G \
            --time=1:00:00 \
            --job-name="judge_agg" \
            --output="$LOGS_DIR/castor_judge_agg_%j.out" \
            --error="$LOGS_DIR/castor_judge_agg_%j.err" \
            --dependency="afterok:${JUDGE_IDS_JOINED}" \
            --parsable \
            "$SCRIPT_DIR/aggregate_job.sh" "$RN")
        echo "job=$J_AGG (after ${JUDGE_IDS_JOINED})"
    else
        echo "[DRY] after judge jobs"
    fi

    echo " Output: $REPO/results/p5_judge/$RN/"
    echo "==========================================="
}

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

for RN in "${RUN_NAMES[@]}"; do
    submit_run "$RN"
done

echo ""
echo "Monitor : squeue -u $USER"
echo "Logs    : $LOGS_DIR/castor_judge_<KEY>_<JOBID>.out"
