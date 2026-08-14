#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 7 (assertion coverage) — orchestration script.
#
# Submits one SLURM job array (one task per run) that runs check_assertions.py
# for each inference JSONL staged in p7_to_check/ using Selene 8B AWQ.
# Resume-safe: tasks skip already-processed images inside check_assertions.py.
#
# Drop the full-answer JSONL(s) you want scored into p7_to_check/ first
# (e.g. p7_to_check/answers_baseline.jsonl), then:
#
# Run from ~/BenchyBench/Eval_CASTOR/ on the head node (head1.condo.cs.cmu.edu):
#
#   Process every run staged in p7_to_check/:
#     bash containers/submit_assertion_coverage.sh
#
#   Process just one specific run (must be in p7_to_check/):
#     bash containers/submit_assertion_coverage.sh --run answers_baseline
#
# Arguments:
#   --run NAME    (optional) Process only this run name (no .jsonl extension).
#                 Omit to discover and process every *.jsonl in p7_to_check/.
#
# Prerequisites:
#   The castor_judge.sif container must already be built:
#     bash containers/build_judge_container.sh --model salvage_embed
#   Selene weights must be downloaded (handled by judge_panel_submit.sh or manually).
#
# Outputs (one subdirectory per run, gitignored):
#   results/p7_assertion_coverage/<run_name>/<run_name>_per_image.csv
#   results/p7_assertion_coverage/<run_name>/<run_name>_per_assertion.csv
#   results/p7_assertion_coverage/<run_name>/<run_name>_summary.csv
#   results/p7_assertion_coverage/eval_summary_assertion.csv  (appended per run)
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/p7_coverage_<ARRAY_JOBID>_<TASK_INDEX>.out
#   cat /data/$USER/logs/p7_manifest_<PID>.txt   # task index -> run name
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
TO_CHECK_DIR="$REPO/p7_to_check"
mkdir -p "$LOGS_DIR"

RUN_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
SIF="$DATA_DIR/castor_judge.sif"
if [ ! -f "$SIF" ]; then
    echo "ERROR: container not found: $SIF" >&2
    echo "       Run: bash containers/build_judge_container.sh --model salvage_embed" >&2
    exit 1
fi

if [ ! -d "$TO_CHECK_DIR" ]; then
    echo "ERROR: staging directory not found: $TO_CHECK_DIR" >&2
    echo "       Create it and drop the JONLs you want scored there." >&2
    exit 1
fi

# ── Discover runs ─────────────────────────────────────────────────────────────
if [ -n "$RUN_NAME" ]; then
    if [ ! -f "$TO_CHECK_DIR/${RUN_NAME}.jsonl" ]; then
        echo "ERROR: $TO_CHECK_DIR/${RUN_NAME}.jsonl not found" >&2
        exit 1
    fi
    RUN_NAMES=("$RUN_NAME")
else
    mapfile -t RUN_NAMES < <(
        find "$TO_CHECK_DIR" -maxdepth 1 -name '*.jsonl' -printf '%f\n' 2>/dev/null \
            | sed 's/\.jsonl$//' \
            | sort
    )
    if [ "${#RUN_NAMES[@]}" -eq 0 ]; then
        echo "ERROR: no *.jsonl files found in $TO_CHECK_DIR" >&2
        echo "       Drop full-answer run JSONLs there first." >&2
        exit 1
    fi
fi

N=${#RUN_NAMES[@]}
MANIFEST="$LOGS_DIR/p7_manifest_$$.txt"
printf '%s\n' "${RUN_NAMES[@]}" > "$MANIFEST"

echo "==========================================="
echo " Pipeline 7 — Assertion Coverage Analysis"
echo " Runs     : ${RUN_NAMES[*]}"
echo " N tasks  : $N"
echo " Manifest : $MANIFEST"
echo " SIF      : $SIF"
echo "==========================================="

echo "[$(date)] Submitting array of $N task(s) ..."
ARRAY_JOB=$(sbatch --parsable \
    -p pleiades \
    --array=0-$((N - 1)) \
    --output="$LOGS_DIR/p7_coverage_%A_%a.out" \
    --error="$LOGS_DIR/p7_coverage_%A_%a.err" \
    "$SCRIPT_DIR/assertion_coverage_job.sh" "$MANIFEST")

echo "  Job array -> $ARRAY_JOB (tasks 0-$((N - 1)))"
echo ""
echo "==========================================="
echo " Submitted: ${RUN_NAMES[*]}"
echo ""
echo " Monitor:"
echo "   squeue -u $USER"
echo "   tail -f $LOGS_DIR/p7_coverage_${ARRAY_JOB}_0.out"
echo "   cat $MANIFEST   # task index -> run name"
echo ""
echo " Outputs (when done):"
echo "   $REPO/results/p7_assertion_coverage/<run_name>/*"
echo "   $REPO/results/p7_assertion_coverage/eval_summary_assertion.csv"
echo "==========================================="
