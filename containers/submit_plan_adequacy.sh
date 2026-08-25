#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 9 (plan adequacy) — orchestration script.
#
# Chains both stages as two SLURM job arrays (one array task per run):
#   Stage 1 : tool-call extraction (guided JSON, vLLM, GPU) -- extract.py
#   Stage 2 : execute + aggregate + report (deterministic rule engine +
#             CSV rollup + narrative, CPU) -- run_executor.py / aggregate.py
#             / report.py, chained inside plan_adequacy_stage2_job.sh
# The second array depends on the first via --dependency=aftercorr, which is
# element-wise: array task N of Stage 2 starts as soon as array task N of
# Stage 1 succeeds, without waiting for the rest of the batch -- same pattern
# as submit_salvage.sh:133-137.
#
# Run from ~/BenchyBench/Eval_CASTOR/ on the head node (head1.condo.cs.cmu.edu):
#
#   Process every plan JSONL staged in pipelines/plan_adequacy/inbox/ (drop
#   full-answer JSONLs there -- one submit_plan_adequacy.sh call processes
#   all of them, as one batch job array rather than N separate submissions):
#     bash containers/submit_plan_adequacy.sh --model glm4_32b
#
#   Process just one run by name (must be one of the files in
#   pipelines/plan_adequacy/inbox/; submitted as a size-1 array):
#     bash containers/submit_plan_adequacy.sh --run answers_baseline --model glm4_32b
#
# Arguments:
#   --run NAME    (optional) Process only this run. Omit to discover and
#                 process every *.jsonl file sitting directly in
#                 pipelines/plan_adequacy/inbox/, each as its own run.
#   --model KEY   (required) Extraction model key -- glm4_32b, llama_3_3_70b,
#                 or phi4_14b, whichever cleared (or was accepted despite not
#                 clearing) calibration. No default -- see calibrate.py.
#
# Prerequisites:
#   bash containers/build_judge_container.sh --model salvage_embed
#   A calibration pass having run for --model (not strictly required to
#   execute, but running this before calibration has been reviewed defeats
#   the point of calibrating -- see the P9 end-to-end-pipeline plan).
#
# Both stages' outputs land in one subdirectory per run under
# Eval_CASTOR/results/p9_plan_adequacy/<run_name>/: tool_calls.jsonl,
# per_step.csv, per_image.csv, summary.csv, report.md, case_studies.md.
# eval_summary_adequacy.csv accumulates one row per run at the
# results/p9_plan_adequacy/ level for cross-arm comparison.
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/p9_stage1_<ARRAY_JOBID>_<TASK_INDEX>.out
#   tail -f /data/$USER/logs/p9_stage2_<ARRAY_JOBID>_<TASK_INDEX>.out
#   cat /data/$USER/logs/p9_manifest_<PID>.txt   # run name <-> task index mapping
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
INBOX_DIR="$REPO/pipelines/plan_adequacy/inbox"
mkdir -p "$LOGS_DIR"

RUN_NAME=""
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN_NAME="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required (glm4_32b, llama_3_3_70b, or phi4_14b -- no default)." >&2
    exit 1
fi

if [ -n "$RUN_NAME" ]; then
    RUN_NAMES=("$RUN_NAME")
else
    # Bash-native discovery (no python3 -- the login node doesn't have one),
    # same convention as submit_salvage.sh: every *.jsonl directly in
    # INBOX_DIR is its own run, no shard-detection heuristic.
    mapfile -t RUN_NAMES < <(
        find "$INBOX_DIR" -maxdepth 1 -name '*.jsonl' -printf '%f\n' 2>/dev/null \
            | sed 's/\.jsonl$//' \
            | sort
    )
    if [ "${#RUN_NAMES[@]}" -eq 0 ]; then
        echo "ERROR: no plan JSONL files found in $INBOX_DIR" >&2
        echo "       Drop the runs you want checked there first (e.g. answers_baseline.jsonl)." >&2
        exit 1
    fi
fi

N=${#RUN_NAMES[@]}
MANIFEST="$LOGS_DIR/p9_manifest_$$.txt"
printf '%s\n' "${RUN_NAMES[@]}" > "$MANIFEST"

echo "==========================================="
echo " Pipeline 9 — Plan Adequacy Checker"
echo " Runs      : ${RUN_NAMES[*]}"
echo " Model     : $MODEL"
echo " Manifest  : $MANIFEST"
echo "==========================================="

echo "[$(date)] Submitting Stage 1 array (tool-call extraction, $MODEL/vLLM, guided JSON), $N task(s) ..."
J_STAGE1=$(sbatch --parsable \
    -p pleiades \
    --array=0-$((N - 1)) \
    --output="$LOGS_DIR/p9_stage1_%A_%a.out" \
    --error="$LOGS_DIR/p9_stage1_%A_%a.err" \
    "$SCRIPT_DIR/plan_adequacy_stage1_job.sh" "$MANIFEST" "$MODEL")
echo "  Stage 1 array -> job $J_STAGE1 (tasks 0-$((N - 1)))"

echo "[$(date)] Submitting Stage 2 array (dependency: aftercorr:$J_STAGE1 -- element-wise, not whole-batch) ..."
J_STAGE2=$(sbatch --parsable \
    -p pleiades \
    --array=0-$((N - 1)) \
    --dependency="aftercorr:${J_STAGE1}" \
    --output="$LOGS_DIR/p9_stage2_%A_%a.out" \
    --error="$LOGS_DIR/p9_stage2_%A_%a.err" \
    "$SCRIPT_DIR/plan_adequacy_stage2_job.sh" "$MANIFEST")
echo "  Stage 2 array -> job $J_STAGE2 (tasks 0-$((N - 1)), each starts once its matching Stage 1 task succeeds)"

echo "==========================================="
echo " Batch submitted for: ${RUN_NAMES[*]}"
echo ""
echo " Monitor:"
echo "   squeue -u $USER"
echo "   tail -f $LOGS_DIR/p9_stage1_${J_STAGE1}_<TASK_INDEX>.out"
echo "   tail -f $LOGS_DIR/p9_stage2_${J_STAGE2}_<TASK_INDEX>.out"
echo "   cat $MANIFEST   # task index -> run name"
echo ""
echo " Output:"
echo "   Eval_CASTOR/results/p9_plan_adequacy/<run_name>/report.md when each finishes"
echo "   Eval_CASTOR/results/p9_plan_adequacy/eval_summary_adequacy.csv accumulates cross-run"
echo "==========================================="
