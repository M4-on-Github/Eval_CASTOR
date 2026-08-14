#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 6 (salvage plan analysis) — orchestration script.
#
# Chains all four stages as two SLURM job arrays (one array task per run):
#   Stage 1     : element extraction via qwen25_72b (vLLM, GPU) -- there is
#                 no Ollama on the cluster.
#   Stage 2+3+4 : clustering (local sentence-transformers embedding, CPU) +
#                 contingency table + stats + report, all inside
#                 castor_judge.sif (already has pandas/scipy/scikit-learn).
# The second array depends on the first via --dependency=aftercorr, which is
# element-wise: array task N of Stage 2+3+4 starts as soon as array task N of
# Stage 1 succeeds, without waiting for the rest of the batch.
#
# Run from ~/BenchyBench/Eval_CASTOR/ on the head node (head1.condo.cs.cmu.edu):
#
#   Process every run staged in p6_plans_to_judge/ (drop full-answer JSONLs
#   there -- one submit_salvage.sh call judges all of them, as one batch job
#   array rather than N separate job submissions):
#     bash containers/submit_salvage.sh --threshold 0.15 --min-generic-pct 0.5
#
#   Process just one run by name (still must be one of the files in
#   p6_plans_to_judge/; submitted as a size-1 array):
#     bash containers/submit_salvage.sh --run answers_baseline --threshold 0.15 --min-generic-pct 0.5
#
# Arguments:
#   --run NAME             (optional) Process only this run. Omit to discover
#                          and process every *.jsonl file sitting directly in
#                          p6_plans_to_judge/, each as its own separate run --
#                          no shard-detection heuristic.
#   --threshold N          (required) Stage 2 clustering distance threshold,
#                          shared across every run processed by this
#                          invocation. No default on purpose -- inspect
#                          elements.json per run before trusting downstream
#                          stats; re-run Stage 2+3+4 alone with a different
#                          value if categories look wrong.
#   --min-generic-pct N    (required) Minimum overall prevalence (0-1) for an
#                          element to be flagged as a generic/boilerplate
#                          template in generic_elements.csv -- elements used
#                          this often overall AND never significant for any
#                          single state. No default on purpose, same reason
#                          as --threshold.
#
# Prerequisites:
#   bash containers/build_judge_container.sh --model qwen25_72b
#   bash containers/build_judge_container.sh --model salvage_embed
#
# All four stages' outputs land in one subdirectory per run under
# Eval_CASTOR/results/p6_salvage_plan/<run_name>/: raw_elements.jsonl,
# elements.json, contingency.csv, tests.csv, omnibus.csv, dunn.csv,
# generic_elements.csv, report.txt (plus combined_input.jsonl if
# separated-into-parts shards were auto-combined).
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/salvage_stage1_<ARRAY_JOBID>_<TASK_INDEX>.out
#   tail -f /data/$USER/logs/salvage_stage234_<ARRAY_JOBID>_<TASK_INDEX>.out
#   cat /data/$USER/logs/salvage_manifest_<PID>.txt   # run name <-> task index mapping
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="/data/$USER"
LOGS_DIR="$DATA_DIR/logs"
PLANS_DIR="$REPO/p6_plans_to_judge"
mkdir -p "$LOGS_DIR"

RUN_NAME=""
THRESHOLD=""
MIN_GENERIC_PCT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run) RUN_NAME="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --min-generic-pct) MIN_GENERIC_PCT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$THRESHOLD" ]; then
    echo "ERROR: --threshold is required (Stage 2 clustering threshold, no default)." >&2
    exit 1
fi

if [ -z "$MIN_GENERIC_PCT" ]; then
    echo "ERROR: --min-generic-pct is required (generic-element prevalence cutoff, no default)." >&2
    exit 1
fi

if [ -n "$RUN_NAME" ]; then
    RUN_NAMES=("$RUN_NAME")
else
    # Bash-native discovery (no python3 -- the login node doesn't have one).
    # Mirrors pipelines/salvage_analysis/combine_shards.py::discover_run_names:
    # every *.jsonl directly in PLANS_DIR is its own run, no shard-detection
    # heuristic -- two files can coincidentally match a "looks like a shard"
    # naming pattern while having unrelated job IDs and no real sibling
    # shards to combine with, so guessing was wrong more often than it helped.
    mapfile -t RUN_NAMES < <(
        find "$PLANS_DIR" -maxdepth 1 -name '*.jsonl' -printf '%f\n' 2>/dev/null \
            | sed 's/\.jsonl$//' \
            | sort
    )
    if [ "${#RUN_NAMES[@]}" -eq 0 ]; then
        echo "ERROR: no full-answer JSONL files found in $PLANS_DIR" >&2
        echo "       Drop the runs you want judged there first (e.g. answers_baseline.jsonl)." >&2
        exit 1
    fi
fi

N=${#RUN_NAMES[@]}
MANIFEST="$LOGS_DIR/salvage_manifest_$$.txt"
printf '%s\n' "${RUN_NAMES[@]}" > "$MANIFEST"

echo "==========================================="
echo " Pipeline 6 — Salvage Plan Templating Analysis"
echo " Runs             : ${RUN_NAMES[*]}"
echo " Threshold        : $THRESHOLD"
echo " Min generic pct  : $MIN_GENERIC_PCT"
echo " Manifest         : $MANIFEST"
echo "==========================================="

echo "[$(date)] Submitting Stage 1 array (element extraction, qwen25_72b/vLLM, guided JSON decoding), $N task(s) ..."
J_STAGE1=$(sbatch --parsable \
    -p pleiades \
    --array=0-$((N - 1)) \
    --output="$LOGS_DIR/salvage_stage1_%A_%a.out" \
    --error="$LOGS_DIR/salvage_stage1_%A_%a.err" \
    "$SCRIPT_DIR/salvage_stage1_job.sh" "$MANIFEST")
echo "  Stage 1 array -> job $J_STAGE1 (tasks 0-$((N - 1)))"

echo "[$(date)] Submitting Stage 2+3+4 array (dependency: aftercorr:$J_STAGE1 -- element-wise, not whole-batch) ..."
J_STAGE234=$(sbatch --parsable \
    -p pleiades \
    --array=0-$((N - 1)) \
    --dependency="aftercorr:${J_STAGE1}" \
    --output="$LOGS_DIR/salvage_stage234_%A_%a.out" \
    --error="$LOGS_DIR/salvage_stage234_%A_%a.err" \
    "$SCRIPT_DIR/salvage_stage234_job.sh" "$MANIFEST" "$THRESHOLD" "$MIN_GENERIC_PCT")
echo "  Stage 2+3+4 array -> job $J_STAGE234 (tasks 0-$((N - 1)), each starts once its matching Stage 1 task succeeds)"

echo "==========================================="
echo " Batch submitted for: ${RUN_NAMES[*]}"
echo ""
echo " Monitor:"
echo "   squeue -u $USER"
echo "   tail -f $LOGS_DIR/salvage_stage1_${J_STAGE1}_<TASK_INDEX>.out"
echo "   tail -f $LOGS_DIR/salvage_stage234_${J_STAGE234}_<TASK_INDEX>.out"
echo "   cat $MANIFEST   # task index -> run name"
echo ""
echo " Output:"
echo "   Eval_CASTOR/results/p6_salvage_plan/<run_name>/report.txt when each finishes"
echo "==========================================="
