#!/bin/bash
# =============================================================================
# run_all.sh — CASTOR improved pipeline: inference → judge → aggregate
# Usage:  sbatch run_all.sh
#
# Stage 1 (inference) runs inside castor_qwen.sif (transformers>=4.51 + qwen-vl-utils).
# Stages 2-3 (judge, aggregate) run inside castor_judge.sif (vLLM 0.8.5 + pandas/scipy).
# =============================================================================

#SBATCH --job-name=castor_improved
#SBATCH --partition=pleiades
#SBATCH --nodelist=pleiades-0-17,pleiades-0-23
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=150G
#SBATCH --output=/data/%u/logs/castor_improved_%j.out
#SBATCH --error=/data/%u/logs/castor_improved_%j.err

set -euo pipefail

mkdir -p "/data/$USER/logs"

# ---------------------------------------------------------------------------
# 0. Setup — locate config.yaml relative to where sbatch was called
# ---------------------------------------------------------------------------
# SLURM copies this script to /var/spool/slurmd/jobXXX/ before running it,
# so BASH_SOURCE[0] resolves to that staging path, not the actual script location.
# SLURM_SUBMIT_DIR is always set to the directory where sbatch was invoked — use that.
# Submit from improved/:   cd ~/Eval_CASTOR/pipelines/plan_coherence/improved && sbatch run_all.sh
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CONFIG="${SCRIPT_DIR}/config.yaml"

# ---------------------------------------------------------------------------
# Resolve the BenchyBench root, which config.yaml refers to as
# ${BENCHYBENCH_ROOT}. This pipeline sits four levels below it:
#   <root>/Eval_CASTOR/pipelines/plan_coherence/improved
#
# The root is confirmed by probing for the image set rather than assumed, so a
# wrong guess fails here with a clear message instead of surfacing as a missing
# file deep inside a GPU job. config.yaml previously hardcoded
# /home/$USER/ONLY/CASTOR/shipwreck_wiki_images, which exists in no current
# layout — the images live once at the BenchyBench root.
# ---------------------------------------------------------------------------
if [[ -z "${BENCHYBENCH_ROOT:-}" ]]; then
    BENCHYBENCH_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
export BENCHYBENCH_ROOT

if [[ ! -d "${BENCHYBENCH_ROOT}/shipwreck_wiki_images/sorted_images" ]]; then
    echo "ERROR: BENCHYBENCH_ROOT does not contain the image set." >&2
    echo "       BENCHYBENCH_ROOT = ${BENCHYBENCH_ROOT}" >&2
    echo "       expected         = ${BENCHYBENCH_ROOT}/shipwreck_wiki_images/sorted_images" >&2
    echo "       Set BENCHYBENCH_ROOT to the directory containing" >&2
    echo "       shipwreck_wiki_images/ and re-submit." >&2
    exit 1
fi
echo "[$(date)] BenchyBench root: ${BENCHYBENCH_ROOT}"

# os.path.expandvars() expands ${USER} / ${HOME} / ${BENCHYBENCH_ROOT} at runtime.
_pypath() {
    python3 -c "import yaml, os; c=yaml.safe_load(open('${CONFIG}')); print(os.path.expandvars(c${1}))"
}

PIPELINE_DIR=$(_pypath "['paths']['pipeline_dir']")
CONTAINER_INFERENCE=$(_pypath "['paths']['container_inference']")
CONTAINER_JUDGE=$(_pypath "['paths']['container_judge']")

echo "============================================================"
echo "CASTOR improved pipeline — $(date)"
echo "Config:              $CONFIG"
echo "Pipeline dir:        $PIPELINE_DIR"
echo "Container inference: $CONTAINER_INFERENCE"
echo "Container judge:     $CONTAINER_JUDGE"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Node:                ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:                ${CUDA_VISIBLE_DEVICES:-all}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Apptainer wrappers
#   - Inference uses castor_qwen.sif (Qwen3-VL + qwen-vl-utils)
#   - Judge/aggregate uses castor_judge.sif (vLLM)
#
#   --containall  fully isolates the container so its %environment PATH is used
#   --home $HOME  mounts the real home dir (required; --env HOME= is not permitted)
#   PYTHON        explicit conda python — not on host PATH inside --containall
# ---------------------------------------------------------------------------
DATA_DIR="/data/${USER}"
HF_HOME="${DATA_DIR}/.cache/huggingface"
TORCH_HOME="${DATA_DIR}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

PYTHON=python3

_APT_OPTS="--containall --nv \
    --pwd ${PIPELINE_DIR} \
    --home ${HOME} \
    --bind /tmp:/tmp \
    --bind ${PIPELINE_DIR}:${PIPELINE_DIR} \
    --bind ${DATA_DIR}:${DATA_DIR} \
    --bind ${HOME}:${HOME} \
    --env USER=${USER} \
    --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=${HF_HOME} \
    --env TRANSFORMERS_CACHE=${HF_HOME} \
    --env TORCH_HOME=${TORCH_HOME} \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1"

RUN_INFERENCE="apptainer exec ${_APT_OPTS} ${CONTAINER_INFERENCE}"
RUN_JUDGE="apptainer exec ${_APT_OPTS} ${CONTAINER_JUDGE}"

# ---------------------------------------------------------------------------
# 1. Inference  (Qwen3-VL 8B, all 3 conditions, ~110 images each)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Stage 1: Inference"
echo "    $(date)"

$RUN_INFERENCE $PYTHON "${PIPELINE_DIR}/inference/run_inference.py" \
    --config "$CONFIG" \
    2>&1 | tee "/data/$USER/logs/inference_${SLURM_JOB_ID:-0}.log"

echo "    Inference done — $(date)"

for COND in standard_v2 control_v2 ablation_v2; do
    JFILE="${PIPELINE_DIR}/results/answers_qwen3vl8b_baseline_${COND}_improved.jsonl"
    if [[ ! -s "$JFILE" ]]; then
        echo "[ERROR] Missing or empty: $JFILE"
        exit 1
    fi
    echo "    ${COND}: $(wc -l < "$JFILE") records"
done

# ---------------------------------------------------------------------------
# 1.5. Assertion coverage  (Selene 8B, STANDARD + CONTROL ref sets, dual-track)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Stage 1.5: Assertion coverage"
echo "    $(date)"

$RUN_JUDGE $PYTHON "${PIPELINE_DIR}/eval/check_assertion_coverage.py" \
    --config "$CONFIG" \
    2>&1 | tee "/data/$USER/logs/coverage_${SLURM_JOB_ID:-0}.log"

echo "    Coverage done — $(date)"

COVERAGE_FILE="${PIPELINE_DIR}/results/coverage_per_image.csv"
if [[ ! -s "$COVERAGE_FILE" ]]; then
    echo "[ERROR] Missing or empty: $COVERAGE_FILE"
    exit 1
fi
echo "    Coverage rows: $(wc -l < "$COVERAGE_FILE")"

# ---------------------------------------------------------------------------
# 2. Judge  (Llama-3.3-70B, SEQ + MTH + SPC, dual-track)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Stage 2: Judge evaluation"
echo "    $(date)"

$RUN_JUDGE $PYTHON "${PIPELINE_DIR}/eval/run_judge_v2.py" \
    --config "$CONFIG" \
    2>&1 | tee "/data/$USER/logs/judge_${SLURM_JOB_ID:-0}.log"

echo "    Judge done — $(date)"

SCORES_FILE="${PIPELINE_DIR}/results/judge_scores_improved.jsonl"
if [[ ! -s "$SCORES_FILE" ]]; then
    echo "[ERROR] Missing or empty: $SCORES_FILE"
    exit 1
fi
echo "    Scored rows: $(wc -l < "$SCORES_FILE")"

# ---------------------------------------------------------------------------
# 3. Aggregate  (summaries + case studies + report)
# ---------------------------------------------------------------------------
echo ""
echo ">>> Stage 3: Aggregation"
echo "    $(date)"

$RUN_JUDGE $PYTHON "${PIPELINE_DIR}/eval/aggregate.py" \
    --config "$CONFIG" \
    2>&1 | tee "/data/$USER/logs/aggregate_${SLURM_JOB_ID:-0}.log"

echo "    Aggregation done — $(date)"

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "CASTOR improved pipeline COMPLETE — $(date)"
echo ""
echo "Output files:"
RESULTS="${PIPELINE_DIR}/results"
for F in \
    "${RESULTS}/answers_qwen3vl8b_baseline_standard_v2_improved.jsonl" \
    "${RESULTS}/answers_qwen3vl8b_baseline_control_v2_improved.jsonl" \
    "${RESULTS}/answers_qwen3vl8b_baseline_ablation_v2_improved.jsonl" \
    "${RESULTS}/judge_scores_improved.jsonl" \
    "${RESULTS}/coverage_per_image.csv" \
    "${RESULTS}/coverage_summary.csv" \
    "${RESULTS}/summary_by_condition.csv" \
    "${RESULTS}/summary_by_condition_state.csv" \
    "${RESULTS}/case_studies.md" \
    "${RESULTS}/report.md"; do
    if [[ -f "$F" ]]; then
        echo "  + $F"
    else
        echo "  MISSING: $F"
    fi
done
echo "============================================================"
