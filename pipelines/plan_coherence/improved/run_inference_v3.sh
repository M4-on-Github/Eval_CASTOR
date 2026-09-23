#!/bin/bash
# =============================================================================
# run_inference_v3.sh — inference only, procedural_v3 arm
# Usage:  cd ~/BenchyBench/Eval_CASTOR/pipelines/plan_coherence/improved && sbatch run_inference_v3.sh
#
# Stage 1 of run_all.sh, and nothing else: Qwen3-VL 8B over all 110 images with
# prompts/prompt_procedural_v3.txt, driven by config_procedural_v3.yaml. The
# coverage / judge / aggregate stages are P8's and are not run; the output goes
# to P9 instead (pipelines/plan_adequacy/inbox/ + submit_plan_adequacy.sh).
#
# Resumes like run_all.sh: images already in the output JSONL are skipped.
# =============================================================================

#SBATCH --job-name=castor_v3_infer
#SBATCH --partition=pleiades
#SBATCH --constraint=RTX6000ADA
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/data/%u/logs/castor_v3_infer_%j.out
#SBATCH --error=/data/%u/logs/castor_v3_infer_%j.err

set -euo pipefail

mkdir -p "/data/$USER/logs"

# SLURM stages this script under /var/spool, so locate the config from the
# submit directory, as run_all.sh does.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CONFIG="${SCRIPT_DIR}/config_procedural_v3.yaml"

if [[ -z "${BENCHYBENCH_ROOT:-}" ]]; then
    BENCHYBENCH_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
export BENCHYBENCH_ROOT

if [[ ! -d "${BENCHYBENCH_ROOT}/shipwreck_wiki_images/sorted_images" ]]; then
    echo "ERROR: BENCHYBENCH_ROOT does not contain the image set: ${BENCHYBENCH_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/prompts/prompt_procedural_v3.txt" ]]; then
    echo "ERROR: prompts/prompt_procedural_v3.txt not found under ${SCRIPT_DIR}" >&2
    exit 1
fi

_pypath() {
    python3 -c "import yaml, os; c=yaml.safe_load(open('${CONFIG}')); print(os.path.expandvars(c${1}))"
}

PIPELINE_DIR=$(_pypath "['paths']['pipeline_dir']")
CONTAINER_INFERENCE=$(_pypath "['paths']['container_inference']")

echo "============================================================"
echo "CASTOR procedural_v3 inference — $(date)"
echo "Config:              $CONFIG"
echo "Pipeline dir:        $PIPELINE_DIR"
echo "Container inference: $CONTAINER_INFERENCE"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "Node:                ${SLURMD_NODENAME:-$(hostname)}"
echo "============================================================"

# --containall drops the host environment, so BENCHYBENCH_ROOT must be passed
# in explicitly: run_inference.py expands it in config paths inside the
# container. Without it, paths stay as the literal "${BENCHYBENCH_ROOT}/...".
DATA_DIR="/data/${USER}"
HF_HOME="${DATA_DIR}/.cache/huggingface"
TORCH_HOME="${DATA_DIR}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

apptainer exec --containall --nv \
    --pwd "${PIPELINE_DIR}" \
    --home "${HOME}" \
    --bind /tmp:/tmp \
    --bind "${PIPELINE_DIR}:${PIPELINE_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${HOME}:${HOME}" \
    --env USER="${USER}" \
    --env BENCHYBENCH_ROOT="${BENCHYBENCH_ROOT}" \
    --env PYTHONUNBUFFERED=1 \
    --env HF_HOME="${HF_HOME}" \
    --env TRANSFORMERS_CACHE="${HF_HOME}" \
    --env TORCH_HOME="${TORCH_HOME}" \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "${CONTAINER_INFERENCE}" \
    python3 "${PIPELINE_DIR}/inference/run_inference.py" --config "$CONFIG" \
    2>&1 | tee "/data/$USER/logs/inference_v3_${SLURM_JOB_ID:-0}.log"

JFILE="${PIPELINE_DIR}/results/answers_qwen3vl8b_baseline_procedural_v3_improved.jsonl"
if [[ ! -s "$JFILE" ]]; then
    echo "[ERROR] Missing or empty: $JFILE"
    exit 1
fi
echo "procedural_v3: $(wc -l < "$JFILE") records -> $JFILE"
echo "Next: cp it into Eval_CASTOR/pipelines/plan_adequacy/inbox/ and run"
echo "  bash containers/submit_plan_adequacy.sh --run answers_qwen3vl8b_baseline_procedural_v3_improved --model glm4_32b"
