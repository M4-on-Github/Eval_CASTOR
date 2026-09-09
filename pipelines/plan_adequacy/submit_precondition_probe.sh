#!/bin/bash
# =============================================================================
# submit_precondition_probe.sh -- Phase 2 + Phase 3 of the revised Direction-4
# protocol (see reports/p9/precondition_position_phase1.txt for Phase 1,
# already run locally with no cluster time).
#
# Runs precondition_probe.py (same directory) twice: once over
# phase2_candidates.jsonl (isolated fact-check questions, no image) and once
# over phase3_items.jsonl (truncated-plan continuations, no image). Both are
# small (271 and 53 rows respectively as of the 2026-09-09 build) and
# text-only, so this needs a fraction of the GPU-time and none of the image
# I/O that the original 330-image generation job (run_all.sh, under
# plan_coherence/improved/) does -- a single GPU, ~1h, is generously sized.
#
# phase3_items.jsonl's prefix_plan_text is real, verified-clean plan text
# (truncated from inject.py's own test-verified base plans, not fabricated)
# for the 16 of 29 registry relationships that have a verified base to draw
# from -- see precondition_probe_prompts.py's realizable_relationships() for
# which 13 were excluded and why. Nothing further needs filling in before
# submitting.
#
# Usage (from Eval_CASTOR/pipelines/plan_adequacy/):
#   sbatch submit_precondition_probe.sh
#
# Lives under plan_adequacy/ (this is P9's own diagnostic work), but reuses
# castor_qwen.sif -- NOT P9's own castor_judge.sif container. P9's usual
# container has no Qwen3-VL stack (transformers>=4.51 + qwen-vl-utils); that
# only exists in castor_qwen.sif, the container plan_coherence/improved's
# run_inference.py uses to generate the corpus this probe is diagnosing.
# Both containers/weights live under /data/$USER/, so this cross-pipeline
# dependency is just a path, not a build -- no new container, no new
# model download.
# =============================================================================

#SBATCH --job-name=precondition_probe
#SBATCH --partition=pleiades
#SBATCH --constraint=RTX6000ADA
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/data/%u/logs/precondition_probe_%j.out
#SBATCH --error=/data/%u/logs/precondition_probe_%j.err

set -euo pipefail
mkdir -p "/data/$USER/logs"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [[ -z "${BENCHYBENCH_ROOT:-}" ]]; then
    # plan_adequacy/ is 3 levels below the BenchyBench root
    # (root/Eval_CASTOR/pipelines/plan_adequacy) -- one shallower than
    # plan_coherence/improved/, which needs 4 ("../../../..").
    BENCHYBENCH_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
export BENCHYBENCH_ROOT

DATA_DIR="/data/${USER}"
VLM_DIR="${DATA_DIR}/qwen3vl-8b"
SIF="${DATA_DIR}/castor_qwen.sif"
if [[ ! -f "$SIF" ]]; then
    echo "ERROR: $SIF not found -- run QWEN-Maritime/CASTOR/submit.sh once first" \
         "(it builds this container) or build_container.sh directly." >&2
    exit 1
fi
if [[ ! -d "$VLM_DIR" ]]; then
    echo "ERROR: $VLM_DIR not found." >&2
    exit 1
fi

REPORTS_DIR="${BENCHYBENCH_ROOT}/reports/p9"
PHASE2_IN="${REPORTS_DIR}/phase2_candidates.jsonl"
PHASE3_IN="${REPORTS_DIR}/phase3_items.jsonl"
if [[ ! -s "$PHASE2_IN" ]]; then
    echo "ERROR: $PHASE2_IN missing -- run precondition_probe_prompts.py first." >&2
    exit 1
fi
if [[ ! -s "$PHASE3_IN" ]]; then
    echo "ERROR: $PHASE3_IN missing -- run precondition_probe_prompts.py first." >&2
    exit 1
fi

HF_HOME="${DATA_DIR}/.cache/huggingface"
TORCH_HOME="${DATA_DIR}/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

APT_OPTS="--containall --nv \
    --pwd ${SCRIPT_DIR} \
    --home ${HOME} \
    --bind /tmp:/tmp \
    --bind ${BENCHYBENCH_ROOT}:${BENCHYBENCH_ROOT} \
    --bind ${DATA_DIR}:${DATA_DIR} \
    --env USER=${USER} \
    --env PYTHONUNBUFFERED=1 \
    --env HF_HOME=${HF_HOME} \
    --env TRANSFORMERS_CACHE=${HF_HOME} \
    --env TORCH_HOME=${TORCH_HOME} \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1"

RUN="apptainer exec ${APT_OPTS} ${SIF} python3"
PROBE="${SCRIPT_DIR}/precondition_probe.py"

echo "============================================================"
echo "Phase 2/3 precondition probe -- $(date)"
echo "Job ID   : ${SLURM_JOB_ID:-local}"
echo "Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo "VLM dir  : $VLM_DIR"
echo "============================================================"

echo ""
echo ">>> Phase 2: isolated fact-check questions ($(wc -l < "$PHASE2_IN") rows)"
$RUN "$PROBE" \
    --vlm-dir "$VLM_DIR" \
    --input   "$PHASE2_IN" \
    --output  "${REPORTS_DIR}/phase2_responses.jsonl" \
    --field-prompt question \
    --max-new-tokens 200

echo ""
echo ">>> Phase 3: truncated-plan continuations ($(wc -l < "$PHASE3_IN") rows)"
$RUN "$PROBE" \
    --vlm-dir "$VLM_DIR" \
    --input   "$PHASE3_IN" \
    --output  "${REPORTS_DIR}/phase3_responses.jsonl" \
    --field-prompt continuation_prompt \
    --field-context prefix_plan_text \
    --max-new-tokens 400

echo ""
echo "Done -- $(date)"
echo "  ${REPORTS_DIR}/phase2_responses.jsonl"
echo "  ${REPORTS_DIR}/phase3_responses.jsonl"
