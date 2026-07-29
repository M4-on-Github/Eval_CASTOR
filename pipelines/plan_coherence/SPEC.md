# P8 — Plan Coherence Analysis: Spec

## Objective

Evaluate whether each VLM salvage plan is logically coherent and correctly
sequenced, using a 5-judge LLM panel with no image and only the GT disaster
state as an anchor. Unlike P5 (factual accuracy vs GT) and P7 (domain concept
coverage), P8 asks: does the plan make operational sense step-by-step?

---

## Design Decisions (locked)

| Decision | Choice |
|----------|--------|
| Step extraction | Regex on numbered list — no LLM needed |
| Judge input | Plan steps + GT state label only (no image, no full GT) |
| Output per step per judge | Binary `valid` (bool) + `reason` (1 sentence) |
| Raw panel signal | `n_invalid` (0–5) — always stored |
| Default threshold | `majority_invalid = n_invalid >= 3` (3/5) |
| Recomputation | `n_invalid` column allows any threshold offline |
| Step context | All prior steps included in each prompt (full chain) |
| Termination | None — all steps scored regardless of earlier failures |

---

## Judges (5 models)

| Key | HF repo | Local dir | VRAM (AWQ) | Architecture |
|-----|---------|-----------|-----------|-------------|
| `deepseek_r1_32b` | `casperhansen/deepseek-r1-distill-qwen-32b-awq` | `deepseek-r1-distill-qwen-32b-awq` | ~20 GB | `Qwen2ForCausalLM` |
| `glm4_32b` | `mratsim/GLM-4-32B-0414.w4a16-gptq` | `glm-4-32b-0414-gptq` | ~20 GB | `ChatGLMModel` |
| `llama_3_3_70b` | `hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` | `llama-3.3-70b-instruct-awq` | ~38 GB | `LlamaForCausalLM` |
| `phi4_14b` | `RedHatAI/phi-4-quantized.w4a16` | `phi-4-w4a16` | ~8 GB | `Phi3ForCausalLM` |
| `gemma3_27b` | `RedHatAI/gemma-3-27b-it-quantized.w4a16` | `gemma-3-27b-it-w4a16` | ~14 GB | `Gemma3ForCausalLM` |

All architectures confirmed in vLLM 0.8.5 supported models list.
DeepSeek and GLM weights already on cluster. Llama/Phi/Gemma need downloading.

> **Gap 1**: Confirm exact HF repo for Llama-3.3-70B AWQ — `hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` is the most likely but needs verification before `download_job.sh` is written.

---

## Step Parsing

Plans are already numbered lists. Regex split on `^\d+\.\s+`:

```python
import re
STEP_RE = re.compile(r'(?m)(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.|\Z)', re.DOTALL)
```

Each step: strip `**bold**` markers, collapse internal newlines to spaces.
Edge cases:
- 0 steps extracted → skip image, log warning
- 1 step → still valid, no "prior steps" context
- Steps with sub-bullets → treat whole block as one step

> **Gap 2**: Validate regex against all 4 prompt variants (IMPROVED, ABLATION, CONTROL, VISUAL_GROUNDED) to confirm no variant uses a non-numeric list format.

---

## Prompt Design

```
System:
  You are a maritime salvage operations expert. Evaluate whether a specific
  step in a salvage plan is operationally valid and correctly sequenced.
  Answer only in the JSON format specified.

User:
  CASUALTY TYPE: {gt_state}   (e.g. "capsized")

  STEPS COMPLETED SO FAR:
  1. {step_1_text}
  2. {step_2_text}
  ...

  STEP TO EVALUATE:
  {k}. {step_k_text}

  Is this step (a) operationally valid for maritime salvage of a {gt_state}
  vessel, and (b) correctly sequenced given the prior steps?

  Respond: {"valid": true, "reason": "one sentence"} or
           {"valid": false, "reason": "one sentence"}
```

Guided JSON schema:
```python
{
  "type": "object",
  "properties": {
    "valid":  {"type": "boolean"},
    "reason": {"type": "string", "maxLength": 300},
  },
  "required": ["valid", "reason"],
  "additionalProperties": False,
}
```

> **Gap 3**: Token budget — worst case is step 7 of 7 with all prior steps in
> context. Estimate: system (~150 tokens) + prior steps (~900 tokens) + step to
> evaluate (~150 tokens) = ~1200 tokens. Fits within 2048 for all models.
> Needs smoke-test confirmation on cluster before full run.

---

## vLLM Model Configs

```python
_MODEL_CONFIG = {
    "deepseek_r1_32b": {
        "dir": "deepseek-r1-distill-qwen-32b-awq",
        "max_model_len": 4096, "max_tokens": 64, "guided_json": True,
    },
    "glm4_32b": {
        "dir": "glm-4-32b-0414-gptq",
        "max_model_len": 4096, "max_tokens": 64, "guided_json": True,
    },
    "llama_3_3_70b": {
        "dir": "llama-3.3-70b-instruct-awq",
        "max_model_len": 4096, "max_tokens": 64, "guided_json": True,
    },
    "phi4_14b": {
        "dir": "phi-4-w4a16",
        "max_model_len": 4096, "max_tokens": 64, "guided_json": True,
    },
    "gemma3_27b": {
        "dir": "gemma-3-27b-it-w4a16",
        "max_model_len": 4096, "max_tokens": 64, "guided_json": True,
    },
}
```

> **Gap 4**: `max_model_len` for Llama-3.3-70B AWQ on 48GB is tight. At 4096
> the model uses ~41 GB (weights + KV cache). May need to reduce to 2048 if OOM.
> Requires smoke test on cluster.

---

## Pipeline Structure

```
Eval_CASTOR/
  pipelines/plan_coherence/
    SPEC.md                      ← this file
    parse_steps.py               ← regex step extractor (no LLM)
    run_coherence_judge.py       ← per-model vLLM batch judge
    aggregate_coherence.py       ← merge 5 judge outputs → consensus
  containers/
    submit_coherence.sh          ← P8 orchestrator (single entry point)
    coherence_judge_job.sh       ← SLURM job: one model, one run
    coherence_aggregate_job.sh   ← SLURM job: CPU-only aggregation
  p8_to_check/                   ← staging folder (drop JONLs here)
    .gitkeep
  results/p8_plan_coherence/     ← gitignored outputs
    <run>/
      <run>_per_step.csv
      <run>_per_image.csv
      <run>_summary.csv
    eval_summary_coherence.csv   ← cumulative across runs
```

---

## Output Schemas

### `_per_step.csv` (one row per image per step)
```
image, gt_state, step_num, step_text,
n_invalid,              # int 0-5: how many judges said invalid
majority_invalid,       # bool: n_invalid >= 3
deepseek_r1_32b_valid,  deepseek_r1_32b_reason,
glm4_32b_valid,         glm4_32b_reason,
llama_3_3_70b_valid,    llama_3_3_70b_reason,
phi4_14b_valid,         phi4_14b_reason,
gemma3_27b_valid,       gemma3_27b_reason
```
Values: `"1"` / `"0"` / `"error"` for _valid columns; `""` if judge not yet run.

### `_per_image.csv` (one row per image)
```
image, gt_state, n_steps,
n_majority_invalid,             # steps where n_invalid >= 3
coherence_pct,                  # (n_steps - n_majority_invalid) / n_steps
first_majority_invalid_step,    # step_num of first failure, or "" if none
```

### `_summary.csv` + `eval_summary_coherence.csv`
```
run, n_images, mean_coherence_pct,
pct_fully_coherent,             # fraction of plans with 0 invalid steps
mean_first_invalid_step,        # avg step where plans first break
coverage_aground/capsized/on_fire/sunken
```

---

## SLURM Job Flow

```
submit_coherence.sh
  ├── coherence_judge_job.sh (deepseek_r1_32b)  ─┐
  ├── coherence_judge_job.sh (glm4_32b)           │
  ├── coherence_judge_job.sh (llama_3_3_70b)      ├── all parallel
  ├── coherence_judge_job.sh (phi4_14b)            │
  ├── coherence_judge_job.sh (gemma3_27b)         ─┘
  └── coherence_aggregate_job.sh  (--dependency=afterok:J1:J2:J3:J4:J5)
```

Each judge job: `--gpus=1 --constraint=RTX6000ADA --time=2:00:00`
Aggregate job: CPU-only, `--mem=8G --time=0:30:00`

One run at a time (not an array) — mirrors P5's pattern. To run multiple runs,
call `submit_coherence.sh` once per run.

> **Gap 5**: Should the orchestrator handle model download/detection like
> `judge_panel_submit.sh` (auto-detect missing weights, submit download job
> first with dependency chain)? Or require weights to be pre-downloaded?
> Recommendation: require pre-download for simplicity; add a
> `--download-models` flag later if needed.

---

## Resume Safety

Per-image resume inside `run_coherence_judge.py`:
- Read existing `_per_step.csv` for this run + model on startup
- Collect already-processed images
- Skip them; only process pending images
- Append new rows to the CSV

Aggregate reads all 5 per-model outputs and takes union of images seen.

---

## Known Gaps Summary

| # | Gap | Blocking? |
|---|-----|-----------|
| 1 | Confirm exact HF repo for Llama-3.3-70B AWQ INT4 | Before download |
| 2 | Validate step regex against all 4 prompt variants | Before full run |
| 3 | Token budget smoke test (step 7 of 7 prompt length) | Before full run |
| 4 | Llama-3.3-70B max_model_len OOM risk at 4096 on 48GB | Before full run |
| 5 | Decide: auto-download in submit_coherence.sh or manual | Before impl |
