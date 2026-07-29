# P8 — Plan Coherence Analysis: Implementation Plan

Spec: `pipelines/plan_coherence/SPEC.md`

## Dependency Graph

```
T1 (parse_steps.py)
  └── T2 (run_coherence_judge.py)
        └── T4 (coherence_judge_job.sh)
              └── T6 (submit_coherence.sh) ──────┐
T3 (aggregate_coherence.py)                       │
  └── T5 (coherence_aggregate_job.sh) ────────────┤
                                                   │
T7 (p8_to_check/ + .gitignore + CLAUDE.md) ───────┤
                                                   ▼
                                          Checkpoint A (local review)
                                                   │
T8a,T8b,T8c (download 3 new model weights) ────────┤
                                                   ▼
                                          Checkpoint B (cluster smoke test)
                                                   │
                                          T9 (full runs)
```

---

## Tasks

### T1 — `parse_steps.py` (no deps)

Regex step extractor. No LLM, no vLLM, runs locally.

**What it does:**
- Strip `**bold**` markers
- Split on `^\d+\.\s+` pattern
- Return list of `(step_num: int, step_text: str)`
- Edge cases: 0 steps → return `[]`; 1 step → valid (no prior context)

**File:** `pipelines/plan_coherence/parse_steps.py`

**Verify:** Run against 3 records from each of IMPROVED / ABLATION / CONTROL /
VISUAL_GROUNDED JSONLs in `temp_plans/`. All should yield ≥2 steps with
correct text (closes Gap 2).

---

### T2 — `run_coherence_judge.py` (needs T1)

Per-model vLLM judge. One SLURM job per model. Mirrors `judge_panel/run_judge.py`.

**What it does:**
- Load inference JSONL + GT CSV (state column only)
- For each image: parse steps (T1), build flat prompt list per step
- vLLM batch: guided JSON `{"valid": bool, "reason": str}`
- Write `results/p8_plan_coherence/<run>/<run>_<model_key>.csv`
  - Columns: `image, gt_state, step_num, step_text, valid, reason, parse_ok`
- Resume-safe: skip images already in output CSV

**Model config dict** (5 entries, `max_tokens=64`, `guided_json=True`):
- deepseek_r1_32b: max_model_len=4096
- glm4_32b: max_model_len=4096
- llama_3_3_70b: max_model_len=4096 (may need 2048 — Gap 4)
- phi4_14b: max_model_len=4096
- gemma3_27b: max_model_len=4096

**CLI:** `python3 run_coherence_judge.py --model KEY --model-dir PATH --input JSONL --out DIR --gt CSV [--limit N]`

**Verify:** `--limit 3` produces valid CSV with correct columns.

---

### T3 — `aggregate_coherence.py` (no deps on T1/T2 code, but needs T2 output)

Merges 5 per-model CSVs into consensus. Mirrors `judge_panel/aggregate.py`.

**What it does:**
- Load all 5 `<run>_<model>.csv` files (warn + skip missing)
- Per `(image, step_num)`: count `n_invalid` (valid==False across judges)
- `majority_invalid = n_invalid >= 3`
- Write `_per_step.csv` (all raw judge columns + n_invalid + majority_invalid)
- Write `_per_image.csv` (n_steps, n_majority_invalid, coherence_pct, first_majority_invalid_step)
- Write `_summary.csv` + append to `eval_summary_coherence.csv`

**CLI:** `python3 aggregate_coherence.py --run RUN_NAME --dir DIR`

**Verify:** Correct n_invalid counts on a hand-crafted 3-row test input.

---

### T4 — `coherence_judge_job.sh` (needs T2)

SLURM batch script for one model × one run. Mirrors `submit_judge_job.sh`.

**SBATCH headers:** `--gpus=1 --constraint=RTX6000ADA --cpus-per-task=8 --time=2:00:00`
- Note: `--mem` is set by `submit_coherence.sh` at submission time (varies per model)

**What it does:**
- Args: `MODEL_KEY RUN_NAME`
- Resolve model dir from MODEL_KEY (case statement, 5 models)
- Check: SIF exists, model dir exists, input JSONL exists
- `apptainer exec --containall --nv` calling `run_coherence_judge.py`
- Same bind mounts as `submit_judge_job.sh`: `/tmp`, `$REPO`, `$DATA_DIR`

**Input path:** `$REPO/p8_to_check/${RUN_NAME}.jsonl`
**Output dir:** `$REPO/results/p8_plan_coherence/`

**Verify:** `--dry-run` equivalent — check paths resolve correctly before submitting.

---

### T5 — `coherence_aggregate_job.sh` (needs T3)

CPU-only SLURM job. Runs after all 5 judge jobs succeed.

**SBATCH headers:** `--cpus-per-task=4 --mem=8G --time=0:30:00 -p pleiades`

**What it does:**
- Args: `RUN_NAME`
- Calls `aggregate_coherence.py --run RUN_NAME --dir $REPO/results/p8_plan_coherence`

---

### T6 — `submit_coherence.sh` (needs T4, T5)

Single entry point. Mirrors `judge_panel_submit.sh` but simpler (no auto-download).

**Usage:**
```bash
bash containers/submit_coherence.sh RUN_NAME [--limit N] [--dry-run]
```

**What it does:**
1. Sanity checks: SIF, `p8_to_check/RUN_NAME.jsonl`, all 5 model dirs
2. Submit 5 judge jobs in parallel (different `--mem` per model):
   - deepseek_r1_32b: `--mem=52G`
   - glm4_32b: `--mem=52G`
   - llama_3_3_70b: `--mem=52G`
   - phi4_14b: `--mem=20G`
   - gemma3_27b: `--mem=24G`
3. Submit aggregate job: `--dependency=afterok:J1:J2:J3:J4:J5`
4. Print monitor commands

**Missing model dir:** print exact `download_job.sh` command to run, exit 1.

**Verify:** `--dry-run` prints job plan without submitting.

---

### T7 — Repo scaffolding (independent)

- Create `p8_to_check/.gitkeep`
- Add `results/p8_plan_coherence/` to `.gitignore`
- Update `Eval_CASTOR/CLAUDE.md`:
  - P8 row in pipeline table
  - P8 in Running the Pipelines section
  - P8 output schema in Output Schemas section
  - 3 new models in Container & Model Quick Reference table

---

### T8a/b/c — Download new model weights (cluster, parallel, independent)

Run on cluster after code is pushed:
```bash
# T8a
sbatch containers/download_job.sh \
    hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
    /data/$USER/llama-3.3-70b-instruct-awq

# T8b
sbatch containers/download_job.sh \
    RedHatAI/phi-4-quantized.w4a16 \
    /data/$USER/phi-4-w4a16

# T8c
sbatch containers/download_job.sh \
    RedHatAI/gemma-3-27b-it-quantized.w4a16 \
    /data/$USER/gemma-3-27b-it-w4a16
```

> **Gap 1:** Llama HF repo needs verification on cluster before T8a.
> Run: `python3 -c "from huggingface_hub import model_info; print(model_info('hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4'))"` on head node.

---

## Checkpoints

### Checkpoint A — Local review (after T1–T7)

- [ ] `parse_steps.py` correctly extracts steps from all 4 prompt variants
- [ ] `run_coherence_judge.py --limit 3` produces valid CSV (local, no vLLM — just test parsing/prompt-building with a mock)
- [ ] `submit_coherence.sh --dry-run` prints correct 6-job plan
- [ ] `CLAUDE.md` updated with P8 section

### Checkpoint B — Cluster smoke test (after T8a/b/c)

- [ ] `bash containers/submit_coherence.sh RUN_NAME --limit 5` completes all 5 judges + aggregate
- [ ] Inspect `_per_step.csv`: all 5 judge columns populated, n_invalid in 0–5
- [ ] Check Llama-3.3-70B log for OOM — if seen, reduce `max_model_len` to 2048 (Gap 4)
- [ ] Check longest prompt fits in context (Gap 3)

---

## Execution Order

| Step | What | Where | Parallel with |
|------|------|-------|---------------|
| T1 | `parse_steps.py` | local | — |
| T2 | `run_coherence_judge.py` | local | T3, T7 |
| T3 | `aggregate_coherence.py` | local | T2, T7 |
| T7 | Scaffolding + docs | local | T2, T3 |
| T4 | `coherence_judge_job.sh` | local | T5 |
| T5 | `coherence_aggregate_job.sh` | local | T4 |
| T6 | `submit_coherence.sh` | local | — |
| **Checkpoint A** | Review | — | — |
| T8a/b/c | Download weights | cluster | each other |
| **Checkpoint B** | Smoke test | cluster | — |
