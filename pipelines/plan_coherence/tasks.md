# P8 Task Breakdown

## Phase 0 — Resolve gaps (do first, unblocks everything)

- [ ] **0a** Confirm Llama-3.3-70B AWQ HF repo ID (Gap 1)
      Acceptance: exact repo string ready for `download_job.sh`

- [ ] **0b** Validate step regex on all 4 prompt variants (Gap 2)
      Acceptance: parse 3 sample records from each of IMPROVED / ABLATION /
      CONTROL / VISUAL_GROUNDED; all produce ≥ 2 steps with correct text

- [ ] **0c** Decide auto-download vs manual (Gap 5)
      Acceptance: decision recorded here; picked manual-pre-download for now

---

## Phase 1 — Core Python (no SLURM needed, testable locally)

- [ ] **1a** `parse_steps.py` — regex extractor
      Input: plan text string
      Output: list of (step_num, step_text) tuples
      Acceptance: handles IMPROVED bold headers, ABLATION plain text,
      single-step plans, plans with sub-bullets

- [ ] **1b** `run_coherence_judge.py` — per-model vLLM judge
      Mirrors P5's `run_judge.py` structure
      Reads: inference JSONL + GT CSV
      Calls parse_steps → builds flat (image, step_num, prompt) batch
      vLLM guided JSON → `{"valid": bool, "reason": str}`
      Writes: `results/p8_plan_coherence/<run>/<run>_<model>.csv`
      Columns: image, step_num, step_text, gt_state, valid, reason
      Resume-safe: skip images already in output CSV
      Acceptance: smoke test with `--limit 5` produces valid CSV

- [ ] **1c** `aggregate_coherence.py` — merge 5 judge CSVs
      Reads: all 5 `_<model>.csv` files for a run
      Computes: n_invalid, majority_invalid per (image, step_num)
      Writes: `_per_step.csv`, `_per_image.csv`, `_summary.csv`,
              appends to `eval_summary_coherence.csv`
      Acceptance: produces correct n_invalid counts on known test input

---

## Phase 2 — SLURM wrappers

- [ ] **2a** `coherence_judge_job.sh` — single model, single run
      Args: MODEL_KEY RUN_NAME
      Sets up Apptainer bind mounts, calls `run_coherence_judge.py`
      SBATCH headers: `--gpus=1 --constraint=RTX6000ADA --time=2:00:00`
      Acceptance: runs cleanly with `--limit 5` on pleiades

- [ ] **2b** `coherence_aggregate_job.sh` — CPU-only aggregation
      Args: RUN_NAME
      Calls `aggregate_coherence.py`
      SBATCH headers: `--cpus-per-task=4 --mem=8G --time=0:30:00`

- [ ] **2c** `submit_coherence.sh` — orchestrator
      Usage: `bash containers/submit_coherence.sh RUN_NAME [--limit N] [--dry-run]`
      Submits 5 parallel judge jobs + 1 aggregate job with
      `--dependency=afterok:J1:J2:J3:J4:J5`
      Sanity checks: SIF exists, `p8_to_check/RUN_NAME.jsonl` exists,
      all 5 model weight dirs exist under `/data/$USER/`
      Acceptance: `--dry-run` prints correct job plan without submitting

---

## Phase 3 — Model downloads (cluster, one-time)

- [ ] **3a** Download Llama-3.3-70B AWQ
      `sbatch containers/download_job.sh <HF_REPO> /data/$USER/llama-3.3-70b-instruct-awq`

- [ ] **3b** Download Phi-4 w4a16
      `sbatch containers/download_job.sh RedHatAI/phi-4-quantized.w4a16 /data/$USER/phi-4-w4a16`

- [ ] **3c** Download Gemma-3-27B w4a16
      `sbatch containers/download_job.sh RedHatAI/gemma-3-27b-it-quantized.w4a16 /data/$USER/gemma-3-27b-it-w4a16`

---

## Phase 4 — Smoke test on cluster

- [ ] **4a** Run `submit_coherence.sh --limit 5` on one run
      Confirm all 5 judge jobs complete, aggregate runs, output CSVs exist

- [ ] **4b** Inspect token budget (Gap 3)
      Check log for longest prompt; confirm < max_model_len for all models

- [ ] **4c** Check Llama-3.3-70B OOM (Gap 4)
      If OOM: reduce `max_model_len` to 2048 in model config

---

## Phase 5 — Documentation

- [ ] **5a** Update `Eval_CASTOR/CLAUDE.md`
      Add P8 row to pipeline table
      Add P8 to Running the Pipelines section
      Add P8 output schema
      Add new model dirs to Container & Model Quick Reference table

- [ ] **5b** Add `p8_to_check/.gitkeep` to git

- [ ] **5c** Commit everything

---

## Dependency order

```
0a,0b,0c → 1a,1b,1c → 2a,2b,2c → 3a,3b,3c (parallel) → 4a → 4b,4c → 5a,5b,5c
```

Phase 3 (downloads) can start in parallel with Phase 2 (SLURM wrappers) since
they're independent. Downloads take ~1 hour each; start them early.
```
