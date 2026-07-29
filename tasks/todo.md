# P8 Todo

## Phase 1 — Core Python (local)
- [ ] T1  `pipelines/plan_coherence/parse_steps.py` — regex step extractor
- [ ] T2  `pipelines/plan_coherence/run_coherence_judge.py` — vLLM per-model judge
- [ ] T3  `pipelines/plan_coherence/aggregate_coherence.py` — merge 5 judge CSVs
- [ ] T7  `p8_to_check/.gitkeep` + `.gitignore` entry + `CLAUDE.md` update

## Phase 2 — SLURM wrappers (local)
- [ ] T4  `containers/coherence_judge_job.sh`
- [ ] T5  `containers/coherence_aggregate_job.sh`
- [ ] T6  `containers/submit_coherence.sh`

## Checkpoint A — local review before cluster work
- [ ] parse_steps validates against all 4 prompt variants (Gap 2)
- [ ] submit_coherence.sh --dry-run prints correct job plan

## Phase 3 — Cluster (after push + Checkpoint A)
- [ ] T8a  Verify Llama-3.3-70B HF repo (Gap 1), then download
- [ ] T8b  Download phi-4-w4a16
- [ ] T8c  Download gemma-3-27b-it-w4a16

## Checkpoint B — smoke test on cluster
- [ ] submit_coherence.sh --limit 5 completes all 6 jobs
- [ ] Check Llama OOM (Gap 4) — reduce max_model_len to 2048 if needed
- [ ] Inspect _per_step.csv output
