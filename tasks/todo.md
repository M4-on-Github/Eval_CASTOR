# Pipeline 5 — Task Checklist

## T1: Prompt files + Preprocessor ✅
- [x] Write `pipelines/judge_panel/prompts/castor_judge_system.txt`
- [x] Write `pipelines/judge_panel/prompts/castor_judge_user.txt`
- [x] Write `pipelines/judge_panel/preprocess.py` (`preprocess()` → `PreprocessResult`)
- [x] Verify: markdown stripping, numeral normalization, verbosity flag (13/13 tests pass)

## T2: `run_judge.py` — single-model inference path ✅
- [x] Write `pipelines/judge_panel/run_judge.py`
- [x] Implement Ollama backend (reuse `shared/ollama.py`)
- [x] Implement Apptainer backend (`call_apptainer` via subprocess)
- [x] Implement `--limit N` flag for smoke tests
- [x] Implement resume support (skip already-scored images)
- [ ] Write `pipelines/eval_judge_panel.py` (local sequential wrapper) — deferred to post-T3

## ✅ CHECKPOINT A
- [ ] `run_judge.py --limit 5 --backend ollama` runs without crash
- [ ] Output has `score ∈ {1,2,3,null}` for every record
- [ ] Re-run skips already-scored records

## T3: Aggregation + summary ✅
- [x] Write `pipelines/judge_panel/aggregate.py`
- [x] Implement consensus: mean, std, status, hallucination_union
- [x] Write flagged JSONL (score_std > 0.8)
- [x] Add `panel_score_summary()` to `shared/metrics.py`
- [x] Append row to `eval_summary_judge.csv`
- [x] Test with 3 mock JONLs (12/12 tests pass)

## ✅ CHECKPOINT B
- [ ] `eval_judge_panel.py --run answers_baseline --limit 10` completes (needs Ollama running)
- [ ] `_consensus.jsonl`, `_flagged.jsonl`, and `eval_summary_judge.csv` all written
- [ ] Record count in consensus == record count in input

## T4: Container definitions (cluster) ✅
- [x] Write `containers/container_judge.def` — single vLLM container (vllm/vllm-openai:v0.5.5)
      covers all three models; weights bind-mounted at runtime from /data/$USER/
- [x] Write `containers/build_judge_container.sh` — hash-gated build + HF model download
- [x] Removed Ollama backend from `run_judge.py`; replaced with vLLM batch mode
- [x] Confirm GPT-OSS 120B VRAM budget — `--gpus=2 --mem=104G --tp=2` (2× 48 GB = 96 GB)

## T5: SLURM orchestration ✅
- [x] Write `containers/submit_judges.sh` (3 parallel + afterok dependency for aggregation)
- [x] Write `containers/submit_judge_job.sh` (generic; MODEL + RUN_NAME as positional args)
- [x] Write `containers/aggregate_job.sh` (CPU-only; runs aggregate.py + panel_score_summary)
- [x] `bash -n` all four scripts — no syntax errors

## ✅ CHECKPOINT C
- [x] `container_judge.def` uses vllm/vllm-openai:v0.5.5 + adds pandas/scikit-learn
- [x] All `.sh` scripts pass `bash -n`
- [x] VRAM + memory budgets confirmed: 1-GPU (52G) for 70B models, 2-GPU (104G) for 120B MoE
- [x] 37/37 unit tests pass after removing Ollama from run_judge.py

## T6: Cluster integration
- [ ] Build all three containers on pleiades
- [ ] Submit `submit_judges.sh answers_baseline`
- [ ] Verify 3 judge JONLs + consensus + flagged produced
- [ ] Regression: ≥ 80% of P1-correct records have mean_score ≥ 2.0
