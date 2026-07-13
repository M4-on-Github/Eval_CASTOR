# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Eval_CASTOR is a standalone evaluation harness for the CASTOR maritime disaster classification task (aground / capsized / on_fire / sunken, classified by LLaVA 1.5-7B). Seven pipelines operate on inference JSONL files from `/data/$USER/castor_results/` (on the cluster) or `../../results/castor_results/` (local).

| Pipeline | What it measures | Entry point | Backend |
|----------|-----------------|-------------|---------|
| P1 | Regex extraction accuracy | `pipelines/eval_castor.py` | None |
| P2 | LLM-extracted field accuracy | `pipelines/extract_gemma.py` + `eval_castor.py --pre-parsed` | Ollama |
| P3 | Semantic judge (binary correct/wrong) | `pipelines/judge_castor.py` | Ollama |
| P4 | Separated-parts format accuracy | `pipelines/eval_separated.py` | None |
| P5 | LLM-as-a-Judge panel (quality 1–3 + hallucination) | `containers/judge_panel_submit.sh` | vLLM on cluster |
| P6 | Salvage plan templating analysis | `containers/submit_salvage.sh` | vLLM + local embeddings on cluster |
| P7 | Assertion coverage (did the plan address each domain concept?) | `containers/submit_assertion_coverage.sh` | vLLM on cluster |

All cluster pipelines share a single Apptainer container (`castor_judge.sif`) built from `containers/container_judge.def` (vLLM 0.8.5 + pandas/scipy/scikit-learn/sentence-transformers).

## Running the Pipelines

All scripts run from the repo root (`Eval_CASTOR/`). Paths resolve relative to the script file — working directory doesn't matter.

**P1 — regex extraction:**
```bash
python pipelines/eval_castor.py
```

**P2 — LLM extraction then regex eval (requires Ollama):**
```bash
python pipelines/extract_gemma.py [--runs file.jsonl ...] [--model MODEL]
python pipelines/eval_castor.py --pre-parsed
```

**P3 — LLM semantic judge (requires Ollama):**
```bash
python pipelines/judge_castor.py [--pre-parsed] [--eval-only]
```

**P4 — separated-parts format:**
```bash
python pipelines/eval_separated.py
```

**P5 — LLM-as-a-Judge panel (cluster only):**

Three judges (DeepSeek-R1-32B-AWQ, GLM-4-32B-GPTQ, Selene-1-Mini-8B-AWQ) score each VLM output 1–3 and list hallucinations. One-shot submission:
```bash
# Download judge model weights (one-time per model):
bash containers/judge_panel_submit.sh --download-only

# Submit all runs in castor_results/ as a parallel job array:
bash containers/judge_panel_submit.sh

# Or a single run:
bash containers/judge_panel_submit.sh --run answers_baseline

# After judges complete, run aggregation:
sbatch --dependency=afterok:J1:J2:J3 containers/aggregate_job.sh <run_name>
```
Outputs land in `/data/$USER/castor_results/p5_judge/<run_name>/`: per-judge JONLs, `_consensus.jsonl`, `_flagged.jsonl` (high std), `eval_summary_judge.csv`.

Consensus fields: `mean_score` (1–3), `score_std`, `consensus_status` ("consensus"/"flagged_for_review"/"parse_error"), `judge_verdict` ("accurate" if mean≥2.5 else "inaccurate"), `hallucination_union`.

**P6 — salvage plan templating analysis (cluster):**

Tests whether VLM salvage plans are templated on the predicted disaster state rather than grounded in the specific image. Drop JONLs to judge into `p6_plans_to_judge/` first.
```bash
bash containers/build_judge_container.sh --model qwen25_72b    # one-time
bash containers/build_judge_container.sh --model salvage_embed # one-time
bash containers/submit_salvage.sh --threshold <float> --min-generic-pct <float>
bash containers/submit_salvage.sh --run answers_baseline --threshold 0.15 --min-generic-pct 0.5
```
Both `--threshold` (Stage 2 clustering) and `--min-generic-pct` (generic element cutoff) are required — no defaults, inspect `elements.json` per run before trusting downstream stats.

Local dev (requires Ollama):
```bash
python pipelines/salvage_analysis/extract.py --run <run_name>
python pipelines/salvage_analysis/normalize.py --run <run_name> --threshold <float>
python pipelines/eval_salvage_plan.py --run <run_name> --min-generic-pct <float>
```

**P7 — assertion coverage analysis (cluster):**

For each VLM salvage plan, checks per-assertion whether the plan addresses each domain-specific concept from `all_prompts/IMPROVED_assertion_registry.csv`. Uses Selene 8B, one LLM call per (image, assertion) pair (~3000 calls/run, ~5–10 min with vLLM prefix caching). Also keyword-scans for wrong-casualty-type terminology (contamination).
```bash
# Drop the JONLs you want scored into p7_to_check/ first, then:
bash containers/submit_assertion_coverage.sh

# Single run (must be in p7_to_check/):
bash containers/submit_assertion_coverage.sh --run answers_baseline

# Smoke test (10 images):
sbatch --partition=pleiades --gpus=1 --constraint=RTX6000ADA \
       --cpus-per-task=4 --mem=40G --time=1:00:00 \
       containers/assertion_coverage_job.sh answers_baseline --limit 10
```
Outputs in `results/p7_assertion_coverage/<run_name>/`: `_per_image.csv` (boolean per assertion + contamination), `_per_assertion.csv` (coverage % per assertion), `_summary.csv`. A cumulative `eval_summary_assertion.csv` is appended.

Key output columns: `coverage_pct` (all relevant assertions), `high_disc_pct` (high-discriminative only — most meaningful), `weighted_score` (weighted by discriminativeness: high=3, medium=2, low=1), `contam_count`/`contam_list` (wrong-casualty terms found).

## Assertion Registry

`all_prompts/IMPROVED_assertion_registry.csv` — five columns:

| Column | Meaning |
|--------|---------|
| `id` | Assertion ID (e.g. A1, C3, R2, X1) |
| `casualty_type` | `aground`/`capsized`/`on_fire`/`sunken`/`resources`/`cross-cutting` |
| `assertion_text` | Full natural-language assertion checked by LLM |
| `checkable_keyword` | `/`-separated keywords for contamination scan |
| `discriminative` | `high`/`medium`/`low` — how domain-specific the term is |

`resources` and `cross-cutting` assertions apply to every image regardless of GT state. State-specific ones apply only to matching images. Contamination scan checks wrong-state assertions via keyword regex — no LLM needed.

## Data Paths

| Source | Default path |
|--------|-------------|
| P1/P2/P3 input | `../../results/castor_results/*.jsonl` (resolves to `#ONR_CAI/results/`) |
| P4 input | `../../results/separated_into_parts/separated_into_parts_*/` |
| P5 input | `/data/$USER/castor_results/<run>.jsonl` (cluster) |
| P6 input | `p6_plans_to_judge/*.jsonl` (inside repo — curated staging area) |
| P7 input | `p7_to_check/*.jsonl` (inside repo — curated staging area, like P6) |
| P5 output | `/data/$USER/castor_results/p5_judge/<run>/` |
| P6 output | `results/p6_salvage_plan/<run>/` (repo-local, gitignored) |
| P7 output | `results/p7_assertion_coverage/<run>/` (repo-local, gitignored) |

## Architecture

```
Eval_CASTOR/
  pipelines/
    eval_castor.py          ← P1 + P2 (--pre-parsed)
    extract_gemma.py        ← P2 Step 1: Ollama field extraction
    judge_castor.py         ← P3: Ollama semantic judge
    eval_separated.py       ← P4: one-JSONL-per-field format
    eval_salvage_plan.py    ← P6 Stage 3+4: contingency table + stats + report
    judge_panel/
      run_judge.py          ← P5: per-model vLLM batch inference + guided JSON scoring
      aggregate.py          ← P5: merge 3 judge JONLs → consensus + flagged
      preprocess.py         ← P5: strip markdown, normalize numerals, verbosity flag
      quantize_model.py     ← AutoAWQ 4-bit quantization helper (called by quantize_job.sh)
      prompts/              ← P5 judge system + user prompt templates
    salvage_analysis/       ← P6 package
      paths.py              ← single source of truth for P6 per-run output paths
      records.py            ← read salvage plan text from full-answer or combined-shard JSONL
      combine_shards.py     ← auto-combine separated-into-parts shards; also a discovery CLI
      extract.py            ← P6 Stage 1: LLM element extraction (--backend ollama|vllm)
      normalize.py          ← P6 Stage 2: embedding + clustering (--backend ollama|local)
      contingency.py        ← P6 Stage 3: typicality score, modal element set
      prompts/              ← Stage 1 extraction prompt
    assertion_coverage/
      check_assertions.py   ← P7: per-assertion LLM coverage + contamination keyword scan
  p6_plans_to_judge/        ← P6 input staging (drop run JSONLs here)
  p7_to_check/              ← P7 input staging (drop run JSONLs here)
  containers/
    container_judge.def           ← Apptainer def: vLLM 0.8.5 + eval deps
    build_judge_container.sh      ← build SIF + download model weights
    judge_panel_submit.sh         ← P5 orchestrator: download → quantize → 3 judge jobs → agg
    submit_judge_job.sh           ← P5 single-judge SLURM batch task
    aggregate_job.sh              ← P5 aggregation SLURM job (CPU-only)
    submit_salvage.sh             ← P6 orchestrator: Stage 1 GPU array → Stage 2+3+4 CPU array
    salvage_stage1_job.sh         ← P6 Stage 1 array task (qwen25_72b, vLLM, guided JSON)
    salvage_stage234_job.sh       ← P6 Stage 2+3+4 array task (local embedding + stats)
    submit_assertion_coverage.sh  ← P7 orchestrator: one job array across all runs
    assertion_coverage_job.sh     ← P7 array task (Selene 8B, vLLM, guided JSON)
    download_job.sh               ← HuggingFace weight download SLURM job
    quantize_job.sh               ← AutoAWQ quantization SLURM job
  shared/
    loaders.py      ← JSONL loading, GT loading, ministral /n-format handling
    metrics.py      ← normalization, scoring, per_state_report, panel_score_summary
    ollama.py       ← Ollama REST client + embed_ollama (P6 local dev)
    stats.py        ← P6: Fisher's exact, BH-FDR, Kruskal-Wallis, Dunn's test
  all_prompts/                          ← prompt variants + assertion registry (repo root, symlinked or accessed via ../../)
  human_ground_truth_label/human_gt.csv
  docs/decisions/ADR-001-salvage-plan-statistical-tests.md
  results/          ← gitignored; all outputs
    p1_regex/
    p2_llm_extract/extracted/
    p3_llm_judge/verdicts/
    p4_separated/
    p6_salvage_plan/<run>/{raw_elements.jsonl, elements.json, contingency.csv, tests.csv, omnibus.csv, dunn.csv, generic_elements.csv, report.txt}
    p7_assertion_coverage/<run>/{_per_image.csv, _per_assertion.csv, _summary.csv}
    p7_assertion_coverage/eval_summary_assertion.csv
```

## Container & Model Quick Reference

All cluster pipelines share `castor_judge.sif`:
```bash
bash containers/build_judge_container.sh --model salvage_embed  # builds SIF + caches all-MiniLM-L6-v2
bash containers/build_judge_container.sh --model qwen25_72b     # downloads Qwen 72B AWQ weights
```

| Model key | HF repo | Local dir | VRAM | Used by |
|-----------|---------|-----------|------|---------|
| `deepseek_r1_32b` | `casperhansen/deepseek-r1-distill-qwen-32b-awq` | `deepseek-r1-distill-qwen-32b-awq` | ~20 GB | P5 |
| `glm4_32b` | `mratsim/GLM-4-32B-0414.w4a16-gptq` | `glm-4-32b-0414-gptq` | ~20 GB | P5 |
| `selene_mini_8b` | `AtlaAI/Selene-1-Mini-Llama-3.1-8B` | `selene-1-mini-llama-3.1-8b-awq` | ~6 GB | P5, P7 |
| `qwen25_72b` | `Qwen/Qwen2.5-72B-Instruct-AWQ` | `qwen25-72b-instruct-awq` | ~40 GB | P6 Stage 1 |

P5 judge model weights are downloaded by `judge_panel_submit.sh` (not `build_judge_container.sh`). Selene FP16 → AWQ quantization runs automatically via `quantize_job.sh` if AWQ weights are missing.

## Key Behaviors

**Guided JSON decoding (P5, P7):** `SamplingParams(guided_decoding=GuidedDecodingParams(json=schema))` — vLLM 0.8.5 API. Enforced by outlines. Eliminates JSON parse failures for well-supported tokenizers.

**vLLM prefix caching (P7):** `enable_prefix_caching=True` in `check_assertions.py`. Consecutive (image, assertion) calls for the same image share the cached plan-text KV, making ~3000 calls feasible in ~5–10 min.

**Ministral /n format:** Some inference files use `}/n{` as record separator with `/n` for newlines inside strings. `shared/loaders._load_slash_n_jsonl` handles this; `load_run` falls back automatically when standard line-by-line parsing yields 0 records.

**Resume support:** P5 `run_judge.py` and P7 `check_assertions.py` skip already-processed images on re-run. P2 `extract_gemma.py` and P3 `judge_castor.py` skip `parse_ok=True` records.

**P6 auto-combine:** `extract.py` and `contingency.py` auto-combine per-field shard JONLs if no full-answer JSONL is found, caching the result as `<run>_combined.jsonl`.

**SLURM_SUBMIT_DIR path resolution:** Build scripts check `SLURM_JOB_ID` to distinguish real SLURM jobs (use `SLURM_SUBMIT_DIR`) from interactive calls (use `BASH_SOURCE[0]`). This avoids the `/var/spool/slurmd/` staging path contaminating file lookups.

## Ollama Config (local dev P2/P3/P6)

| Env var | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `CASTOR_GEMMA_MODEL` | `gemma4:31b-cloud` | Extraction model (P2) |
| `CASTOR_JUDGE_MODEL` | `gemma4:31b-cloud` | Judge model (P3) |
| `CASTOR_SALVAGE_MODEL` | `gemma4:31b-cloud` | Element extraction + embedding (P6 Stages 1-2) |

All Ollama calls: `temperature: 0`, `format: "json"`, `num_predict: 1024`, no streaming.

## Output Schemas

**P1–P4 per-entry CSV** (shared columns):
`image, gt_state, pred_state, state_correct, gt_q1..gt_q5, pred_q1..pred_q5, q1_correct..q5_correct, gt_size_bucket, pred_size_bucket, size_correct, vessel_jaccard, cargo_match, parse_error, parse_fail_reason, infer_s`

**P5 consensus JSONL** (per image):
`image, gt_state, pred_text, verbosity_flagged, scores{model→int}, rationales{model→str}, hallucinations{model→[str]}, mean_score, score_std, consensus_status, judge_verdict, hallucination_union`

**P7 per-image CSV** (per image):
`image, gt_state, coverage_pct, high_disc_pct, weighted_score, n_covered, n_relevant, contam_count, contam_list, <assertion_id>...`
Values per assertion: `"1"` (covered), `"0"` (not covered), `""` (not applicable to this casualty type), `"error"` (LLM parse failure).

## Dependencies

```
pandas, scikit-learn   ← metrics; AgglomerativeClustering (P6 Stage 2)
scipy                  ← P6: fisher_exact, kruskal, norm, rankdata
sentence-transformers  ← P6 Stage 2 --backend local (cluster); P5/P7 container only
vllm                   ← P5, P6 Stage 1, P7 (cluster only, inside castor_judge.sif)
```
