# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Eval_CASTOR is a standalone evaluation harness for the CASTOR project — it measures how well LLaVA 1.5-7b classifies maritime disasters (aground, capsized, on_fire, sunken) from images. Four evaluation pipelines (P1-P4) operate on inference JSONL files produced by CASTOR and compare them against human-annotated ground truth. Pipeline 6 (`pipelines/salvage_analysis/` + `eval_salvage_plan.py`) is a separate analysis over the same inference output: it tests whether a VLM's salvage/recovery plan text is templated on its predicted state rather than grounded in the specific image — see `SPEC_salvage_analysis.md` and `docs/decisions/ADR-001-salvage-plan-statistical-tests.md`.

## Running the Pipelines

All scripts are run from the repo root (`Eval_CASTOR/`). Paths are computed relative to the script file, so working directory doesn't matter.

**Pipeline 1 — regex extraction (no Ollama needed):**
```
python pipelines/eval_castor.py
```

**Pipeline 2 — LLM extraction then regex eval (requires Ollama):**
```
python pipelines/extract_gemma.py [--runs file.jsonl ...] [--model MODEL]
python pipelines/eval_castor.py --pre-parsed
```

**Pipeline 3 — LLM reference judge (requires Ollama):**
```
python pipelines/judge_castor.py [--pre-parsed] [--eval-only]
```

**Pipeline 4 — separated-parts format (one JSONL per field):**
```
python pipelines/eval_separated.py
```

**Pipeline 6 — salvage plan templating analysis (requires Ollama for Stages 1-2):**

Drop the full-answer JSONL(s) you want judged into `p6_plans_to_judge/` first (e.g. `p6_plans_to_judge/answers_baseline.jsonl`), then:
```
python pipelines/salvage_analysis/extract.py --run <run_name>          # Stage 1: element extraction
python pipelines/salvage_analysis/normalize.py --run <run_name> --threshold <float>  # Stage 2: clustering (no default threshold — pick deliberately, review output)
python pipelines/eval_salvage_plan.py --run <run_name>                 # Stage 3+4: contingency table + stats + report
```
`eval_salvage_plan.py` (Stage 4) with no `--run` processes every run currently sitting in `p6_plans_to_judge/`.

**Pipeline 6 on the cluster (pleiades, SLURM) — no Ollama there:**
```
bash containers/build_judge_container.sh --model deepseek_r1     # one-time, if not already built for the Judge Panel
bash containers/build_judge_container.sh --model salvage_embed   # one-time: caches the local embedding model
bash containers/submit_salvage.sh --threshold <float>             # judges every run in p6_plans_to_judge/
bash containers/submit_salvage.sh --run <run_name> --threshold <float>  # or just one
```
There is no Ollama on the cluster, so the cluster path uses different backends than local dev:
- **Stage 1** (`extract.py --backend vllm`): loads `deepseek_r1` directly via vLLM inside `castor_judge.sif` (same model already built for the Judge Panel — see `containers/build_judge_container.sh`), GPU job, one batch generation pass over all pending records. `clean_and_parse_json` strips deepseek's `<think>` blocks/code fences before parsing.
- **Stage 2** (`normalize.py --backend local`): embeds phrases with a small local `sentence-transformers/all-MiniLM-L6-v2` model (no network at runtime, cached under `/data/$USER/all-minilm-l6-v2`) instead of Ollama's `/api/embeddings`. Same `cluster_phrases`/`AgglomerativeClustering` math either way — only the vector source changes.
- Local dev keeps the original Ollama-based backends (`--backend ollama`, the default for both scripts) for interactive testing against a local Ollama server.

`containers/submit_salvage.sh` submits Stage 1 (GPU, `salvage_stage1_job.sh`) and Stage 2+3+4 (CPU-only, `salvage_stage234_job.sh`, reuses `castor_judge.sif` for pandas/scipy/scikit-learn/sentence-transformers) as two SLURM **job arrays** (one array task per run, `--array=0-N-1`) rather than N separate job submissions — the Stage 2+3+4 array depends on the Stage 1 array via `--dependency=aftercorr`, which is element-wise (task *i* starts once task *i* of Stage 1 succeeds, not the whole batch). Each array task looks up its run name by `SLURM_ARRAY_TASK_ID` from a manifest file (`/data/$USER/logs/salvage_manifest_<PID>.txt`, one run name per line) written by `submit_salvage.sh` at submission time.

**Auto-combine for separated-into-parts input**: `extract.py` and `contingency.py` (via `pipelines/salvage_analysis/combine_shards.py::resolve_input_path`) first look for `<run_name>.jsonl` in the results directory; if that full-answer file doesn't exist, they scan the same directory for per-field shard files (`answers_..._<N>_<field>_j<job>.jsonl`, same convention as `tempp/group_answers.py`) matching `run_name`, merge them by image, and cache the result as `results/p6_salvage_plan/<run_name>/<run_name>_combined.jsonl`. `records.get_field_text` reads both the combined shape (plain top-level field keys) and the original CoT-wrapped `text` shape.

**`CASTOR_SALVAGE_RESULTS_DIR`** env var overrides the input search directory (default `p6_plans_to_judge/`, under this repo — deliberately separate from the shared `results/castor_results/` that P1-P4 read, since P6 is meant to run over a curated subset you drop in yourself, not every experimental variant sitting in the shared results directory). The SLURM scripts default it to `$REPO/p6_plans_to_judge` on the cluster too (same relative location, no NAS path needed).

**`pipelines/salvage_analysis/combine_shards.py` doubles as a discovery CLI**: `python pipelines/salvage_analysis/combine_shards.py --dir p6_plans_to_judge` prints one full-answer run name per line (skipping shard files) — used by `eval_salvage_plan.py` when no `--run` is given. `submit_salvage.sh` itself discovers runs with a bash-native equivalent instead (the login node has no `python3` at all), then writes them to the manifest file the job array tasks read from.

## Data Paths

The scripts expect inference data **outside** this repo:

| Variable | Default path |
|----------|-------------|
| `RESULTS_IN` (P1/P2/P3) | `../../results/castor_results/*.jsonl` |
| `RESULTS_IN` (P4) | `../../results/separated_into_parts/separated_into_parts_*/` |
| `RESULTS_IN` (P6) | `p6_plans_to_judge/*.jsonl` (inside this repo — deliberately curated, not the full shared `castor_results/`) |

P1-P4's paths resolve to `#ONR_CAI/results/` two directories above `Eval_CASTOR/`. The `results/` subtree inside this repo (gitignored) holds only **outputs**, not inputs. P6 is the exception: its input directory (`p6_plans_to_judge/`) lives inside this repo, since it's a deliberate staging area, not a firehose of every experimental variant.

**Note:** `#ONR_CAI/results/castor_results/` now exists at the canonical path (holds every experimental variant's full-answer JSONL). For Pipeline 6, copy just the specific run(s) you want judged into `Eval_CASTOR/p6_plans_to_judge/` rather than pointing at `castor_results/` directly.

## Architecture

```
Eval_CASTOR/
  pipelines/          ← runnable entry points
    eval_castor.py    ← P1 (regex) and P2 (--pre-parsed); separate output dirs
    extract_gemma.py  ← P2 step 1: Ollama-based field extraction
    judge_castor.py   ← P3: Ollama-based semantic judge
    eval_separated.py ← P4: one-JSONL-per-field format
    eval_salvage_plan.py    ← P6 Stage 3+4: contingency table, stats, report
    salvage_analysis/       ← P6 package
      paths.py               ← single source of truth for P6's per-run output paths
      records.py            ← pulls a field out of a full-answer text blob or a combined-shard record
      combine_shards.py      ← auto-combines separated-into-parts shards into one JSONL per run; also a discovery CLI (--dir)
      extract.py            ← P6 Stage 1: LLM element extraction (--backend ollama|vllm)
      normalize.py           ← P6 Stage 2: embedding + clustering into canonical elements (--backend ollama|local)
      contingency.py         ← P6 Stage 3 helpers (typicality score, modal element set)
      prompts/               ← Stage 1 extraction prompts
  p6_plans_to_judge/  ← P6 input staging: drop full-answer run JSONLs here to have them judged
  containers/         ← Apptainer/SLURM (Judge Panel P5 + Pipeline 6 cluster runs)
    container_judge.def       ← vLLM + pandas/scipy/scikit-learn/sentence-transformers
    submit_salvage.sh          ← P6 cluster orchestrator (Stage 1 GPU array -> Stage 2+3+4 CPU array, aftercorr)
    salvage_stage1_job.sh       ← P6 Stage 1 array task (qwen25_72b via vLLM, guided JSON decoding)
    salvage_stage234_job.sh     ← P6 Stage 2+3+4 array task (local embedding + stats)
  shared/             ← shared Python package (add to sys.path via EVAL_ROOT)
    loaders.py        ← JSONL loading, GT loading, ministral /n-format handling
    metrics.py        ← normalization, scoring, per_state_report, summary_row
    ollama.py         ← Ollama REST client (strips fences, unescapes LaTeX, returns dict; also embed_ollama for P6)
    stats.py          ← P6: Fisher's exact, BH-FDR, Kruskal-Wallis, Dunn's test
  human_ground_truth_label/human_gt.csv
  docs/decisions/     ← ADRs (see ADR-001 for P6's statistical test rationale)
  results/            ← gitignored; all outputs go here
    p1_regex/
    p2_llm_extract/extracted/   ← Gemma output JSOLs
    p3_llm_judge/verdicts/
    p4_separated/
    p6_salvage_plan/    ← one subdirectory per run: <run_name>/{raw_elements.jsonl, elements.json, contingency.csv, tests.csv, omnibus.csv, dunn.csv, report.txt}
```

Each pipeline script sets `EVAL_ROOT = Path(__file__).parent.parent` and does `sys.path.insert(0, str(EVAL_ROOT))` to make `shared` importable.

## Key Behaviors

**Ministral /n format**: Some inference files (ministral3-3B, ministral3-8B) use a non-standard format where the entire file is one line, records separated by `}/n{`, and `/n` within strings represents a newline. `shared/loaders._load_slash_n_jsonl` handles this; `load_run` falls back to it automatically when standard line-by-line parsing yields 0 records.

**Resume support**: `extract_gemma.py` and `judge_castor.py` skip records already present in the output JSONL (`gemma_parse_ok=True` / `judge_ok=True`). On resume, failed records are removed from the output and retried.

**P1 vs P2 output isolation**: Both use `eval_castor.py` but write to different directories (`results/p1_regex/` vs `results/p2_llm_extract/`), determined by whether `--pre-parsed` is set.

**Gemma UNKNOWN sentinel**: `shared/metrics.gemma_val()` converts the string `"UNKNOWN"` → `None`. Downstream, `None` state → `"UNPARSEABLE"`, `None` q-answer → `None` (excluded from accuracy).

**State normalization**: `normalize_state()` maps near-synonyms (grounded→aground, sinking→sunken) and returns `"UNPARSEABLE"` for anything unrecognized. The `STATE_MAP` in `shared/metrics.py` is the single source of truth.

## Ollama Config

| Env var | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint (extract, judge, and P6 salvage extraction/embedding) |
| `CASTOR_GEMMA_MODEL` | `gemma4:31b-cloud` | Extraction model (P2) |
| `CASTOR_JUDGE_MODEL` | `gemma4:31b-cloud` | Judge model (P3) |
| `CASTOR_SALVAGE_MODEL` | `gemma4:31b-cloud` | Element extraction + embedding model (P6 Stages 1-2) |

All Ollama calls use `temperature: 0`, `format: "json"`, `num_predict: 1024`, no streaming.

## Output Schema

Per-entry CSVs from all four pipelines share the same column schema so results are directly comparable:

`image, gt_state, pred_state, state_correct, gt_q1..gt_q5, pred_q1..pred_q5, q1_correct..q5_correct, gt_size_bucket, pred_size_bucket, size_correct, vessel_jaccard, cargo_match, parse_error, parse_fail_reason, infer_s`

P4 (separated format) sets `pred_q1`–`pred_q5` and `q*_correct` to `None` since that format has no Q prompts.

P3 (judge) uses a different schema: `judge_{field}` boolean columns + `judge_all_correct`.

## Dependencies

```
pandas, scikit-learn   ← metrics and classification_report; AgglomerativeClustering (P6 Stage 2)
scipy                  ← P6: fisher_exact, kruskal, norm, rankdata (shared/stats.py)
sentence-transformers  ← P6 Stage 2 --backend local (cluster runs; no Ollama there)
```

Ollama HTTP calls use `urllib.request` (stdlib). `scipy` is a transitive dependency of scikit-learn that was already installed but not previously imported directly; Pipeline 6 imports it directly, so it's now a real (not just transitive) dependency — see `docs/decisions/ADR-001-salvage-plan-statistical-tests.md`. `sentence-transformers` (and the `vllm`/`torch`/`transformers` already present in `castor_judge.sif`) are only needed for Pipeline 6's cluster backends (`--backend vllm` / `--backend local`), not for local dev against Ollama.
