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
```
python pipelines/salvage_analysis/extract.py --run <run_name>          # Stage 1: element extraction
python pipelines/salvage_analysis/normalize.py --run <run_name> --threshold <float>  # Stage 2: clustering (no default threshold — pick deliberately, review output)
python pipelines/eval_salvage_plan.py --run <run_name>                 # Stage 3+4: contingency table + stats + report
```

## Data Paths

The scripts expect inference data **outside** this repo:

| Variable | Default path |
|----------|-------------|
| `RESULTS_IN` (P1/P2/P3) | `../../results/castor_results/*.jsonl` |
| `RESULTS_IN` (P4) | `../../results/separated_into_parts/separated_into_parts_*/` |
| `RESULTS_IN` (P6) | `../../results/castor_results/*.jsonl` (same as P1-P3) |

These paths resolve to `#ONR_CAI/results/` two directories above `Eval_CASTOR/`. The `results/` subtree inside this repo (gitignored) holds only **outputs**, not inputs.

**Note:** as of this writing, `#ONR_CAI/results/castor_results/` does not exist yet at that canonical path — only `#ONR_CAI/tempp/*.jsonl` and `#ONR_CAI/tempp/castor_results/*.jsonl` (duplicates) hold full-answer inference data. Until real inference output is copied into place, pass `--input` explicitly to point Pipeline 6's scripts at `tempp/`.

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
      records.py            ← pulls a field out of a full-answer text blob
      extract.py            ← P6 Stage 1: LLM element extraction
      normalize.py           ← P6 Stage 2: embedding + clustering into canonical elements
      contingency.py         ← P6 Stage 3 helpers (typicality score, modal element set)
      prompts/               ← Stage 1 extraction prompts
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
    p6_salvage_plan/    ← raw_elements_*.jsonl, elements_*.json, contingency_*.csv, tests_*.csv, report_*.txt
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
```

Ollama HTTP calls use `urllib.request` (stdlib). `scipy` is a transitive dependency of scikit-learn that was already installed but not previously imported directly; Pipeline 6 imports it directly, so it's now a real (not just transitive) dependency — see `docs/decisions/ADR-001-salvage-plan-statistical-tests.md`.
