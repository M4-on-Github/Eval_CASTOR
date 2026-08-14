# Eval_CASTOR

Evaluation harness for the CASTOR maritime disaster classification task
(`aground / capsized / on_fire / sunken`). Eight pipelines measure different
aspects of VLM inference output, from classification accuracy to plan
coherence. All pipelines consume inference JSONL files produced by the
DeGF, ONLY, or QWEN repos.

## Pipelines

| Pipeline | What it measures | Entry point | Backend |
|----------|-----------------|-------------|---------|
| P1 | Regex extraction accuracy | `pipelines/eval_castor.py` | None |
| P2 | LLM-extracted field accuracy | `pipelines/extract_gemma.py` + `eval_castor.py --pre-parsed` | Ollama |
| P3 | Semantic judge (binary correct/wrong) | `pipelines/judge_castor.py` | Ollama |
| P4 | Separated-parts format accuracy | `pipelines/eval_separated.py` | None |
| P5 | LLM-as-a-Judge panel (quality 1–3 + hallucinations) | `containers/judge_panel_submit.sh` | vLLM on cluster |
| P6 | Salvage plan templating analysis | `containers/submit_salvage.sh` | vLLM + embeddings on cluster |
| P7 | Assertion coverage (domain concept coverage) | `containers/submit_assertion_coverage.sh` | vLLM on cluster |
| P8 | Plan coherence (step-by-step logical sequencing) | `containers/submit_coherence.sh` | vLLM on cluster |
| P8+ | Plan coherence, improved assertions | `pipelines/plan_coherence/improved/run_all.sh` | vLLM on cluster |

**P8+** supersedes P8 and is self-contained under
`pipelines/plan_coherence/improved/` — its own `config.yaml`, prompts,
assertions, and a one-shot `run_all.sh` chaining inference → assertion coverage
→ judge → aggregation. It has its own
[README](pipelines/plan_coherence/improved/README.md). Edit the `paths:` block
in `config.yaml` before the first run; paths are expressed relative to
`${BENCHYBENCH_ROOT}`, which `run_all.sh` resolves and verifies.

All cluster pipelines share a single Apptainer container (`castor_judge.sif`)
built from `containers/container_judge.def` (vLLM 0.8.5 + pandas/scipy/
scikit-learn/sentence-transformers).

## Data Flow

```
DeGF/ or ONLY/ or QWEN/ inference output (.jsonl)
  → drop into staging folder (p5_to_judge/, p7_to_check/, etc.)
  → cluster pipeline reads it, writes results to /data/$USER/castor_results/
  → compare against human_ground_truth_label/human_gt.csv
```

## Running the Pipelines

All scripts run from the repo root (`Eval_CASTOR/`).

**P1 — regex extraction:**
```bash
python pipelines/eval_castor.py
```

**P2 — LLM extraction + regex eval (requires Ollama):**
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

Three judges (DeepSeek-R1-32B-AWQ, GLM-4-32B-GPTQ, Selene-1-Mini-8B-AWQ)
score each VLM output 1–3 and list hallucinations. Drop JSONLs into
`p5_to_judge/` first:

```bash
bash containers/judge_panel_submit.sh                  # all JSONLs in p5_to_judge/
bash containers/judge_panel_submit.sh --run answers_baseline
```

Outputs: `/data/$USER/castor_results/p5_judge/<run>/` — per-judge JSOLs,
`_consensus.jsonl`, `_flagged.jsonl` (high std), `eval_summary_judge.csv`.

Consensus fields: `mean_score` (1–3), `score_std`, `consensus_status`
(`"consensus"/"flagged_for_review"/"parse_error"`), `judge_verdict`
(`"accurate"` if mean ≥ 2.5 else `"inaccurate"`), `hallucination_union`.

**P6 — salvage plan templating analysis (cluster):**

Tests whether plans are templated on the predicted state rather than image-grounded.
Drop JSONLs into `p6_plans_to_judge/` first:

```bash
bash containers/build_judge_container.sh --model qwen25_72b    # one-time
bash containers/build_judge_container.sh --model salvage_embed # one-time
bash containers/submit_salvage.sh --run answers_baseline --threshold 0.15 --min-generic-pct 0.5
```

Both `--threshold` and `--min-generic-pct` are required — inspect
`elements.json` per run before trusting downstream stats.

**P7 — assertion coverage (cluster):**

Checks whether each salvage plan addresses all required domain concepts.
Drop JSONLs into `p7_to_check/` first:

```bash
bash containers/submit_assertion_coverage.sh
bash containers/submit_assertion_coverage.sh --run answers_baseline
```

Output: `/data/$USER/castor_results/p7_assertions/<run>/`

**P8 — plan coherence (cluster):**

Five LLM judges evaluate whether each step is operationally valid and
correctly sequenced (no image, GT state label as anchor). Drop JSONLs into
`p8_to_check/` first:

```bash
bash containers/submit_coherence.sh --run answers_self_verify
```

Each judge runs in parallel; aggregation auto-chains via `--dependency=afterok`.
Output: `/data/$USER/castor_results/p8_plan_coherence/<run>/`

## Cluster Setup

```bash
ssh head1.condo.cs.cmu.edu
# Interactive CPU node (P1–P4 local; P5–P8 cluster-only)
squeue -u $USER
```

Container build (one-time, needed before P5–P8):
```bash
bash containers/build_judge_container.sh
```

Model weights for P5: DeepSeek-R1-32B-AWQ and GLM-4-32B-GPTQ are already on
the cluster. Llama/Phi/Gemma weights for P8 must be downloaded separately.

## Cluster Storage

| What | Path |
|------|------|
| Apptainer container | `/data/$USER/castor_judge.sif` |
| P5 results | `/data/$USER/castor_results/p5_judge/` |
| P6 results | `/data/$USER/castor_results/p6_salvage/` |
| P7 results | `/data/$USER/castor_results/p7_assertions/` |
| P8 results | `/data/$USER/castor_results/p8_plan_coherence/` |
| Logs | `/data/$USER/logs/` |

## Ground Truth

`human_ground_truth_label/human_gt.csv` — primary GT labels used by P1–P4.
`human_ground_truth_label/real_human_gt.csv` — secondary annotation set.

## Tests

```bash
python -m pytest tests/
```
