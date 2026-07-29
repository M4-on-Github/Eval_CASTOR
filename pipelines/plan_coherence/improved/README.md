# CASTOR Improved — Assertion Experiment

Tests whether specific visual-grounding assertions in the VLM prompt improve salvage plan coherence.

**Hypothesis:** STANDARD assertions > CONTROL assertions ≈ ABLATION (no assertions)

---

## Directory layout

```
improved/
├── config.yaml                        # All cluster paths and model settings — edit before running
├── run_all.sh                         # One-command SLURM script  (sbatch run_all.sh)
│
├── assertions/
│   ├── rubric.md                      # Defines what makes an assertion STANDARD vs CONTROL
│   ├── standard_v2.txt                # 8 specific, discriminative assertions × 4 casualty types
│   └── control_v2.txt                 # 8 garbage (tautological/vague) assertions × 4 casualty types
│
├── prompts/
│   ├── prompt_standard_v2.txt         # Base task + STANDARD assertions
│   ├── prompt_control_v2.txt          # Base task + CONTROL assertions
│   └── prompt_ablation_v2.txt         # Base task only (no assertions)
│
├── inference/
│   └── run_inference.py               # Qwen3-VL 8B inference, all 3 conditions
│
├── eval/
│   ├── parse_steps_v2.py              # Step parser with quality flag (ok/gaps/failed)
│   ├── extract_condition.py           # Extracts VLM-stated casualty type from plan text
│   ├── check_seq.py                   # 14-chain deterministic SEQ checker
│   ├── check_assertion_coverage.py    # Stage 1.5: Selene 8B assertion coverage + contamination
│   ├── run_judge_v2.py                # Llama-3.3-70B judge for MTH + SPC, dual-track
│   └── aggregate.py                   # Summaries, case studies, narrative report
│
├── results/                           # Created at runtime
│   ├── answers_qwen3vl8b_baseline_{condition}_improved.jsonl   (inference output)
│   ├── coverage_per_image.csv         (Stage 1.5 — per-image assertion coverage)
│   ├── coverage_summary.csv           (Stage 1.5 — mean ± SD per condition × ref × track)
│   ├── judge_scores_improved.jsonl    (Stage 2 judge output)
│   ├── summary_by_condition.csv
│   ├── summary_by_condition_state.csv
│   ├── case_studies.md
│   └── report.md
│
└── logs/                              # SLURM stdout/stderr per job
```

---

## Setup

Edit `config.yaml` — the paths section uses `${USER}` / `${HOME}` which are expanded at runtime:

```yaml
paths:
  images_dir:         /home/${USER}/ONLY/CASTOR/shipwreck_wiki_images/sorted_images
  user_models_dir:    /data/${USER}          # VLM: $USER/qwen3vl-8b
                                             # Judge: $USER/llama-3.3-70b-instruct-w4a16
                                             # Selene: $USER/selene-1-mini-llama-3.1-8b-awq
  pipeline_dir:       /home/${USER}/Eval_CASTOR/pipelines/plan_coherence/improved
  gt_csv:             /home/${USER}/Eval_CASTOR/human_ground_truth_label/human_gt.csv
  container_inference: /data/${USER}/castor_qwen.sif
  container_judge:    /data/${USER}/castor_judge.sif
```

The `models.selene_dir` key in config.yaml must name the Selene AWQ weights directory under `user_models_dir`.
Default: `selene-1-mini-llama-3.1-8b-awq` (Selene 8B fits on 1 GPU, ~6 GB VRAM).

---

## Run

```bash
cd ~/Eval_CASTOR/pipelines/plan_coherence/improved
sbatch run_all.sh
```

Runs four stages in sequence. Inference uses `castor_qwen.sif`; all other stages use `castor_judge.sif`.

| Stage | Script | Container | Time (est.) |
|---|---|---|---|
| 1. Inference | `inference/run_inference.py` | castor_qwen.sif | ~4–6 h (330 plans, greedy) |
| 1.5. Coverage | `eval/check_assertion_coverage.py` | castor_judge.sif | ~20–40 min (Selene 8B, ~6k calls) |
| 2. Judge | `eval/run_judge_v2.py` | castor_judge.sif | ~1–2 h (660 rows, Llama-70B) |
| 3. Aggregate | `eval/aggregate.py` | castor_judge.sif | < 1 min |

Both inference and coverage are resumable — if the job is preempted, resubmit and each picks up from where it left off.

---

## Running coverage standalone

`check_assertion_coverage.py` can be run on its own against any completed inference output:

```bash
# All conditions, both reference sets
python improved/eval/check_assertion_coverage.py --config improved/config.yaml

# Single condition
python improved/eval/check_assertion_coverage.py --config improved/config.yaml \
    --condition standard_v2

# Single reference set
python improved/eval/check_assertion_coverage.py --config improved/config.yaml \
    --ref standard

# Smoke test (10 images per condition)
python improved/eval/check_assertion_coverage.py --config improved/config.yaml \
    --limit 10
```

---

## Experiment design

### Conditions

| Condition | File | Content |
|---|---|---|
| `standard_v2` | `prompt_standard_v2.txt` | 8 specific assertions per casualty type |
| `control_v2` | `prompt_control_v2.txt` | 8 vague/tautological assertions per casualty type |
| `ablation_v2` | `prompt_ablation_v2.txt` | No assertions — base prompt only |

All three prompts share identical task instructions and output format requirements. The only difference is the assertion block.

### Assertion design

**STANDARD** assertions must satisfy at least one of:
- Named technique + the condition that selects it
- Safety-critical guard → trigger sequence
- Named resource type + context for choosing it over alternatives
- Named crew role + specific task

**CONTROL** assertions must fail all four criteria AND be tautological, non-discriminative, name only generic resource categories, or identify a factor with no mechanism or threshold.

Both conditions have exactly 8 assertions per casualty type. See `assertions/rubric.md` for full criteria and examples.

### Evaluation metrics

**SEQ** — rule-based, deterministic. 14 precondition chains (4 aground, 4 capsized, 4 sunken, 2 on_fire). A chain fails if its trigger action appears in the plan but its guard action does not, or appears after the trigger.

**MTH** — LLM judge (Llama-3.3-70B). Checks that the technique and resources match the actual casualty type and conditions, and that no explicitly wrong method is applied.

**SPC** — LLM judge. Checks that the plan names specific techniques, specific resource types, and actionable decision criteria (not just "assess" and "coordinate").

**Coherence score** = 0.40 × SEQ + 0.35 × MTH + 0.25 × SPC

**Assertion coverage (Stage 1.5):**
- `recall` = n_covered / n_relevant — fraction of applicable assertions addressed by the plan
- `precision` = n_covered / (n_covered + contam_count) — penalises plans containing wrong-casualty terminology; None if denom = 0
- `f1` = harmonic mean of recall and precision; None if either is None
- `contam_count` / `contam_list` — count and list of wrong-state keywords found in the plan

### Assertion reference sets

Coverage is evaluated against two reference sets:

| Reference set | Source | What it measures |
|---|---|---|
| `standard` | `assertions/standard_v2.txt` + hardcoded R_*/C_* sub-assertions | Domain concept recall — does the plan address discriminative, state-specific concepts? |
| `control` | `assertions/control_v2.txt` + hardcoded CR_*/CC_ALL sub-assertions | Baseline coverage — a well-formed plan should cover these regardless of prompt condition |

STANDARD includes hardcoded resource sub-assertions (R_AG_*, R_CA_*, R_SU_*, R_OF_*) and crew
sub-assertions (C_AG_*, C_CA_*, C_SU_*, C_OF_*) derived from the RESOURCES and CREW blocks in
`standard_v2.txt`. CONTROL includes vague per-state resource assertions (CR_AG / CR_CA / CR_SU / CR_OF)
and one universal crew assertion (CC_ALL — same text for all states).

The key comparison is **STANDARD recall** across the three prompt conditions: if the STANDARD assertions
in the prompt raise recall on the STANDARD reference set (without inflating contamination), that is
direct evidence the assertions are doing causal work.

### Dual-track evaluation

Each plan is scored twice:
- **GT track** — using the ground-truth casualty state (from image path prefix or `human_gt.csv`)
- **Predicted track** — using the state the VLM declared in its own output header

If the VLM-declared state cannot be extracted, the predicted track row is written with empty metrics
(NaN) rather than skipped — it remains in the CSV to flag the parse failure.

---

## Output files

| File | Contents |
|---|---|
| `results/answers_qwen3vl8b_baseline_{condition}_improved.jsonl` | Raw VLM plans (one per image per condition) |
| `results/coverage_per_image.csv` | Per-image assertion coverage: one row per (image × condition × ref_set × track). Columns: `image, condition, reference_set, track, gt_state, predicted_state, n_relevant, n_covered, contam_count, contam_list, recall, precision, f1` + per-assertion `1/0/""` columns |
| `results/coverage_summary.csv` | Mean ± SD of recall / precision / f1 / contam_count per (condition × reference_set × track), plus per-state mean recall breakdown |
| `results/judge_scores_improved.jsonl` | Per-plan SEQ + MTH + SPC + coherence, both tracks |
| `results/summary_by_condition.csv` | Mean coherence scores × condition × track |
| `results/summary_by_condition_state.csv` | Mean coherence scores × condition × casualty state × track |
| `results/case_studies.md` | Best / worst / STANDARD-vs-CONTROL flip cases per condition |
| `results/report.md` | Narrative report with hypothesis assessment and per-state breakdown |

---

## Key design decisions

**Single large judge instead of panel.** The original 5-model panel reached F1=0.59. Root causes: `max_tokens=64` truncated reasoning chains, no rubric in prompts, smaller models suppressed recall. This pipeline uses one Llama-3.3-70B-w4a16 judge with `max_tokens=256` and the rubric injected verbatim into the system prompt.

**Rule-based SEQ.** Sequencing is the most deterministic dimension — 14 specific guard→trigger chains can be checked with keyword matching at 100% reliability. No LLM needed for SEQ.

**Decomposed judge schema.** The judge outputs `{method_valid: bool, specific: bool, reason: str}` rather than a single validity flag. This makes the source of any failure (wrong method vs. too vague) directly readable in the results.

**Equal assertion count across conditions.** STANDARD and CONTROL both have exactly 8 assertions per casualty type. This controls for prompt length and assertion count, isolating assertion quality as the only variable.

**Selene 8B for coverage (Stage 1.5).** Selene fits on one GPU alongside the vLLM prefix cache. With `enable_prefix_caching=True` and prompts ordered by image (all assertions for the same plan are consecutive), the plan KV-prefix is cached across ~13 assertion calls per image, reducing effective compute to roughly 1 full prompt per image.

**Contamination scan is keyword-based.** Wrong-casualty-type terminology is checked with a hard-coded keyword list per state, not an LLM. This is deterministic, instant, and avoids inflating the number of vLLM calls. The contamination score feeds into precision: a plan that describes parbuckling (capsized technique) for an aground vessel has lower precision even if it covers all applicable aground assertions.
