# P9: Plan Adequacy Checker -- Progress Report

Slide-mappable: one `##` section per slide, ≤5 bullets, every number in a
table (pastes into PowerPoint or a LaTeX `tabular` with minimal editing).
`report.py` generates a per-run version of the Results section in this same
skeleton, so a fresh cluster run's numbers drop in without re-layout.

---

## What this is

- A deterministic pipeline that grades VLM-generated maritime salvage plans
  for validity -- sequencing, method fit, route recognition, and
  hedging/commitment-avoidance -- not just fluency or plausibility.
- Two-stage: (1) an LLM parses each plan step into a structured tool call
  (extraction, not agentic tool use -- no loop, no live invocation); (2) a
  deterministic Python executor walks those calls against a route-scoped
  world model and derives a verdict per step and per plan.
- Same task both papers in this repo already share (CASTOR): classify and
  now also *evaluate the response quality of* maritime disaster plans.

---

## Assumptions

- **Static plan checking, not agentic simulation.** The plan is already
  fully written; there is no live loop, no tool execution with real
  effects, no environment feedback.
- **A plan is graded against whichever of several valid routes it
  recognisably follows**, not one canonical sequence -- refloating an
  aground vessel has ≥7 legitimate answers; grading against one would fail
  correct plans.
- **Rules read from what the plan established, never from ground truth** --
  a plan is scored on what it demonstrated it knew, not on whether it
  happened to be right by luck.
- **Numeric magnitude grading is deferred (phase 2).** Measured across the
  full real corpus: essentially zero stated action magnitudes (tug counts,
  forces, masses). Confirmed a second time via the gold-set params audit
  (337/338 records genuinely have no stated value). Every graded step reads
  `SPECIFIED_UNGRADED`, never `SPECIFIED_ADEQUATE`, until `physics.py`
  exists.

---

## Experiment setup

| Component | Size | Detail |
|---|---|---|
| Tool registry | 47 tools | `registry/tools.json`, each sourced to Salvor's Handbook, Salvage Manual, FSS Code, IS Code, Stability Reference Guide, Operational Guidelines, or flagged `[AUTHORED]` |
| Route registry | 24 routes | aground 7, capsized 6, on_fire 6, sunken 5 |
| Gold calibration set | 338 records | 72 real-corpus (headline), 166 synthetic (failure-type stratified), 100 authored (tool-coverage probe) |
| Calibration candidates | 3 models | `glm4_32b`, `llama_3_3_70b`, `phi4_14b` |
| Cluster | `pleiades` / RTX6000ADA | `castor_judge.sif`, weights reused from P8's judge panel where possible |

- **Gold-set audit**: two full manual passes over 338 records found that two
  heuristically-labeled "high confidence" tiers were 40-57% wrong before
  review. This is a reportable finding about calibration-set construction
  in its own right.
- **Three real pipeline bugs found and fixed during calibration** (not
  model-quality issues): a JSON-Schema type-array shorthand the
  guided-decoding backend didn't honor, a `parse_extraction` line that
  silently deleted every correctly-null parameter before it could be
  scored, and a prompt with no worked examples for its own hardest rules.

---

## What we measure

**Calibration** (is the extractor trustworthy):

| Metric | What it catches |
|---|---|
| `tool_id_micro/macro_accuracy` | right tool picked, overall / averaged per tool |
| `null_fidelity` | model invents a value the plan never stated |
| `conditional_f1` | model correctly detects genuine commitment-withholding |
| `condition_var_accuracy` | correct reason given for a withheld commitment |
| `no_match_f1` | correctly distinguishes real actions from filler |
| `parse_failure_rate` | guided JSON decoding actually produced valid JSON |

**Plan validity** (is the plan good):

| Level | Signal |
|---|---|
| Per step | one of six verdicts: `SPECIFIED_UNGRADED` / `UNSPECIFIED` / `CONDITIONAL_UNRESOLVED` / `SEQUENCE_VIOLATION` / `METHOD_ERROR` / `NO_MATCH` |
| Per plan | `route_name` / `route_score` / `route_admissible`, `route_coherence` (shotgun-plan detector), `route_completeness`, `gate_rate` / `unresolved_gate_count` (hedging), `self_contradictory_on_size`, `unused_assessments` (hollow diagnostics) |

---

## How to interpret the outputs

- A high `UNSPECIFIED` rate is **not** itself a failure -- it is the
  primary discriminator this project measures: does a prompt variant make
  plans commit to magnitudes, given the corpus baseline is ~0.
- Low `route_coherence` flags a plan naming every technique without
  committing to one -- the same avoided-commitment pattern as hedging, a
  different surface form.
- `unused_assessments` catches plans that go through the motions of
  diligence without it changing anything downstream -- subtler than an
  outright hedge because it reads as more careful, not less.
- Calibration "FAIL" is diagnostic, not disqualifying by itself: a failing
  run here still produced three fixed bugs and moved two metrics
  substantially.
- **A metric pinned at exactly 0.0 is a structural signature, not a
  performance reading** -- see the null_fidelity story below. Check the
  measurement path before concluding a model behavior.

---

## Results (calibration bake-off, all 3 models, post-fix)

Full three-model run under the fixed code (HEAD `b9f0c92`, job 26797-26800).
**FAIL overall for every candidate** -- `glm4_32b` is closest, `phi4_14b` a
close second, `llama_3_3_70b` unusable as-is.

| Metric (Layer A headline) | `glm4_32b` | `phi4_14b` | `llama_3_3_70b` | Target |
|---|---|---|---|---|
| tool_id_micro_accuracy | 0.639 | 0.611 | 0.139 | ≥ 0.90 |
| tool_id_macro_accuracy | 0.636 | 0.642 | 0.042 | ≥ 0.85 |
| null_fidelity | 0.864 | 0.651 | N/A | ≥ 0.97 |
| conditional_f1 | 0.737 | 0.667 | N/A | ≥ 0.85 |
| condition_var_accuracy | 0.667 | 0.667 | 0.333 | ≥ 0.80 |
| no_match_f1 | 0.500 | 0.588 | 0.244 | ≥ 0.85 |
| parse_failure_rate | 0.000 | 0.028 | **1.000** | ≤ 0.01 |

- **The remaining failure (for the two usable models) is tool
  disambiguation, not extraction integrity.** Both `glm4_32b` and
  `phi4_14b` parse reliably (≤3% failure), respect nulls, and read stated
  values correctly -- they pick the wrong tool from a 47-way vocabulary.
  The confusion pairs are **consistent across both models**:
  `survey_seabed -> survey_hull` and (`phi4_14b` only) `sonar_search ->
  survey_hull` -- `survey_hull` acts as an attractor for any assessment
  step, regardless of which survey type is actually named. Both models'
  single per-tool-floor failure is the same tool too: `no_match` recall is
  0.6 (`glm4_32b`) / 0.5 (`phi4_14b`) -- both over-call `no_match` on steps
  that do name a real action. This is now a concrete, reproducible
  registry-design finding, not a vague "accuracy is low": the fix is
  merging/disambiguating the `survey_*` tool cluster and tightening the
  `no_match` decision boundary in the prompt, not a bigger model.
- **The `null_fidelity` lesson**: it read exactly 0.0 across three cluster
  runs and two fix attempts. That flatness was the signature of a
  structurally unreachable value (a JSON-Schema shorthand bug, then a
  params-filtering bug), not model behavior -- fixing both moved it to
  0.864 (`glm4_32b`) / 0.651 (`phi4_14b`) in this run.
- **`llama_3_3_70b`'s 100% parse failure is now diagnosed, not just
  observed.** Raw response capture (added specifically for this) shows a
  distinctive, repeatable pathology, not random truncation:
  - It emits property names that **do not exist in the schema**
    (`"tug_class"`, `"type"`) -- the registry's actual param vocabulary
    never includes these. Guided decoding's `additionalProperties: false`
    should make this grammatically impossible; seeing it means guided
    decoding is not actually constraining this model's output, despite the
    code requesting it.
  - Once ungoverned, generation degenerates into repeating the tool name as
    every param value and array entry (`"secondary_tools": ["pull",
    "pull", "pull", "pull", ...]` dozens of times), then hits `max_tokens`
    before ever closing the JSON object -- a repetition-loop failure mode,
    not a formatting quirk.
  - Conclusion: this looks like a guided-decoding/backend incompatibility
    specific to this model (possibly its built-in tool-calling chat
    template interfering with the grammar compiler), not a fixable prompt
    or schema issue. Consistent with the earlier decision to deprioritize
    it rather than spend further cluster runs chasing it -- one working
    extractor is sufficient to proceed, and this run confirms `glm4_32b`
    is that one (with `phi4_14b` as a viable second option, not a distant
    third).

---

## Results (pipeline robustness fix, independent of model choice)

- `UNSPECIFIED` vs `SPECIFIED_UNGRADED` used to be decided **entirely from
  the extracted params dict** -- at `null_fidelity = 0.864`, roughly 1 in 7
  parameter slots still gets a value the plan text never stated, and every
  one of those silently flips a step toward apparent commitment (the error
  is directional, always inflating).
- **Fixed**: specificity is now read from the step text itself (a digit
  actually present in the sentence), not from the extractor's params dict.
  `UNSPECIFIED` is now structurally immune to extraction hallucination,
  regardless of which model is chosen or how well it calibrates.

---

## Status / what's built

| Piece | Status |
|---|---|
| Tool + route registry, world-state executor | Built, tested |
| Gold calibration set (338 records, hand-audited) | Built, tested |
| Calibration bake-off (`calibrate.py`) | Built, run 5x on cluster (4 iterative fixes + 1 full 3-model bake-off), `glm4_32b` leading but not passing; `llama_3_3_70b` diagnosed as unusable |
| Extraction CLI (`extract.py`) | Built, tested (resume-safe, batches whole run in one vLLM load) |
| Stage 2 link (`run_executor.py`) | Built, tested |
| CSV rollup (`aggregate.py`) | Built, tested |
| Narrative report (`report.py` + `case_studies.md`) | Built, tested, deterministic example selection |
| Cluster wiring (`submit_plan_adequacy.sh`) | Built, not yet run end-to-end on a real plan folder |

---

## Next steps

- Run `submit_plan_adequacy.sh --model glm4_32b` on a real plan folder
  (`--limit`-style smoke test first) to produce the first real `report.md`
  / `case_studies.md`.
- Address the now-concrete tool-disambiguation finding: merge or
  disambiguate the `survey_hull` / `survey_seabed` / `sonar_search` cluster
  in the registry, and tighten the `no_match` decision boundary in the
  prompt (both models under-call real actions as `no_match`). This is a
  registry/prompt change, not a bigger-model problem -- see Results above.
- `llama_3_3_70b` is out of consideration (guided decoding does not appear
  to actually constrain its output); no further cluster time planned on it.
- Numeric adequacy grading (`physics.py`, phase 2) remains out of scope
  until magnitude-stating prompts exist in the corpus to grade.
