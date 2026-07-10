# Spec: Salvage Plan Element & Templating Analysis (Pipeline 6)

## Objective

Add a new evaluation pipeline to `Eval_CASTOR` that tests whether a VLM's
free-text salvage/recovery plan (`recovery_considerations`, elicited by
"how would you salvage this situation?") is **grounded in the specific
image** or **templated on the state label** — i.e. does the model just
emit a stock plan ("call a fireboat, contain the spill") whenever it
outputs `on_fire`/`aground`/etc., regardless of what's actually different
about that particular vessel/cargo/situation.

**User**: the ONR research team (you), doing hallucination-mitigation
analysis on DeGF/ONLY outputs for the CASTOR task. Not a public-facing
tool — output is a report + CSVs consumed by the same people who read
the existing P1–P4 pipeline reports.

**Success looks like**: for a given inference run, a report that says,
per salvage element (e.g. "fireboat", "tug", "crane", "divers"),
whether its presence is statistically associated with predicted state
and/or ground-truth state (with effect size and FDR-corrected
significance), plus an overall verdict on whether plans for a given
state are more "templated" (similar to each other) than plans across
different states.

Numbered **Pipeline 6** — the judge panel already claims "Pipeline 5"
internally (`results/p5_judge/` on the cluster), even though it's not
yet reflected in `Eval_CASTOR/CLAUDE.md`'s pipeline table.

## Pipeline Design

### Stage 1 — Element Extraction (LLM, open vocabulary)

For each record's `recovery_considerations` text, call an LLM (via the
existing `shared/ollama.py` client) to extract a list of salvage
entities/actions mentioned (e.g. `["fireboat", "containment boom"]`).
No predefined taxonomy — whatever the model surfaces is kept as-is at
this stage.

### Stage 2 — Normalization (cluster open vocabulary into canonical elements)

Raw phrases vary ("call a fireboat" vs "dispatch fireboat" vs
"fireboat response"). Embed each unique extracted phrase (Ollama
`/api/embeddings`, new addition to `shared/ollama.py`), then cluster
with `sklearn.cluster.AgglomerativeClustering` (cosine distance,
distance threshold tunable) to collapse near-duplicates into one
canonical element label per cluster. Output: a mapping
`raw phrase -> canonical element`, reviewable/editable as a checkpoint
before Stage 3 runs (so you can eyeball the clusters once before
committing to them for a given run).

### Stage 3 — Contingency Data & Stats

Build a per-record table: `image, predicted_state, gt_state,
element_1_present, element_2_present, ..., typicality_score`.

- **Typicality score**: for each state (predicted and separately GT),
  compute the modal element set across all records with that state —
  the "template." Score each record as the Jaccard similarity between
  its own element set and its state's template. High score = generic/
  templated; low score = distinctive to that image.

- **Primary test — Fisher's exact, one element vs. one state at a
  time, one-vs-rest.** For each (element, state) pair, build a 2×2
  table {element present/absent} × {this state / all other states}.
  Report p-value, odds ratio (effect size), and the same pair computed
  against both `predicted_state` and `gt_state`. Apply
  **Benjamini-Hochberg FDR correction** across the full set of
  (element × state × {pred,gt}) tests before calling anything
  significant.

- **Secondary test — Kruskal-Wallis** on the typicality score across
  state groups (run once against predicted-state grouping, once against
  GT-state grouping). If significant (corrected p < .05), follow up
  with **Dunn's test** (pairwise, rank-based) to identify which specific
  state pairs differ. Chosen over one-way ANOVA because per-state
  sample sizes (~20-30 of 110 images) are too small to assume normality.

- **Documented, not built in v1**: PERMANOVA (permutation-based
  multivariate test on the full element-presence vector via Jaccard
  distance) as a stretch goal if the per-element univariate approach
  turns out too fragmented to give a clean overall verdict.

### Stage 4 — Report

CSV of per-record data (Stage 3 table), CSV of per-(element,state) test
results (p, corrected p, odds ratio), a text report summarizing
significant associations plus the Kruskal-Wallis/Dunn's verdict per
state-grouping, mirroring the existing `eval_summary_*` report style.

## Tech Stack

- Python 3, same environment as the rest of `Eval_CASTOR`
  (`pandas`, `scikit-learn` already present).
- **New dependency: `scipy`** (`fisher_exact`, `kruskal`). Dunn's
  post-hoc and Benjamini-Hochberg are hand-rolled in `shared/stats.py`
  (both are short, well-defined algorithms — avoids pulling in
  `statsmodels`/`scikit-posthocs` for two functions).
- LLM calls via the existing Ollama REST client
  (`shared/ollama.py::call_ollama`), extended with an `embed_ollama()`
  helper for Stage 2.
- Env vars follow the existing convention: `CASTOR_SALVAGE_MODEL`
  (extraction LLM, default `gemma4:31b-cloud`), reuses `OLLAMA_HOST`.

## Commands

```
# Stage 1+2: extract + normalize elements for a run (writes a checkpoint
# file you can review before stats run)
python pipelines/salvage_analysis/extract.py --run <run_name>

# Stage 3+4: build contingency table, run stats, write report
python pipelines/eval_salvage_plan.py --run <run_name>

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_salvage_stats.py -v      # this feature only
```

## Project Structure

```
Eval_CASTOR/
  pipelines/
    eval_salvage_plan.py        ← new entry point (Stage 3+4 driver)
    salvage_analysis/
      __init__.py
      extract.py                ← Stage 1: LLM element extraction
      normalize.py               ← Stage 2: embed + cluster into canonical elements
      contingency.py             ← Stage 3: build per-record table, typicality score
      aggregate.py                ← Stage 4: report writer
      prompts/
        salvage_extract_system.txt
        salvage_extract_user.txt
  shared/
    ollama.py                    ← add embed_ollama()
    stats.py                     ← NEW: fisher_exact wrapper, benjamini_hochberg(),
                                    kruskal_wallis wrapper, dunn_test()
  tests/
    test_salvage_extract.py
    test_salvage_normalize.py
    test_salvage_contingency.py
    test_salvage_stats.py
  p6_plans_to_judge/    ← input staging: drop full-answer run JSONLs here
  results/
    p6_salvage_plan/
      <run_name>/
        raw_elements.jsonl   ← Stage 1 checkpoint
        elements.json        ← Stage 2 checkpoint (raw->canonical map)
        contingency.csv      ← Stage 3 per-record table
        tests.csv            ← per-(element,state) Fisher's results
        report.txt           ← Stage 4 human-readable summary
```

## Code Style

Match existing `Eval_CASTOR` conventions — dataclasses for structured
results (see `judge_panel/preprocess.py::PreprocessResult`), pure
functions over classes, `EVAL_ROOT`/`sys.path.insert` pattern for
running scripts standalone from any working directory.

```python
# shared/stats.py — style example
from dataclasses import dataclass
from scipy.stats import fisher_exact

@dataclass
class ElementStateTest:
    element: str
    state: str
    state_source: str       # "predicted" or "gt"
    odds_ratio: float
    p_value: float
    p_corrected: float | None = None

def fisher_one_vs_rest(present: list, in_state: list) -> tuple:
    """present, in_state: parallel boolean lists. Returns (odds_ratio, p_value)."""
    a = sum(p and s for p, s in zip(present, in_state))
    b = sum(p and not s for p, s in zip(present, in_state))
    c = sum(not p and s for p, s in zip(present, in_state))
    d = sum(not p and not s for p, s in zip(present, in_state))
    return fisher_exact([[a, b], [c, d]])
```

## Testing Strategy

- **Framework**: `pytest`, matching existing `tests/test_preprocess.py`
  style — no test framework config beyond what's already in the repo.
- **Unit tests** (no Ollama, no network) for everything deterministic:
  `shared/stats.py` (Fisher's, BH correction, Dunn's — check against
  hand-computed small examples), `contingency.py` (typicality score
  computation on a fixed toy element/state table), `normalize.py`
  clustering logic (mock embeddings, verify expected clusters merge).
- **No live-Ollama tests** in the default suite — `extract.py`'s
  LLM call is mocked/stubbed in tests the same way `run_judge.py`'s
  tests presumably stub `call_ollama` (verify actual convention in
  `tests/test_run_judge.py` before writing these).
- **Coverage expectation**: every new function in `shared/stats.py` and
  `contingency.py` gets at least one test; `extract.py`/`normalize.py`
  get tests for the deterministic parts (parsing, clustering) not the
  LLM call itself.

## Boundaries

- **Always do**: run `pytest tests/` before considering a task done;
  keep new pipeline consistent with the `EVAL_ROOT` / `sys.path` /
  `results/<pN>_name/` conventions already established; apply BH-FDR
  correction before reporting any element/state association as
  "significant."
- **Ask first**: adding `scipy` to whatever `requirements.txt`/env spec
  the project uses (CLAUDE.md currently documents zero additional pip
  packages beyond pandas/scikit-learn); choosing/changing the
  clustering distance threshold in Stage 2 (affects what counts as
  "the same element" — a judgment call, not a default to silently
  pick); picking the extraction LLM model name/env var default.
- **Never do**: report a chi-squared/Fisher's result as significant
  without the FDR-corrected p-value; silently fall back to ANOVA/t-test
  on data that hasn't been checked for the group-size assumptions those
  tests require; hardcode the element taxonomy (must stay open-vocab →
  clustered, not a fixed list, per your Stage 1 answer).

## Success Criteria

- Running the pipeline on one run staged in `p6_plans_to_judge/` (copied
  there from the canonical `results/castor_results/` path) produces all
  four output files listed under Project Structure, in that run's own
  subdirectory, without error.
- At least one element/state association in the fireboat/on_fire style
  is either confirmed (FDR-corrected p < .05, odds ratio reported) or
  explicitly ruled out — not left ambiguous.
- Kruskal-Wallis + Dunn's post-hoc results are reported for both
  predicted-state and GT-state groupings.
- All new code in `shared/stats.py` has passing unit tests verified
  against hand-computed expected values (not just "doesn't crash").

## Open Questions

1. Which specific inference run(s) should be the first target — one run,
   or every run currently staged in `p6_plans_to_judge/`? (Resolved:
   `submit_salvage.sh`/`eval_salvage_plan.py` process every run staged
   there when no `--run` is given.)
2. Extraction LLM: reuse whatever model `CASTOR_JUDGE_MODEL` /
   `CASTOR_GEMMA_MODEL` currently points at, or a distinct model?
3. Cluster distance threshold for Stage 2 — needs a first pass on real
   data before a sensible default can be picked; treat the first run's
   clustering output as provisional and review before trusting the
   stats built on top of it.
4. Should PERMANOVA be scheduled as explicit follow-up work now, or
   revisited only if the Fisher's-exact-per-element results turn out
   too fragmented to answer the "is this state templated overall"
   question cleanly?
