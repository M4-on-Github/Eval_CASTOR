# Implementation Plan: Salvage Plan Element & Templating Analysis (Pipeline 6)

## Context

`SPEC_salvage_analysis.md` (already written and approved) defines a new Eval_CASTOR
pipeline to test whether a VLM's salvage plan (`recovery_considerations` field) is
grounded in the specific image or templated on the state label (e.g. "always mention
a fireboat when the model says on_fire"). This plan breaks that spec into ordered,
independently-verifiable tasks.

Research during planning surfaced a few facts that shape the task breakdown:

- **Input format confirmed**: full-answer JSONL records (`tempp/answers_baseline.jsonl`,
  `answers_degf.jsonl`, `answers_only_*.jsonl`) have `text` = chain-of-thought reasoning
  ending in an embedded JSON blob with **backslash-escaped keys**
  (`vessel\_type`, `recovery\_considerations`, ...). `shared/metrics.py` already has
  the exact tools needed to pull this out: `extract_json_block(text) -> (dict|None, reason)`
  and `_unescape(text)` (undoes the `\_` escaping). No new extraction logic needed —
  reuse these.
- **`results/castor_results/` doesn't exist yet** at the repo root — only
  `tempp/*.jsonl` and `tempp/castor_results/*.jsonl` (duplicates) currently hold
  full-answer data. The pipeline's default `RESULTS_IN` should follow the same
  `EVAL_ROOT.parent / "results" / "castor_results"` convention as `eval_castor.py`
  for when real data lands there, with `--input`/`--run` CLI overrides for now.
- **scipy is already installed** (1.17.1, alongside sklearn 1.9.0) even though
  `Eval_CASTOR/CLAUDE.md` doesn't list it as a direct dependency — safe to import
  directly, just needs documenting.
- **No mocking convention exists in this repo for LLM calls.** `test_run_judge.py`/
  `test_aggregate.py` only test pure functions (prompt building, response parsing,
  record building) with plain fixtures — never mock `call_ollama` or vLLM. New tests
  for `extract.py`/`normalize.py` follow this same pattern: test everything
  deterministic, skip testing the actual network call.
- **User decisions locked in during planning**: process all three runs (baseline,
  degf, only) as **separate** outputs (not merged); extraction/embedding LLM defaults
  to `gemma4:31b-cloud` (same as `CASTOR_JUDGE_MODEL`/`CASTOR_GEMMA_MODEL`) via a new
  `CASTOR_SALVAGE_MODEL` env var; Stage 2's cluster distance threshold is a
  **required CLI flag with no default** (forces conscious choice every run, per the
  spec's "ask first" boundary).

## Architecture Decisions

- Reuse `shared/metrics.extract_json_block` + `_unescape` for pulling both
  `recovery_considerations` and `state` (re-derived predicted state) out of the raw
  `text` blob — one shared helper (`records.get_field_text`), not duplicated logic.
- Reuse `shared/loaders.load_ground_truth` for GT state lookup — no new GT parsing.
- `shared/ollama.py::call_ollama` (existing) for Stage 1 extraction calls; new
  `embed_ollama()` added to the same file for Stage 2, following its exact
  `urllib.request`-based style (no new HTTP library).
- New `shared/stats.py` holds all statistics as pure, independently testable
  functions — no dependency on any pipeline module, so it can be built and fully
  unit-tested first, in parallel with nothing (true foundation layer).
- Clustering in `normalize.py` is split into a pure function
  `cluster_phrases(phrase_to_vector: dict, threshold: float) -> dict` separate from
  the embedding-fetching code, so clustering logic is unit-testable with fixed fake
  vectors (no network needed in tests) even though the embedding fetch itself isn't
  mocked (matches repo convention).
- `eval_salvage_plan.py` follows `eval_separated.py`'s discover-runs-in-a-directory
  pattern so "process all three runs, keep outputs separate" falls out naturally
  from existing conventions rather than needing new multi-run logic.

## Task List

### Phase 1: Foundation (pure functions, no LLM/network, fully unit-testable now)

- [ ] **Task 1: `shared/stats.py` — statistical primitives**
  - Acceptance: `fisher_one_vs_rest(present, in_state) -> (odds_ratio, p_value)`; `benjamini_hochberg(p_values) -> list[float]` (corrected p-values, same order); `kruskal_wallis(groups) -> (H, p_value)` (thin wrapper on `scipy.stats.kruskal`); `dunn_test(groups: dict) -> list[dict]` (pairwise rank-based post-hoc); `ElementStateTest` dataclass per the spec's Code Style section.
  - Verify: `python -m pytest Eval_CASTOR/tests/test_salvage_stats.py -v`; each function checked against a hand-computed value.
  - Dependencies: None.
  - Files: `Eval_CASTOR/shared/stats.py`, `Eval_CASTOR/tests/test_salvage_stats.py`.
  - Scope: S.

- [ ] **Task 2: `pipelines/salvage_analysis/records.py` — shared field-extraction helper**
  - Acceptance: `get_field_text(record, field) -> str | None` — extracts embedded JSON via `shared.metrics.extract_json_block`, unescapes keys, returns `parsed.get(field)`; graceful `None` on failure. Verified against `recovery_considerations` and `state`.
  - Verify: `python -m pytest Eval_CASTOR/tests/test_salvage_records.py -v` using the real sample text from `tempp/answers_baseline.jsonl` plus a deliberately malformed blob.
  - Dependencies: None.
  - Files: `Eval_CASTOR/pipelines/salvage_analysis/__init__.py`, `records.py`, `Eval_CASTOR/tests/test_salvage_records.py`.
  - Scope: S.

### Checkpoint: Foundation
- [ ] `python -m pytest Eval_CASTOR/tests/test_salvage_stats.py Eval_CASTOR/tests/test_salvage_records.py -v` — all green.
- [ ] No network/Ollama access needed for anything so far.
- [ ] Human review before proceeding to LLM-integration phase.

### Phase 2: LLM Integration (Stage 1 extraction, Stage 2 normalization)

- [ ] **Task 3: `shared/ollama.py::embed_ollama()` + extraction prompts**
  - Acceptance: `embed_ollama(text, model, url) -> list[float] | None`, mirroring `call_ollama`'s `urllib.request` pattern against `/api/embeddings`. New prompt files: `pipelines/salvage_analysis/prompts/salvage_extract_system.txt`, `salvage_extract_user.txt` (`{recovery_text}` placeholder).
  - Verify: file-existence + placeholder tests in `test_salvage_extract.py` (mirrors `test_preprocess.py`'s prompt-file tests).
  - Dependencies: None.
  - Files: `Eval_CASTOR/shared/ollama.py`, `Eval_CASTOR/pipelines/salvage_analysis/prompts/*.txt`.
  - Scope: S.

- [ ] **Task 4: `pipelines/salvage_analysis/extract.py` — Stage 1 driver**
  - Acceptance: argparse CLI (`--run`, `--input`, `--out`, `--model` default `CASTOR_SALVAGE_MODEL` env var → `gemma4:31b-cloud`, `--limit`); per record: `records.get_field_text(record, "recovery_considerations")` → `call_ollama` → parse JSON-list response into `raw_elements` → checkpoint JSONL; resume/skip logic mirroring `run_judge.py`.
  - Verify: `python -m pytest Eval_CASTOR/tests/test_salvage_extract.py -v` — prompt-building + response-parsing as pure functions, fixed fixtures (valid/malformed/fenced JSON), no real Ollama call.
  - Dependencies: Task 2, Task 3.
  - Files: `Eval_CASTOR/pipelines/salvage_analysis/extract.py`, `Eval_CASTOR/tests/test_salvage_extract.py`.
  - Scope: M.

- [ ] **Task 5: `pipelines/salvage_analysis/normalize.py` — Stage 2 driver**
  - Acceptance: `cluster_phrases(phrase_to_vector, threshold) -> dict` (pure function, `sklearn.cluster.AgglomerativeClustering`, cosine distance, threshold has no default anywhere). Driver: loads Stage 1 checkpoint, embeds unique phrases via `embed_ollama`, clusters, writes `elements_<run_name>.json`. CLI: `--run`, `--threshold` (required), `--model`.
  - Verify: `python -m pytest Eval_CASTOR/tests/test_salvage_normalize.py -v` — fixed fake vectors (no network) verifying expected merges/splits at a known threshold.
  - Dependencies: Task 3, Task 4.
  - Files: `Eval_CASTOR/pipelines/salvage_analysis/normalize.py`, `Eval_CASTOR/tests/test_salvage_normalize.py`.
  - Scope: M.

### Checkpoint: LLM Integration
- [ ] `python -m pytest Eval_CASTOR/tests/ -v` — all still green (unit tests only, no network).
- [ ] **Manual, human-in-the-loop step** (cannot be automated): run `extract.py` then `normalize.py` against one real run (e.g. `tempp/answers_baseline.jsonl`) with live Ollama, and visually review `elements_<run_name>.json` before trusting downstream stats — the spec's explicit "ask first" boundary on threshold choice.
- [ ] Human review/approval of clustering output before Phase 3.

### Phase 3: Stats & Report

- [ ] **Task 6: `pipelines/salvage_analysis/contingency.py` — Stage 3 driver**
  - Acceptance: one row per image — `image`, `predicted_state` (via `records.get_field_text` + `shared.metrics.normalize_state`), `gt_state` (via `shared.loaders.load_ground_truth`), one boolean column per canonical element, `typicality_score_pred` and `typicality_score_gt` (Jaccard vs. modal element set for that state, predicted and GT separately). Returns a `pandas.DataFrame`; CLI wrapper writes `contingency_<run_name>.csv`.
  - Verify: `python -m pytest Eval_CASTOR/tests/test_salvage_contingency.py -v` — small fixed toy dataset, modal set + every Jaccard score hand-computed and asserted exactly.
  - Dependencies: Task 2, Task 5.
  - Files: `Eval_CASTOR/pipelines/salvage_analysis/contingency.py`, `Eval_CASTOR/tests/test_salvage_contingency.py`.
  - Scope: M.

- [ ] **Task 7: `pipelines/eval_salvage_plan.py` — Stage 4 entry point**
  - Acceptance: discovers runs like `eval_separated.py::discover_runs` (or `--run` for one); per run, builds contingency table, computes `fisher_one_vs_rest` for every (element × state × source ∈ {predicted, gt}), applies `benjamini_hochberg` across the full set before writing `tests_<run_name>.csv`; runs `kruskal_wallis` on typicality scores per grouping, `dunn_test` post-hoc if significant; writes `report_<run_name>.txt`.
  - Verify: `python -m pytest Eval_CASTOR/tests/test_eval_salvage_plan.py -v` — synthetic contingency table with one planted association (e.g. fireboat/on_fire) asserted significant post-FDR, a noise element asserted not significant.
  - Dependencies: Task 1, Task 6.
  - Files: `Eval_CASTOR/pipelines/eval_salvage_plan.py`, `Eval_CASTOR/tests/test_eval_salvage_plan.py`.
  - Scope: M.

### Checkpoint: Full Pipeline
- [ ] `python -m pytest Eval_CASTOR/tests/ -v` — full suite green.
- [ ] Run `python pipelines/eval_salvage_plan.py --run answers_baseline` (and separately `answers_degf`, `answers_only`) end-to-end against real data (`--input` pointed at `tempp/`) — confirm all four output files produced per run.
- [ ] Manually read `report_<run_name>.txt` for plausibility — fireboat/on_fire-style pattern confirmed or explicitly ruled out, not left ambiguous.
- [ ] Human review before calling this feature done.

### Phase 4: Polish

- [ ] **Task 8: Documentation update**
  - Acceptance: `Eval_CASTOR/CLAUDE.md` pipeline table gets a Pipeline 6 row; dependency list corrected to include `scipy`; note that `results/castor_results/` is expected-but-currently-empty, `tempp/` is the interim dev-data location.
  - Verify: manual read-through.
  - Dependencies: Task 7.
  - Files: `Eval_CASTOR/CLAUDE.md`.
  - Scope: XS.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| No live Ollama access during automated build/test | Med | Keep network call thin/isolated; unit-test everything else; defer live verification to human checkpoint after Phase 2 |
| Small sample size (~110 images, ~20-30/state) → few/no significant results after FDR | Med | Document as expected honest outcome; report odds ratios/raw p-values alongside corrected ones |
| Clustering threshold is a genuine judgment call | Med | Required CLI flag (no silent default) + mandatory manual review checkpoint |
| `results/castor_results/` doesn't exist yet at canonical path | Low | Use `--input`/`--run` overrides pointing at `tempp/`; document gap in Task 8 |

## Open Questions

- None blocking — all three items from the spec's Open Questions were resolved during planning. PERMANOVA remains explicitly deferred, not scheduled.
