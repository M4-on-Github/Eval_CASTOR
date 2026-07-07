# Todo: Salvage Plan Element & Templating Analysis (Pipeline 6)

Full plan: `tasks/plan_salvage_analysis.md`. Spec: `SPEC_salvage_analysis.md`.

## Phase 1: Foundation
- [x] Task 1 — `shared/stats.py` (fisher_one_vs_rest, benjamini_hochberg, kruskal_wallis, dunn_test, ElementStateTest) + `tests/test_salvage_stats.py`
- [x] Task 2 — `pipelines/salvage_analysis/records.py` (get_field_text) + `tests/test_salvage_records.py`
- [x] **Checkpoint**: `pytest tests/test_salvage_stats.py tests/test_salvage_records.py -v` green; human review

## Phase 2: LLM Integration
- [x] Task 3 — `shared/ollama.py::embed_ollama()` + extraction prompt files
- [x] Task 4 — `pipelines/salvage_analysis/extract.py` (Stage 1) + tests
- [x] Task 5 — `pipelines/salvage_analysis/normalize.py` (Stage 2, cluster_phrases) + tests
- [ ] **Checkpoint (STILL OPEN — needs your live Ollama access)**: full test suite is green (88 passed), but the **manual** run of extract.py + normalize.py on real data (e.g. `tempp/answers_baseline.jsonl`) with live Ollama, plus visually reviewing `elements_*.json` clustering before trusting it, has not been done — I have no Ollama access in this environment. Run these two commands yourself and eyeball the clustering before relying on Phase 3's output.

## Phase 3: Stats & Report
- [x] Task 6 — `pipelines/salvage_analysis/contingency.py` (Stage 3, contingency table + typicality scores) + tests
- [x] Task 7 — `pipelines/eval_salvage_plan.py` (Stage 4, Fisher's+FDR, Kruskal-Wallis+Dunn's, report) + tests
- [ ] **Checkpoint (STILL OPEN — depends on the Phase 2 checkpoint above)**: code is built and unit-tested against synthetic data, but hasn't been run end-to-end on `answers_baseline`/`answers_degf`/`answers_only` against real data yet; do this after the Phase 2 checkpoint and read the reports for plausibility.

## Phase 4: Polish
- [x] Task 8 — Update `Eval_CASTOR/CLAUDE.md` (Pipeline 6 row, scipy dependency, results/ path note)

## Decisions locked in during planning
- Process baseline/degf/only as **separate** outputs (not merged)
- Extraction + embedding model: `gemma4:31b-cloud` via new `CASTOR_SALVAGE_MODEL` env var
- Cluster threshold: **required CLI flag, no default**
- Input format: full-answer JSONL only (v1) — reuse `shared/metrics.extract_json_block` + unescape
- PERMANOVA: explicitly deferred, not built in v1
