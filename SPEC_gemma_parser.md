# CASTOR Gemma Parser — SPEC

## Objective

Preprocess LLaVA 1.5 inference outputs using **Ollama gemma4:31b-cloud** as a pure
extraction engine, converting raw (often truncated) text into structured JSONL files that
`eval_castor.py` can consume instead of its regex extractor.

**Motivation**: The current regex extractor fails on 40–67% of chain-of-thought records
because LLaVA hits its token budget before reaching the Step 10 JSON block. Gemma reads
the partial CoT text and extracts whatever is explicitly stated — no inference, no
hallucination.

**Target users**: CASTOR researchers running the offline evaluation pipeline.

---

## Inputs

| Source | Path |
|--------|------|
| Inference runs | `results/castor_results/*.jsonl` (auto-discovered) |
| Ollama model | `gemma4:31b-cloud` via `http://localhost:11434` |

Each JSONL record has at minimum: `image`, `text`.

---

## Extraction Fields

Gemma extracts **all seven fields** from every record, regardless of prompt style:

| Field | Type | Notes |
|-------|------|-------|
| `state` | string | One of: aground, capsized, on_fire, sunken, good — or UNKNOWN |
| `vessel_type` | string | Free text as it appears; UNKNOWN if absent |
| `size_estimate` | string | Free text (small/medium/large/etc.); UNKNOWN if absent |
| `cargo` | string | Free text or UNKNOWN |
| `q1`–`q5` | string | `yes`, `no`, or UNKNOWN |

**Strict extraction rule**: Gemma must only copy values explicitly stated in the source
text. If a field is not present, output `UNKNOWN`. Never infer, paraphrase, or fill gaps.

---

## Ollama Prompt Design

### System prompt
```
You are a strict data extraction assistant for a maritime disaster classification task.
Your only job is to find and copy values that are explicitly stated in the input text.
Do NOT infer, guess, paraphrase, or hallucinate any value.
If a field is not clearly present in the text, output exactly: UNKNOWN
Output a single valid JSON object with the keys listed by the user. No explanation.
```

### User prompt (per record)
```
Extract the following fields from the text below. Copy values EXACTLY as stated.
For q1–q5: output "yes" or "no" only if that exact word appears as an answer to
that question. Otherwise UNKNOWN.

Fields: state, vessel_type, size_estimate, cargo, q1, q2, q3, q4, q5

TEXT:
<rec["text"]>
```

### Model params
- `temperature: 0` (deterministic)
- `format: "json"` (Ollama structured output)
- No streaming

---

## Outputs

### Pre-parsed JSONL per run
**Path**: `DeGF/Eval_CASTOR/results/gemma_parsed/<run_name>_gemma.jsonl`

One line per input record:
```json
{
  "image": "aground/00017.jpg",
  "state": "aground",
  "vessel_type": "cargo ship",
  "size_estimate": "large",
  "cargo": "UNKNOWN",
  "q1": "yes",
  "q2": "no",
  "q3": "UNKNOWN",
  "q4": "yes",
  "q5": "no",
  "gemma_parse_ok": true,
  "gemma_raw": "<raw Ollama response string>"
}
```

If Ollama fails or returns malformed JSON for a record:
```json
{
  "image": "aground/00017.jpg",
  "gemma_parse_ok": false,
  "gemma_raw": "<raw response or error message>"
}
```

### Parse failure log
**Path**: `results/gemma_parsed/<run_name>_gemma_errors.txt`

Lists records where `gemma_parse_ok=false` with the raw response.

### Progress display (console)
```
[answers_baseline] 1/110  aground/00017.jpg  -> state=aground  (0.8s)
[answers_baseline] 2/110  aground/00034.jpg  -> state=UNKNOWN  (1.1s)
...
[answers_baseline] Done: 108/110 parsed, 2 failures  (avg 0.9s/rec)
```

---

## Integration with eval_castor.py

`eval_castor.py` is extended (non-breaking) with a `--pre-parsed` flag:

```
python DeGF/Eval_CASTOR/eval_castor.py [--pre-parsed]
```

When `--pre-parsed` is set:
1. For each discovered run, check if `results/gemma_parsed/<run>_gemma.jsonl` exists.
2. If yes: load it and use `gemma_parsed` fields directly (skip `extract_json_block`).
3. If no: fall back to regex extraction as before.

The `evaluate_run()` function gains an optional `pre_parsed_dict` argument:
```python
def evaluate_run(records, gt_dict, prompt_style, pre_parsed_dict=None):
    for rec in records:
        img = rec["image"]
        if pre_parsed_dict and img in pre_parsed_dict:
            pp = pre_parsed_dict[img]
            parsed = pp if pp.get("gemma_parse_ok") else None
            parse_fail_reason = "" if parsed else "gemma_parse_failed"
        else:
            parsed, parse_fail_reason = extract_json_block(rec["text"])
        ...
```

UNKNOWN values from Gemma are handled identically to missing values:
- `state=UNKNOWN` → `normalize_state(None)` → `UNPARSEABLE`
- `q1=UNKNOWN` → treated as `None` (no answer found)

---

## Script Architecture

Single new file: `DeGF/Eval_CASTOR/parse_with_gemma.py`

```
parse_with_gemma.py
  build_prompt(text)        -> (system_str, user_str)
  call_ollama(prompt)       -> (dict | None, raw_str, elapsed_s)
  parse_record(rec)         -> gemma_record dict
  process_run(jsonl_path)   -> writes gemma JSONL + error log
  main()                    -> discovers runs, processes each
```

eval_castor.py changes:
- Add `--pre-parsed` CLI flag (argparse)
- Add `load_pre_parsed(path)` → `dict[image -> gemma_record]`
- Extend `evaluate_run()` with `pre_parsed_dict` kwarg
- Call `load_pre_parsed` in `main()` when flag is set

---

## Tech Stack

- Python 3.x stdlib: `json`, `re`, `pathlib`, `argparse`, `time`, `urllib.request`
- No new pip dependencies (use `urllib.request` for Ollama HTTP, or `requests` if already available)
- Same `pandas` / `sklearn` as eval_castor.py

---

## File Layout

```
DeGF/Eval_CASTOR/
  parse_with_gemma.py          <- new script
  eval_castor.py               <- extended with --pre-parsed flag
  results/
    gemma_parsed/              <- gitignored (inside results/)
      answers_baseline_gemma.jsonl
      answers_baseline_gemma_errors.txt
      ...
```

---

## Boundaries

| Rule | Detail |
|------|--------|
| **Always do** | `temperature=0` to suppress hallucination |
| **Always do** | Output `UNKNOWN` (not null/empty/None) for absent fields |
| **Always do** | Preserve `gemma_raw` for auditability |
| **Always do** | `gemma_parse_ok=false` records still appear in output JSONL |
| **Never do** | Modify source JSONL files |
| **Never do** | Send image bytes to Gemma — text only |
| **Never do** | Let Gemma correct or normalize state names — copy verbatim |
| **Ask first** | If Ollama model name changes or endpoint is non-local |

---

## Acceptance Criteria

- `parse_with_gemma.py` runs from any directory, processes all `*.jsonl` in `results/castor_results/`
- Output JSONL has one line per input record, every record has `gemma_parse_ok` field
- `eval_castor.py --pre-parsed` produces a summary CSV with lower parse failure rate than regex baseline
- UNKNOWN fields propagate correctly to UNPARSEABLE / N/A in eval output
- No new pip dependencies required
