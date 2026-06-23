"""
CASTOR Gemma Parser
Reads LLaVA 1.5 inference JSONL files and uses Ollama (gemma4:31b-cloud) as a
pure extraction engine to lift structured fields from each record's text output.

Writes pre-parsed JSONL to DeGF/Eval_CASTOR/results/gemma_parsed/ for use with:
    python DeGF/Eval_CASTOR/eval_castor.py --pre-parsed

Run from anywhere:
    python DeGF/Eval_CASTOR/parse_with_gemma.py [--runs <name> ...] [--model <model>]
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (mirror eval_castor.py layout)
# ---------------------------------------------------------------------------
HERE        = Path(__file__).parent
RESULTS_DIR = HERE.parent.parent / "results" / "castor_results"
OUT_DIR     = HERE / "results" / "gemma_parsed"

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
MODEL      = os.environ.get("CASTOR_GEMMA_MODEL", "gemma4:31b-cloud")

FIELDS = ["state", "vessel_type", "size_estimate", "cargo", "q1", "q2", "q3", "q4", "q5"]

_SYSTEM_PROMPT = """\
You are a strict data extraction assistant for a maritime disaster classification task.
Your only job is to find and copy values explicitly stated in the input text.
Do NOT infer, guess, paraphrase, or hallucinate any value. If they mention about a specific field but don't have the lcearest, choose the one that they mentioned as most likely or most confident if they have mentioned on.
If a field is not clearly present in the text, output exactly the string: UNKNOWN
Respond with a single valid JSON object only. No explanation, no markdown fences."""

_USER_TEMPLATE = """\
Extract the following fields from the TEXT below. Rules:
- state: one of aground, capsized, on_fire, sunken, good — copy EXACTLY as stated in the text, else UNKNOWN
- vessel_type: copy the vessel description EXACTLY as stated, else UNKNOWN
- size_estimate: copy the size description EXACTLY as stated, else UNKNOWN
- cargo: copy cargo description EXACTLY as stated, else UNKNOWN
- q1 through q5: output "yes" or "no" ONLY if that exact word is the answer to that numbered question in the text, else UNKNOWN

Return JSON with exactly these keys: state, vessel_type, size_estimate, cargo, q1, q2, q3, q4, q5

TEXT:
{text}"""


# ---------------------------------------------------------------------------
# Loaders (duplicated from eval_castor.py to keep this script standalone)
# ---------------------------------------------------------------------------

def discover_runs(results_dir: Path, filter_names: list = None) -> list:
    runs = []
    for path in sorted(results_dir.glob("*.jsonl")):
        if filter_names and path.name not in filter_names:
            continue
        runs.append(path)
    return runs


def load_run(jsonl_path: Path) -> list:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    records.append(json.loads(line, strict=False))
                except json.JSONDecodeError as e:
                    print(f"  WARNING: could not parse line {lineno} in {jsonl_path.name}: {e}")
    return records


def load_existing_output(out_path: Path) -> set:
    """Return set of image paths successfully processed (resume support).
    Only records with gemma_parse_ok=True are counted as done — failed
    records are re-attempted on the next run.
    """
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("gemma_parse_ok") and "image" in rec:
                    done.add(rec["image"])
            except json.JSONDecodeError:
                pass
    return done


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(text: str) -> tuple:
    user = _USER_TEMPLATE.format(text=text)
    return _SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def call_ollama(system: str, user: str, model: str = MODEL, url: str = OLLAMA_URL):
    """POST to Ollama chat API. Returns (parsed_dict | None, raw_str, elapsed_s)."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream":  False,
        # "options": {"temperature": 0, "num_predict": 512*2},
        "format":  "json",
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        elapsed = time.perf_counter() - t0
    except (urllib.error.URLError, TimeoutError) as e:
        elapsed = time.perf_counter() - t0
        return None, f"HTTP_ERROR: {e}", elapsed

    try:
        outer = json.loads(raw)
        content = outer["message"]["content"]
    except (json.JSONDecodeError, KeyError) as e:
        return None, f"RESPONSE_PARSE_ERROR: {e} | raw={raw[:200]}", elapsed

    # Strip markdown fences that Gemma sometimes adds despite format:"json"
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped)
        stripped = re.sub(r'\s*```\s*$', '', stripped).strip()

    # Unescape LaTeX-style escapes (e.g. on\_fire -> on_fire) that break JSON parsing
    stripped = re.sub(r'\\([_\-/])', r'\1', stripped)

    try:
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            return None, f"NOT_DICT: {stripped[:200]}", elapsed
        return parsed, stripped, elapsed
    except json.JSONDecodeError as e:
        return None, f"CONTENT_JSON_ERROR: {e} | content={content[:200]}", elapsed


# ---------------------------------------------------------------------------
# Process a single run
# ---------------------------------------------------------------------------

def process_run(jsonl_path: Path, model: str = MODEL) -> None:
    run_name = jsonl_path.stem
    out_path = OUT_DIR / f"{run_name}_gemma.jsonl"
    err_path = OUT_DIR / f"{run_name}_errors.txt"

    print(f"\n{'='*62}")
    print(f"  Run: {run_name}")
    print(f"  Model: {model}")
    print(f"{'='*62}")

    records = load_run(jsonl_path)
    done    = load_existing_output(out_path)   # only gemma_parse_ok=True entries

    if done:
        print(f"  Resuming: {len(done)}/{len(records)} already successful.")

    # Rewrite the output file keeping only successful records, so failed entries
    # are dropped and will be re-attempted this run.
    if out_path.exists() and done:
        kept = [r for r in _read_jsonl(out_path) if r.get("gemma_parse_ok")]
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")

    errors = []

    with open(out_path, "a", encoding="utf-8") as out_f, \
         open(err_path, "a", encoding="utf-8") as err_f:

        for idx, rec in enumerate(records, 1):
            img  = rec.get("image", "")
            text = rec.get("text", "")

            if img in done:
                continue

            system, user = build_prompt(text)
            parsed, raw, elapsed = call_ollama(system, user, model=model)

            if parsed is not None:
                out_rec = {"image": img, "gemma_parse_ok": True, "gemma_infer_s": round(elapsed, 3)}
                for field in FIELDS:
                    out_rec[field] = parsed.get(field, "UNKNOWN")
                out_rec["gemma_raw"] = raw[:500]   # truncate for file size

                state_display = out_rec.get("state", "UNKNOWN")
                print(f"  [{run_name}] {idx}/{len(records)}  {img}  -> state={state_display}  ({elapsed:.1f}s)")
            else:
                out_rec = {"image": img, "gemma_parse_ok": False, "gemma_raw": raw, "gemma_infer_s": round(elapsed, 3)}
                errors.append(f"{img}: {raw[:300]}")
                err_f.write(f"{img}: {raw[:300]}\n")
                err_f.flush()
                print(f"  [{run_name}] {idx}/{len(records)}  {img}  -> FAILED  ({elapsed:.1f}s)")

            out_f.write(json.dumps(out_rec) + "\n")
            out_f.flush()
            done.add(img)

    n_ok   = sum(1 for r in _read_jsonl(out_path) if r.get("gemma_parse_ok"))
    n_fail = sum(1 for r in _read_jsonl(out_path) if not r.get("gemma_parse_ok"))
    print(f"\n  Done: {n_ok} parsed, {n_fail} failed -> {out_path.relative_to(HERE)}")


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract structured fields from LLaVA outputs via Ollama Gemma.")
    parser.add_argument("--runs",  nargs="+", metavar="FILE",
                        help="Specific JSONL filenames to process (default: all in castor_results/)")
    parser.add_argument("--model", default=MODEL,
                        help=f"Ollama model name (default: {MODEL}, override via CASTOR_GEMMA_MODEL env)")
    parser.add_argument("--url",   default=OLLAMA_URL,
                        help=f"Ollama API URL (default: {OLLAMA_URL}, override via OLLAMA_HOST env)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Ollama URL : {args.url}")
    print(f"Model      : {args.model}")
    print(f"Input dir  : {RESULTS_DIR}")
    print(f"Output dir : {OUT_DIR}")

    runs = discover_runs(RESULTS_DIR, filter_names=args.runs)
    if not runs:
        print("\nNo JSONL files found. Exiting.")
        return

    print(f"\n{len(runs)} run(s) to process: {[r.name for r in runs]}")

    for run_path in runs:
        process_run(run_path, model=args.model)

    print(f"\nAll runs complete. Output in: {OUT_DIR}")


if __name__ == "__main__":
    main()
