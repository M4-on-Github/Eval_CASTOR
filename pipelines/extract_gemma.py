"""
CASTOR LLM Field Extractor — Pipeline 2, Step 1

Uses Ollama (gemma4:31b-cloud by default) as a pure extraction engine to lift
structured fields (state, vessel_type, size_estimate, cargo, q1-q5) from
LLaVA 1.5 inference outputs, even when the CoT is truncated before Step 10.

Output goes to results/p2_llm_extract/extracted/*_gemma.jsonl.
Then run eval_castor.py --pre-parsed to evaluate those extractions.

Run from anywhere:
    python pipelines/extract_gemma.py [--runs name.jsonl ...] [--model model]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

EVAL_ROOT  = Path(__file__).parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_run, read_jsonl
from shared.ollama  import call_ollama

RESULTS_IN = EVAL_ROOT.parent.parent / "results" / "castor_results"
OUT_DIR    = EVAL_ROOT / "results" / "p2_llm_extract" / "extracted"

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
MODEL      = os.environ.get("CASTOR_GEMMA_MODEL", "gemma4:31b-cloud")

FIELDS = ["state", "vessel_type", "size_estimate", "cargo", "q1", "q2", "q3", "q4", "q5"]

_SYSTEM = """\
You are a strict data extraction assistant for a maritime disaster classification task.
Your only job is to find and copy values explicitly stated in the input text.
Do NOT infer, guess, paraphrase, or hallucinate any value.
If they mention about a specific field but lack certainty, choose the one they stated
as most likely or most confident.
If a field is not clearly present in the text, output exactly the string: UNKNOWN
Respond with a single valid JSON object only. No explanation, no markdown fences."""

_USER_TEMPLATE = """\
Extract the following fields from the TEXT below. Rules:
- state: one of aground, capsized, on_fire, sunken, good — copy EXACTLY as stated, else UNKNOWN
- vessel_type: copy the vessel description EXACTLY as stated, else UNKNOWN
- size_estimate: copy the size description EXACTLY as stated, else UNKNOWN
- cargo: copy cargo description EXACTLY as stated, else UNKNOWN
- q1 through q5: output "yes" or "no" ONLY if that exact word is the answer to that
  numbered question in the text, else UNKNOWN

Return JSON with exactly these keys: state, vessel_type, size_estimate, cargo, q1, q2, q3, q4, q5

TEXT:
{text}"""


class ExtractionRuns:
    """Selects which inference files P2 sends to Gemma for field extraction.

    Separate from P1's discovery because the question differs: P1 asks what
    runs EXIST, while P2 asks which to spend Ollama calls on. Extraction is
    the slow, billable step, so `filter_names` exists to re-run one file
    without re-extracting the rest.

    An EMPTY filter list is treated as no filter, not as "nothing". The check
    is `if filter_names and ...`, so [] is falsy and everything passes. That is
    the useful default for a caller building a list conditionally, but it means
    an accidentally-empty filter processes the whole directory rather than
    doing nothing.
    """


def discover_runs(results_dir: Path, filter_names: list = None) -> list:
    """Inference files to extract from, optionally filtered. See ExtractionRuns."""
    runs = []
    for path in sorted(results_dir.glob("*.jsonl")):
        if filter_names and path.name not in filter_names:
            continue
        runs.append(path)
    return runs


def load_existing_output(out_path: Path) -> set:
    """Return image paths already SUCCESSFULLY extracted, for resume.

    Only records with gemma_parse_ok are counted as done. A record that failed
    extraction is deliberately NOT included, so a transient Ollama error is
    retried on the next run rather than being permanently skipped — the file
    would still contain a line for that image, and a naive "seen this image"
    check would drop it forever.

    Malformed lines are skipped rather than fatal: a run interrupted mid-write
    must not prevent resuming.
    """
    done = set()
    if not out_path.exists():
        return done
    for rec in read_jsonl(out_path):
        if rec.get("gemma_parse_ok") and "image" in rec:
            done.add(rec["image"])
    return done


def process_run(jsonl_path: Path, model: str, url: str) -> None:
    run_name = jsonl_path.stem
    out_path = OUT_DIR / f"{run_name}_gemma.jsonl"
    err_path = OUT_DIR / f"{run_name}_errors.txt"

    print(f"\n{'='*62}")
    print(f"  Run  : {run_name}")
    print(f"  Model: {model}")
    print(f"{'='*62}")

    records = load_run(jsonl_path)
    done    = load_existing_output(out_path)

    if done:
        print(f"  Resuming: {len(done)}/{len(records)} already extracted.")

    # Rewrite keeping only successes so failed entries are retried
    if out_path.exists() and done:
        kept = [r for r in read_jsonl(out_path) if r.get("gemma_parse_ok")]
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")

    options = {"temperature": 0, "num_predict": 1024}

    with open(out_path, "a", encoding="utf-8") as out_f, \
         open(err_path, "a", encoding="utf-8") as err_f:

        for idx, rec in enumerate(records, 1):
            img  = rec.get("image", "")
            text = rec.get("text", "")

            if img in done:
                continue

            user   = _USER_TEMPLATE.format(text=text)
            parsed, raw, elapsed = call_ollama(_SYSTEM, user, model=model, url=url,
                                               options=options)

            if parsed is not None:
                out_rec = {"image": img, "gemma_parse_ok": True,
                           "gemma_infer_s": round(elapsed, 3)}
                for field in FIELDS:
                    out_rec[field] = parsed.get(field, "UNKNOWN")
                out_rec["gemma_raw"] = raw[:500]
                print(f"  {idx}/{len(records)}  {img}  -> state={out_rec.get('state')}  ({elapsed:.1f}s)")
            else:
                out_rec = {"image": img, "gemma_parse_ok": False,
                           "gemma_raw": raw, "gemma_infer_s": round(elapsed, 3)}
                err_f.write(f"{img}: {raw[:300]}\n")
                err_f.flush()
                print(f"  {idx}/{len(records)}  {img}  -> FAILED  ({elapsed:.1f}s)")

            out_f.write(json.dumps(out_rec) + "\n")
            out_f.flush()
            done.add(img)

    n_ok   = sum(1 for r in read_jsonl(out_path) if r.get("gemma_parse_ok"))
    n_fail = sum(1 for r in read_jsonl(out_path) if not r.get("gemma_parse_ok"))
    print(f"\n  Done: {n_ok} extracted, {n_fail} failed -> {out_path.relative_to(EVAL_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured fields from LLaVA outputs via Ollama (Pipeline 2 Step 1)."
    )
    parser.add_argument("--runs",  nargs="+", metavar="FILE",
                        help="Specific JSONL filenames to process (default: all in castor_results/)")
    parser.add_argument("--model", default=MODEL,
                        help=f"Ollama model (default: {MODEL}; override via CASTOR_GEMMA_MODEL)")
    parser.add_argument("--url",   default=OLLAMA_URL,
                        help="Ollama API URL (override via OLLAMA_HOST)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Ollama URL : {args.url}")
    print(f"Model      : {args.model}")
    print(f"Input      : {RESULTS_IN}")
    print(f"Output     : {OUT_DIR.relative_to(EVAL_ROOT)}")

    runs = discover_runs(RESULTS_IN, filter_names=args.runs)
    if not runs:
        print("\nNo JSONL files found. Exiting.")
        return

    print(f"\n{len(runs)} run(s): {[r.name for r in runs]}")

    for run_path in runs:
        process_run(run_path, model=args.model, url=args.url)

    print(f"\nAll runs complete. Output in: {OUT_DIR.relative_to(EVAL_ROOT)}")
    print("Next: python pipelines/eval_castor.py --pre-parsed")


if __name__ == "__main__":
    main()
