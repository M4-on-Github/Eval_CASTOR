"""
Pipeline 6 Stage 1 — LLM element extraction from salvage plans.

For each record's recovery_considerations text, asks an LLM (via Ollama) to
extract the concrete salvage entities/actions mentioned, open-vocabulary
(no fixed taxonomy). Writes a checkpoint JSONL of {image, raw_elements}
that Stage 2 (normalize.py) clusters into canonical categories.

Usage:
  python pipelines/salvage_analysis/extract.py --run answers_baseline
  python pipelines/salvage_analysis/extract.py --input path/to/run.jsonl --out path/to/checkpoint.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_run
from shared.ollama import call_ollama
from pipelines.salvage_analysis.records import get_field_text

PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT = (PROMPTS_DIR / "salvage_extract_system.txt").read_text(encoding="utf-8")
_USER_TEMPLATE = (PROMPTS_DIR / "salvage_extract_user.txt").read_text(encoding="utf-8")

RESULTS_IN = EVAL_ROOT.parent / "results" / "castor_results"
OUT_DIR = EVAL_ROOT / "results" / "p6_salvage_plan"

DEFAULT_MODEL = os.environ.get("CASTOR_SALVAGE_MODEL", "gemma4:31b-cloud")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

def build_extract_prompt(recovery_text: str) -> str:
    """Fill the user template with the recovery-plan text to extract from."""
    return _USER_TEMPLATE.format(recovery_text=recovery_text)


def parse_extract_result(parsed: dict) -> dict:
    """Normalize call_ollama's parsed dict (or None on failure) into
    {elements: list[str], parse_ok: bool}."""
    if parsed is None:
        return {"elements": [], "parse_ok": False}
    elements = parsed.get("elements")
    if not isinstance(elements, list):
        return {"elements": [], "parse_ok": False}
    return {"elements": [str(e) for e in elements], "parse_ok": True}


def build_output_record(image: str, result: dict) -> dict:
    """Assemble the Stage 1 checkpoint record for one image."""
    return {
        "image": image,
        "raw_elements": result["elements"],
        "parse_ok": result["parse_ok"],
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(input_path: Path, out_path: Path, model: str, url: str, limit=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: keep only successfully-parsed images, same pattern as
    # judge_panel/run_judge.py's resume logic.
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("parse_ok"):
                            done[r["image"]] = line
                    except json.JSONDecodeError:
                        pass
        if done:
            with open(out_path, "w", encoding="utf-8") as f:
                for l in done.values():
                    f.write(l + "\n")
        else:
            out_path.unlink()
        print(f"  Resume: {len(done)} already extracted.")

    records = load_run(input_path)
    if limit:
        records = records[:limit]

    pending = [r for r in records if r.get("image", "") not in done]
    if not pending:
        print(f"  All {len(done)} records already extracted — nothing to do.")
        return out_path

    print(f"  Extracting elements from {len(pending)} records with {model} ...")

    errors = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(pending):
            image = rec.get("image", "")
            recovery_text = get_field_text(rec, "recovery_considerations") or ""
            user_prompt = build_extract_prompt(recovery_text)
            parsed, _raw, _elapsed = call_ollama(_SYSTEM_PROMPT, user_prompt, model, url)
            result = parse_extract_result(parsed)
            output_rec = build_output_record(image, result)
            out_f.write(json.dumps(output_rec) + "\n")
            if not result["parse_ok"]:
                errors += 1
                print(f"  PARSE_FAIL [{i + 1}/{len(pending)}] {image[-50:]}")

    print(f"\n  Done. extracted={len(pending)}  skipped={len(done)}  parse_errors={errors}")
    print(f"  Output -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Stage 1: extract salvage elements from recovery plans (Pipeline 6)"
    )
    ap.add_argument("--run", help="Run name; resolves --input/--out defaults if not given explicitly")
    ap.add_argument("--input", type=Path, help="Full-answer inference JSONL (default: derived from --run)")
    ap.add_argument("--out", type=Path, help="Output checkpoint JSONL (default: derived from --run)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    ap.add_argument("--limit", type=int, default=None, help="Process only first N records (smoke test)")
    args = ap.parse_args()

    input_path = args.input or (RESULTS_IN / f"{args.run}.jsonl" if args.run else None)
    if input_path is None:
        ap.error("Either --run or --input must be given.")
    run_name = args.run or input_path.stem
    out_path = args.out or (OUT_DIR / f"raw_elements_{run_name}.jsonl")

    run(input_path, out_path, args.model, args.url, args.limit)


if __name__ == "__main__":
    main()
