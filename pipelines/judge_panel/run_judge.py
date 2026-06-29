"""
Pipeline 5 — per-model judge inference.

Loads a CASTOR inference JSONL + ground truth, runs each record through a
judge model, and writes a per-sample JSONL with scores and rationales.

Backends:
  ollama     — calls a local Ollama instance (for local testing)
  apptainer  — calls a model inside an Apptainer container (cluster)

Usage:
  python run_judge.py \\
      --model   qwen25_72b \\
      --input   ../../results/castor_results/answers_baseline.jsonl \\
      --out     ../../results/p5_judge/answers_baseline/ \\
      --backend ollama \\
      [--ollama-model qwen2.5:7b] \\
      [--limit N]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_ground_truth, load_run
from shared.ollama import call_ollama
from pipelines.judge_panel.preprocess import preprocess, PROMPTS_DIR

VALID_SCORES = {1, 2, 3}

_SYSTEM_PROMPT = (PROMPTS_DIR / "castor_judge_system.txt").read_text(encoding="utf-8")
_USER_TEMPLATE = (PROMPTS_DIR / "castor_judge_user.txt").read_text(encoding="utf-8")


def _coerce_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    return [str(val)]

_FENCE_RE = re.compile(r'^```(?:json)?\s*', re.MULTILINE)
_FENCE_END_RE = re.compile(r'\s*```\s*$')
_LATEX_RE = re.compile(r'\\([_\-/])')


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

def build_user_prompt(gt_fields: dict, pred_text: str) -> str:
    """Fill the user template with GT fields and cleaned prediction."""
    return _USER_TEMPLATE.format(
        gt_state=gt_fields.get("state", ""),
        gt_vessel_type=gt_fields.get("vessel_type", ""),
        gt_size_estimate=gt_fields.get("size_estimate", ""),
        gt_cargo=gt_fields.get("cargo", ""),
        pred_text=pred_text,
    )


def parse_judge_response(raw: str) -> dict:
    """Parse a judge model's raw text response into structured fields.

    Returns a dict with keys: score, rationale, hallucinations, parse_ok,
    and optionally raw_response on failure.
    """
    # Strip markdown fences
    cleaned = _FENCE_RE.sub('', raw.strip())
    cleaned = _FENCE_END_RE.sub('', cleaned).strip()
    cleaned = _LATEX_RE.sub(r'\1', cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            data = json.loads(cleaned, strict=False)
        except json.JSONDecodeError as e:
            return {
                "score": None, "rationale": "", "hallucinations": [],
                "parse_ok": False, "raw_response": raw[:500],
            }

    if not isinstance(data, dict):
        return {
            "score": None, "rationale": "", "hallucinations": [],
            "parse_ok": False, "raw_response": raw[:500],
        }

    score = data.get("final_score")
    if score not in VALID_SCORES:
        return {
            "score": None, "rationale": str(data.get("visual_alignment_rationale", "")),
            "hallucinations": _coerce_list(data.get("hallucinations_detected")),
            "parse_ok": False, "raw_response": raw[:500],
        }

    return {
        "score": int(score),
        "rationale": str(data.get("visual_alignment_rationale", "")),
        "hallucinations": _coerce_list(data.get("hallucinations_detected")),
        "parse_ok": True,
    }


def build_output_record(image: str, gt_state: str, pred_text: str,
                        verbosity_flagged: bool, judge_model: str,
                        parse_result: dict, elapsed_s: float) -> dict:
    """Assemble the final per-sample output record."""
    rec = {
        "image":            image,
        "gt_state":         gt_state,
        "pred_text":        pred_text,
        "verbosity_flagged": verbosity_flagged,
        "judge_model":      judge_model,
        "score":            parse_result["score"],
        "rationale":        parse_result.get("rationale", ""),
        "hallucinations":   parse_result.get("hallucinations", []),
        "parse_ok":         parse_result["parse_ok"],
        "elapsed_s":        round(elapsed_s, 3),
    }
    if not parse_result["parse_ok"] and "raw_response" in parse_result:
        rec["raw_response"] = parse_result["raw_response"]
    return rec


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

def _call_ollama_backend(user_prompt: str, model_name: str, ollama_url: str) -> tuple:
    return call_ollama(
        system=_SYSTEM_PROMPT,
        user=user_prompt,
        model=model_name,
        url=ollama_url,
        options={"temperature": 0, "num_predict": 512},
    )


def _call_apptainer_backend(user_prompt: str, sif_path: str,
                             model_dir: str, tp_size: int = 1) -> tuple:
    """Run judge inference inside an Apptainer container via vLLM subprocess."""
    import subprocess, tempfile

    payload = json.dumps({"system": _SYSTEM_PROMPT, "user": user_prompt})
    data_dir = os.path.expandvars("/data/$USER")

    cmd = [
        "apptainer", "exec", "--containall", "--nv",
        "--bind", f"{data_dir}:{data_dir}",
        sif_path,
        "/opt/conda/bin/python3", "-c",
        f"""
import json, sys
from vllm import LLM, SamplingParams
payload = json.loads(sys.stdin.read())
llm = LLM(model="{model_dir}", tensor_parallel_size={tp_size})
params = SamplingParams(temperature=0, max_tokens=512)
out = llm.chat([
    {{"role": "system", "content": payload["system"]}},
    {{"role": "user",   "content": payload["user"]}},
], params)
print(out[0].outputs[0].text)
""",
    ]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, input=payload, capture_output=True, text=True, timeout=300
        )
        elapsed = time.perf_counter() - t0
        raw = proc.stdout.strip()
        return None, raw, elapsed   # parsed dict not returned here; parse_judge_response handles it
    except subprocess.TimeoutExpired:
        return None, "APPTAINER_TIMEOUT", time.perf_counter() - t0
    except Exception as e:
        return None, f"APPTAINER_ERROR: {e}", time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_MODEL_CONFIG = {
    "qwen25_72b":   {"gpus": 1, "tp": 1, "sif": "castor_qwen_judge.sif",  "dir": "qwen25-72b-instruct"},
    "deepseek_r1":  {"gpus": 1, "tp": 1, "sif": "castor_deepseek.sif",    "dir": "deepseek-r1-distill-llama-70b"},
    "gptoss_120b":  {"gpus": 2, "tp": 2, "sif": "castor_gptoss.sif",      "dir": "gpt-oss-120b"},
}


def run(input_path: Path, gt_path: Path, out_dir: Path, model: str,
        backend: str, ollama_model: str, ollama_url: str, limit: int | None):

    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = input_path.stem
    out_path = out_dir / f"{run_name}_{model}.jsonl"

    # Resume: collect already-scored images
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("parse_ok"):
                            done.add(r["image"])
                    except json.JSONDecodeError:
                        pass
        print(f"  Resume: {len(done)} already scored.")

    gt = load_ground_truth(gt_path)
    records = load_run(input_path)
    if limit:
        records = records[:limit]

    data_dir = os.path.expandvars("/data/$USER")
    cfg = _MODEL_CONFIG.get(model, {})

    skipped = errors = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            image = rec.get("image", "")
            if image in done:
                skipped += 1
                continue

            pred_text = rec.get("text", "")
            gt_fields = gt.get(image, {})
            gt_state  = gt_fields.get("state", "")

            pp = preprocess(pred_text, gt_fields)
            user_prompt = build_user_prompt(gt_fields, pp.clean_pred)

            if backend == "ollama":
                _, raw, elapsed = _call_ollama_backend(user_prompt, ollama_model, ollama_url)
            else:
                sif  = f"{data_dir}/{cfg['sif']}"
                mdir = f"{data_dir}/{cfg['dir']}"
                _, raw, elapsed = _call_apptainer_backend(user_prompt, sif, mdir, cfg.get("tp", 1))

            parse_result = parse_judge_response(raw if isinstance(raw, str) else "")
            output_rec = build_output_record(
                image, gt_state, pred_text, pp.verbosity_flagged,
                model, parse_result, elapsed,
            )

            out_f.write(json.dumps(output_rec) + "\n")
            out_f.flush()

            status = f"score={output_rec['score']}" if output_rec["parse_ok"] else "PARSE_FAIL"
            if (i + 1) % 10 == 0 or not output_rec["parse_ok"]:
                print(f"  [{i+1}/{len(records)}] {image[-40:]:40s}  {status}  {elapsed:.1f}s")
            if not output_rec["parse_ok"]:
                errors += 1

    print(f"\n  Done. scored={len(records)-skipped}  skipped={skipped}  parse_errors={errors}")
    print(f"  Output -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Run one judge model over a CASTOR JSONL.")
    ap.add_argument("--model",        required=True, choices=list(_MODEL_CONFIG),
                    help="Judge model key")
    ap.add_argument("--input",        required=True, type=Path,
                    help="Inference JSONL to score")
    ap.add_argument("--out",          required=True, type=Path,
                    help="Output directory")
    ap.add_argument("--gt",           type=Path,
                    default=EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv",
                    help="Ground truth CSV")
    ap.add_argument("--backend",      default="ollama", choices=["ollama", "apptainer"])
    ap.add_argument("--ollama-model", default="qwen2.5:7b",
                    help="Ollama model tag (--backend ollama only)")
    ap.add_argument("--ollama-url",   default=os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat")
    ap.add_argument("--limit",        type=int, default=None,
                    help="Process only first N records (smoke test)")
    args = ap.parse_args()

    run(
        input_path=args.input,
        gt_path=args.gt,
        out_dir=args.out,
        model=args.model,
        backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
