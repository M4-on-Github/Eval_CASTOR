"""
Pipeline 5 — per-model judge inference.

Loads a CASTOR inference JSONL + ground truth, runs all records through a
judge model in one vLLM batch pass, and writes a per-sample JSONL with
scores and rationales.

This script is designed to run INSIDE an Apptainer container on the pleiades
cluster. See containers/submit_judge_job.sh for the SLURM invocation.

Usage (inside container via submit_judge_job.sh):
  python3 run_judge.py \\
      --model   qwen25_72b \\
      --model-dir /data/$USER/qwen25-72b-instruct \\
      --input   ../../results/castor_results/answers_baseline.jsonl \\
      --out     ../../results/p5_judge/answers_baseline/ \\
      [--tp 1] \\
      [--limit N]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_ground_truth, load_run
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
    cleaned = _FENCE_RE.sub('', raw.strip())
    cleaned = _FENCE_END_RE.sub('', cleaned).strip()
    cleaned = _LATEX_RE.sub(r'\1', cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            data = json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
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
        "image":             image,
        "gt_state":          gt_state,
        "pred_text":         pred_text,
        "verbosity_flagged": verbosity_flagged,
        "judge_model":       judge_model,
        "score":             parse_result["score"],
        "rationale":         parse_result.get("rationale", ""),
        "hallucinations":    parse_result.get("hallucinations", []),
        "parse_ok":          parse_result["parse_ok"],
        "elapsed_s":         round(elapsed_s, 3),
    }
    if not parse_result["parse_ok"] and "raw_response" in parse_result:
        rec["raw_response"] = parse_result["raw_response"]
    return rec


# ---------------------------------------------------------------------------
# vLLM batch inference (runs inside the container)
# ---------------------------------------------------------------------------

_MODEL_CONFIG = {
    # AWQ 4-bit quantized variants: ~40 GB each, fit on 1× RTX 6000 Ada (48 GB)
    "qwen25_72b":  {"tp": 1, "dir": "qwen25-72b-instruct-awq"},
    "deepseek_r1": {"tp": 1, "dir": "deepseek-r1-distill-llama-70b-awq"},
    # GPT-OSS 120B: unsupported by vLLM 0.5.5 — see containers/NOTES.md
    "gptoss_120b": {"tp": 2, "dir": "gpt-oss-120b"},
}


def _run_vllm_batch(user_prompts: list, model_dir: str, tp_size: int) -> list:
    """Load model once and score all prompts in one batch via vLLM."""
    from vllm import LLM, SamplingParams

    print(f"  [vLLM] Loading model from {model_dir} (tp={tp_size}) ...")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp_size,
        dtype="auto",           # auto-detects AWQ/GPTQ quantization from model config
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    params = SamplingParams(temperature=0.0, max_tokens=512)

    conversations = [
        [{"role": "system", "content": _SYSTEM_PROMPT},
         {"role": "user",   "content": up}]
        for up in user_prompts
    ]

    print(f"  [vLLM] Scoring {len(conversations)} records ...")
    outputs = llm.chat(conversations, sampling_params=params)
    return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(input_path: Path, gt_path: Path, out_dir: Path, model: str,
        model_dir: str, tp_size: int, limit: int | None):

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

    pending = [r for r in records if r.get("image", "") not in done]
    if not pending:
        print(f"  All {len(done)} records already scored — nothing to do.")
        return out_path

    print(f"  Scoring {len(pending)} records with {model} ...")

    # Preprocess all records and build prompts
    images, gt_states, pred_texts, vflags, user_prompts = [], [], [], [], []
    for rec in pending:
        image     = rec.get("image", "")
        pred_text = rec.get("text", "")
        gt_fields = gt.get(image, {})
        pp        = preprocess(pred_text, gt_fields)
        up        = build_user_prompt(gt_fields, pp.clean_pred)
        images.append(image)
        gt_states.append(gt_fields.get("state", ""))
        pred_texts.append(pred_text)
        vflags.append(pp.verbosity_flagged)
        user_prompts.append(up)

    # Single vLLM batch pass
    t0 = time.perf_counter()
    raw_responses = _run_vllm_batch(user_prompts, model_dir, tp_size)
    total_elapsed = time.perf_counter() - t0
    per_s = total_elapsed / len(raw_responses) if raw_responses else 0.0

    # Parse and append results
    errors = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, (image, gt_state, pred_text, vflag, raw) in enumerate(
            zip(images, gt_states, pred_texts, vflags, raw_responses)
        ):
            parse_result = parse_judge_response(raw if isinstance(raw, str) else "")
            output_rec   = build_output_record(
                image, gt_state, pred_text, vflag, model, parse_result, per_s,
            )
            out_f.write(json.dumps(output_rec) + "\n")
            if not parse_result["parse_ok"]:
                errors += 1
                print(f"  PARSE_FAIL [{i+1}/{len(pending)}] {image[-50:]}")
            elif (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(pending)}] score={output_rec['score']}")

    print(f"\n  Done. scored={len(pending)}  skipped={len(done)}  parse_errors={errors}")
    print(f"  Batch time: {total_elapsed:.1f}s  ({per_s:.2f}s/record avg)")
    print(f"  Output -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Run one judge model over a CASTOR JSONL (runs inside Apptainer container)."
    )
    ap.add_argument("--model",     required=True, choices=list(_MODEL_CONFIG),
                    help="Judge model key (qwen25_72b / deepseek_r1 / gptoss_120b)")
    ap.add_argument("--model-dir", required=True,
                    help="Absolute path to HuggingFace model weights directory")
    ap.add_argument("--tp",        type=int, default=None,
                    help="Tensor-parallel size (default: from model config)")
    ap.add_argument("--input",     required=True, type=Path,
                    help="Inference JSONL to score")
    ap.add_argument("--out",       required=True, type=Path,
                    help="Output directory for judge JSONL")
    ap.add_argument("--gt",        type=Path,
                    default=EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv",
                    help="Ground truth CSV")
    ap.add_argument("--limit",     type=int, default=None,
                    help="Process only first N records (smoke test)")
    args = ap.parse_args()

    cfg     = _MODEL_CONFIG[args.model]
    tp_size = args.tp if args.tp is not None else cfg["tp"]

    run(
        input_path=args.input,
        gt_path=args.gt,
        out_dir=args.out,
        model=args.model,
        model_dir=args.model_dir,
        tp_size=tp_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
