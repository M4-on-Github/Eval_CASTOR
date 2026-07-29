"""
Pipeline 8 — per-model plan coherence judge.

For each VLM salvage plan, parses numbered steps then asks the judge model
whether each step is operationally valid and correctly sequenced, given the
GT disaster state and all prior steps as context. One vLLM call per
(image, step) pair — ~660 calls per run per judge (110 images × ~6 steps avg).

No image is passed to the judge. The GT state label is the only anchor.

Usage (inside Apptainer via coherence_judge_job.sh):
  python3 run_coherence_judge.py \\
      --model     deepseek_r1_32b \\
      --model-dir /data/$USER/deepseek-r1-distill-qwen-32b-awq \\
      --input     /path/to/answers_baseline.jsonl \\
      --out       /path/to/results/p8_plan_coherence/ \\
      --gt        /path/to/human_gt.csv \\
      [--limit N]
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_ground_truth, load_run
from pipelines.plan_coherence.parse_steps import parse_steps

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

_MODEL_CONFIG = {
    "deepseek_r1_32b": {
        "dir": "deepseek-r1-distill-qwen-32b-awq",
        "max_model_len": 4096, "max_tokens": 64,
    },
    "glm4_32b": {
        "dir": "glm-4-32b-0414-gptq",
        "max_model_len": 4096, "max_tokens": 64,
    },
    "llama_3_3_70b": {
        "dir": "llama-3.3-70b-instruct-w4a16",
        "max_model_len": 4096, "max_tokens": 64,
    },
    "phi4_14b": {
        "dir": "phi-4-w4a16",
        "max_model_len": 4096, "max_tokens": 64,
    },
    "gemma4_31b": {
        "dir": "gemma4-31b-it-w4a16",
        "max_model_len": 4096, "max_tokens": 64,
    },
}

_COHERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "valid":  {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 300},
    },
    "required": ["valid", "reason"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are a maritime salvage operations expert. "
    "Evaluate whether a specific step in a salvage plan is operationally "
    "valid and correctly sequenced, given the disaster type and prior steps.\n\n"
    "Mark a step INVALID if any of the following apply:\n"
    "- Wrong order: the step depends on a precondition (stability check, "
    "gas-testing, tank/cargo check, personnel clearance) that a prior step "
    "should have established but did not.\n"
    "- Skipped precondition: the step performs an action (entering a space, "
    "hot work, righting, refloating, suppression) without a required safety "
    "or assessment step coming first, even if no prior steps exist yet.\n"
    "- Wrong method: the step's action or resource is operationally incorrect "
    "for the stated casualty type or space (e.g. wrong suppression agent, "
    "wrong lifting/righting method for vessel size).\n"
    "- Non-actionable filler: the step names no concrete equipment, resource, "
    "or role and describes no specific action (e.g. \"assess the situation\", "
    "\"establish a perimeter\").\n\n"
    "Mark a step VALID if it is a concrete, operationally correct action whose "
    "prerequisites (if any) are satisfied by the steps completed so far.\n\n"
    "Examples:\n\n"
    "CASUALTY TYPE: capsized\n"
    "STEPS COMPLETED SO FAR: (none)\n"
    "STEP TO EVALUATE: 1. Verify the vessel's stability against minimum "
    "righting-arm and metacentric-height (GM) criteria before attempting to "
    "right the vessel.\n"
    "{\"valid\": true, \"reason\": \"Stability must be confirmed before "
    "righting; this is a required precondition and is correctly first.\"}\n\n"
    "CASUALTY TYPE: sunken\n"
    "STEPS COMPLETED SO FAR:\n"
    "  1. Refloat the vessel using lift bags.\n"
    "STEP TO EVALUATE: 2. Assess the hull for structural damage and resolve "
    "any stability problems.\n"
    "{\"valid\": false, \"reason\": \"Structural and stability assessment "
    "must occur before refloating, not after; the steps are reversed.\"}\n\n"
    "CASUALTY TYPE: sunken\n"
    "STEPS COMPLETED SO FAR: (none)\n"
    "STEP TO EVALUATE: 1. Send a diver into the flooded engine room to begin "
    "hot-cutting the damaged hull plate.\n"
    "{\"valid\": false, \"reason\": \"The space must be tested for explosive "
    "gas, then oxygen, then toxic gas before any diver enters or hot work "
    "begins; this step skips that precondition.\"}\n\n"
    "CASUALTY TYPE: on_fire\n"
    "STEPS COMPLETED SO FAR:\n"
    "  1. Confirm the machinery space is clear of personnel.\n"
    "STEP TO EVALUATE: 2. Apply standard foam to suppress the engine room "
    "fire.\n"
    "{\"valid\": false, \"reason\": \"A machinery space fire should be "
    "suppressed by gas flooding, not foam; foam is the wrong method for "
    "this space.\"}\n\n"
    "CASUALTY TYPE: aground\n"
    "STEPS COMPLETED SO FAR: (none)\n"
    "STEP TO EVALUATE: 1. Assess the situation and establish a perimeter.\n"
    "{\"valid\": false, \"reason\": \"This step names no equipment, "
    "resource, or role and gives no concrete action; it is generic filler, "
    "not an operational step.\"}\n\n"
    "CASUALTY TYPE: aground\n"
    "STEPS COMPLETED SO FAR:\n"
    "  1. Confirm all tanks for fuel and hazardous cargo before any cutting "
    "or movement begins.\n"
    "  2. Determine the ground reaction and friction coefficient for the "
    "seabed type.\n"
    "STEP TO EVALUATE: 3. Deploy a salvage tug to apply the calculated "
    "pulling force to free the vessel from a sand seabed.\n"
    "{\"valid\": true, \"reason\": \"Pulling force is correctly applied "
    "after tank checks and ground-reaction calculation, and the resource "
    "matches the required action.\"}\n\n"
    "Now evaluate the following step. Answer only in the JSON format "
    "specified."
)


def _user_prompt(gt_state: str, prior_steps: list[tuple[int, str]],
                 step_num: int, step_text: str) -> str:
    parts = [f"CASUALTY TYPE: {gt_state}\n"]
    if prior_steps:
        parts.append("STEPS COMPLETED SO FAR:")
        for num, text in prior_steps:
            parts.append(f"  {num}. {text}")
        parts.append("")
    parts.append("STEP TO EVALUATE:")
    parts.append(f"  {step_num}. {step_text}")
    parts.append("")
    parts.append(
        f'Is this step (a) operationally valid for maritime salvage of a '
        f'"{gt_state}" casualty, and (b) correctly sequenced given the prior steps?\n'
        f'Respond: {{"valid": true, "reason": "one sentence"}} or '
        f'{{"valid": false, "reason": "one sentence"}}'
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# vLLM batch inference
# ---------------------------------------------------------------------------

def _run_vllm_batch(prompts: list[str], model_dir: str,
                    max_model_len: int, max_tokens: int) -> list[dict | None]:
    from vllm import LLM, SamplingParams

    # API changed in vLLM 0.12: GuidedDecodingParams → StructuredOutputsParams,
    # guided_decoding= → structured_outputs=. Both live in vllm.sampling_params.
    try:
        from vllm.sampling_params import GuidedDecodingParams
        guided_kwargs = {"guided_decoding": GuidedDecodingParams(json=_COHERENCE_SCHEMA)}
    except ImportError:
        from vllm.sampling_params import StructuredOutputsParams
        guided_kwargs = {"structured_outputs": StructuredOutputsParams(json=_COHERENCE_SCHEMA)}

    print(f"  [vLLM] Loading model from {model_dir} ...")
    llm = LLM(
        model=model_dir,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        **guided_kwargs,
    )
    conversations = [
        [{"role": "system", "content": _SYSTEM_PROMPT},
         {"role": "user",   "content": p}]
        for p in prompts
    ]
    print(f"  [vLLM] Scoring {len(conversations)} (image, step) pairs ...")
    t0 = time.perf_counter()
    outputs = llm.chat(conversations, sampling_params=params)
    elapsed = time.perf_counter() - t0
    print(f"  [vLLM] Done in {elapsed:.1f}s ({elapsed/max(len(outputs),1):.2f}s/call avg)")

    results = []
    for o in outputs:
        raw = o.outputs[0].text.strip()
        try:
            results.append(json.loads(raw))
        except (json.JSONDecodeError, AttributeError):
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: Path, gt_path: Path, out_dir: Path,
        model_key: str, model_dir: str, limit: Optional[int]):

    run_name = input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_{model_key}.csv"

    cfg = _MODEL_CONFIG[model_key]

    # ── Resume: skip already-processed images ─────────────────────────────────
    done_images: set[str] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_images.add(row["image"])
                existing_rows.append(row)
        if done_images:
            print(f"  Resume: {len(done_images)} images already processed.")

    gt      = load_ground_truth(gt_path)
    records = load_run(input_path)
    if limit:
        records = records[:limit]

    pending = [r for r in records if r.get("image", "") not in done_images]
    if not pending:
        print("  All records already processed — nothing to do.")
        return

    # ── Build flat prompt list ─────────────────────────────────────────────────
    batch_meta: list[tuple[str, int, str, str]] = []  # (image, step_num, step_text, gt_state)
    batch_prompts: list[str] = []
    skipped_no_steps = 0

    for rec in pending:
        image     = rec.get("image", "")
        plan_text = rec.get("text", "")
        gt_state  = gt.get(image, {}).get("state", "unknown")
        steps     = parse_steps(plan_text)

        if not steps:
            skipped_no_steps += 1
            print(f"  WARNING: no steps parsed for {image} — skipping")
            continue

        for i, (step_num, step_text) in enumerate(steps):
            prior = steps[:i]
            prompt = _user_prompt(gt_state, prior, step_num, step_text)
            batch_meta.append((image, step_num, step_text, gt_state))
            batch_prompts.append(prompt)

    if not batch_prompts:
        print(f"  No prompts to score (all {skipped_no_steps} images had no steps).")
        return

    print(f"  {len(pending) - skipped_no_steps} images → {len(batch_prompts)} (image, step) pairs")

    # ── vLLM batch pass ───────────────────────────────────────────────────────
    results = _run_vllm_batch(
        batch_prompts, model_dir,
        max_model_len=cfg["max_model_len"],
        max_tokens=cfg["max_tokens"],
    )

    # ── Build per-step rows ────────────────────────────────────────────────────
    parse_errors = 0
    new_rows: list[dict] = []
    for (image, step_num, step_text, gt_state), result in zip(batch_meta, results):
        if result is None:
            parse_errors += 1
            valid_val  = "error"
            reason_val = ""
        else:
            valid_val  = "1" if result.get("valid") else "0"
            reason_val = str(result.get("reason", ""))
        new_rows.append({
            "image":    image,
            "gt_state": gt_state,
            "step_num": step_num,
            "step_text": step_text,
            "valid":    valid_val,
            "reason":   reason_val,
        })

    if parse_errors:
        print(f"  WARNING: {parse_errors} parse failures (recorded as 'error')")

    # ── Write CSV (full rewrite — new rows appended after existing) ────────────
    fieldnames = ["image", "gt_state", "step_num", "step_text", "valid", "reason"]
    all_rows = existing_rows + new_rows
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  {len(new_rows)} step rows written → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="P8: per-model plan coherence judge (runs inside Apptainer)"
    )
    ap.add_argument("--model",     required=True, choices=list(_MODEL_CONFIG),
                    help="Judge model key")
    ap.add_argument("--model-dir", required=True,
                    help="Absolute path to model weights directory")
    ap.add_argument("--input",     required=True, type=Path,
                    help="Inference JSONL (text field = salvage plan)")
    ap.add_argument("--out",       required=True, type=Path,
                    help="Output directory")
    ap.add_argument("--gt",        type=Path,
                    default=EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv",
                    help="Ground truth CSV")
    ap.add_argument("--limit",     type=int, default=None,
                    help="Process only first N images (smoke test)")
    args = ap.parse_args()

    run(
        input_path=args.input,
        gt_path=args.gt,
        out_dir=args.out / args.input.stem,
        model_key=args.model,
        model_dir=args.model_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
