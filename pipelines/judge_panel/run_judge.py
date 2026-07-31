"""
Pipeline 5 — per-model judge inference.

Loads a CASTOR inference JSONL + ground truth, runs all records through a
judge model in one vLLM batch pass, and writes a per-sample JSONL with
scores and rationales.

This script is designed to run INSIDE an Apptainer container on the pleiades
cluster. See containers/submit_judge_job.sh for the SLURM invocation.

Usage (inside container via submit_judge_job.sh):
  python3 run_judge.py \\
      --model   deepseek_r1_32b \\
      --model-dir /data/$USER/deepseek-r1-distill-qwen-32b-awq \\
      --input   /data/$USER/castor_results/answers_baseline.jsonl \\
      --out     /data/$USER/castor_results/p5_judge/answers_baseline/ \\
      [--tp 1] \\
      [--limit N]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

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
_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_THINK_OPEN_RE = re.compile(r'<think>.*', re.DOTALL)
# Matches the outermost {...} block — used to extract JSON from responses that
# include preamble or trailing text (e.g. DeepSeek-R1 after the think block).
_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


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
    cleaned = _THINK_RE.sub('', raw).strip()
    cleaned = _THINK_OPEN_RE.sub('', cleaned).strip()
    cleaned = _FENCE_RE.sub('', cleaned)
    cleaned = _FENCE_END_RE.sub('', cleaned).strip()
    cleaned = _LATEX_RE.sub(r'\1', cleaned)

    def _try_parse(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                return json.loads(s, strict=False)
            except json.JSONDecodeError:
                return None

    data = _try_parse(cleaned)
    if data is None:
        # Model output has preamble/trailing text — extract the first {...} block.
        m = _JSON_OBJECT_RE.search(cleaned)
        if m:
            data = _try_parse(m.group(0))
    if data is None:
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
            "state_correct": None, "vessel_type_correct": None,
            "size_correct": None, "cargo_correct": None,
            "parse_ok": False, "raw_response": raw[:500],
        }

    def _bool(key) -> bool | None:
        v = data.get(key)
        return bool(v) if isinstance(v, bool) else None

    return {
        "score":               int(score),
        "rationale":           str(data.get("visual_alignment_rationale", "")),
        "hallucinations":      _coerce_list(data.get("hallucinations_detected")),
        "state_correct":       _bool("state_correct"),
        "vessel_type_correct": _bool("vessel_type_correct"),
        "size_correct":        _bool("size_correct"),
        "cargo_correct":       _bool("cargo_correct"),
        "parse_ok":            True,
    }


def make_record_id(image: str, model_tag: str = "", method: str = "",
                   prompt_stem: str = "") -> str:
    """Stable composite key for a single inference record."""
    return f"{image}||{model_tag}||{method}||{prompt_stem}"


def build_output_record(image: str, gt_state: str, pred_text: str,
                        verbosity_flagged: bool, judge_model: str,
                        parse_result: dict, elapsed_s: float,
                        model_tag: str = "", method: str = "",
                        prompt_stem: str = "") -> dict:
    """Assemble the final per-sample output record."""
    rec = {
        "record_id":           make_record_id(image, model_tag, method, prompt_stem),
        "image":               image,
        "model_tag":           model_tag,
        "method":              method,
        "prompt_stem":         prompt_stem,
        "gt_state":            gt_state,
        "pred_text":           pred_text,
        "verbosity_flagged":   verbosity_flagged,
        "judge_model":         judge_model,
        "score":               parse_result["score"],
        "rationale":           parse_result.get("rationale", ""),
        "hallucinations":      parse_result.get("hallucinations", []),
        "state_correct":       parse_result.get("state_correct"),
        "vessel_type_correct": parse_result.get("vessel_type_correct"),
        "size_correct":        parse_result.get("size_correct"),
        "cargo_correct":       parse_result.get("cargo_correct"),
        "parse_ok":            parse_result["parse_ok"],
        "elapsed_s":           round(elapsed_s, 3),
    }
    if not parse_result["parse_ok"] and "raw_response" in parse_result:
        rec["raw_response"] = parse_result["raw_response"]
    return rec


# ---------------------------------------------------------------------------
# vLLM batch inference (runs inside the container)
# ---------------------------------------------------------------------------

# vLLM guided-decoding schema — applied to the tokens AFTER </think> so the
# final answer is structurally enforced to be valid JSON matching our rubric.
_JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_alignment_rationale": {"type": "string", "maxLength": 600},
        "hallucinations_detected":    {"type": "array", "items": {"type": "string"}},
        "state_correct":              {"type": "boolean"},
        "vessel_type_correct":        {"type": "boolean"},
        "size_correct":               {"type": "boolean"},
        "cargo_correct":              {"type": "boolean"},
        "final_score":                {"type": "integer", "enum": [1, 2, 3]},
    },
    "required": [
        "visual_alignment_rationale", "hallucinations_detected",
        "state_correct", "vessel_type_correct", "size_correct", "cargo_correct",
        "final_score",
    ],
    "additionalProperties": False,
}

_MODEL_CONFIG = {
    # DeepSeek-R1-Distill-Qwen-32B AWQ: ~22 GB weights, 1 GPU.
    # guided_json blocks <think> tokens from token 1 and enforces valid JSON output
    # directly — avoids the 100% PARSE_FAIL seen when the 70B variant emitted prose
    # reasoning outside <think> tags and exhausted max_tokens before reaching JSON.
    "deepseek_r1_32b": {
        "tp": 1, "dir": "deepseek-r1-distill-qwen-32b-awq",
        "quantization": None, "max_model_len": 8192, "max_tokens": 2048,
        "guided_json": _JUDGE_JSON_SCHEMA,
    },
    # GLM-4-32B-0414 GPTQ W4A16: ~22 GB weights, 1 GPU. GPTQ backend supports
    # guided decoding via logit processors (no kernel conflict with marlin).
    "glm4_32b": {
        "tp": 1, "dir": "glm-4-32b-0414-gptq",
        "quantization": None, "max_model_len": 4096, "max_tokens": 1024,
        "guided_json": _JUDGE_JSON_SCHEMA,
    },
    # Atla Selene Mini 8B AWQ (self-quantized from AtlaAI/Selene-1-Mini-Llama-3.1-8B): ~7 GB.
    # Purpose-built judge (LlamaForCausalLM backbone, #1 RewardBench at 8B class).
    "selene_mini_8b": {
        "tp": 1, "dir": "selene-1-mini-llama-3.1-8b-awq",
        "quantization": None, "max_model_len": 4096, "max_tokens": 1024,
        "guided_json": _JUDGE_JSON_SCHEMA,
    },
}


def _run_vllm_batch(user_prompts: list, model_dir: str, tp_size: int,
                    pp_size: int = 1,
                    quantization: Optional[str] = None,
                    max_model_len: int = 4096,
                    max_tokens: int = 512,
                    prefill: Optional[str] = None,
                    guided_json: Optional[dict] = None,
                    tokenizer_mode: str = "auto") -> list:
    """Load model once and score all prompts in one batch via vLLM."""
    from vllm import LLM, SamplingParams

    print(f"  [vLLM] Loading model from {model_dir} (tp={tp_size}, pp={pp_size}, quant={quantization or 'auto'}, max_model_len={max_model_len}, guided_json={guided_json is not None}, tokenizer_mode={tokenizer_mode}) ...")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        dtype="auto",
        quantization=quantization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        tokenizer_mode=tokenizer_mode,
    )
    sampling_kwargs = {"temperature": 0.0, "max_tokens": max_tokens}
    if guided_json is not None:
        from vllm.sampling_params import GuidedDecodingParams
        sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=guided_json)
    params = SamplingParams(**sampling_kwargs)

    if prefill:
        # vLLM's chat() passes add_generation_prompt=True internally, which
        # conflicts with continue_final_message=True in transformers ≥4.44.
        # Workaround: apply the chat template manually, append the prefill
        # text, then call llm.generate() which has no such conflict.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,
                                            local_files_only=True)
        raw_prompts = []
        for up in user_prompts:
            msgs = [{"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": up}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            raw_prompts.append(text + prefill)
        print(f"  [vLLM] Scoring {len(raw_prompts)} records (prefill: {prefill!r}) ...")
        outputs = llm.generate(raw_prompts, sampling_params=params)
    else:
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
        model_dir: str, tp_size: int, limit: Optional[int]):

    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = input_path.stem
    out_path = out_dir / f"{run_name}_{model}.jsonl"

    # Resume: collect already-scored records; key on record_id (composite) so
    # multiple combos per image are all tracked independently.
    done = {}  # record_id -> json line (string)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("parse_ok"):
                            rid = r.get("record_id") or r.get("image", "")
                            done[rid] = line
                    except json.JSONDecodeError:
                        pass
        if done:
            with open(out_path, "w", encoding="utf-8") as f:
                for l in done.values():
                    f.write(l + "\n")
        else:
            out_path.unlink()
        print(f"  Resume: {len(done)} already scored.")

    gt = load_ground_truth(gt_path)
    records = load_run(input_path)
    if limit:
        records = records[:limit]

    pending = [r for r in records
               if make_record_id(r.get("image", ""), r.get("model_tag", ""),
                                 r.get("method", ""), r.get("prompt_stem", ""))
               not in done]
    if not pending:
        print(f"  All {len(done)} records already scored — nothing to do.")
        return out_path

    print(f"  Scoring {len(pending)} records with {model} ...")

    # Preprocess all records and build prompts
    images, gt_states, pred_texts, vflags, user_prompts = [], [], [], [], []
    model_tags, methods, prompt_stems = [], [], []
    for rec in pending:
        image      = rec.get("image", "")
        pred_text  = rec.get("text", "")
        model_tag  = rec.get("model_tag", "")
        method     = rec.get("method", "")
        prompt_stem = rec.get("prompt_stem", "")
        gt_fields  = gt.get(image, {})
        pp         = preprocess(pred_text, gt_fields)
        up         = build_user_prompt(gt_fields, pp.clean_pred)
        images.append(image)
        gt_states.append(gt_fields.get("state", ""))
        pred_texts.append(pred_text)
        vflags.append(pp.verbosity_flagged)
        user_prompts.append(up)
        model_tags.append(model_tag)
        methods.append(method)
        prompt_stems.append(prompt_stem)

    # Single vLLM batch pass
    cfg = _MODEL_CONFIG.get(model, {})
    t0 = time.perf_counter()
    raw_responses = _run_vllm_batch(user_prompts, model_dir, tp_size,
                                    pp_size=cfg.get("pp", 1),
                                    quantization=cfg.get("quantization"),
                                    max_model_len=cfg.get("max_model_len", 4096),
                                    max_tokens=cfg.get("max_tokens", 512),
                                    prefill=cfg.get("prefill"),
                                    guided_json=cfg.get("guided_json"),
                                    tokenizer_mode=cfg.get("tokenizer_mode", "auto"))
    total_elapsed = time.perf_counter() - t0
    per_s = total_elapsed / len(raw_responses) if raw_responses else 0.0

    # Show first raw response so we can see exactly what the model outputs.
    if raw_responses:
        print(f"  [DEBUG] raw_responses[0] ({len(raw_responses[0])} chars): {raw_responses[0][:400]!r}")

    # Parse and append results
    errors = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, (image, gt_state, pred_text, vflag, raw, mt, meth, ps) in enumerate(
            zip(images, gt_states, pred_texts, vflags, raw_responses,
                model_tags, methods, prompt_stems)
        ):
            parse_result = parse_judge_response(raw if isinstance(raw, str) else "")
            output_rec   = build_output_record(
                image, gt_state, pred_text, vflag, model, parse_result, per_s,
                model_tag=mt, method=meth, prompt_stem=ps,
            )
            out_f.write(json.dumps(output_rec) + "\n")
            if not parse_result["parse_ok"]:
                errors += 1
                print(f"  PARSE_FAIL [{i+1}/{len(pending)}] {image[-50:]}")
                print(f"    raw ({len(raw)} chars): {raw[:600]!r}")
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
                    help="Judge model key (deepseek_r1_32b / glm4_32b / selene_mini_8b)")
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
