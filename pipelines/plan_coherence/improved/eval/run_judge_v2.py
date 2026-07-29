"""
run_judge_v2.py — Qwen2.5-72B-Instruct-AWQ judge for MTH + SPC evaluation.

Design decisions vs. original run_coherence_judge.py:
  - Single model (Qwen2.5-72B-AWQ on 1x RTX6000Ada) instead of 5-model panel
  - max_tokens=256 (was 64 — old value truncated DeepSeek-R1 reasoning)
  - Decomposed schema: {method_valid: bool, specific: bool, reason: str}
  - Rubric injected verbatim into system prompt
  - Dual-track: evaluates each plan under both GT state and VLM-predicted state
  - vLLM batched inference with guided JSON decoding

Input:  one or more JSONL files produced by run_inference.py
        Each record must have: question_id, image, text, condition, method

Output: results/judge_scores_improved.jsonl
        One record per (question_id, condition, eval_track) with SEQ + MTH + SPC scores

Usage:
    python improved/eval/run_judge_v2.py --config improved/config.yaml
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# parents[1] = improved/ dir, so "from eval.X" resolves to improved/eval/X.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.parse_steps_v2 import parse_steps_v2
from eval.extract_condition import extract_condition
from eval.check_seq import check_seq


# ---------------------------------------------------------------------------
# Rubric injected into the judge system prompt
# ---------------------------------------------------------------------------

RUBRIC_MTH = """
METHOD (MTH) rubric — mark method_valid=true if ALL apply:
  1. The technique chosen matches the vessel's actual casualty type and observed conditions.
  2. If the plan names a specific resource (tug, crane, foam, CO2, etc.), the resource is
     appropriate for the casualty type and condition (not just generically available).
  3. The plan does NOT apply a technique that is explicitly wrong for the condition
     (e.g. parbuckling a small craft, standard foam on a polar-solvent fire,
      fireboat monitors directed into the fire space, CO2 in an open cargo deck fire).
"""

RUBRIC_SPC = """
SPECIFICITY (SPC) rubric — mark specific=true if ALL apply:
  1. At least one step names a specific technique (not just 'assess', 'plan', or 'coordinate').
  2. At least one step names a specific resource type (not just 'equipment' or 'resources').
  3. The plan gives actionable decision criteria for at least one choice (e.g. "if depth >30 m,
     use crane-and-barge", "if GM < minimum threshold, abort righting").
"""

SYSTEM_PROMPT = f"""You are a maritime salvage expert evaluating AI-generated salvage plans.
You will receive the plan and the vessel's casualty state.
Evaluate METHOD and SPECIFICITY using the rubrics below, then output ONLY valid JSON.

{RUBRIC_MTH}

{RUBRIC_SPC}

Output schema (JSON only, no other text):
{{
  "method_valid": <true|false>,
  "specific": <true|false>,
  "reason": "<one sentence each for method and specificity judgements>"
}}
"""

USER_TEMPLATE = """Casualty state: {state}

Salvage plan:
{plan_text}

Evaluate and return JSON only."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_gt(gt_csv: str) -> dict[str, str]:
    """Return {image_path: gt_state} from human_gt.csv."""
    df = pd.read_csv(gt_csv)
    out = {}
    for _, row in df.iterrows():
        img = str(row.get("image", "")).strip()
        state = str(row.get("state", "unknown")).strip().lower()
        if img:
            out[img] = state
    return out


def state_from_image_path(image: str) -> str:
    """Extract GT state from image path prefix (aground/capsized/sunken/on_fire)."""
    parts = Path(image).parts
    state_map = {"aground": "aground", "capsized": "capsized",
                 "sunken": "sunken", "on_fire": "on_fire", "on fire": "on_fire"}
    for part in parts:
        if part.lower() in state_map:
            return state_map[part.lower()]
    return "unknown"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] skipping bad line in {path}: {e}", file=sys.stderr)
    return records


def truncate_plan(text: str, max_chars: int = 3000) -> str:
    """Truncate plan to fit within context budget."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


# ---------------------------------------------------------------------------
# batch inference
# ---------------------------------------------------------------------------

def build_prompts(records: list[dict], gt_map: dict[str, str]) -> list[dict]:
    """
    For each record, produce two evaluation rows: gt_track and predicted_track.
    Returns list of dicts with all fields needed for judging.
    """
    rows = []
    for rec in records:
        image = rec.get("image", "")
        plan_text = rec.get("text", "")
        qid = rec.get("question_id", image)
        condition = rec.get("condition", "unknown")

        # Steps and SEQ
        steps, parse_flag = parse_steps_v2(plan_text, source_id=qid)
        predicted_state = extract_condition(plan_text)

        # GT state: try gt_map first, then parse from image path
        gt_state = gt_map.get(image) or state_from_image_path(image)

        # SEQ on both tracks
        seq_gt  = check_seq(steps, gt_state)
        seq_pred = check_seq(steps, predicted_state)

        plan_trunc = truncate_plan(plan_text)

        for track, state, seq_res in [
            ("gt",        gt_state,        seq_gt),
            ("predicted", predicted_state, seq_pred),
        ]:
            rows.append({
                "question_id":  qid,
                "image":        image,
                "condition":    condition,
                "eval_track":   track,
                "eval_state":   state,
                "predicted_state": predicted_state,
                "gt_state":     gt_state,
                "parse_flag":   parse_flag,
                "seq_score":    seq_res["seq_score"],
                "seq_chains_applicable": seq_res["chains_applicable"],
                "seq_chains_passed":     seq_res["chains_passed"],
                "seq_chains_failed":     seq_res["chains_failed"],
                "seq_failures": seq_res["failures"],
                # judge prompt fields
                "_plan_trunc":  plan_trunc,
                "_state":       state,
            })
    return rows


def run_judge(llm: LLM, rows: list[dict], cfg: dict) -> list[dict]:
    """Run vLLM batched judge inference and attach MTH+SPC scores."""
    judge_cfg = cfg["judge"]
    sampling = SamplingParams(
        temperature=judge_cfg["temperature"],
        max_tokens=judge_cfg["max_tokens"],
        guided_decoding=GuidedDecodingParams(
            json={
                "type": "object",
                "properties": {
                    "method_valid": {"type": "boolean"},
                    "specific":     {"type": "boolean"},
                    "reason":       {"type": "string", "maxLength": 180},
                },
                "required": ["method_valid", "specific", "reason"],
            }
        ),
    )

    # Build conversation lists for vLLM
    convos = []
    for row in rows:
        user_msg = USER_TEMPLATE.format(
            state=row["_state"] or "unknown",
            plan_text=row["_plan_trunc"],
        )
        convos.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ])

    outputs = llm.chat(convos, sampling_params=sampling)

    _BOOL_RE = {
        "method_valid": re.compile(r'"method_valid"\s*:\s*(true|false)', re.I),
        "specific":     re.compile(r'"specific"\s*:\s*(true|false)', re.I),
    }

    results = []
    for row, out in zip(rows, outputs):
        raw = out.outputs[0].text.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            # JSON truncated mid-reason — rescue the boolean fields via regex
            rescued = {}
            for field, pat in _BOOL_RE.items():
                m = pat.search(raw)
                rescued[field] = (m.group(1).lower() == "true") if m else None
            rescued["reason"] = f"parse_error(rescued): {raw[:120]}"
            parsed = rescued

        result = {k: v for k, v in row.items() if not k.startswith("_")}
        result["method_valid"] = parsed.get("method_valid")
        result["specific"]     = parsed.get("specific")
        result["reason"]       = parsed.get("reason", "")
        # Composite coherence score:
        # SEQ weight 0.4, MTH 0.35, SPC 0.25
        try:
            mth = 1.0 if parsed["method_valid"] else 0.0
            spc = 1.0 if parsed["specific"]     else 0.0
            seq = row["seq_score"]
            result["coherence_score"] = round(0.40 * seq + 0.35 * mth + 0.25 * spc, 4)
        except Exception:
            result["coherence_score"] = None

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="improved/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pipeline_dir    = Path(os.path.expandvars(cfg["paths"]["pipeline_dir"]))
    results_dir     = pipeline_dir / "results"
    gt_csv          = os.path.expandvars(cfg["paths"]["gt_csv"])
    user_models_dir = Path(os.path.expandvars(cfg["paths"]["user_models_dir"]))
    judge_path      = user_models_dir / cfg["models"]["judge_dir"]

    # Load GT
    gt_map: dict[str, str] = {}
    if Path(gt_csv).exists():
        gt_map = load_gt(gt_csv)
    else:
        print(f"[WARN] gt_csv not found: {gt_csv} — will infer state from image path", file=sys.stderr)

    # Load all inference JSONL files
    jsonl_files = sorted(results_dir.glob("answers_qwen3vl8b_baseline_*_improved.jsonl"))
    if not jsonl_files:
        print("[ERROR] No inference JSONL files found in results/. Run run_inference.py first.",
              file=sys.stderr)
        sys.exit(1)

    all_records: list[dict] = []
    for jf in jsonl_files:
        recs = load_jsonl(str(jf))
        print(f"Loaded {len(recs)} records from {jf.name}")
        all_records.extend(recs)

    print(f"Total records to evaluate: {len(all_records)}")

    # Build evaluation rows (2 tracks × N records)
    rows = build_prompts(all_records, gt_map)
    print(f"Evaluation rows (2 tracks × records): {len(rows)}")

    # Load judge
    judge_cfg = cfg["judge"]
    print(f"Loading judge from {judge_path} ...")
    llm = LLM(
        model=str(judge_path),
        max_model_len=judge_cfg["max_model_len"],
        gpu_memory_utilization=judge_cfg["gpu_memory_utilization"],
        tensor_parallel_size=1,
    )

    # Run judge
    results = run_judge(llm, rows, cfg)

    # Write output
    out_path = results_dir / "judge_scores_improved.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    print(f"Judge scores written → {out_path}")
    print(f"Total rows: {len(results)}")


if __name__ == "__main__":
    main()
