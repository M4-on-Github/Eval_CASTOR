"""
calibrate.py -- score a candidate extraction model against the gold
tool-call set (calibration/gold_tool_calls.jsonl) and check it against the
go/no-go thresholds before it's trusted to run at scale.

Two halves, deliberately separable (design plan section 4e):
  - Scoring logic (score_record, aggregate, check_thresholds) is pure
    Python, no model, unit-tested against synthetic ToolCalls in
    tests/test_plan_adequacy_calibrate.py. This is what you'd trust even
    with zero cluster access.
  - The bake-off (run_calibration) actually calls extract.py's vLLM path
    and needs a GPU -- it's just wiring the pure scorer up to real
    extractions, exercising the identical prompt/parse code that will ship
    in the real pipeline (that's the whole point: calibration must test
    what actually runs, not a stand-in).

Usage (inside Apptainer via containers/plan_adequacy_calibrate_job.sh):
  python3 calibrate.py \\
      --model      glm4_32b \\
      --model-dir  /data/$USER/glm-4-32b-0414-gptq \\
      --gold       pipelines/plan_adequacy/calibration/gold_tool_calls.jsonl \\
      --out        results/p9_plan_adequacy/calibration/
"""

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Optional

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.paths import BASE_OUT_DIR, GOLD_TOOL_CALLS_PATH
from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry, build_guided_json_schema

# ---------------------------------------------------------------------------
# Go/no-go thresholds -- design plan section 4f
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "tool_id_micro_accuracy": 0.90,
    "tool_id_macro_accuracy": 0.85,
    "null_fidelity": 0.97,
    "conditional_f1": 0.85,
    "condition_var_accuracy": 0.80,
    "no_match_f1": 0.85,
    "parse_failure_rate_max": 0.01,   # ceiling, not floor
}
#: Layer A only -- see design plan sec 4b, headline accuracy is reported on
#: the real-corpus layer, never on the authored Layer C coverage probe.
HEADLINE_LAYER = "A"
#: Per-tool minimum instance count (any layer) before its worst-case
#: accuracy is held to the 0.70 floor -- a tool with 1-2 gold examples
#: doesn't have enough signal to fail calibration on.
MIN_INSTANCES_FOR_PER_TOOL_FLOOR = 10
PER_TOOL_RECALL_FLOOR = 0.70


# ---------------------------------------------------------------------------
# Gold set loading
# ---------------------------------------------------------------------------

def load_gold(path: Path) -> list:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def gold_to_calls(gold_records: list) -> list:
    """(step_num, step_text, casualty) tuples in gold order, ready for
    extract.extract_steps(). step_num is synthetic (position in file) --
    gold records aren't tied to a real multi-step plan, each is scored
    independently."""
    return [(i + 1, r["step_text"], r["casualty"]) for i, r in enumerate(gold_records)]


# ---------------------------------------------------------------------------
# Per-record scoring (pure, no model)
# ---------------------------------------------------------------------------

def score_record(gold: dict, predicted: ToolCall, parse_ok: bool) -> dict:
    """Compare one gold record against one extracted ToolCall. Returns a
    flat dict of booleans/values that aggregate() reduces over the whole
    set -- kept as simple facts here, not pre-aggregated, so aggregate()
    can slice by layer/failure_type/casualty without re-deriving anything."""
    gold_tool = gold["expected_tool"]
    tool_correct = predicted.tool == gold_tool

    # Null fidelity: of params gold says are ABSENT (null/not given), did
    # the model also leave them null? Checked over the union of param keys
    # either side mentions, since a param the model invents that gold never
    # mentions at all is exactly the hallucination this metric exists to
    # catch. Weighted heaviest of all metrics -- see design plan sec 4d.4.
    gold_params = gold.get("expected_params") or {}
    pred_params = predicted.params or {}
    all_keys = set(gold_params.keys()) | set(pred_params.keys())
    null_checks = []
    value_checks = []
    for k in all_keys:
        gv = gold_params.get(k)
        pv = pred_params.get(k)
        if gv is None:
            null_checks.append(pv is None)
        else:
            value_checks.append(gv == pv)

    gold_conditional = bool(gold.get("expected_conditional", False))
    pred_conditional = bool(predicted.conditional)

    condition_var_correct = None
    if gold_conditional:
        condition_var_correct = (predicted.condition_var == gold.get("expected_condition_var", "none"))

    return {
        "gold_id": gold.get("gold_id"),
        "layer": gold.get("layer"),
        "failure_type": gold.get("failure_type"),
        "casualty": gold.get("casualty"),
        "gold_tool": gold_tool,
        "predicted_tool": predicted.tool,
        "tool_correct": tool_correct,
        "null_checks": null_checks,      # list[bool]
        "value_checks": value_checks,    # list[bool]
        "gold_conditional": gold_conditional,
        "predicted_conditional": pred_conditional,
        "condition_var_correct": condition_var_correct,  # None if gold wasn't conditional
        "gold_is_no_match": gold_tool == "no_match",
        "predicted_is_no_match": predicted.tool == "no_match",
        "parse_ok": parse_ok,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0
          else None)
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate(scored: list) -> dict:
    """Reduce a list of score_record() outputs into the metrics table --
    design plan section 4d. `scored` should already be filtered to the
    layer/subset being reported (aggregate() does not itself split by
    layer; see build_report() for the per-layer breakdown)."""
    n = len(scored)
    if n == 0:
        return {"n": 0}

    tool_correct_n = sum(1 for s in scored if s["tool_correct"])
    micro_accuracy = tool_correct_n / n

    # macro accuracy: mean per-gold-tool recall
    by_gold_tool = collections.defaultdict(list)
    for s in scored:
        by_gold_tool[s["gold_tool"]].append(s["tool_correct"])
    per_tool_recall = {t: sum(v) / len(v) for t, v in by_gold_tool.items()}
    macro_accuracy = sum(per_tool_recall.values()) / len(per_tool_recall) if per_tool_recall else None

    confusion = collections.Counter((s["gold_tool"], s["predicted_tool"]) for s in scored)

    null_all = [c for s in scored for c in s["null_checks"]]
    null_fidelity = sum(null_all) / len(null_all) if null_all else None

    value_all = [c for s in scored for c in s["value_checks"]]
    param_value_accuracy = sum(value_all) / len(value_all) if value_all else None

    cond_tp = sum(1 for s in scored if s["gold_conditional"] and s["predicted_conditional"])
    cond_fp = sum(1 for s in scored if not s["gold_conditional"] and s["predicted_conditional"])
    cond_fn = sum(1 for s in scored if s["gold_conditional"] and not s["predicted_conditional"])
    conditional_prf1 = _prf1(cond_tp, cond_fp, cond_fn)

    cvar_checks = [s["condition_var_correct"] for s in scored if s["condition_var_correct"] is not None]
    condition_var_accuracy = sum(cvar_checks) / len(cvar_checks) if cvar_checks else None

    nm_tp = sum(1 for s in scored if s["gold_is_no_match"] and s["predicted_is_no_match"])
    nm_fp = sum(1 for s in scored if not s["gold_is_no_match"] and s["predicted_is_no_match"])
    nm_fn = sum(1 for s in scored if s["gold_is_no_match"] and not s["predicted_is_no_match"])
    no_match_prf1 = _prf1(nm_tp, nm_fp, nm_fn)

    parse_failures = sum(1 for s in scored if not s["parse_ok"])
    parse_failure_rate = parse_failures / n

    return {
        "n": n,
        "tool_id_micro_accuracy": round(micro_accuracy, 4),
        "tool_id_macro_accuracy": round(macro_accuracy, 4) if macro_accuracy is not None else None,
        "per_tool_recall": {t: round(v, 4) for t, v in sorted(per_tool_recall.items())},
        "per_tool_n": {t: len(v) for t, v in by_gold_tool.items()},
        "confusion_top": confusion.most_common(15),
        "null_fidelity": round(null_fidelity, 4) if null_fidelity is not None else None,
        "param_value_accuracy": round(param_value_accuracy, 4) if param_value_accuracy is not None else None,
        "conditional_precision": conditional_prf1["precision"],
        "conditional_recall": conditional_prf1["recall"],
        "conditional_f1": conditional_prf1["f1"],
        "condition_var_accuracy": round(condition_var_accuracy, 4) if condition_var_accuracy is not None else None,
        "no_match_precision": no_match_prf1["precision"],
        "no_match_recall": no_match_prf1["recall"],
        "no_match_f1": no_match_prf1["f1"],
        "parse_failure_rate": round(parse_failure_rate, 4),
    }


def stratify(scored: list, key: str) -> dict:
    """aggregate() computed separately per distinct value of `key`
    (e.g. "layer", "failure_type", "casualty") -- design plan sec 4d.8:
    a model strong overall but blind on one failure_type is a different,
    hidden problem that a single aggregate number would mask."""
    buckets = collections.defaultdict(list)
    for s in scored:
        buckets[s.get(key)].append(s)
    return {k: aggregate(v) for k, v in buckets.items() if k is not None}


# ---------------------------------------------------------------------------
# Threshold checking
# ---------------------------------------------------------------------------

def check_thresholds(headline_metrics: dict) -> dict:
    """Pass/fail against THRESHOLDS, evaluated on the HEADLINE (Layer A)
    metrics only -- design plan sec 4f. Per-tool floor is checked
    separately since it needs per_tool_recall + per_tool_n together."""
    results = {}
    for key, floor in THRESHOLDS.items():
        if key == "parse_failure_rate_max":
            val = headline_metrics.get("parse_failure_rate")
            results[key] = {"value": val, "threshold": floor, "passed": val is not None and val <= floor}
            continue
        val = headline_metrics.get(key)
        results[key] = {"value": val, "threshold": floor, "passed": val is not None and val >= floor}

    per_tool_failures = []
    for tool, n in headline_metrics.get("per_tool_n", {}).items():
        if n >= MIN_INSTANCES_FOR_PER_TOOL_FLOOR:
            recall = headline_metrics["per_tool_recall"].get(tool, 0.0)
            if recall < PER_TOOL_RECALL_FLOOR:
                per_tool_failures.append((tool, n, recall))
    results["per_tool_floor"] = {
        "threshold": PER_TOOL_RECALL_FLOOR,
        "min_instances": MIN_INSTANCES_FOR_PER_TOOL_FLOOR,
        "failures": per_tool_failures,
        "passed": len(per_tool_failures) == 0,
    }

    results["overall_pass"] = all(r["passed"] for r in results.values())
    return results


# ---------------------------------------------------------------------------
# Bake-off orchestration (cluster only -- calls extract.py's vLLM path)
# ---------------------------------------------------------------------------

def run_calibration(model_key: str, model_dir: str, gold_path: Path,
                     max_model_len: int = 4096, max_tokens: int = 256) -> dict:
    """Extract every gold step with the given model, score against gold,
    and return the full report (headline + per-layer + per-failure-type +
    per-casualty + threshold check). This is the only function in this
    module that touches vLLM (via extract.py, imported lazily inside it)."""
    from pipelines.plan_adequacy.extract import (
        build_system_prompt, build_user_prompt, _run_vllm_batch, parse_extraction,
    )

    registry = ToolRegistry.load()
    gold_records = load_gold(gold_path)
    schema = build_guided_json_schema(registry)
    system = build_system_prompt(registry)

    prompts = [
        (system, build_user_prompt(r["casualty"], i + 1, r["step_text"]))
        for i, r in enumerate(gold_records)
    ]
    raw_results = _run_vllm_batch(prompts, model_dir, schema, max_model_len, max_tokens)

    scored = []
    for gold, raw in zip(gold_records, raw_results):
        parse_ok = raw is not None
        call = parse_extraction(raw, 0, gold["step_text"])
        scored.append(score_record(gold, call, parse_ok))

    headline = aggregate([s for s in scored if s["layer"] == HEADLINE_LAYER])
    thresholds = check_thresholds(headline)

    return {
        "model": model_key,
        "model_dir": model_dir,
        "headline": headline,
        "thresholds": thresholds,
        "by_layer": stratify(scored, "layer"),
        "by_failure_type": stratify([s for s in scored if s["layer"] == "B"], "failure_type"),
        "by_casualty": stratify(scored, "casualty"),
        "overall": aggregate(scored),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="P9 extraction-model calibration bake-off")
    ap.add_argument("--model", required=True, help="Model key (for the report filename)")
    ap.add_argument("--model-dir", required=True, help="Absolute path to model weights")
    ap.add_argument("--gold", type=Path, default=GOLD_TOOL_CALLS_PATH)
    ap.add_argument("--out", type=Path, default=BASE_OUT_DIR / "calibration",
                     help="Output directory")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    report = run_calibration(args.model, args.model_dir, args.gold,
                              args.max_model_len, args.max_tokens)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"calibration_{args.model}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    h = report["headline"]
    t = report["thresholds"]
    print(f"\n=== {args.model} headline (Layer A, n={h['n']}) ===")
    print(f"  tool_id_micro_accuracy : {h['tool_id_micro_accuracy']}  (>= {THRESHOLDS['tool_id_micro_accuracy']})")
    print(f"  tool_id_macro_accuracy : {h['tool_id_macro_accuracy']}  (>= {THRESHOLDS['tool_id_macro_accuracy']})")
    print(f"  null_fidelity          : {h['null_fidelity']}  (>= {THRESHOLDS['null_fidelity']})")
    print(f"  conditional_f1         : {h['conditional_f1']}  (>= {THRESHOLDS['conditional_f1']})")
    print(f"  condition_var_accuracy : {h['condition_var_accuracy']}  (>= {THRESHOLDS['condition_var_accuracy']})")
    print(f"  no_match_f1            : {h['no_match_f1']}  (>= {THRESHOLDS['no_match_f1']})")
    print(f"  parse_failure_rate     : {h['parse_failure_rate']}  (<= {THRESHOLDS['parse_failure_rate_max']})")
    print(f"  per-tool floor failures: {t['per_tool_floor']['failures']}")
    print(f"\n  OVERALL: {'PASS' if t['overall_pass'] else 'FAIL'}")
    print(f"\n  Report written to {out_path}")


if __name__ == "__main__":
    main()
