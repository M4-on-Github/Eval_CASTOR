"""
Build Layer B of the P9 gold tool-call set: add a tool-call annotation layer
to the EXISTING 166 hand-labeled steps in p8_to_check/synthetic_calibration.jsonl.

That file already has judgment labels (sequencing_valid, method_valid,
specific, overall_valid, reason) for a different purpose (calibrating the
P8-style coherence judge) -- it does NOT have tool-call labels. This script
reuses its step text and failure_type stratification, and reuses the same
heuristic drafter from build_gold_layer_a.py, so the review process is
identical: draft, then a human (or a trusted reader, not the model under
test) corrects the low-confidence ones.

Usage (from Eval_CASTOR/):
  python3 pipelines/plan_adequacy/calibration/build_gold_layer_b.py

Writes gold_layer_b_scaffold.jsonl next to this file.
"""

import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.gates import detect_gates
from pipelines.plan_adequacy.vocab import ToolRegistry
from pipelines.plan_adequacy.calibration.build_gold_layer_a import _draft_tool

SOURCE_PATH = EVAL_ROOT / "p8_to_check" / "synthetic_calibration.jsonl"
OUT_PATH = Path(__file__).parent / "gold_layer_b_scaffold.jsonl"


def main():
    reg = ToolRegistry.load()
    recs = [json.loads(l) for l in SOURCE_PATH.open(encoding="utf-8") if l.strip()]

    out_records = []
    for r in recs:
        plan_text = r["text"]
        gates_in_plan = detect_gates(plan_text)

        for step in r["expected"]:
            step_num = step["step_num"]
            # step_text isn't stored separately in this file -- pull it back
            # out of the plan text by locating the numbered line.
            step_text = _extract_step_text(plan_text, step_num)
            if step_text is None:
                continue

            tool_guess, secondary_guess, confidence = _draft_tool(step_text)
            step_gates = [g for g in gates_in_plan if g.condition_text[:40] in step_text[:200]]
            conditional_guess = bool(step_gates)
            condition_var_guess = step_gates[0].condition_var if step_gates else "none"

            out_records.append({
                "cal_id": r["cal_id"],
                "gt_state": r["gt_state"],
                "failure_type": r["failure_type"],
                "step_num": step_num,
                "step_text": step_text,
                # carried forward from the EXISTING judgment labels -- not
                # tool-call labels, kept for stratification/cross-check.
                "existing_sequencing_valid": step["sequencing_valid"],
                "existing_method_valid": step["method_valid"],
                "existing_specific": step["specific"],
                "existing_overall_valid": step["overall_valid"],
                "existing_reason": step["reason"],
                # new layer -- what this script actually adds.
                "expected_tool": tool_guess,
                "expected_secondary_tools": secondary_guess,
                "expected_params": {},
                "expected_conditional": conditional_guess,
                "expected_condition_var": condition_var_guess,
                "expected_family": reg.family(tool_guess) if reg.has(tool_guess) else "no_match",
                "confidence": confidence,
                "reviewed": False,
            })

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
        encoding="utf-8",
    )
    import collections
    counts = collections.Counter(r["confidence"] for r in out_records)
    print(f"{len(out_records)} step records across {len(recs)} synthetic plans -> {OUT_PATH}")
    print(f"  medium (single clean hit): {counts['medium']}")
    print(f"  no_match_confident (zero hits, likely filler): {counts['no_match_confident']}")
    print(f"  low (genuine multi-hit ambiguity -- review these): {counts['low']}")


def _extract_step_text(plan_text: str, step_num: int) -> str:
    """synthetic_calibration.jsonl's `expected` list carries step_num but
    not the step text itself -- recover it from the numbered plan text with
    the same numbered-list convention parse_steps.py uses."""
    import re
    m = re.search(
        rf'(?:^|\n)\s*{step_num}\.\s+(.*?)(?=\n\s*\d+\.|\Z)',
        plan_text, re.DOTALL,
    )
    if not m:
        return None
    return " ".join(m.group(1).split())


if __name__ == "__main__":
    main()
