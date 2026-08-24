"""
Combine the three gold-set layers (Layer A: real corpus, Layer B: existing
synthetic plans, Layer C: authored coverage fill) into the single file
calibrate.py actually reads: calibration/gold_tool_calls.jsonl.

All three source files are fully hand-reviewed (see memory:
gold-set-audit-findings.md) -- this script does no drafting or judgment of
its own, just normalizes field names and stamps each record with which
layer it came from, since calibrate.py grades differently per layer
(design plan section 4b): headline accuracy is Layer A only, Layer B adds
failure-type stratification, Layer C is a coverage probe never counted
toward headline numbers.

Usage (from Eval_CASTOR/):
  python3 pipelines/plan_adequacy/calibration/build_gold_combined.py
"""

import json
from pathlib import Path

CAL_DIR = Path(__file__).parent
OUT_PATH = CAL_DIR / "gold_tool_calls.jsonl"


def _normalize(r: dict, layer: str, gold_id: str) -> dict:
    return {
        "gold_id": gold_id,
        "layer": layer,
        "step_text": r["step_text"],
        "expected_tool": r["expected_tool"],
        "expected_secondary_tools": r.get("expected_secondary_tools", []),
        "expected_params": r.get("expected_params", {}),
        "expected_conditional": r.get("expected_conditional", False),
        "expected_condition_var": r.get("expected_condition_var", "none"),
        "expected_family": r.get("expected_family", ""),
        # Layer A: casualty. Layer B: gt_state. Layer C: none (synthetic,
        # not tied to a real casualty scenario).
        "casualty": r.get("casualty") or r.get("gt_state") or "",
        # Layer B only -- None elsewhere. This is what enables
        # per-failure-type calibration stratification (design plan sec 4d.8).
        "failure_type": r.get("failure_type"),
    }


def main():
    out_records = []

    a_path = CAL_DIR / "gold_layer_a_scaffold.jsonl"
    for i, line in enumerate(a_path.open(encoding="utf-8")):
        if not line.strip():
            continue
        out_records.append(_normalize(json.loads(line), "A", f"A{i:04d}"))

    b_path = CAL_DIR / "gold_layer_b_scaffold.jsonl"
    for i, line in enumerate(b_path.open(encoding="utf-8")):
        if not line.strip():
            continue
        out_records.append(_normalize(json.loads(line), "B", f"B{i:04d}"))

    c_path = CAL_DIR / "gold_layer_c_scaffold.jsonl"
    for i, line in enumerate(c_path.open(encoding="utf-8")):
        if not line.strip():
            continue
        out_records.append(_normalize(json.loads(line), "C", f"C{i:04d}"))

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
        encoding="utf-8",
    )

    import collections
    by_layer = collections.Counter(r["layer"] for r in out_records)
    print(f"{len(out_records)} gold records -> {OUT_PATH}")
    print(f"  Layer A (headline accuracy): {by_layer['A']}")
    print(f"  Layer B (failure-type stratified): {by_layer['B']}")
    print(f"  Layer C (coverage probe only): {by_layer['C']}")


if __name__ == "__main__":
    main()
