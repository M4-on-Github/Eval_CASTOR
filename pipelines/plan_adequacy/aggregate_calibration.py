"""
Compare calibration_<model>.json reports (one per candidate model, produced
by calibrate.py) and report which candidates clear the go/no-go thresholds
-- design plan section 4g: "pick empirically; do not pre-commit on
reputation."

Usage (inside Apptainer via containers/plan_adequacy_calibrate_compare_job.sh,
or directly once all calibration_*.json files exist):
  python3 aggregate_calibration.py --dir results/p9_plan_adequacy/calibration/
"""

import argparse
import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))


def load_reports(directory: Path) -> list:
    reports = []
    for p in sorted(directory.glob("calibration_*.json")):
        reports.append(json.loads(p.read_text(encoding="utf-8")))
    return reports


def build_comparison(reports: list) -> dict:
    rows = []
    for r in reports:
        h = r["headline"]
        t = r["thresholds"]
        rows.append({
            "model": r["model"],
            "n": h["n"],
            "tool_id_micro_accuracy": h["tool_id_micro_accuracy"],
            "tool_id_macro_accuracy": h["tool_id_macro_accuracy"],
            "null_fidelity": h["null_fidelity"],
            "conditional_f1": h["conditional_f1"],
            "condition_var_accuracy": h["condition_var_accuracy"],
            "no_match_f1": h["no_match_f1"],
            "parse_failure_rate": h["parse_failure_rate"],
            "per_tool_floor_failures": len(t["per_tool_floor"]["failures"]),
            "overall_pass": t["overall_pass"],
        })

    passing = [r["model"] for r in rows if r["overall_pass"]]
    # Among passing models, prefer higher micro accuracy; among failing
    # models (if none pass), report the closest one so the fallback ladder
    # (design plan sec 4g) has somewhere concrete to start.
    ranked = sorted(rows, key=lambda r: r["tool_id_micro_accuracy"] or 0, reverse=True)

    return {
        "models_evaluated": [r["model"] for r in rows],
        "models_passing": passing,
        "recommended": passing[0] if passing else None,
        "closest_if_none_passing": ranked[0]["model"] if not passing and ranked else None,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare P9 calibration reports across candidate models")
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()

    reports = load_reports(args.dir)
    if not reports:
        print(f"No calibration_*.json files found in {args.dir}")
        return

    comparison = build_comparison(reports)
    out_path = args.dir / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n=== P9 Calibration Comparison ===")
    header = ("model", "n", "micro", "macro", "null_fid", "cond_f1", "cvar_acc", "no_match_f1", "pass")
    print("  " + "  ".join(f"{h:>10s}" for h in header))
    for r in comparison["rows"]:
        vals = (r["model"], r["n"], r["tool_id_micro_accuracy"], r["tool_id_macro_accuracy"],
                 r["null_fidelity"], r["conditional_f1"], r["condition_var_accuracy"],
                 r["no_match_f1"], "PASS" if r["overall_pass"] else "FAIL")
        print("  " + "  ".join(f"{str(v):>10s}" for v in vals))

    if comparison["recommended"]:
        print(f"\n  RECOMMENDED: {comparison['recommended']}")
    else:
        print(f"\n  NO MODEL PASSED. Closest: {comparison['closest_if_none_passing']}")
        print("  See design plan sec 4g fallback ladder: merge confusable tool")
        print("  clusters first, then try the two-stage (S2) schema, then escalate model size.")

    print(f"\n  Full comparison -> {out_path}")


if __name__ == "__main__":
    main()
