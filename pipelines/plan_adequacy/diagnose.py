"""
diagnose.py -- the diagnosis table, and the only summary P9 reports.

Everything else in this pipeline exists to make this one table's cells
defensible. Six failure classes down the side; seven columns across, grouped
into three questions:

    WHAT HAPPENED   first-fail %, hazard rank, mean EPL
    TRUST           diagnostic recall, % attributable to planner
    WORTH           delta EPL if repaired
                    -> direction (read off a pre-declared remedy mapping)

HOW TO READ IT. A row is actionable only if RECALL and % PLANNER are both
high -- those two are the trust filter, and a row failing either is a
statement about this instrument rather than about any planner. Among the rows
that pass, sort by delta EPL. If the pre-execution rows dominate, the
bottleneck is upstream of planning entirely (perception, or route vocabulary)
and no amount of sequencing work will move the number.

Recall and % planner are NOT computed here: they come from the injection study
(inject.py) and from hand adjudication respectively, and are merged in when
those exist. Until then they print as "--", which is the honest rendering --
an unvalidated prevalence number should look unvalidated.

Usage:
    python -m pipelines.plan_adequacy.diagnose \\
        --runs answers_..._ablation_v2_improved answers_..._standard_v2_improved \\
        --dir  /path/to/results [--repair] [--out diagnosis.csv]
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.classify import (FAILURE_CLASSES,
                                              PRE_EXECUTION_CLASSES, classify)
from pipelines.plan_adequacy.hazard import (hazard_rank, hazard_table,
                                            mean_epl_by_class, prevalence)
from pipelines.plan_adequacy.methods import RouteRegistry
from pipelines.plan_adequacy.repair import (delta_epl_by_class,
                                            repair_to_exhaustion,
                                            transition_matrix)
from pipelines.plan_adequacy.run_executor import group_tool_calls
from pipelines.plan_adequacy.scenario import load_scenarios
from pipelines.plan_adequacy.vocab import ToolRegistry

DIAGNOSIS_FIELDS = ["failure_class", "n", "first_fail_pct", "hazard_rank",
                    "mean_epl", "recall", "pct_planner", "delta_epl", "direction"]

#: Pre-declared remedy mapping (reports/p9/redesign.tex, Table 4). Fixed
#: before results existed, which is the whole point: a direction read off a
#: table written in advance is a prediction, one written afterwards is a
#: story.
DIRECTIONS = {
    "NO_PROCEDURE":        "route vocabulary coverage",
    "STRATEGY_PERCEPTION": "visual grounding / casualty ID",
    "STRATEGY_TECHNIQUE":  "vessel-conditioned technique selection",
    "PROCEDURE":           "sequencing / dependency structure",
    "COMMITMENT":          "magnitude elicitation",
    "INCOMPLETE":          "goal-directed termination",
    "VALID":               "--",
}


def rows_from_per_image(path: Path) -> list:
    """Read the diagnosis fields back out of an aggregate.py per_image.csv.

    Reading the CSV rather than re-executing is deliberate: it forces the
    reported table to be a function of the artefact on disk, so a number in
    the write-up can always be traced to a file rather than to a rerun that
    may not reproduce.
    """
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append({
                "image": r["image"],
                "casualty": r["casualty"],
                "failure_class": r["failure_class"],
                "failure_step": int(r["failure_step"]) if r["failure_step"] else None,
                "epl": int(r["epl"]),
                "epl_is_structural": r["epl_is_structural"] == "True",
                "foreign_casualty": r.get("foreign_casualty", ""),
            })
    return out


def build_diagnosis(rows: list, deltas: dict = None) -> list:
    """The table. `deltas` is delta_epl_by_class() output, or None when the
    repair pass has not been run."""
    prev = prevalence(rows)
    ranks = hazard_rank(rows)
    means = mean_epl_by_class(rows)
    deltas = deltas or {}

    table = []
    for cls in FAILURE_CLASSES:
        mean = means[cls]
        delta = deltas.get(cls, {}).get("mean_delta_epl")
        table.append({
            "failure_class": cls,
            "n": prev[cls]["n"],
            "first_fail_pct": round(prev[cls]["pct"] * 100, 1),
            # None for pre-execution classes: their events are counted
            # against the whole corpus while step-wise hazards use a
            # shrinking risk set, so the two are not on a common scale.
            "hazard_rank": ranks.get(cls) if ranks.get(cls) is not None else "--",
            # None where EPL is structural -- averaging a definition
            # produces a number that looks like evidence.
            "mean_epl": "--" if mean is None else round(mean, 2),
            "recall": "--",        # inject.py, not yet run
            "pct_planner": "--",   # hand adjudication, not yet run
            "delta_epl": "--" if delta is None else round(delta, 2),
            "direction": DIRECTIONS.get(cls, "--"),
        })
    return table


def run_repairs(tool_calls_path: Path, gt_path: Path = None) -> list:
    """Neutralise-and-continue over every plan in a run.

    Pure function of the tool calls, so this runs over the FULL corpus rather
    than a hand-repaired subsample -- the original design budgeted 60
    hand-repaired plans, which the operator makes unnecessary.
    """
    tool_registry, route_registry = ToolRegistry.load(), RouteRegistry.load()
    scenarios = load_scenarios(gt_path)
    out = []
    for image, calls in group_tool_calls(tool_calls_path).items():
        scenario = scenarios.get(image)
        if scenario is None:
            continue
        plan_text = "\n".join(c.step_text for c in calls)
        res = repair_to_exhaustion(calls, scenario.state, scenario,
                                   tool_registry, route_registry, plan_text)
        res["image"] = image
        out.append(res)
    return out


def _fmt_table(table: list) -> str:
    head = (f"{'class':<21}{'n':>5}{'first-fail%':>13}{'haz.rank':>10}"
            f"{'meanEPL':>9}{'recall':>8}{'%planner':>10}{'dEPL':>7}  direction")
    lines = [head, "-" * len(head)]
    for r in table:
        lines.append(f"{r['failure_class']:<21}{r['n']:>5}{r['first_fail_pct']:>12.1f}%"
                     f"{str(r['hazard_rank']):>10}{str(r['mean_epl']):>9}"
                     f"{r['recall']:>8}{r['pct_planner']:>10}{str(r['delta_epl']):>7}"
                     f"  {r['direction']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="P9 diagnosis table")
    ap.add_argument("--runs", nargs="+", required=True, help="run directory names")
    ap.add_argument("--dir", required=True, help="results base directory")
    ap.add_argument("--gt", default=None, help="ground-truth CSV (scenario lookup)")
    ap.add_argument("--repair", action="store_true",
                    help="also run the neutralise-and-continue repair pass")
    ap.add_argument("--out", default=None, help="write the table to this CSV")
    args = ap.parse_args()

    base = Path(args.dir)
    rows, repair_rows = [], []
    for run in args.runs:
        rows += rows_from_per_image(base / run / "per_image.csv")
        if args.repair:
            repair_rows += run_repairs(base / run / "tool_calls.jsonl",
                                       Path(args.gt) if args.gt else None)

    deltas = delta_epl_by_class(repair_rows) if repair_rows else None
    table = build_diagnosis(rows, deltas)

    epls = [r["epl"] for r in rows]
    zeros = sum(1 for e in epls if e == 0)
    print(f"\nmean EPL = {sum(epls)/len(epls):.2f} of 6   "
          f"({zeros}/{len(epls)} = {zeros/len(epls)*100:.0f}% of plans at 0)\n")
    print(_fmt_table(table))

    if repair_rows:
        print("\nfirst-repair transitions (repaired class -> what failed next):")
        for (a, b), n in sorted(transition_matrix(repair_rows).items(),
                                 key=lambda kv: -kv[1]):
            wall = "   <- wall" if a == b else ""
            print(f"  {a:<21} -> {b:<21} {n:>4}{wall}")
        solved = [r for r in repair_rows if r["repairs_to_valid"] is not None]
        if solved:
            dist = Counter(r["repairs_to_valid"] for r in solved)
            print(f"\nrepairs-to-clean: {dict(sorted(dist.items()))}  "
                  f"({len(solved)}/{len(repair_rows)} reach a clean execution at all)")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DIAGNOSIS_FIELDS)
            w.writeheader()
            w.writerows(table)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
