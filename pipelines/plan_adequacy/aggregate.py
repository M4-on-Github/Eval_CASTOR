"""
aggregate.py -- Stage 2's CSV rollup: list[PlanResult] -> per_step.csv,
per_image.csv, summary.csv.

Mirrors pipelines/plan_coherence/aggregate_coherence.py's shape (see the P9
end-to-end-pipeline plan, Part 1) -- same three-file-per-run + cumulative-
append pattern, same _safe_mean reuse. What's different is entirely driven
by PlanResult.summary() (executor.py:72-92) not being CSV-scalar in four of
its 13 keys; see _flatten_plan_row()'s docstring for exactly how each is
handled.

Usage:
  python aggregate.py --run answers_baseline --dir results/p9_plan_adequacy/
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.executor import STEP_VERDICTS, PlanResult
from pipelines.plan_adequacy.paths import CUMULATIVE_SUMMARY_PATH, RunPaths
from pipelines.plan_adequacy.run_executor import run_executor
from pipelines.plan_coherence.aggregate_coherence import _safe_mean

#: The four casualty states this corpus covers -- see scenario.py /
#: human_gt.csv. A tuple, not one hardcoded summary line per state like
#: aggregate_coherence.py:263-266, since P9's summary has more numeric
#: fields to break out per-casualty than P8's single coherence_pct.
CASUALTIES = ("aground", "capsized", "on_fire", "sunken")

#: PlanResult.summary()'s three list[str] fields, and the column-name
#: prefix each gets (n_<field> for the count, <field>_text for the
#: semicolon-joined text) -- see _flatten_plan_row().
_LIST_FIELDS = ("not_attempted", "sequence_violations", "unused_assessments")


# ---------------------------------------------------------------------------
# per_step.csv
# ---------------------------------------------------------------------------

def build_per_step_rows(results: list) -> list:
    """list[PlanResult] -> list[dict], one row per StepResult. Field
    selection mirrors StepResult (executor.py:43-53) plus the owning
    image/casualty, matching the per_step.csv shape declared in the P9
    end-to-end-pipeline plan Part 1."""
    rows = []
    for r in results:
        for s in r.steps:
            rows.append({
                "image": r.image,
                "casualty": r.casualty,
                "step_num": s.n,
                "step_text": s.text,
                "tool": s.tool,
                "verdict": s.verdict,
                "detail": s.detail,
                "conditional": s.conditional,
            })
    return rows


PER_STEP_FIELDNAMES = ["image", "casualty", "step_num", "step_text", "tool", "verdict",
                        "detail", "conditional"]


# ---------------------------------------------------------------------------
# per_image.csv
# ---------------------------------------------------------------------------

def _flatten_plan_row(result: PlanResult) -> dict:
    """One PlanResult -> one CSV-scalar dict. Four of summary()'s 13 keys
    need explicit handling (see the P9 end-to-end-pipeline plan, Part 1):

      counts                 sparse {verdict: n} -- read via .get(v, 0) over
                              the FULL STEP_VERDICTS tuple, never counts.keys(),
                              so a verdict with zero occurrences this run still
                              gets its column and the column set never drifts
                              between runs (that drift is what would silently
                              break the cumulative CSV).
      not_attempted /
      sequence_violations /
      unused_assessments     list[str] of pre-formatted strings -- each gets
                              BOTH a count column (n_<field>, for aggregation/
                              summary.csv) and a ";"-joined text column (for
                              inspection). NOTE: not_attempted may be the
                              single sentinel ["NO_RECOGNISABLE_ROUTE"]
                              (executor.py:113) -- a plan-level verdict
                              wearing a list's clothing. This function does
                              NOT special-case it (that's report.py's job,
                              per the plan); n_not_attempted will read 1 and
                              not_attempted_text will read the sentinel
                              verbatim, which is enough for a human to spot
                              in per_image.csv even before report.py exists.
      self_contradictory_
      on_size                 real Python bool (PlanResult's own field type)
                              -- csv.DictWriter renders it "True"/"False".
                              Decision (see plan): P9 does NOT mirror P8's
                              "1"/"0" string convention (aggregate_coherence.
                              py:189) -- keeping a real bool's str() output is
                              less silently-compatible with P8 but avoids
                              inventing a third boolean convention across the
                              two pipelines. Readers must compare against the
                              string "True", not the int 1.

    route_score is summary()'s only pre-rounded field; route_coherence/
    route_completeness come through unrounded and None-able -- rounded here,
    None becomes "" (csv.DictWriter's native None rendering).
    """
    s = result.summary()
    row = {
        "image": s["image"],
        "casualty": s["casualty"],
        "route_name": s["route_name"] or "",
        "route_score": s["route_score"],
        "route_admissible": s["route_admissible"],
        "route_coherence": round(s["route_coherence"], 3) if s["route_coherence"] is not None else "",
        "route_completeness": round(s["route_completeness"], 3) if s["route_completeness"] is not None else "",
        "gate_rate": s["gate_rate"],
        "unresolved_gate_count": s["unresolved_gate_count"],
        "self_contradictory_on_size": s["self_contradictory_on_size"],
    }
    for v in STEP_VERDICTS:
        row[f"n_{v}"] = s["counts"].get(v, 0)
    for field in _LIST_FIELDS:
        items = s[field]
        row[f"n_{field}"] = len(items)
        row[f"{field}_text"] = "; ".join(items)
    return row


def build_per_image_rows(results: list) -> list:
    return [_flatten_plan_row(r) for r in results]


def _per_image_fieldnames() -> list:
    """Built from the known column list, NOT rows[0].keys() -- guards
    against aggregate_coherence.py:237's latent crash on an empty run
    (IndexError from indexing an empty per_image_rows list). See the P9
    end-to-end-pipeline plan, Part 1."""
    fields = ["image", "casualty", "route_name", "route_score", "route_admissible",
              "route_coherence", "route_completeness", "gate_rate",
              "unresolved_gate_count", "self_contradictory_on_size"]
    fields += [f"n_{v}" for v in STEP_VERDICTS]
    for field in _LIST_FIELDS:
        fields += [f"n_{field}", f"{field}_text"]
    return fields


PER_IMAGE_FIELDNAMES = _per_image_fieldnames()


# ---------------------------------------------------------------------------
# summary.csv
# ---------------------------------------------------------------------------

#: Numeric per_image.csv columns averaged into summary.csv -- everything
#: except the join keys (image/casualty), the two text columns, and
#: route_admissible/route_name (categorical, not meaningfully mean-able).
_SUMMARY_NUMERIC_FIELDS = (
    ["route_score", "route_coherence", "route_completeness", "gate_rate",
     "unresolved_gate_count"]
    + [f"n_{v}" for v in STEP_VERDICTS]
    + [f"n_{f}" for f in _LIST_FIELDS]
)


def build_summary(run_name: str, per_image_rows: list) -> dict:
    """One row: means of every per_image.csv numeric column, plus
    per-casualty breakdowns via a local _state_mean closure -- same idiom
    as aggregate_coherence.py:249-252, generalized over CASUALTIES rather
    than one hardcoded line per state.

    Reuses _safe_mean (aggregate_coherence.py:57-58): None for an empty
    list ("not measured"), never silently 0.0 ("measured zero") -- every
    consumer of this dict depends on that distinction holding.
    """
    n_images = len(per_image_rows)

    def _mean_of(field, rows=None):
        rows = per_image_rows if rows is None else rows
        vals = [float(r[field]) for r in rows if r.get(field) not in ("", None)]
        return _safe_mean(vals)

    def _state_mean(field, casualty):
        rows = [r for r in per_image_rows if r["casualty"] == casualty]
        return _mean_of(field, rows)

    n_route_recognised = sum(1 for r in per_image_rows if r["route_name"])
    self_contra_rate = (
        sum(1 for r in per_image_rows if str(r["self_contradictory_on_size"]) == "True") / n_images
        if n_images else None
    )

    summary = {
        "run": run_name,
        "n_images": n_images,
        "pct_route_recognised": round(n_route_recognised / n_images, 3) if n_images else None,
        "self_contradictory_on_size_rate": round(self_contra_rate, 3) if self_contra_rate is not None else None,
    }
    for field in _SUMMARY_NUMERIC_FIELDS:
        summary[f"mean_{field}"] = _mean_of(field)
    for casualty in CASUALTIES:
        summary[f"mean_route_score_{casualty}"] = _state_mean("route_score", casualty)
        summary[f"mean_gate_rate_{casualty}"] = _state_mean("gate_rate", casualty)
        summary[f"mean_n_UNSPECIFIED_{casualty}"] = _state_mean("n_UNSPECIFIED", casualty)
    return summary


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def write_csv(path: Path, fieldnames: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def append_cumulative(path: Path, summary: dict) -> None:
    """Same summary dict object written to both summary.csv and here, so
    the two column sets can never drift -- the load-bearing part of
    aggregate_coherence.py:281-287's pattern, not an incidental one.
    Append-only, no dedup -- re-running a run appends a duplicate row,
    matching P8's convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def aggregate_run(run_name: str, tool_calls_path: Path, out_dir: Path = None,
                   gt_path: Optional[Path] = None,
                   cumulative_path: Optional[Path] = None) -> dict:
    """Run executor over tool_calls_path, write per_step/per_image/summary
    CSVs for this run, append to the cumulative CSV. Returns the summary
    dict (report.py's cross-arm narration reads this same shape back out of
    summary.csv, so keep the two in lockstep if either changes).

    cumulative_path defaults to <out_dir>/eval_summary_adequacy.csv, NOT
    unconditionally to the module-level CUMULATIVE_SUMMARY_PATH -- a
    caller passing a non-default out_dir (a test, or an alternate results
    location) must not have its cumulative row silently land in the
    production results/p9_plan_adequacy/ tree instead. Pass cumulative_path
    explicitly to opt out of this pairing."""
    paths = RunPaths(run_name, base_dir=out_dir) if out_dir else RunPaths(run_name)
    if cumulative_path is None:
        cumulative_path = (out_dir / "eval_summary_adequacy.csv") if out_dir else CUMULATIVE_SUMMARY_PATH

    results = run_executor(tool_calls_path, gt_path)
    print(f"  Executed {len(results)} plans")

    per_step_rows = build_per_step_rows(results)
    write_csv(paths.per_step, PER_STEP_FIELDNAMES, per_step_rows)
    print(f"  Per-step  -> {paths.per_step}  ({len(per_step_rows)} rows)")

    per_image_rows = build_per_image_rows(results)
    write_csv(paths.per_image, PER_IMAGE_FIELDNAMES, per_image_rows)
    print(f"  Per-image -> {paths.per_image}  ({len(per_image_rows)} rows)")

    summary = build_summary(run_name, per_image_rows)
    write_csv(paths.summary, list(summary.keys()), [summary])
    print(f"  Summary   -> {paths.summary}")

    append_cumulative(cumulative_path, summary)
    print(f"  Cumulative -> {cumulative_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="P9 stage 2: execute extracted tool calls and roll up per_step/per_image/summary CSVs"
    )
    ap.add_argument("--run", required=True, help="Run name (matches Stage 1's output subdirectory)")
    ap.add_argument("--dir", type=Path, default=None,
                     help="Base output directory (default: paths.BASE_OUT_DIR)")
    ap.add_argument("--gt", type=Path, default=None,
                     help="human_gt.csv path (default: scenario.py's default)")
    args = ap.parse_args()

    from pipelines.plan_adequacy.paths import RunPaths as _RunPaths
    run_paths = _RunPaths(args.run, base_dir=args.dir) if args.dir else _RunPaths(args.run)

    aggregate_run(args.run, run_paths.tool_calls, out_dir=args.dir, gt_path=args.gt)


if __name__ == "__main__":
    main()
