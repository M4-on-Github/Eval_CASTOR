"""
report.py -- Stage 2's narrative half: summary.csv -> report.md, plus
case_studies.md (the record-level evidence a report.md claim traces back
to). See the P9 end-to-end-pipeline plan, Part 1 ("report.py") and Part 1c
("example mining: numbers -> records").

Mechanism mirrors pipelines/plan_coherence/improved/eval/aggregate.py's
report generator (aggregate.py:176-307): no template file, no Jinja -- a
list[str] of markdown lines, "\\n".join-ed once at the end. That module uses
pandas DataFrames; this one deliberately does not (P9's other stage-2
modules are stdlib-csv/dict based throughout -- aggregate.py, run_executor.py
-- and report.py stays pure dict-in/str-out so it's testable against
synthetic rows with no CSV/pandas round trip required).

Narration is threshold-based on raw deltas/rates with three verdict glyphs
(pass/marginal/fail), same honesty as P8's hypothesis section -- no p-value,
no bootstrap. Wrapped in try/except so one missing field degrades a section
instead of failing the whole report, mirroring aggregate.py:227,243-244.
Does NOT mirror aggregate.py:241's bug (a `.format()` nested inside an
f-string) -- fmt() is used everywhere instead.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.executor import STEP_VERDICTS
from pipelines.plan_adequacy.paths import RunPaths
from shared.loaders import read_jsonl


def fmt(val) -> str:
    """N/A for missing, 3-decimal fixed point otherwise -- mirrors
    improved/eval/aggregate.py:47-50's fmt(), used in every table cell and
    every narrative sentence here so a None-able metric never renders as
    the string "None" on a slide."""
    if val is None or val == "":
        return "N/A"
    try:
        return f"{float(val):.3f}"
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# Loading (thin CSV/JSONL wrappers -- report.py's own module boundary)
# ---------------------------------------------------------------------------

def load_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_records_jsonl(path: Optional[Path]) -> list:
    """calibrate.py --dump-records output (records_<model>.jsonl) -- see
    Part 1c. Returns [] (not an error) when path is None or missing, so a
    report can still be generated for a run with no calibration dump."""
    if path is None or not path.exists():
        return []
    return list(read_jsonl(path))


# ---------------------------------------------------------------------------
# Example mining -- Part 1c: numbers -> records, deterministically
# ---------------------------------------------------------------------------

def pick_step_examples(per_step_rows: list, verdict: str, n: int = 3) -> list:
    """Deterministic: sort candidates by (image, step_num) ascending, take
    the first n. Same input rows in any order produce the same output --
    see test_plan_adequacy_report.py's determinism test, which is what
    makes these examples reportable rather than anecdotal (Part 1c)."""
    candidates = [r for r in per_step_rows if r.get("verdict") == verdict]
    candidates.sort(key=lambda r: (r.get("image", ""), int(r.get("step_num", 0))))
    return candidates[:n]


def pick_confusion_examples(records: list, n_pairs: int = 3, per_pair: int = 1) -> list:
    """From calibrate.py's --dump-records rows: the most frequent
    (gold_tool, predicted_tool) MISTAKEN pairs, deterministically. Sort
    order is frequency descending, then the pair name ascending as a fixed
    tie-break -- never insertion order, so a re-run of the exact same
    calibration data always narrates the same examples. Deliberately does
    NOT exclude the top (most embarrassing) pair -- see Part 1c's selection
    rule: a progress report that only shows the method working is not
    evidence."""
    wrong = [r for r in records if r.get("tool_correct") is False]

    pair_counts = {}
    pair_rows = {}
    for r in wrong:
        pair = (r.get("gold_tool"), r.get("predicted_tool"))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        pair_rows.setdefault(pair, []).append(r)

    ranked_pairs = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    examples = []
    for pair, _count in ranked_pairs[:n_pairs]:
        rows = sorted(pair_rows[pair], key=lambda r: str(r.get("gold_id", "")))
        examples.extend(rows[:per_pair])
    return examples


# ---------------------------------------------------------------------------
# case_studies.md
# ---------------------------------------------------------------------------

def format_step_case(row: dict, label: str = "") -> str:
    header = f"### {label}" if label else "### Step example"
    return "\n".join([
        header,
        f"- **Image / step**: `{row.get('image', 'N/A')}` step {row.get('step_num', 'N/A')}",
        f"- **Step text**: {row.get('step_text', '')!r}",
        f"- **Tool**: {row.get('tool', 'N/A')}",
        f"- **Verdict**: {row.get('verdict', 'N/A')}",
        f"- **Detail**: {row.get('detail', '')}",
    ])


def format_confusion_case(row: dict) -> str:
    return "\n".join([
        f"### Confusion: gold `{row.get('gold_tool')}` -> predicted `{row.get('predicted_tool')}`",
        f"- **Gold ID**: `{row.get('gold_id', 'N/A')}`",
        f"- **Step text**: {row.get('step_text', '')!r}",
        f"- **Gold params**: {row.get('gold_params', {})}",
        f"- **Predicted params**: {row.get('predicted_params', {})}",
    ])


#: Ordered by how actionable/interesting a failure category is, not by
#: STEP_VERDICTS' declared order -- SEQUENCE_VIOLATION and METHOD_ERROR are
#: genuine plan defects worth leading with; NO_MATCH is closer to noise.
_CASE_STUDY_VERDICTS = (
    "SEQUENCE_VIOLATION", "METHOD_ERROR", "CONDITIONAL_UNRESOLVED", "UNSPECIFIED", "NO_MATCH",
)


def write_case_studies(per_step_rows: list, calibration_records: list, out_path: Path,
                        n_per_verdict: int = 2, n_confusion_pairs: int = 3) -> None:
    lines = ["# Case Studies", "",
             "Record-level evidence behind report.md's claims. Selection is deterministic "
             "(see pick_step_examples/pick_confusion_examples) -- re-running against the same "
             "data reproduces the same examples.", ""]

    lines.append("## Plan-validity examples")
    lines.append("")
    for verdict in _CASE_STUDY_VERDICTS:
        examples = pick_step_examples(per_step_rows, verdict, n=n_per_verdict)
        lines.append(f"### {verdict}")
        if not examples:
            lines.append("_No examples in this run._")
            lines.append("")
            continue
        for ex in examples:
            lines.append(format_step_case(ex))
            lines.append("")

    lines.append("## Calibration confusion examples")
    lines.append("")
    confusions = pick_confusion_examples(calibration_records, n_pairs=n_confusion_pairs)
    if not confusions:
        lines.append("_No calibration records available -- pass --records to include this "
                      "section (see calibrate.py --dump-records)._")
    else:
        for row in confusions:
            lines.append(format_confusion_case(row))
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Case studies -> {out_path}")


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

#: Effect-size cutoffs for the narrative glyphs -- see build_narrative().
#: Same honesty convention as P8's hypothesis section (aggregate.py:226-244):
#: raw deltas against a declared cutoff, not a significance test.
_PASS_GLYPH, _MARGINAL_GLYPH, _FAIL_GLYPH = "✓", "~", "✗"


def _step_total(summary: dict) -> Optional[float]:
    """Sum of mean_n_<verdict> across STEP_VERDICTS -- summary.csv carries
    per-verdict means but not a "total steps" field directly, so the
    denominator for a rate is derived here rather than adding a new
    aggregate.py column for one report-only convenience."""
    vals = [summary.get(f"mean_n_{v}") for v in STEP_VERDICTS]
    vals = [float(v) for v in vals if v not in (None, "")]
    return sum(vals) if vals else None


def build_narrative(summary: dict) -> list:
    """The findings-assessment section's lines. Wrapped by the caller in
    try/except so a missing/malformed summary degrades this ONE section
    rather than failing the whole report -- mirrors improved/eval/
    aggregate.py:227,243-244."""
    lines = []

    pct_goal = summary.get("pct_goal_reached")
    if pct_goal is not None:
        glyph = _PASS_GLYPH if float(pct_goal) > 0.5 else _MARGINAL_GLYPH if float(pct_goal) > 0.1 else _FAIL_GLYPH
        lines.append(
            f"- Goal reached: {fmt(pct_goal)} of plans ({glyph}) -- zero violations anywhere in the plan "
            f"AND the casualty's terminal state actually established. The strictest single number this "
            f"report produces; a plan can score well on every other metric and still fail this one."
        )

    total = _step_total(summary)
    mean_unspecified = summary.get("mean_n_UNSPECIFIED")
    if total and mean_unspecified is not None:
        rate = float(mean_unspecified) / total
        glyph = _FAIL_GLYPH if rate > 0.5 else _MARGINAL_GLYPH if rate > 0.15 else _PASS_GLYPH
        lines.append(
            f"- UNSPECIFIED rate: {fmt(rate)} of steps ({glyph}) -- the primary discriminator "
            f"this project measures. High is not itself a failure; it means plans are naming "
            f"actions without committing to a magnitude."
        )

    gate_rate = summary.get("mean_gate_rate")
    if gate_rate is not None:
        lines.append(f"- Mean gate_rate (hedging language per plan): {fmt(gate_rate)}.")

    route_coh = summary.get("mean_route_coherence")
    if route_coh is not None:
        glyph = _FAIL_GLYPH if float(route_coh) < 0.5 else _MARGINAL_GLYPH if float(route_coh) < 0.8 else _PASS_GLYPH
        lines.append(
            f"- Mean route_coherence: {fmt(route_coh)} ({glyph}) -- low values flag plans "
            f"naming every technique without committing to one, the shotgun-plan pattern."
        )

    n_unused = summary.get("mean_n_unused_assessments")
    if n_unused is not None:
        lines.append(
            f"- Mean unused assessments per plan: {fmt(n_unused)} -- assessments performed but "
            f"never consumed by a later step (the \"hollow diagnostic\" pattern)."
        )

    if not lines:
        lines.append("_Could not assess: summary is missing the fields this section reads._")
    return lines


def build_results_table(summary_rows: list) -> list:
    """One row per run (summary_rows may be a single run's [summary] or
    several rows read from the cumulative eval_summary_adequacy.csv for
    cross-arm comparison)."""
    lines = [
        "| Run | N | Route recognised | Mean UNSPECIFIED | Mean gate_rate | Mean route_coherence |",
        "|---|---|---|---|---|---|",
    ]
    for s in summary_rows:
        lines.append(
            f"| {s.get('run', 'N/A')} | {s.get('n_images', 'N/A')} | "
            f"{fmt(s.get('pct_route_recognised'))} | {fmt(s.get('mean_n_UNSPECIFIED'))} | "
            f"{fmt(s.get('mean_gate_rate'))} | {fmt(s.get('mean_route_coherence'))} |"
        )
    return lines


def write_report(summary_rows: list, out_path: Path, case_studies_filename: str = "case_studies.md") -> None:
    lines = [
        "# P9 Plan Adequacy -- Evaluation Report",
        "",
        "## Experiment summary",
        "",
        "- Static plan checking (extraction, not agentic tool use): each step is parsed into a "
        "structured tool call, then walked deterministically against a route-scoped world model.",
        "- Plans are graded against whichever of several valid routes they recognisably follow, "
        "not one canonical sequence.",
        "- Numeric adequacy grading (whether a stated magnitude is itself correct) is a phase-2 "
        "placeholder -- every graded step reads SPECIFIED_UNGRADED, never SPECIFIED_ADEQUATE, "
        "until physics.py exists.",
        "",
        "## Results",
        "",
    ]
    lines += build_results_table(summary_rows)
    lines += ["", "## Findings assessment", ""]

    if summary_rows:
        if len(summary_rows) > 1:
            # NOTE: this narrates the LAST row only -- no cross-arm delta
            # computation (a la P8's STANDARD-vs-CONTROL deltas) exists yet.
            # Said explicitly rather than silently comparing one arm and
            # implying the others were considered -- see the Results table
            # above for every arm's raw numbers.
            lines.append(
                f"_{len(summary_rows)} runs in the Results table above; findings below narrate "
                f"only the most recent ({summary_rows[-1].get('run', 'N/A')}). Cross-arm delta "
                f"narration is not yet implemented -- compare the table rows directly._"
            )
            lines.append("")
        try:
            lines += build_narrative(summary_rows[-1])
        except Exception as e:
            lines.append(f"_Could not assess findings: {e}_")
    else:
        lines.append("_No summary rows available._")

    lines += [
        "",
        "---",
        f"_See {case_studies_filename} for the record-level examples behind these numbers._",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="P9 stage 2: generate report.md + case_studies.md from a run's CSVs"
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--records", type=Path, default=None,
                     help="Optional calibrate.py --dump-records JSONL for confusion examples")
    args = ap.parse_args()

    paths = RunPaths(args.run, base_dir=args.dir) if args.dir else RunPaths(args.run)

    per_step_rows = load_csv_rows(paths.per_step)
    summary_rows = load_csv_rows(paths.summary)
    calibration_records = load_records_jsonl(args.records)

    write_case_studies(per_step_rows, calibration_records, paths.case_studies)
    write_report(summary_rows, paths.report, case_studies_filename=paths.CASE_STUDIES)


if __name__ == "__main__":
    main()
