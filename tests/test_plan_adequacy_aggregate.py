"""
Tests for pipelines/plan_adequacy/aggregate.py

Pure CSV-shape/rollup logic, driven by hand-built PlanResult objects (via
executor.execute_plan(), same executor@oracle convention as
test_plan_adequacy_executor.py) -- no model, no cluster, no file I/O beyond
pytest's tmp_path for the write_csv/append_cumulative tests.
Run: python -m pytest tests/test_plan_adequacy_aggregate.py -v
"""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.aggregate import (
    PER_IMAGE_FIELDNAMES,
    PER_STEP_FIELDNAMES,
    aggregate_run,
    append_cumulative,
    build_per_image_rows,
    build_per_step_rows,
    build_summary,
    write_csv,
)
from pipelines.plan_adequacy.executor import STEP_VERDICTS, execute_plan
from pipelines.plan_adequacy.methods import RouteRegistry
from pipelines.plan_adequacy.paths import RunPaths
from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry

_TOOL_REG = ToolRegistry.load()
_ROUTE_REG = RouteRegistry.load()


def _scenario(**kw):
    kw.setdefault("image", "aground/1.jpg")
    return SimpleNamespace(**kw)


def _call(n, tool, text="", params=None, conditional=False,
          condition_text=None, condition_var="none"):
    return ToolCall(step_num=n, step_text=text or tool, tool=tool,
                     params=params or {}, conditional=conditional,
                     condition_text=condition_text, condition_var=condition_var)


def _clean_tug_pull_plan(image="aground/1.jpg"):
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", text="Deploy 2 tugs at 4000 shp.", params={"count": 2, "shp": 4000.0}),
        _call(6, "pull", text="Pull with 90 tons of force.", params={"force_t": 90}),
    ]
    plan_text = "\n".join(c.step_text for c in calls)
    return execute_plan(calls, "aground", _scenario(image=image), _TOOL_REG, _ROUTE_REG, plan_text=plan_text)


# ── build_per_step_rows ──────────────────────────────────────────────────

def test_per_step_rows_one_row_per_step_across_plans():
    r1 = _clean_tug_pull_plan("aground/1.jpg")
    r2 = _clean_tug_pull_plan("aground/2.jpg")
    rows = build_per_step_rows([r1, r2])
    assert len(rows) == len(r1.steps) + len(r2.steps)
    assert set(rows[0].keys()) == set(PER_STEP_FIELDNAMES)


def test_per_step_rows_carry_image_and_casualty_onto_every_step():
    r = _clean_tug_pull_plan("aground/1.jpg")
    rows = build_per_step_rows([r])
    assert all(row["image"] == "aground/1.jpg" for row in rows)
    assert all(row["casualty"] == "aground" for row in rows)


# ── build_per_image_rows / _flatten_plan_row ─────────────────────────────

def test_per_image_row_has_one_column_per_step_verdict():
    r = _clean_tug_pull_plan()
    row = build_per_image_rows([r])[0]
    for v in STEP_VERDICTS:
        assert f"n_{v}" in row


def test_per_image_row_verdict_columns_are_zero_not_missing_when_unused():
    """The sparse-counts-dict guard: a verdict this plan never produced
    must still get an explicit 0 column, not be absent -- see
    _flatten_plan_row's docstring on why counts.get(v, 0) is used instead
    of counts.keys()."""
    r = _clean_tug_pull_plan()  # every step is SPECIFIED_UNGRADED
    row = build_per_image_rows([r])[0]
    assert row["n_SPECIFIED_UNGRADED"] == 6
    assert row["n_UNSPECIFIED"] == 0
    assert row["n_METHOD_ERROR"] == 0


def test_per_image_row_list_fields_get_count_and_text_columns():
    r = _clean_tug_pull_plan()
    row = build_per_image_rows([r])[0]
    assert row["n_not_attempted"] == len(r.not_attempted)
    assert row["not_attempted_text"] == "; ".join(r.not_attempted)


def test_per_image_row_self_contradictory_is_real_bool_not_1_0_string():
    """P9's explicit decision (see the plan, Part 1): unlike P8's
    aggregate_coherence.py:189 "1"/"0" strings, self_contradictory_on_size
    stays a real Python bool through _flatten_plan_row -- csv.DictWriter
    renders it as the string "True"/"False", not "1"/"0"."""
    r = _clean_tug_pull_plan()
    row = build_per_image_rows([r])[0]
    assert row["self_contradictory_on_size"] is False


def test_per_image_fieldnames_do_not_depend_on_rows_existing():
    """Guards against aggregate_coherence.py:237's latent IndexError on an
    empty run (fieldnames derived from rows[0].keys()) -- PER_IMAGE_FIELDNAMES
    must be constructible with zero PlanResults."""
    assert len(PER_IMAGE_FIELDNAMES) > 0
    assert "image" in PER_IMAGE_FIELDNAMES


# ── build_summary ────────────────────────────────────────────────────────

def test_summary_n_images_matches_row_count():
    rows = build_per_image_rows([_clean_tug_pull_plan("aground/1.jpg"),
                                  _clean_tug_pull_plan("aground/2.jpg")])
    summary = build_summary("test_run", rows)
    assert summary["n_images"] == 2


def test_summary_is_none_not_zero_on_an_empty_run():
    """_safe_mean's None-vs-0.0 distinction must survive into summary.csv:
    an empty run means "not measured", never "measured zero"."""
    summary = build_summary("empty_run", [])
    assert summary["n_images"] == 0
    assert summary["mean_route_score"] is None
    assert summary["pct_route_recognised"] is None


def test_summary_per_casualty_breakdown_only_averages_matching_rows():
    aground_rows = build_per_image_rows([_clean_tug_pull_plan("aground/1.jpg")])
    summary = build_summary("test_run", aground_rows)
    assert summary["mean_route_score_aground"] == 1.0
    assert summary["mean_route_score_capsized"] is None


# ── CSV writing / cumulative append ──────────────────────────────────────

def test_write_csv_round_trips_rows(tmp_path):
    rows = build_per_image_rows([_clean_tug_pull_plan()])
    out = tmp_path / "per_image.csv"
    write_csv(out, PER_IMAGE_FIELDNAMES, rows)
    with out.open(encoding="utf-8") as f:
        read_back = list(csv.DictReader(f))
    assert read_back[0]["image"] == "aground/1.jpg"


def test_append_cumulative_writes_header_only_once(tmp_path):
    cum = tmp_path / "eval_summary_adequacy.csv"
    s1 = build_summary("run1", build_per_image_rows([_clean_tug_pull_plan("aground/1.jpg")]))
    s2 = build_summary("run2", build_per_image_rows([_clean_tug_pull_plan("aground/2.jpg")]))
    append_cumulative(cum, s1)
    append_cumulative(cum, s2)
    with cum.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert [r["run"] for r in rows] == ["run1", "run2"]


def test_append_cumulative_column_set_matches_summary_csv_exactly(tmp_path):
    """The load-bearing property from aggregate_coherence.py:281-287: the
    SAME summary dict object must be written to both summary.csv and the
    cumulative file, so the two column sets can never drift."""
    summary = build_summary("run1", build_per_image_rows([_clean_tug_pull_plan()]))
    summary_csv = tmp_path / "summary.csv"
    cumulative_csv = tmp_path / "eval_summary_adequacy.csv"
    write_csv(summary_csv, list(summary.keys()), [summary])
    append_cumulative(cumulative_csv, summary)
    with summary_csv.open(encoding="utf-8") as f:
        summary_fields = csv.DictReader(f).fieldnames
    with cumulative_csv.open(encoding="utf-8") as f:
        cumulative_fields = csv.DictReader(f).fieldnames
    assert summary_fields == cumulative_fields


# ── aggregate_run: the actual CLI orchestration function ────────────────
#
# Everything above tests the pure building blocks aggregate_run() chains
# together; NONE of it previously exercised aggregate_run() itself -- the
# function main() actually calls. Added on a recheck pass specifically
# because that gap let a real bug (cumulative_path silently ignoring a
# custom out_dir) ship untested. See the fix in aggregate.py: cumulative_path
# now defaults to <out_dir>/eval_summary_adequacy.csv, not unconditionally
# to the production CUMULATIVE_SUMMARY_PATH.

_GT_HEADER = "image,state,vessel_type,,cargo,error,q1,q2,q3,q4,q5,size_estimate\n"


def _write_gt(tmp_path, rows):
    path = tmp_path / "human_gt.csv"
    lines = [_GT_HEADER]
    for image, state, vessel_type in rows:
        lines.append(f"{image},{state},{vessel_type},,,,yes,no,no,no,no,medium (10-50m)\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_tool_calls(tmp_path, rows):
    path = tmp_path / "tool_calls.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _tc_row(image, step_num, tool, text="", params=None, casualty="aground"):
    return {
        "image": image, "casualty": casualty, "step_num": step_num,
        "step_text": text or tool, "tool": tool, "params": params or {},
        "conditional": False, "condition_text": None, "condition_var": "none",
        "secondary_tools": [],
    }


def test_aggregate_run_writes_all_four_artifacts_under_out_dir(tmp_path):
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship")])
    tc = _write_tool_calls(tmp_path, [_tc_row("aground/1.jpg", 1, "sound_tanks")])
    out_dir = tmp_path / "out"

    aggregate_run("test_run", tc, out_dir=out_dir, gt_path=gt)

    paths = RunPaths("test_run", base_dir=out_dir)
    assert paths.per_step.exists()
    assert paths.per_image.exists()
    assert paths.summary.exists()


def test_aggregate_run_cumulative_csv_lands_under_the_passed_out_dir_not_the_default(tmp_path):
    """Regression anchor: aggregate_run's cumulative_path used to default
    unconditionally to the production CUMULATIVE_SUMMARY_PATH regardless of
    out_dir, so a caller using a non-default out_dir (any test, or an
    alternate results location) would silently write its cumulative row
    into results/p9_plan_adequacy/ instead of alongside its own output."""
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship")])
    tc = _write_tool_calls(tmp_path, [_tc_row("aground/1.jpg", 1, "sound_tanks")])
    out_dir = tmp_path / "out"

    aggregate_run("test_run", tc, out_dir=out_dir, gt_path=gt)

    cumulative = out_dir / "eval_summary_adequacy.csv"
    assert cumulative.exists()
    with cumulative.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["run"] == "test_run"


def test_aggregate_run_summary_row_matches_build_summary_output(tmp_path):
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship")])
    tc = _write_tool_calls(tmp_path, [
        _tc_row("aground/1.jpg", 1, "sound_tanks", params={"tank_ids": ["1"]}),
    ])
    out_dir = tmp_path / "out"

    returned_summary = aggregate_run("test_run", tc, out_dir=out_dir, gt_path=gt)

    paths = RunPaths("test_run", base_dir=out_dir)
    with paths.summary.open(encoding="utf-8") as f:
        written = list(csv.DictReader(f))[0]
    assert written["run"] == returned_summary["run"] == "test_run"
    assert written["n_images"] == str(returned_summary["n_images"])
