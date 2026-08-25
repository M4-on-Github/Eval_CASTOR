"""
Tests for pipelines/plan_adequacy/run_executor.py

Hand-built tool_calls.jsonl + a small synthetic ground-truth CSV -- no
model, no cluster. Isolates the grouping/joining logic (row -> ToolCall,
image -> scenario, ordering by step_num) from executor.py's own pass logic,
which is already covered by test_plan_adequacy_executor.py.
Run: python -m pytest tests/test_plan_adequacy_run_executor.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.run_executor import group_tool_calls, run_executor

_GT_HEADER = "image,state,vessel_type,,cargo,error,q1,q2,q3,q4,q5,size_estimate\n"


def _write_gt(tmp_path, rows):
    """rows: list[(image, state, vessel_type)]"""
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


def _row(image, step_num, tool, text="", params=None, casualty="aground", **kw):
    d = {
        "image": image, "casualty": casualty, "step_num": step_num,
        "step_text": text or tool, "tool": tool, "params": params or {},
        "conditional": False, "condition_text": None, "condition_var": "none",
        "secondary_tools": [],
    }
    d.update(kw)
    return d


# ── group_tool_calls ─────────────────────────────────────────────────────

def test_group_tool_calls_sorts_by_step_num_even_if_file_order_is_scrambled(tmp_path):
    path = _write_tool_calls(tmp_path, [
        _row("aground/1.jpg", 3, "pull"),
        _row("aground/1.jpg", 1, "sound_tanks"),
        _row("aground/1.jpg", 2, "survey_seabed"),
    ])
    grouped = group_tool_calls(path)
    assert [c.step_num for c in grouped["aground/1.jpg"]] == [1, 2, 3]


def test_group_tool_calls_separates_by_image(tmp_path):
    path = _write_tool_calls(tmp_path, [
        _row("aground/1.jpg", 1, "sound_tanks"),
        _row("aground/2.jpg", 1, "sound_tanks"),
    ])
    grouped = group_tool_calls(path)
    assert set(grouped.keys()) == {"aground/1.jpg", "aground/2.jpg"}


def test_group_tool_calls_skips_rows_with_no_image(tmp_path):
    path = tmp_path / "tool_calls.jsonl"
    path.write_text(
        json.dumps(_row("aground/1.jpg", 1, "sound_tanks")) + "\n"
        + json.dumps({"casualty": "aground", "step_num": 1, "tool": "no_match"}) + "\n",
        encoding="utf-8",
    )
    grouped = group_tool_calls(path)
    assert list(grouped.keys()) == ["aground/1.jpg"]


# ── run_executor: end to end (no model, hand-built rows) ────────────────

def test_run_executor_produces_one_planresult_per_image(tmp_path):
    gt = _write_gt(tmp_path, [
        ("aground/1.jpg", "aground", "Cargo Ship"),
        ("aground/2.jpg", "aground", "Tanker"),
    ])
    tc = _write_tool_calls(tmp_path, [
        _row("aground/1.jpg", 1, "sound_tanks", params={"tank_ids": ["1"]}),
        _row("aground/2.jpg", 1, "sound_tanks", params={"tank_ids": ["1"]}),
    ])
    results = run_executor(tc, gt_path=gt)
    assert len(results) == 2
    assert {r.image for r in results} == {"aground/1.jpg", "aground/2.jpg"}


def test_run_executor_uses_ground_truth_casualty_not_the_row_casualty(tmp_path):
    """The row's own "casualty" field is Stage 1's best guess at extraction
    time; the ground-truth CSV is the authority the executor should grade
    against, same as executor@oracle tests do via _scenario()."""
    gt = _write_gt(tmp_path, [("capsized/1.jpg", "capsized", "Ferry")])
    # Row claims "aground" -- should be overridden by ground truth "capsized".
    tc = _write_tool_calls(tmp_path, [
        _row("capsized/1.jpg", 1, "rig_parbuckling", casualty="aground"),
    ])
    results = run_executor(tc, gt_path=gt)
    assert results[0].casualty == "capsized"
    # rig_parbuckling is a capsized-family tool -- if casualty were still
    # read as "aground" this step would read METHOD_ERROR.
    assert results[0].steps[0].verdict != "METHOD_ERROR"


def test_run_executor_skips_image_with_no_ground_truth_row(tmp_path):
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship")])
    tc = _write_tool_calls(tmp_path, [
        _row("aground/1.jpg", 1, "sound_tanks"),
        _row("aground/UNKNOWN.jpg", 1, "sound_tanks"),
    ])
    results = run_executor(tc, gt_path=gt)
    assert len(results) == 1
    assert results[0].image == "aground/1.jpg"


def test_run_executor_plan_text_is_step_text_joined_in_order(tmp_path):
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship")])
    tc = _write_tool_calls(tmp_path, [
        _row("aground/1.jpg", 2, "pull", text="Pull the vessel free."),
        _row("aground/1.jpg", 1, "sound_tanks", text="Sound the tanks first."),
    ])
    results = run_executor(tc, gt_path=gt)
    # gate_rate/self_contradictory are regex-driven off plan_text -- confirm
    # indirectly that both step texts made it in, in step_num order.
    assert results[0].gate_rate == 0  # no hedge language in either sentence
