"""
run_executor.py -- the missing link: extracted ToolCalls -> PlanResults.

Stage 2's first half (see the P9 end-to-end-pipeline plan, Part 1). Reads
`tool_calls.jsonl` (Stage 1's output, one record per (image, step)), groups
rows back into per-image ToolCall lists in step order, and calls
executor.execute_plan() once per image. aggregate.py consumes the returned
PlanResults; this module does no CSV writing itself.

`tool_calls.jsonl` schema, one JSON object per line (the contract extract.py's
CLI writes and this module reads):
    {"image": str, "casualty": str, "step_num": int, "step_text": str,
     "tool": str, "params": dict, "conditional": bool,
     "condition_text": str|None, "condition_var": str,
     "secondary_tools": list[str]}

Deliberately mirrors executor@oracle's ToolCall shape (vocab.ToolCall) field
for field, so a hand-written gold set and a real extraction run go through
the identical grouping/execution code -- same rationale as extract.py's
docstring: calibration and production must exercise the same path or neither
proves anything about the other.
"""

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.executor import PlanResult, execute_plan
from pipelines.plan_adequacy.methods import RouteRegistry
from pipelines.plan_adequacy.scenario import load_scenarios
from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry
from shared.loaders import read_jsonl


def _row_to_call(row: dict) -> ToolCall:
    return ToolCall(
        step_num=row["step_num"],
        step_text=row.get("step_text", ""),
        tool=row.get("tool", "no_match"),
        params=dict(row.get("params") or {}),
        conditional=bool(row.get("conditional", False)),
        condition_text=row.get("condition_text"),
        condition_var=row.get("condition_var") or "none",
        secondary_tools=tuple(row.get("secondary_tools") or ()),
    )


def group_tool_calls(tool_calls_path: Path) -> dict:
    """{image -> list[ToolCall]}, each list sorted by step_num. Rows for an
    image need not already be contiguous or ordered in the file -- Stage 1
    can be resumed/re-run per-image, so this makes no assumption about file
    ordering beyond "all rows for one image share its `image` value"."""
    by_image = {}
    for row in read_jsonl(tool_calls_path):
        image = row.get("image")
        if not image:
            continue
        by_image.setdefault(image, []).append(row)

    grouped = {}
    for image, rows in by_image.items():
        rows.sort(key=lambda r: r.get("step_num", 0))
        grouped[image] = [_row_to_call(r) for r in rows]
    return grouped


def run_executor(tool_calls_path: Path, gt_path: Path = None) -> list:
    """Read tool_calls.jsonl, execute every plan, return list[PlanResult] in
    the same image order group_tool_calls() produced them (dict insertion
    order -- Python 3.7+, and stable since tool_calls.jsonl is read once
    top to bottom).

    `tool_registry`/`route_registry`/scenario lookups are loaded ONCE for
    the whole run, not per image -- see the plan's Part 1 note on this;
    ToolRegistry.load()/RouteRegistry.load() both re-parse a JSON file, and
    a run can cover 100+ images.
    """
    tool_registry = ToolRegistry.load()
    route_registry = RouteRegistry.load()
    scenarios = load_scenarios(gt_path)

    grouped = group_tool_calls(tool_calls_path)

    results = []
    for image, calls in grouped.items():
        scenario = scenarios.get(image)
        if scenario is None:
            # No ground-truth row for this image -- can't determine
            # casualty/size/habitat-sensitivity, so this plan can't be
            # graded. Skip rather than guess; aggregate.py's row count will
            # visibly undercount versus tool_calls.jsonl if this fires a
            # lot, which is the intended signal that inputs are mismatched.
            print(f"  WARNING: no scenario for '{image}' in ground truth -- skipping.")
            continue
        casualty = scenario.state
        # plan_text is reconstructed from the extracted steps' own text
        # (joined in step order) rather than re-reading the original answer
        # JSONL here -- gate_rate/is_self_contradictory_on_size (gates.py)
        # are plain regex scans over raw text, and step_text already IS the
        # plan's raw sentences, just pre-split. Avoids a second file read
        # and a second (image -> plan_text) join.
        plan_text = "\n".join(c.step_text for c in calls)
        result = execute_plan(calls, casualty, scenario, tool_registry,
                               route_registry, plan_text=plan_text)
        results.append(result)
    return results
