"""
Tests for pipelines/plan_adequacy/calibrate.py

Pure scoring/aggregation logic only -- run_calibration() (the vLLM bake-off)
is not exercised here, matching the repo convention of testing pipeline
logic against synthetic data rather than a real model. See design plan
section 4e: this is exactly the "scoring logic buildable without cluster
access" half.
Run: python -m pytest tests/test_plan_adequacy_calibrate.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.calibrate import (
    THRESHOLDS,
    aggregate,
    check_thresholds,
    load_gold,
    score_record,
    stratify,
)
from pipelines.plan_adequacy.vocab import ToolCall

GOLD_PATH = Path(__file__).parent.parent / "pipelines" / "plan_adequacy" / "calibration" / "gold_tool_calls.jsonl"


def _gold(tool="attach_tug", params=None, conditional=False, condition_var="none",
          layer="A", failure_type=None, casualty="aground", gold_id="x0001"):
    return {
        "gold_id": gold_id, "layer": layer, "failure_type": failure_type,
        "casualty": casualty, "expected_tool": tool,
        "expected_params": params or {}, "expected_conditional": conditional,
        "expected_condition_var": condition_var,
    }


def _call(tool="attach_tug", params=None, conditional=False, condition_var="none",
          secondary=()):
    return ToolCall(step_num=1, step_text="x", tool=tool, params=params or {},
                     conditional=conditional, condition_var=condition_var,
                     secondary_tools=tuple(secondary))


# ── score_record ──────────────────────────────────────────────────────────

def test_score_record_exact_match():
    s = score_record(_gold(tool="attach_tug"), _call(tool="attach_tug"), parse_ok=True)
    assert s["tool_correct"] is True


def test_score_record_wrong_tool():
    s = score_record(_gold(tool="attach_tug"), _call(tool="pull"), parse_ok=True)
    assert s["tool_correct"] is False


def test_score_record_null_fidelity_correct_when_both_null():
    s = score_record(_gold(params={"count": None}), _call(params={}), parse_ok=True)
    assert s["null_checks"] == [True]


def test_score_record_null_fidelity_wrong_when_model_hallucinates_value():
    s = score_record(_gold(params={"count": None}), _call(params={"count": 4}), parse_ok=True)
    assert s["null_checks"] == [False]


def test_score_record_value_check_when_gold_has_a_value():
    s = score_record(_gold(params={"count": 2}), _call(params={"count": 2}), parse_ok=True)
    assert s["value_checks"] == [True]
    s2 = score_record(_gold(params={"count": 2}), _call(params={"count": 3}), parse_ok=True)
    assert s2["value_checks"] == [False]


def test_score_record_condition_var_only_checked_when_gold_conditional():
    s_unconditional = score_record(_gold(conditional=False), _call(condition_var="draft"), parse_ok=True)
    assert s_unconditional["condition_var_correct"] is None

    s_conditional = score_record(
        _gold(conditional=True, condition_var="vessel_size"),
        _call(conditional=True, condition_var="vessel_size"),
        parse_ok=True,
    )
    assert s_conditional["condition_var_correct"] is True


def test_score_record_no_match_flags():
    s = score_record(_gold(tool="no_match"), _call(tool="no_match"), parse_ok=True)
    assert s["gold_is_no_match"] is True
    assert s["predicted_is_no_match"] is True


# ── aggregate ──────────────────────────────────────────────────────────────

def test_aggregate_empty_returns_n_zero():
    assert aggregate([]) == {"n": 0}


def test_aggregate_micro_accuracy():
    scored = [
        score_record(_gold(tool="attach_tug"), _call(tool="attach_tug"), True),
        score_record(_gold(tool="attach_tug"), _call(tool="pull"), True),
        score_record(_gold(tool="pull"), _call(tool="pull"), True),
        score_record(_gold(tool="pull"), _call(tool="pull"), True),
    ]
    agg = aggregate(scored)
    assert agg["n"] == 4
    assert agg["tool_id_micro_accuracy"] == 0.75


def test_aggregate_macro_accuracy_weights_rare_tools_equally():
    # attach_tug: 1/1 correct. pull: 1/3 correct. Macro should NOT be
    # dominated by pull's larger n (micro would be 2/4=0.5; macro should
    # differ since it's the mean of per-tool recalls: (1.0 + 1/3)/2).
    scored = [
        score_record(_gold(tool="attach_tug"), _call(tool="attach_tug"), True),
        score_record(_gold(tool="pull"), _call(tool="pull"), True),
        score_record(_gold(tool="pull"), _call(tool="attach_tug"), True),
        score_record(_gold(tool="pull"), _call(tool="attach_tug"), True),
    ]
    agg = aggregate(scored)
    assert agg["tool_id_micro_accuracy"] == 0.5
    assert abs(agg["tool_id_macro_accuracy"] - (1.0 + 1 / 3) / 2) < 1e-3  # aggregate() rounds to 4dp


def test_aggregate_null_fidelity():
    scored = [
        score_record(_gold(params={"count": None}), _call(params={}), True),          # correct null
        score_record(_gold(params={"count": None}), _call(params={"count": 4}), True),  # hallucinated
    ]
    agg = aggregate(scored)
    assert agg["null_fidelity"] == 0.5


def test_aggregate_conditional_f1_perfect():
    scored = [
        score_record(_gold(conditional=True), _call(conditional=True), True),
        score_record(_gold(conditional=False), _call(conditional=False), True),
    ]
    agg = aggregate(scored)
    assert agg["conditional_f1"] == 1.0


def test_aggregate_no_match_f1():
    scored = [
        score_record(_gold(tool="no_match"), _call(tool="no_match"), True),
        score_record(_gold(tool="attach_tug"), _call(tool="no_match"), True),  # false positive
        score_record(_gold(tool="no_match"), _call(tool="attach_tug"), True),  # false negative
    ]
    agg = aggregate(scored)
    # tp=1, fp=1, fn=1 -> precision=0.5, recall=0.5, f1=0.5
    assert agg["no_match_precision"] == 0.5
    assert agg["no_match_recall"] == 0.5
    assert agg["no_match_f1"] == 0.5


def test_aggregate_parse_failure_rate():
    scored = [
        score_record(_gold(), _call(), parse_ok=True),
        score_record(_gold(), _call(tool="no_match"), parse_ok=False),
    ]
    agg = aggregate(scored)
    assert agg["parse_failure_rate"] == 0.5


# ── stratify ──────────────────────────────────────────────────────────────

def test_stratify_by_layer():
    scored = [
        score_record(_gold(tool="attach_tug", layer="A"), _call(tool="attach_tug"), True),
        score_record(_gold(tool="attach_tug", layer="B"), _call(tool="pull"), True),
    ]
    strat = stratify(scored, "layer")
    assert strat["A"]["tool_id_micro_accuracy"] == 1.0
    assert strat["B"]["tool_id_micro_accuracy"] == 0.0


def test_stratify_by_failure_type_skips_none():
    scored = [
        score_record(_gold(failure_type="method"), _call(), True),
        score_record(_gold(failure_type=None), _call(), True),
    ]
    strat = stratify(scored, "failure_type")
    assert "method" in strat
    assert None not in strat


# ── check_thresholds ──────────────────────────────────────────────────────

def test_check_thresholds_pass_when_all_above_floor():
    metrics = {
        "tool_id_micro_accuracy": 0.95, "tool_id_macro_accuracy": 0.90,
        "null_fidelity": 0.99, "conditional_f1": 0.90,
        "condition_var_accuracy": 0.85, "no_match_f1": 0.90,
        "parse_failure_rate": 0.0, "per_tool_recall": {}, "per_tool_n": {},
    }
    result = check_thresholds(metrics)
    assert result["overall_pass"] is True


def test_check_thresholds_fail_when_below_floor():
    metrics = {
        "tool_id_micro_accuracy": 0.50, "tool_id_macro_accuracy": 0.90,
        "null_fidelity": 0.99, "conditional_f1": 0.90,
        "condition_var_accuracy": 0.85, "no_match_f1": 0.90,
        "parse_failure_rate": 0.0, "per_tool_recall": {}, "per_tool_n": {},
    }
    result = check_thresholds(metrics)
    assert result["tool_id_micro_accuracy"]["passed"] is False
    assert result["overall_pass"] is False


def test_check_thresholds_per_tool_floor_ignores_low_n_tools():
    metrics = {
        "tool_id_micro_accuracy": 0.95, "tool_id_macro_accuracy": 0.90,
        "null_fidelity": 0.99, "conditional_f1": 0.90,
        "condition_var_accuracy": 0.85, "no_match_f1": 0.90,
        "parse_failure_rate": 0.0,
        "per_tool_recall": {"rare_tool": 0.0}, "per_tool_n": {"rare_tool": 2},
    }
    result = check_thresholds(metrics)
    # only 2 instances -- below MIN_INSTANCES_FOR_PER_TOOL_FLOOR, must not fail on it
    assert result["per_tool_floor"]["passed"] is True


def test_check_thresholds_per_tool_floor_flags_common_tool_below_recall():
    metrics = {
        "tool_id_micro_accuracy": 0.95, "tool_id_macro_accuracy": 0.90,
        "null_fidelity": 0.99, "conditional_f1": 0.90,
        "condition_var_accuracy": 0.85, "no_match_f1": 0.90,
        "parse_failure_rate": 0.0,
        "per_tool_recall": {"common_tool": 0.40}, "per_tool_n": {"common_tool": 15},
    }
    result = check_thresholds(metrics)
    assert result["per_tool_floor"]["passed"] is False
    assert result["per_tool_floor"]["failures"] == [("common_tool", 15, 0.40)]


# ── self-consistency sanity check against the real gold set ─────────────

def test_oracle_self_score_against_real_gold_set_is_perfect():
    """Feeding each gold record's OWN expected fields back in as the
    'prediction' must score 1.0 everywhere -- this is the executor@oracle
    sanity check applied to the scorer itself: if this doesn't score
    perfectly, the scorer (not the model) has a bug."""
    if not GOLD_PATH.exists():
        return
    gold_records = load_gold(GOLD_PATH)
    scored = []
    for g in gold_records:
        call = ToolCall(
            step_num=1, step_text=g["step_text"], tool=g["expected_tool"],
            params={k: v for k, v in (g.get("expected_params") or {}).items() if v is not None},
            conditional=g.get("expected_conditional", False),
            condition_var=g.get("expected_condition_var", "none"),
            secondary_tools=tuple(g.get("expected_secondary_tools", [])),
        )
        scored.append(score_record(g, call, parse_ok=True))
    agg = aggregate(scored)
    assert agg["tool_id_micro_accuracy"] == 1.0
    assert agg["no_match_f1"] in (1.0, None)
    assert agg["parse_failure_rate"] == 0.0
