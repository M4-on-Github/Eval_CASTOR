"""
Tests for pipelines/plan_adequacy/gates.py

Includes the regression anchor from the plan-adequacy design plan: gate
detection over the real p7_to_check/*.jsonl corpus must reproduce the
2026-08-19 measured gates/plan spread (2.39 / 4.24 / 5.90 / 5.60 across
ABLATION / CONTROL / IMPROVED / self_verify) that motivated dropping the
magnitude-based specification_rate metric. See memory:
castor-plans-have-no-magnitudes.md.
Run: python -m pytest tests/test_plan_adequacy_gates.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.gates import (
    detect_gates,
    gate_rate,
    guess_condition_var,
    has_definite_size_claim,
    has_size_gate,
    is_self_contradictory_on_size,
    resolve_conditional,
)
from pipelines.plan_adequacy.worldstate import WorldState
from pipelines.plan_adequacy.vocab import ToolRegistry

EVAL_ROOT = Path(__file__).parent.parent
P7_DIR = EVAL_ROOT / "p7_to_check"

# (filename substring, expected gates/plan, tolerance)
_REGRESSION_TARGETS = [
    ("baseline_ABLATION_visual_grounded", 2.39),
    ("baseline_CONTROL_visual_grounded", 4.24),
    ("baseline_visual_grounded_netural_assertions_j4708", 5.90),
    ("self_verify_s_visual_grounded", 5.60),
]


def _load(path):
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


# ── pure detection ────────────────────────────────────────────────────────────

def test_gate_rate_counts_if_then_constructions():
    text = "If the vessel is large, deploy additional tugs. Otherwise proceed with two."
    assert gate_rate(text) >= 1


def test_gate_rate_zero_on_plain_declarative_text():
    text = "Deploy two harbor tugs and apply 90 tons of pull to free the vessel."
    assert gate_rate(text) == 0


def test_detect_gates_returns_condition_text_and_var():
    text = "If the substrate is rock, dredge before pulling."
    gates = detect_gates(text)
    assert len(gates) == 1
    assert gates[0].condition_var == "substrate"


def test_guess_condition_var_vessel_size():
    assert guess_condition_var("if the vessel is large") == "vessel_size"


def test_guess_condition_var_none_when_unclear():
    assert guess_condition_var("if necessary") == "none"


# ── self-contradiction ────────────────────────────────────────────────────────

def test_self_contradiction_true_when_both_present():
    text = ("The vessel is large (>50m) and aground on a sand bank. "
            "If the vessel is large, deploy additional tugs.")
    assert has_definite_size_claim(text)
    assert has_size_gate(text)
    assert is_self_contradictory_on_size(text)


def test_self_contradiction_false_with_only_definite_claim():
    text = "The vessel is large (>50m) and aground on a sand bank."
    assert not is_self_contradictory_on_size(text)


# ── resolution against WorldState ────────────────────────────────────────────

def test_resolve_conditional_none_var_always_resolved():
    ws = WorldState()
    assert resolve_conditional("none", ws) == "resolved"


def test_resolve_conditional_unresolved_when_fact_never_established():
    ws = WorldState()
    assert resolve_conditional("vessel_size", ws) == "unresolved"


def test_resolve_conditional_resolved_after_establishing_call():
    reg = ToolRegistry.load()
    ws = WorldState()

    class FakeCall:
        tool = "survey_hull"
        params = {}

    ws.apply(FakeCall(), reg)
    assert ws.knows("vessel_size")
    assert resolve_conditional("vessel_size", ws) == "resolved"


# ── regression anchor against the real corpus ────────────────────────────────

def test_gate_rate_regression_anchor_against_p7_corpus():
    if not P7_DIR.exists():
        return  # skip silently outside a full checkout
    found_any = False
    for substr, expected in _REGRESSION_TARGETS:
        matches = list(P7_DIR.glob(f"*{substr}*.jsonl"))
        if not matches:
            continue
        found_any = True
        recs = _load(matches[0])
        actual = sum(gate_rate(r["text"]) for r in recs) / len(recs)
        assert abs(actual - expected) < 0.05, (
            f"{matches[0].name}: gate_rate regressed, expected ~{expected}, got {actual:.2f}"
        )
    assert found_any, "none of the expected p7_to_check run files were found -- corpus moved?"
