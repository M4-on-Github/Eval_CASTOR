"""
Tests for pipelines/plan_adequacy/inject.py

The load-bearing test here is that every BASE plan is clean. An injection
study measures "one defect of known class in an otherwise-correct plan"; if a
base silently carries its own defect -- a route inadmissible for its scenario,
a step out of order, a magnitude missing from the text -- then every case
built on it measures two defects at once and the confusion matrix is
meaningless. Registry edits are the realistic way that happens, so this
asserts it rather than trusting it.

Run: python -m pytest tests/test_plan_adequacy_inject.py -v
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.classify import classify
from pipelines.plan_adequacy.executor import execute_plan
from pipelines.plan_adequacy.inject import (BASES, INJECTORS, SCENARIOS,
                                            confusion_matrix, recall_by_class,
                                            run_injections)
from pipelines.plan_adequacy.methods import RouteRegistry
from pipelines.plan_adequacy.vocab import ToolRegistry

#: Recall floor below which a class's prevalence is not reportable. Declared
#: in reports/p9/redesign.tex before any injection was run.
RECALL_FLOOR = 0.7


def _reg():
    return ToolRegistry.load(), RouteRegistry.load()


def test_every_base_plan_is_clean():
    """No base may carry a defect of its own -- otherwise the injection on
    top of it is measuring two things."""
    tr, rr = _reg()
    for casualty, base in BASES.items():
        calls = base()
        scenario = SimpleNamespace(image=f"{casualty}/base.jpg", state=casualty,
                                   **SCENARIOS[casualty])
        result = execute_plan(calls, casualty, scenario, tr, rr,
                              "\n".join(c.step_text for c in calls))
        diag = classify(result)
        assert diag["failure_class"] in ("VALID", "INCOMPLETE"), (
            f"{casualty} base is not clean: {diag['failure_class']} at step "
            f"{diag['failure_step']} -- "
            + "; ".join(f"{s.n}:{s.tool}:{s.verdict}" for s in result.steps
                        if s.verdict != "SPECIFIED_UNGRADED"))
        assert diag["epl"] == 6
        assert result.foreign_casualty is None


def test_every_base_route_is_admissible_for_its_scenario():
    """A base on an inadmissible route would classify STRATEGY_TECHNIQUE
    before any injected defect could be seen."""
    tr, rr = _reg()
    for casualty, base in BASES.items():
        calls = base()
        scenario = SimpleNamespace(image=f"{casualty}/base.jpg", state=casualty,
                                   **SCENARIOS[casualty])
        result = execute_plan(calls, casualty, scenario, tr, rr)
        assert result.route_name is not None, f"{casualty} base matches no route"
        assert result.route_admissible != "no", (
            f"{casualty} base is on inadmissible route {result.route_name}")


def test_the_full_injection_sweep_covers_every_base_and_defect():
    records = run_injections()
    assert len(records) == len(BASES) * len(INJECTORS)
    assert {r["casualty"] for r in records} == set(BASES)
    assert {r["defect"] for r in records} == {name for name, _ in INJECTORS}


def test_no_class_falls_below_the_declared_recall_floor():
    """The gate on whether a class's prevalence may be reported at all."""
    for cls, v in recall_by_class(run_injections()).items():
        assert v["recall"] >= RECALL_FLOOR, (
            f"{cls} recall {v['recall']:.2f} < {RECALL_FLOOR}: its prevalence "
            f"is not reportable until the checker is fixed")


def test_confusion_matrix_is_diagonal_for_every_covered_class():
    records = run_injections()
    off_diagonal = {k: v for k, v in confusion_matrix(records).items() if k[0] != k[1]}
    assert not off_diagonal, f"misdiagnoses: {off_diagonal}"


def test_sequence_injection_targets_a_step_that_actually_has_preconditions():
    """Guards the bug this injector was written twice to fix: picking the
    moved step by POSITION produced still-clean plans on bases whose last
    step has an empty `requires`."""
    tr, _ = _reg()
    from pipelines.plan_adequacy.inject import inject_sequence_violation
    for casualty, base in BASES.items():
        calls, expected = inject_sequence_violation(base(), casualty, tr)
        assert expected == "PROCEDURE"
        assert tr.spec(calls[0].tool).requires, (
            f"{casualty}: moved step {calls[0].tool} has no preconditions to violate")


def test_commitment_injection_leaves_params_populated():
    """Specificity must be read from the step TEXT. Stripping digits from the
    text while leaving params intact is what proves the extractor cannot
    satisfy the check by inventing a value."""
    from pipelines.plan_adequacy.inject import inject_commitment
    calls, expected = inject_commitment(BASES["aground"](), "aground")
    assert expected == "COMMITMENT"
    assert any(c.params for c in calls)
    assert not any(ch.isdigit() for c in calls for ch in c.step_text)
