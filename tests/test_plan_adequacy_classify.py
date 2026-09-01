"""
Tests for pipelines/plan_adequacy/classify.py

Two kinds of test live here. The unit tests drive classify() with
hand-assembled PlanResults so each precedence rung is exercised in isolation.
The property test drives it over a generated cross-product of plan shapes and
asserts the partition holds -- exactly one class, always, with no residual --
because MECE is the property the whole diagnosis table rests on and a
counterexample would invalidate every prevalence number downstream rather
than just failing one case.

Run: python -m pytest tests/test_plan_adequacy_classify.py -v
"""
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.classify import (FAILURE_CLASSES, NON_EXECUTABLE,
                                              PRE_EXECUTION_CLASSES, classify,
                                              first_failure, is_cascade)
from pipelines.plan_adequacy.executor import STEP_VERDICTS


def _step(n, verdict):
    return SimpleNamespace(n=n, verdict=verdict, tool="t", text="", params={},
                           conditional=False, condition_text=None, detail="")


def _plan(verdicts, route_name="beach_gear", admissible="yes",
          foreign=None, goal_reached=False):
    """A PlanResult-shaped stand-in. classify() reads only these five
    attributes, so a namespace is a truer test subject than a real
    PlanResult -- it fails loudly if classify() ever starts depending on a
    field it has no business reading."""
    return SimpleNamespace(
        steps=[_step(i + 1, v) for i, v in enumerate(verdicts)],
        route_name=route_name, route_admissible=admissible,
        foreign_casualty=foreign, goal_reached=goal_reached,
    )


_CLEAN = ["SPECIFIED_UNGRADED"] * 6


# ── precedence, rung by rung ─────────────────────────────────────────────────

def test_no_route_is_no_procedure_regardless_of_step_verdicts():
    # A plan with no recognisable route has steps that were still graded, so
    # this also pins that step verdicts cannot outrank the route check.
    plan = _plan(["NO_MATCH"] * 6, route_name=None)
    got = classify(plan)
    assert got["failure_class"] == "NO_PROCEDURE"
    assert got["epl"] == 0 and got["epl_is_structural"] is True


def test_foreign_casualty_outranks_a_step_level_failure():
    plan = _plan(["SPECIFIED_UNGRADED", "SEQUENCE_VIOLATION"] + _CLEAN[:4],
                 foreign="aground")
    assert classify(plan)["failure_class"] == "STRATEGY_PERCEPTION"


def test_foreign_casualty_outranks_inadmissibility():
    # The load-bearing precedence choice: a plan solving the wrong accident
    # is usually ALSO on a route inadmissible for the real vessel. Reporting
    # that as a technique error would aim remediation at planning when the
    # failure is perception.
    plan = _plan(_CLEAN, admissible="no", foreign="sunken")
    assert classify(plan)["failure_class"] == "STRATEGY_PERCEPTION"


def test_inadmissible_route_is_structural_strategy_technique():
    got = classify(_plan(_CLEAN, admissible="no"))
    assert got["failure_class"] == "STRATEGY_TECHNIQUE"
    assert got["epl"] == 0 and got["epl_is_structural"] is True


def test_method_error_is_step_level_strategy_technique_with_measured_epl():
    plan = _plan(["SPECIFIED_UNGRADED", "SPECIFIED_UNGRADED", "METHOD_ERROR"])
    got = classify(plan)
    assert got["failure_class"] == "STRATEGY_TECHNIQUE"
    assert got["failure_step"] == 3
    assert got["epl"] == 2 and got["epl_is_structural"] is False


def test_unspecified_is_commitment_not_procedure():
    plan = _plan(["SPECIFIED_UNGRADED", "UNSPECIFIED"] + _CLEAN[:4])
    got = classify(plan)
    assert got["failure_class"] == "COMMITMENT"
    assert got["epl"] == 1


def test_clean_plan_that_never_reaches_the_goal_is_incomplete():
    got = classify(_plan(_CLEAN, goal_reached=False))
    assert got["failure_class"] == "INCOMPLETE"
    assert got["epl"] == 6


def test_clean_plan_that_reaches_the_goal_is_valid():
    got = classify(_plan(_CLEAN, goal_reached=True))
    assert got["failure_class"] == "VALID"
    assert got["epl"] == 6


# ── EPL is a projection of failure_step, not an independent quantity ─────────

def test_epl_is_exactly_one_less_than_the_first_failing_step():
    for k in range(1, 7):
        verdicts = ["SPECIFIED_UNGRADED"] * (k - 1) + ["NO_MATCH"]
        verdicts += ["SPECIFIED_UNGRADED"] * (6 - len(verdicts))
        got = classify(_plan(verdicts))
        assert got["failure_step"] == k
        assert got["epl"] == k - 1


def test_first_failure_reads_declared_step_order_not_list_order():
    plan = _plan(_CLEAN)
    plan.steps = list(reversed(plan.steps))
    plan.steps[0].verdict = "NO_MATCH"        # this is step n=6, sitting first
    plan.steps[-1].verdict = "SEQUENCE_VIOLATION"   # this is step n=1
    assert first_failure(plan) == 1


def test_unspecified_counts_as_non_executable_but_bad_verdicts_does_not_move():
    from pipelines.plan_adequacy.executor import BAD_VERDICTS
    assert "UNSPECIFIED" in NON_EXECUTABLE
    assert "UNSPECIFIED" not in BAD_VERDICTS      # route_completeness must not shift
    assert BAD_VERDICTS < NON_EXECUTABLE


# ── the partition property ───────────────────────────────────────────────────

def test_every_reachable_plan_shape_gets_exactly_one_class():
    """MECE over the cross-product of route state, foreign flag,
    admissibility, goal state, and first-failing verdict. If any combination
    fell through, the prevalence table would silently not sum to n."""
    seen = set()
    for route, foreign, adm, goal, verdict in itertools.product(
            [None, "beach_gear"], [None, "sunken"], ["yes", "no", "unknown", "n/a"],
            [False, True], list(STEP_VERDICTS)):
        plan = _plan(["SPECIFIED_UNGRADED", verdict] + _CLEAN[:4],
                     route_name=route, admissible=adm, foreign=foreign,
                     goal_reached=goal)
        got = classify(plan)
        assert got["failure_class"] in FAILURE_CLASSES
        assert isinstance(got["epl"], int) and 0 <= got["epl"] <= 6
        seen.add(got["failure_class"])
    # Every class except INCOMPLETE is reachable from this cross-product;
    # INCOMPLETE needs an all-clean plan, which the product never generates.
    assert seen >= {"NO_PROCEDURE", "STRATEGY_PERCEPTION", "STRATEGY_TECHNIQUE",
                    "PROCEDURE", "COMMITMENT"}


def test_pre_execution_classes_always_report_structural_epl():
    for cls, plan in [("NO_PROCEDURE", _plan(_CLEAN, route_name=None)),
                      ("STRATEGY_PERCEPTION", _plan(_CLEAN, foreign="aground"))]:
        got = classify(plan)
        assert got["failure_class"] == cls
        assert cls in PRE_EXECUTION_CLASSES
        assert got["epl_is_structural"] is True


# ── cascade flag ─────────────────────────────────────────────────────────────

def test_cascade_flags_a_step_downstream_of_an_earlier_no_match():
    plan = _plan(["NO_MATCH", "SPECIFIED_UNGRADED", "SEQUENCE_VIOLATION"])
    assert is_cascade(plan, 3) is True
    assert is_cascade(plan, 1) is False


def test_cascade_is_false_when_the_earlier_failure_is_not_a_state_loss():
    # UNSPECIFIED and CONDITIONAL_UNRESOLVED do not remove facts from the
    # world state, so a later violation is not attributable to them.
    plan = _plan(["UNSPECIFIED", "CONDITIONAL_UNRESOLVED", "SEQUENCE_VIOLATION"])
    assert is_cascade(plan, 3) is False
