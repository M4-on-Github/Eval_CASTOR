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


def test_unspecified_no_longer_stops_a_plan():
    # Quantities were dropped: an unquantified step is still labelled, but
    # the plan runs on through it. COMMITMENT is empty by construction.
    plan = _plan(["SPECIFIED_UNGRADED", "UNSPECIFIED"] + _CLEAN[:4], goal_reached=True)
    got = classify(plan)
    assert got["failure_class"] == "VALID"
    assert got["failure_step"] is None and got["epl"] == 6


def test_support_steps_neither_stop_a_plan_nor_count_towards_epl():
    plan = _plan(["SUPPORT", "SPECIFIED_UNGRADED", "SUPPORT", "SPECIFIED_UNGRADED"],
                 goal_reached=True)
    got = classify(plan)
    assert got["failure_class"] == "VALID"
    assert got["epl"] == 2                        # graded steps only


def test_epl_counts_graded_steps_before_failure_but_step_keeps_its_number():
    plan = _plan(["SUPPORT", "SPECIFIED_UNGRADED", "SUPPORT", "NO_MATCH",
                  "SPECIFIED_UNGRADED"])
    got = classify(plan)
    assert got["failure_class"] == "PROCEDURE"
    assert got["failure_step"] == 4               # the plan's own numbering
    assert got["epl"] == 1                        # one graded step before it


def test_clean_plan_that_never_reaches_the_goal_is_incomplete():
    got = classify(_plan(_CLEAN, goal_reached=False))
    assert got["failure_class"] == "INCOMPLETE"
    assert got["epl"] == 6


def test_clean_plan_that_reaches_the_goal_is_valid():
    got = classify(_plan(_CLEAN, goal_reached=True))
    assert got["failure_class"] == "VALID"
    assert got["epl"] == 6


def test_clean_plan_epl_is_its_own_length_not_six():
    # procedural_v3 lets the model choose plan length. A hard-coded 6 would
    # give a clean 10-step plan EPL 6, below a plan that fails at step 9.
    ten = ["SPECIFIED_UNGRADED"] * 10
    assert classify(_plan(ten, goal_reached=True))["epl"] == 10
    assert classify(_plan(ten, goal_reached=False))["epl"] == 10
    assert classify(_plan(["SPECIFIED_UNGRADED"] * 4, goal_reached=True))["epl"] == 4

    fails_at_9 = ten[:8] + ["SEQUENCE_VIOLATION", "SPECIFIED_UNGRADED"]
    assert classify(_plan(fails_at_9))["epl"] == 8 < 10


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


def test_the_stop_set_is_bad_verdicts_and_excludes_unspecified_and_support():
    from pipelines.plan_adequacy.executor import BAD_VERDICTS
    assert NON_EXECUTABLE == BAD_VERDICTS
    assert "UNSPECIFIED" not in NON_EXECUTABLE
    assert "SUPPORT" not in NON_EXECUTABLE


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
    # COMMITMENT is unreachable by construction now that quantities are
    # dropped; a clean second step (SPECIFIED_UNGRADED, UNSPECIFIED, SUPPORT)
    # reaches VALID or INCOMPLETE depending on the goal flag.
    assert seen == {"NO_PROCEDURE", "STRATEGY_PERCEPTION", "STRATEGY_TECHNIQUE",
                    "PROCEDURE", "INCOMPLETE", "VALID"}


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


# ── the vocabulary gap, labelled not closed ──────────────────────────────────

def test_no_match_category_labels_the_six_missing_capabilities():
    from pipelines.plan_adequacy.classify import no_match_category as cat
    cases = {
        "Establish a safety perimeter using buoys and warning flares": "site_control",
        "Coordinate with local maritime authorities and port control": "liaison",
        "Install temporary mooring lines from the beach to the bow": "temporary_stabilisation",
        "Monitor weather and sea state continuously during the lift": "ongoing_monitoring",
        "Mobilize appropriate salvage equipment including heavy tugs": "logistics",
        "Document the operation and report for insurance purposes": "documentation",
    }
    for text, expected in cases.items():
        assert cat(text, "NO_MATCH") == expected, text


def test_unmatched_steps_fall_to_the_honest_residual():
    from pipelines.plan_adequacy.classify import UNCATEGORISED, no_match_category
    assert no_match_category("Deploy divers to cut the anchor chain", "NO_MATCH") == UNCATEGORISED


def test_category_labels_support_steps_too():
    from pipelines.plan_adequacy.classify import no_match_category as cat
    assert cat("Establish a safety perimeter", "SUPPORT") == "site_control"


def test_category_is_empty_for_any_graded_step():
    """The column must not invite the reading that a graded step was also
    'really' something else."""
    from pipelines.plan_adequacy.classify import no_match_category as cat
    for verdict in ("SPECIFIED_UNGRADED", "UNSPECIFIED", "METHOD_ERROR",
                    "SEQUENCE_VIOLATION", "CONDITIONAL_UNRESOLVED"):
        assert cat("Establish a safety perimeter", verdict) == ""


def test_categorising_cannot_change_a_diagnosis():
    """The leak guard at the classify layer. Whether a step is SUPPORT is
    decided once, by the executor (support.is_support); classify() reads
    only verdicts. Rewriting step text after execution must therefore move
    nothing -- if it did, the label would be reaching into grading through a
    second door."""
    plan = _plan(["NO_MATCH", "SPECIFIED_UNGRADED"] + _CLEAN[:4])
    baseline = classify(plan)
    for step in plan.steps:
        step.text = "Establish a safety perimeter and coordinate with authorities"
    assert classify(plan) == baseline


# ── all-errors mode ──────────────────────────────────────────────────────────

_FLAG_KEYS = ("no_match", "hedged", "method", "sequence", "unquantified")


def _flagged(n, *on, verdict="SPECIFIED_UNGRADED"):
    step = _step(n, verdict)
    step.flags = {k: k in on for k in _FLAG_KEYS}
    return step


def _errplan(steps, route_name="beach_gear", admissible="yes", foreign=None,
             not_attempted=()):
    return SimpleNamespace(steps=steps, route_name=route_name,
                           route_admissible=admissible, foreign_casualty=foreign,
                           not_attempted=list(not_attempted))


def test_all_errors_counts_every_flag_not_just_the_first():
    from pipelines.plan_adequacy.classify import all_errors
    plan = _errplan([_flagged(1, "sequence", "hedged", verdict="CONDITIONAL_UNRESOLVED"),
                     _flagged(2, "method", "sequence", verdict="METHOD_ERROR"),
                     _flagged(3, "sequence", verdict="SEQUENCE_VIOLATION")])
    got = all_errors(plan)
    assert got["n_err_sequence"] == 3            # verdict alone shows 1
    assert got["n_err_hedged"] == 1 and got["n_err_method"] == 1
    assert got["n_error_types"] == 3


def test_only_a_genuine_no_match_knocks_later_steps_on():
    """METHOD_ERROR still applies its tool's effects, so it removes nothing
    from the world state and cannot explain a later sequence error."""
    from pipelines.plan_adequacy.classify import all_errors
    after_method = _errplan([_flagged(1, "method", verdict="METHOD_ERROR"),
                             _flagged(2, "sequence", verdict="SEQUENCE_VIOLATION")])
    got = all_errors(after_method)
    assert got["n_err_sequence_independent"] == 1

    after_no_match = _errplan([_flagged(1, "no_match", verdict="NO_MATCH"),
                               _flagged(2, "sequence", verdict="SEQUENCE_VIOLATION"),
                               _flagged(3, "hedged", verdict="CONDITIONAL_UNRESOLVED")])
    got = all_errors(after_no_match)
    assert got["n_err_sequence"] == 1 and got["n_err_sequence_independent"] == 0
    assert got["n_err_hedged"] == 1 and got["n_err_hedged_independent"] == 0


def test_support_steps_are_neither_errors_nor_knock_on_sources():
    from pipelines.plan_adequacy.classify import all_errors
    plan = _errplan([_flagged(1, verdict="SUPPORT"),
                     _flagged(2, "sequence", verdict="SEQUENCE_VIOLATION")])
    got = all_errors(plan)
    assert got["n_err_no_match"] == 0
    assert got["n_err_sequence_independent"] == 1
    assert got["n_support_steps"] == 1 and got["n_graded_steps"] == 1


def test_unquantified_steps_are_not_errors_in_all_errors_mode():
    from pipelines.plan_adequacy.classify import all_errors
    got = all_errors(_errplan([_flagged(1, "unquantified", verdict="UNSPECIFIED")]))
    assert got["n_error_types"] == 0
    assert not any("unquantified" in k for k in got)


def test_plan_level_errors_are_all_reported_together():
    """classify() would stop at perception; all_errors() also reports that
    the plan's own route was inadmissible and left a required action out."""
    from pipelines.plan_adequacy.classify import all_errors
    plan = _errplan([_flagged(1)], admissible="no", foreign="sunken",
                    not_attempted=["route:pull", "C1:fuel_offloaded"])
    got = all_errors(plan)
    assert (got["err_perception"], got["err_technique_inadmissible"],
            got["err_technique_incomplete"], got["err_no_technique"]) == (1, 1, 1, 0)
    assert got["n_error_types"] == 3


def test_a_missed_universal_obligation_is_not_an_incomplete_technique():
    from pipelines.plan_adequacy.classify import all_errors
    got = all_errors(_errplan([_flagged(1)], not_attempted=["C1:fuel_offloaded"]))
    assert got["err_technique_incomplete"] == 0


def test_all_errors_emits_exactly_its_declared_columns():
    from pipelines.plan_adequacy.classify import ALL_ERRORS_FIELDS, all_errors
    got = all_errors(_errplan([_flagged(1)]))
    assert list(got) == ALL_ERRORS_FIELDS
