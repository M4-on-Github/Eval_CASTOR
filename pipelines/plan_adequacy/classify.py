"""
The failure taxonomy: one PlanResult in, exactly one diagnosis out.

This module is the summary layer of P9's redesign (reports/p9/redesign.tex).
The old endpoint, goal_reached, is a four-way conjunction that returned 0 for
every plan in the corpus -- a binary with no variance, which supports no
comparison and no diagnosis. What replaces it is not a different threshold on
the same idea but a different question: not "is this plan valid" (answered:
no) but "what stopped it, how far did it get, and whose fault is that".

Two objects come out of here:

  * failure_class -- six mutually exclusive classes, assigned in STRICT
    PRECEDENCE order. Precedence is what makes the taxonomy MECE; without it
    a plan on an inadmissible route whose third step is also UNSPECIFIED has
    two defensible labels and the prevalence table stops meaning anything.

  * EPL (Executable Prefix Length) -- how many leading steps a deterministic
    executor could actually carry out before the plan stopped telling it what
    to do. Counts GRADED steps only -- SUPPORT steps (pure scaffolding, see
    support.py) are neither executed actions nor failures -- so it runs from
    0 to the plan's graded-step count. It is NOT a second variable: EPL is
    the number of graded steps before failure_step (failure_step - 1 when a
    plan has no SUPPORT steps), a projection of the same record. It exists
    because it contains the old endpoint as its top bin (goal_reached <=>
    every graded step executed and the terminal fact established), and
    because instrument error can be measured in the same unit, which the
    1 - p**6 argument it replaces could not do.

Why first-failure only. 72.7% (ablation) and 74.5% (control) of sequence
violations in the corpus occur downstream of an earlier NO_MATCH or
METHOD_ERROR -- i.e. later verdicts are conditionally dependent on earlier
ones and are contaminated observations. The first failure is the only
statistically independent failure event in a plan.

The known cost of that choice, stated because it biases a headline number:
first-failure counting MASKS late classes. A COMMITMENT failure at step 5 is
not caused by a PROCEDURE failure at step 2, it is merely hidden by it, and
since hard failures cluster early (median first violation: step 2) raw
prevalence understates COMMITMENT. hazard.py exists to correct exactly this
and must be read alongside any prevalence table produced from here.
"""

from typing import Optional

from pipelines.plan_adequacy.executor import BAD_VERDICTS

#: The six classes plus the (currently empty) success label, in the order
#: they are evaluated. Rows of the diagnosis table are drawn from this tuple,
#: so its order is the table's order.
FAILURE_CLASSES = (
    "NO_PROCEDURE",
    "STRATEGY_PERCEPTION",
    "STRATEGY_TECHNIQUE",
    "PROCEDURE",
    "COMMITMENT",
    "INCOMPLETE",
    "VALID",
)

#: Classes that fail before any step executes. Their EPL is 0 BY
#: CONSTRUCTION, not by measurement, so mean-EPL must be reported as n/a for
#: them rather than as 0.0 -- printing a structural zero as if it were an
#: observation is how a table starts lying.
PRE_EXECUTION_CLASSES = frozenset({"NO_PROCEDURE", "STRATEGY_PERCEPTION"})

#: Step verdicts that stop a plan. This used to be BAD_VERDICTS plus
#: UNSPECIFIED (a step whose tool wants a magnitude and whose text never
#: states one). Quantities were dropped from P9's question: the CASTOR plans
#: state essentially none (337 of 338 gold steps), so UNSPECIFIED fired on
#: almost every numeric-param tool and measured the registry's parameter list
#: rather than the planner. UNSPECIFIED is still assigned per step and
#: scan_unquantified is still in per_step.csv; it just no longer stops a plan,
#: so COMMITMENT is now empty by construction. The separate name is kept so
#: the stop set can diverge from BAD_VERDICTS again without touching callers.
NON_EXECUTABLE = BAD_VERDICTS

#: Verdicts a step can carry without being graded at all: pure scaffolding
#: (support.py). They never stop a plan and never count towards EPL.
UNGRADED = frozenset({"SUPPORT"})

#: Which class a first-failing step's verdict implies. METHOD_ERROR is a
#: technique failure detected at step level rather than at route level --
#: "this approach does not suit this vessel" is the same diagnosis whether
#: the registry caught it on the route or the executor caught it on a step.
_VERDICT_CLASS = {
    "METHOD_ERROR": "STRATEGY_TECHNIQUE",
    "SEQUENCE_VIOLATION": "PROCEDURE",
    "NO_MATCH": "PROCEDURE",
    "CONDITIONAL_UNRESOLVED": "PROCEDURE",
}


def first_failure(plan) -> Optional[int]:
    """Step number (`n`, the plan's own numbering) of the first
    non-executable step, or None if every step executes cleanly. Steps are
    consulted in their own declared order rather than list order, so a caller
    that assembled them out of order cannot silently shift the answer."""
    for step in sorted(plan.steps, key=lambda s: s.n):
        if step.verdict in NON_EXECUTABLE:
            return step.n
    return None


def graded_steps_before(plan, step_n: Optional[int] = None) -> int:
    """How many GRADED steps precede step `step_n` (all of them if None).

    This is EPL. SUPPORT steps are skipped on both sides: a perimeter step
    before the failure is not an executed salvage action, and a plan should
    not score higher for padding itself with scaffolding.
    """
    return sum(1 for s in plan.steps
               if s.verdict not in UNGRADED and (step_n is None or s.n < step_n))


def classify(plan) -> dict:
    """Diagnose one plan.

    Returns {failure_class, failure_step, epl, epl_is_structural}, where
    failure_step is None for a plan with no failing step, and
    epl_is_structural marks the pre-execution classes whose EPL of 0 is a
    definition rather than a measurement.

    Precedence, top to bottom:

      1. STRATEGY_PERCEPTION -- the plan's tools fit a DIFFERENT casualty's
                                routes strictly better than its own
                                (executor.foreign_casualty, computed by
                                methods.detect_perception_mismatch). The plan
                                is competently solving the wrong accident: a
                                failure upstream of planning.
      2. NO_PROCEDURE        -- no route recognised, for this casualty or any
                                other. The output is not a recognisable
                                salvage procedure at all.
      3. STRATEGY_TECHNIQUE  -- right casualty, wrong approach for THIS
                                vessel: an inadmissible route, or a
                                METHOD_ERROR as the first failing step.
      4. PROCEDURE           -- right route, unexecutable as written.
      5. COMMITMENT          -- action named, magnitude never decided.
                                EMPTY BY CONSTRUCTION since quantities were
                                dropped (see NON_EXECUTABLE); kept in the
                                tuple so column sets do not drift.
      6. INCOMPLETE          -- every step executed and the plan still never
                                established the casualty's terminal fact.
                                Expected to be empty at present and defined
                                anyway: a class that only populates once
                                specificity improves is still part of the
                                partition, and adding it later would be a
                                taxonomy revision rather than a reading.

    Perception is checked FIRST, ahead of both no-route and inadmissibility.
    Two reasons, and the first was found by running the taxonomy over the real
    corpus rather than by reasoning:

      * Ordering NO_PROCEDURE first put 99 of 126 foreign-casualty matches
        into NO_PROCEDURE, because a plan solving the wrong accident usually
        matches NOTHING in its own casualty's library -- route_name is None
        precisely when perception has failed hardest. "No recognisable
        procedure" has to mean recognisable to no casualty, or the class
        swallows the very failures the taxonomy was extended to separate.
      * A plan solving the wrong accident will also, usually, be on a route
        inadmissible for the real vessel. "Wrong accident" is the upstream and
        more actionable diagnosis; reporting it as a technique error would
        point remediation at planning when the problem is perception.
    """
    step = first_failure(plan)

    if plan.foreign_casualty is not None:
        return _result("STRATEGY_PERCEPTION", None, 0, structural=True)

    if plan.route_name is None:
        return _result("NO_PROCEDURE", None, 0, structural=True)

    if plan.route_admissible == "no":
        # Route-level technique failure: EPL 0 is structural in the same way
        # as classes 1-2 (the approach is wrong before execution begins),
        # even though STRATEGY_TECHNIQUE can ALSO arise at step level below
        # with a measured EPL. The two sub-cases are distinguished by
        # epl_is_structural so the diagnosis table can split the row.
        return _result("STRATEGY_TECHNIQUE", None, 0, structural=True)

    if step is not None:
        verdict = next(s.verdict for s in plan.steps if s.n == step)
        return _result(_VERDICT_CLASS[verdict], step,
                       graded_steps_before(plan, step), structural=False)

    # A plan that never failed executed every graded step it has. The length
    # is the plan's own, not a constant: the procedural_v3 arm lets the model
    # choose how many steps to write, and a hard-coded 6 would rank a plan
    # failing at step 9 (EPL 8) above a clean 10-step plan.
    n_steps = graded_steps_before(plan)

    if not plan.goal_reached:
        # Nothing failed, route is fine, and the plan still did not get
        # there. goal_reached is reused rather than re-deriving the terminal
        # fact check, so the two can never disagree.
        return _result("INCOMPLETE", None, n_steps, structural=False)

    return _result("VALID", None, n_steps, structural=False)


def _result(failure_class: str, failure_step: Optional[int], epl: int,
            structural: bool) -> dict:
    return {
        "failure_class": failure_class,
        "failure_step": failure_step,
        "epl": epl,
        "epl_is_structural": structural,
    }


def is_cascade(plan, step_n: int) -> bool:
    """Is the step at `step_n` downstream of an earlier NO_MATCH or
    METHOD_ERROR?

    A step flagged here is a suspect observation, not a confirmed one: an
    earlier extraction failure can leave the world state missing a fact that
    a later, perfectly good step then appears to violate. Promoted from a
    post-hoc script to a per-row field so the contamination argument in the
    results document is checkable against the data rather than recomputed
    each time someone wants to quote it.
    """
    for step in sorted(plan.steps, key=lambda s: s.n):
        if step.n >= step_n:
            return False
        if step.verdict in ("NO_MATCH", "METHOD_ERROR"):
            return True
    return False


# ── all-errors mode ──────────────────────────────────────────────────────────

#: Every error type all_errors() can report, plan-level first. n_error_types
#: counts how many of these a plan has at least one of.
ERROR_TYPES = ("perception", "no_technique", "technique_inadmissible",
               "technique_incomplete", "method", "sequence", "hedged", "no_match")

#: The step-level types whose check reads the world state, and so can be
#: knocked on by an earlier step that failed to add a fact to it. method and
#: no_match are judged on the step alone.
STATE_DEPENDENT = ("sequence", "hedged")


def all_errors(plan) -> dict:
    """Every error in a plan, not just the first -- the counterpart to
    classify(), which answers "is this plan valid" and so stops at the
    first failure.

    Plan-level errors are 0/1 (err_*); step-level errors are counts of the
    executor's raw scan flags (n_err_*), so a step that is both out of order
    and hedged counts once under each. Unquantified steps are not errors
    here: quantities were dropped from P9's question (see NON_EXECUTABLE).

    Knock-on. A step is knocked on if a genuine NO_MATCH comes before it:
    that step added no fact to the world state, so a later sequence or hedge
    error may only be reporting the gap. The *_independent counts exclude
    those. METHOD_ERROR is not a source -- the executor still applies a
    wrong-family tool's effects -- and SUPPORT steps carry no flags at all.

    technique_incomplete is a required action the plan never calls
    (not_attempted's route: entries), NOT route_completeness < 1: that ratio
    also drops core tools that were called but failed a step check, which
    would count those errors a second time.
    """
    err = {
        "perception": int(plan.foreign_casualty is not None),
        "no_technique": int(plan.route_name is None),
        "technique_inadmissible": int(plan.route_admissible == "no"),
        "technique_incomplete": int(any(x.startswith("route:")
                                        for x in plan.not_attempted)),
    }
    counts = dict.fromkeys(("method", "sequence", "hedged", "no_match"), 0)
    independent = dict.fromkeys(STATE_DEPENDENT, 0)
    knocked_on = False
    n_support = 0
    for step in sorted(plan.steps, key=lambda s: s.n):
        if step.verdict in UNGRADED:
            n_support += 1
            continue
        for k in counts:
            if step.flags.get(k):
                counts[k] += 1
                if k in independent and not knocked_on:
                    independent[k] += 1
        if step.flags.get("no_match"):
            knocked_on = True

    row = {f"err_{k}": v for k, v in err.items()}
    for k, v in counts.items():
        row[f"n_err_{k}"] = v
        if k in independent:
            row[f"n_err_{k}_independent"] = independent[k]
    present = {k for k, v in err.items() if v} | {k for k, v in counts.items() if v}
    row["n_error_types"] = len(present)
    row["n_support_steps"] = n_support
    row["n_graded_steps"] = len(plan.steps) - n_support
    return row


#: per_image.csv column order for all_errors(). Built from ERROR_TYPES so it
#: cannot drift from the function.
ALL_ERRORS_FIELDS = (
    [f"err_{k}" for k in ERROR_TYPES[:4]]
    + [c for k in ERROR_TYPES[4:] for c in
       ([f"n_err_{k}"] + ([f"n_err_{k}_independent"] if k in STATE_DEPENDENT else []))]
    + ["n_error_types", "n_support_steps", "n_graded_steps"]
)


# The categoriser lives in support.py, because the executor now uses it to
# decide which unmapped steps are SUPPORT -- and executor imports nothing from
# here. Re-exported so existing imports keep working.
from pipelines.plan_adequacy.support import (NO_MATCH_CATEGORIES,  # noqa: E402,F401
                                             UNCATEGORISED, no_match_category)
