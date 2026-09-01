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
    to do. Integer 0-6. This is NOT a second variable: EPL = failure_step - 1,
    so it is a projection of the same record. It exists because it contains
    the old endpoint as its top bin (goal_reached <=> EPL == 6 and terminal
    fact established), and because instrument error can be measured in the
    same unit, which the 1 - p**6 argument it replaces could not do.

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

#: Step verdicts that make a step non-executable. Note this is BAD_VERDICTS
#: *plus* UNSPECIFIED: a step whose tool wants a magnitude and whose text
#: never states one is not something an executor can carry out. UNSPECIFIED
#: deliberately stays out of BAD_VERDICTS itself (route_completeness also
#: consumes that set, and is about route execution rather than magnitude),
#: which is why this is a separate constant rather than a reuse.
NON_EXECUTABLE = BAD_VERDICTS | {"UNSPECIFIED"}

#: Which class a first-failing step's verdict implies. METHOD_ERROR is a
#: technique failure detected at step level rather than at route level --
#: "this approach does not suit this vessel" is the same diagnosis whether
#: the registry caught it on the route or the executor caught it on a step.
_VERDICT_CLASS = {
    "METHOD_ERROR": "STRATEGY_TECHNIQUE",
    "SEQUENCE_VIOLATION": "PROCEDURE",
    "NO_MATCH": "PROCEDURE",
    "CONDITIONAL_UNRESOLVED": "PROCEDURE",
    "UNSPECIFIED": "COMMITMENT",
}


def first_failure(plan) -> Optional[int]:
    """1-based index of the first non-executable step, or None if all six
    execute cleanly. Steps are consulted in their own declared order (`n`)
    rather than list order, so a caller that assembled them out of order
    cannot silently shift the answer."""
    for step in sorted(plan.steps, key=lambda s: s.n):
        if step.verdict in NON_EXECUTABLE:
            return step.n
    return None


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
        return _result(_VERDICT_CLASS[verdict], step, step - 1, structural=False)

    if not plan.goal_reached:
        # Nothing failed, route is fine, and the plan still did not get
        # there. goal_reached is reused rather than re-deriving the terminal
        # fact check, so the two can never disagree.
        return _result("INCOMPLETE", None, 6, structural=False)

    return _result("VALID", None, 6, structural=False)


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
