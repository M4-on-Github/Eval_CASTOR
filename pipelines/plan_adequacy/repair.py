"""
Counterfactual repair: what would solving one failure class actually buy?

The failure profile (classify.py) says how often each class stops a plan. It
cannot say whether fixing that class would help. A COMMITMENT failure might be
a speed bump -- state the magnitude and the plan runs to completion -- or a
wall, where the very next step fails anyway. Those have opposite implications
for what to build next, and prevalence alone cannot tell them apart.

The operator is NEUTRALISE, not author. The tempting alternative is to rewrite
the failing step correctly and re-run, but then the measurement depends on how
good a salvor the person doing the rewriting is that afternoon. Instead the
executor is told to treat that one step's failure as not having happened --
gate resolved, method fitting, preconditions granted, magnitude stated -- and
the walk continues. The answer falls out as

    delta_epl = next_failure_step - repaired_step

i.e. simply the distance to the next failure.

HOW TO READ THE NUMBER. Neutralising grants the fix for free; no real
intervention works that cleanly. So delta_epl is a CEILING on the gain, not an
expected gain, which makes this a rule-OUT instrument: if granting COMMITMENT
for free buys 0.3 steps, then any real remedy for commitment buys at most 0.3
and the direction can be closed with confidence. A high value only permits a
direction; it never establishes one. Every reported delta_epl must carry that
reading with it.

Two known asymmetries, both disclosed rather than corrected:

  * NO_MATCH is repaired by SKIPPING, not fixing -- there is no tool, so
    there are no effects to apply. Its delta_epl is therefore measured under a
    strictly weaker repair than every other verdict and is biased DOWNWARD.
    Hand-assigned intended tools for a subsample (the Phase 3 NO_MATCH
    adjudication) are the intended correction.
  * Pre-execution classes (NO_PROCEDURE, STRATEGY_PERCEPTION, and route-level
    STRATEGY_TECHNIQUE) have no failing step to neutralise. They are not
    repairable here and return None. STRATEGY_PERCEPTION's real delta comes
    from the perception-controlled arm -- regeneration with the casualty
    given -- which is evidence rather than simulation and is strictly better
    than anything this module could produce for it.
"""

from pipelines.plan_adequacy.classify import PRE_EXECUTION_CLASSES, classify
from pipelines.plan_adequacy.executor import execute_plan

#: Cap on repair iterations. Six steps means at most six repairs, and the
#: guard exists only so a future executor change that fails to advance cannot
#: spin forever.
MAX_REPAIRS = 6


def repair_once(calls, casualty, scenario, tool_registry, route_registry,
                plan_text="", already_repaired=frozenset()):
    """Neutralise the first failure and re-run. Returns None when there is
    nothing repairable (a clean plan, or a pre-execution failure)."""
    before = execute_plan(calls, casualty, scenario, tool_registry,
                          route_registry, plan_text, repaired_steps=already_repaired)
    diag = classify(before)
    if diag["failure_class"] in PRE_EXECUTION_CLASSES or diag["failure_step"] is None:
        return None
    if diag["failure_class"] == "STRATEGY_TECHNIQUE" and diag["epl_is_structural"]:
        return None

    step = diag["failure_step"]
    repaired = already_repaired | {step}
    after = execute_plan(calls, casualty, scenario, tool_registry,
                         route_registry, plan_text, repaired_steps=repaired)
    after_diag = classify(after)

    return {
        "repaired_step": step,
        "repaired_class": diag["failure_class"],
        "epl_before": diag["epl"],
        "epl_after": after_diag["epl"],
        "delta_epl": after_diag["epl"] - diag["epl"],
        "next_class": after_diag["failure_class"],
        "repaired_steps": repaired,
    }


def repair_to_exhaustion(calls, casualty, scenario, tool_registry,
                          route_registry, plan_text="", max_repairs=MAX_REPAIRS):
    """Repair, continue, repair again, until the plan stops failing at step
    level or the cap is hit.

    Produces three things at once, all from the same operator:

      * the per-repair delta_epl chain, whose first element is the marginal
        value of that plan's actual first failure;
      * repairs_to_valid -- how many repairs the plan is away from executing
        cleanly. This is a far more informative answer to "are these plans
        valid" than the flat 0/330 the binary endpoint gives, because it is a
        distance rather than a verdict;
      * the class-transition chain, which recovers the joint failure structure
        that first-failure counting necessarily discards -- and doubles as an
        empirical check on hazard.py's masking correction, since a class that
        only ever appears as a SECOND repair is precisely a masked class.
    """
    chain, repaired = [], frozenset()
    for _ in range(max_repairs):
        step = repair_once(calls, casualty, scenario, tool_registry,
                           route_registry, plan_text, repaired)
        if step is None:
            break
        chain.append(step)
        repaired = step["repaired_steps"]

    final = classify(execute_plan(calls, casualty, scenario, tool_registry,
                                  route_registry, plan_text,
                                  repaired_steps=repaired))
    return {
        "chain": chain,
        "repairs_to_valid": len(chain) if final["failure_class"] in ("VALID", "INCOMPLETE") else None,
        "terminal_class": final["failure_class"],
        "final_epl": final["epl"],
        "class_sequence": [c["repaired_class"] for c in chain],
    }


def delta_epl_by_class(repair_rows: list) -> dict:
    """Mean first-repair delta_epl per class, with n.

    Only the FIRST repair of each plan is aggregated here. Later repairs in a
    chain are conditional on the earlier ones having been granted, so pooling
    them would mix a marginal effect with a joint one and the resulting mean
    would answer no question anyone asked.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in repair_rows:
        if row["chain"]:
            first = row["chain"][0]
            buckets[first["repaired_class"]].append(first["delta_epl"])
    return {cls: {"n": len(v), "mean_delta_epl": sum(v) / len(v)}
            for cls, v in sorted(buckets.items())}


def transition_matrix(repair_rows: list) -> dict:
    """{(repaired_class, next_class): n} over first repairs.

    A large diagonal is the diagnostic worth watching: repairing a class and
    immediately hitting the SAME class again means the failure is pervasive
    rather than localised, and a remedy would have to fire on every step
    rather than once.
    """
    from collections import Counter
    return dict(Counter(
        (row["chain"][0]["repaired_class"], row["chain"][0]["next_class"])
        for row in repair_rows if row["chain"]))
