"""
Cause-specific hazard over the failure taxonomy.

classify.py assigns each plan the class of its FIRST failure, which is the
only statistically independent failure event a plan has (see that module's
docstring on cascades). The cost of that choice is masking: a class that
tends to strike late is hidden whenever some other class strikes early, so
raw prevalence understates it. In this corpus hard failures cluster at step 2,
so COMMITMENT -- which can only surface on a step that actually executed --
is the class most understated by a naive count.

The correction is the standard competing-risks object:

    h_C(k) = P(fail with class C at step k | survived to step k)

read over only the plans still at risk at k, rather than over all 330. A
class whose hazard rank differs from its prevalence rank was being masked,
and the prevalence table alone would have mis-ranked the remedies.

Pre-execution classes (NO_PROCEDURE, STRATEGY_PERCEPTION, and route-level
STRATEGY_TECHNIQUE) fail at k = 0, before any step is at risk. They are
reported at k = 0 with the full corpus as the risk set, and are excluded from
the step-wise hazards so they cannot dilute them.
"""

from collections import defaultdict

from pipelines.plan_adequacy.classify import FAILURE_CLASSES

MAX_STEPS = 6


def hazard_table(rows: list, max_steps: int = MAX_STEPS) -> dict:
    """Cause-specific hazard by (class, step).

    `rows` are dicts carrying at least `failure_class` and `failure_step`
    (as emitted by classify.classify()); a None failure_step means the plan
    either failed before execution or never failed at all, which the
    epl/class pair already distinguishes.

    Returns {"at_risk": {k: n}, "events": {(cls, k): n},
             "hazard": {(cls, k): float}, "k0": {cls: n}}.
    """
    k0 = defaultdict(int)          # pre-execution failures, k = 0
    events = defaultdict(int)      # (class, k) -> count
    for row in rows:
        step = row.get("failure_step")
        if step is None:
            if row["failure_class"] not in ("VALID", "INCOMPLETE"):
                k0[row["failure_class"]] += 1
            continue
        events[(row["failure_class"], step)] += 1

    # Risk set at step k: plans that reached step k without having already
    # failed. A plan that failed pre-execution was never at risk at any step
    # and is excluded from every denominator -- including it would make the
    # step-wise hazards a function of how many plans never got a route,
    # which is a different question.
    survivors = sum(1 for r in rows if r.get("failure_step") is not None
                    or r["failure_class"] in ("VALID", "INCOMPLETE"))
    at_risk, hazard = {}, {}
    for k in range(1, max_steps + 1):
        at_risk[k] = survivors
        if survivors:
            for cls in FAILURE_CLASSES:
                n = events.get((cls, k), 0)
                if n:
                    hazard[(cls, k)] = n / survivors
        survivors -= sum(events.get((cls, k), 0) for cls in FAILURE_CLASSES)

    return {"at_risk": at_risk, "events": dict(events),
            "hazard": dict(hazard), "k0": dict(k0)}


def hazard_rank(rows: list, max_steps: int = MAX_STEPS) -> dict:
    """Rank the STEP-WISE classes by summed cause-specific hazard, 1 = most
    hazardous. Pre-execution classes get None.

    Summing h_C(k) over k weights each event by how few plans were still at
    risk when it happened, which is exactly the reweighting that undoes
    masking. It answers a different question from prevalence: not "how often
    does this happen" but "conditional on a plan getting the chance to fail
    this way, how likely is it to". A class with few events but a hazard near
    1.0 is saying something real -- every plan that got that far failed this
    way -- and prevalence alone would bury it.

    Rank rather than the raw sum goes in the diagnosis table, because the sum
    is not on an interpretable scale and inviting anyone to read it as one
    would be worse than useless.

    Pre-execution classes are deliberately NOT ranked here. Their events
    occur at k = 0 against the whole corpus, while step-wise hazards use a
    shrinking risk set, so the two are not on a common scale: a first attempt
    that summed them ranked a class hitting 10 plans (hazard 1.0, because
    only 10 were at risk) above one hitting 90 (0.9 of the corpus). Reporting
    None is the same treatment mean_epl_by_class() gives their structural
    zero, and for the same reason -- a number that cannot be compared should
    not be printed in a column that invites comparison. Their prevalence is
    unambiguous and is read straight off prevalence().
    """
    table = hazard_table(rows, max_steps)
    summed = defaultdict(float)
    for (cls, _k), h in table["hazard"].items():
        summed[cls] += h

    ordered = sorted(summed.items(), key=lambda kv: (-kv[1], kv[0]))
    ranks = {cls: i + 1 for i, (cls, _) in enumerate(ordered)}
    for cls in table["k0"]:
        ranks.setdefault(cls, None)
    return ranks


def prevalence(rows: list) -> dict:
    """Raw first-failure counts and proportions per class, all classes
    present (zeros included) so column sets never drift between runs."""
    counts = {cls: 0 for cls in FAILURE_CLASSES}
    for row in rows:
        counts[row["failure_class"]] = counts.get(row["failure_class"], 0) + 1
    n = len(rows) or 1
    return {cls: {"n": c, "pct": c / n} for cls, c in counts.items()}


def mean_epl_by_class(rows: list) -> dict:
    """Mean EPL per class, or None where the class's EPL is structural.

    Returning None rather than 0.0 for the pre-execution classes is the
    point of the function: their zero is a definition, and averaging
    definitions produces a number that looks like evidence and is not.
    """
    buckets = defaultdict(list)
    structural = set()
    for row in rows:
        if row.get("epl_is_structural"):
            structural.add(row["failure_class"])
            continue
        buckets[row["failure_class"]].append(row["epl"])

    out = {}
    for cls in FAILURE_CLASSES:
        vals = buckets.get(cls, [])
        if vals:
            out[cls] = sum(vals) / len(vals)
        elif cls in structural:
            out[cls] = None
        else:
            out[cls] = None
    return out
