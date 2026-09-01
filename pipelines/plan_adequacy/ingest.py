"""
ingest.py -- read hand-coded worksheet exports back into the diagnosis.

worksheet.py writes the coding tasks; this reads the JSON they export and
turns it into the two columns no amount of code can fill: whether a label is
correct on real plans, and whether a failure belongs to the planner or to us.

The coder is recorded on every ingest and carried into the reported table.
That is not bookkeeping. The taxonomy, the checker and the first coding pass
were all produced by the same author, so agreement between them measures
self-consistency, not validity. A pass coded by the model is an ERROR
ANALYSIS -- it can find labels that are wrong, and it cannot establish that
the remaining ones are right. Anything reported from a model-coded pass has to
say so, which is why `coder` is required rather than defaulted.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

#: Attribution choices that mean "this is a finding about the planner" versus
#: "this is a finding about our own pipeline". `ambiguous` counts as neither
#: and is reported separately -- folding it into either side would
#: manufacture certainty the coder explicitly declined to claim.
PLANNER = "planner"
INSTRUMENT = ("registry", "extractor")


def read_coded(path) -> dict:
    """One worksheet export: {"task", "coder", "labels": {id: {choice, note}}}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"task": data.get("task", ""), "coder": data.get("coder", "unknown"),
            "labels": data.get("labels", {})}


def attribution_by_class(coded: dict, per_image_rows: list) -> dict:
    """{failure_class: {planner, registry, extractor, ambiguous, n, pct_planner}}.

    pct_planner is computed over CODED items only, and excludes `ambiguous`
    from the denominator, so it reads as "of the items where a call could be
    made, how many were the planner's". n_ambiguous is reported alongside so
    the size of that exclusion is visible rather than buried.
    """
    cls_of = {r["_key"] if "_key" in r else r["image"]: r["failure_class"]
              for r in per_image_rows}
    buckets = defaultdict(Counter)
    for item_id, rec in coded["labels"].items():
        cls = cls_of.get(item_id)
        if cls:
            buckets[cls][rec["choice"]] += 1

    out = {}
    for cls, c in buckets.items():
        decided = c[PLANNER] + sum(c[k] for k in INSTRUMENT)
        out[cls] = {
            "planner": c[PLANNER],
            "registry": c["registry"],
            "extractor": c["extractor"],
            "ambiguous": c["ambiguous"],
            "n": sum(c.values()),
            "pct_planner": (c[PLANNER] / decided) if decided else None,
        }
    return out


def perception_precision(coded: dict, groups: dict) -> dict:
    """Precision of the perception detector at each threshold.

    `groups` maps item id -> "confirmed" (flagged at the shipped threshold) or
    "disputed" (flagged only by the permissive one). A label of "same" means
    the coder judged the plan to match its real casualty, i.e. the detector
    fired wrongly.
    """
    hit = Counter()
    tot = Counter()
    for item_id, rec in coded["labels"].items():
        g = groups.get(item_id)
        if g:
            tot[g] += 1
            hit[g] += int(rec["choice"] != "same")

    strict_n, strict_h = tot["confirmed"], hit["confirmed"]
    both_n = strict_n + tot["disputed"]
    both_h = strict_h + hit["disputed"]
    return {
        "shipped_threshold": {"n": strict_n,
                              "precision": strict_h / strict_n if strict_n else None},
        "disputed_band": {"n": tot["disputed"],
                          "precision": hit["disputed"] / tot["disputed"] if tot["disputed"] else None},
        "permissive_threshold": {"n": both_n,
                                 "precision": both_h / both_n if both_n else None},
    }


def estimate_true_perception_count(precisions: dict, n_shipped: int,
                                   n_permissive: int) -> dict:
    """Extrapolate a corpus count from the two measured precisions.

    Returned as a LOWER bound, and labelled as one. Precision says what
    fraction of detections are real; it says nothing about failures neither
    threshold caught, and a perception failure severe enough to match no route
    at all is invisible to both. The estimate can therefore only move the
    number up from here.
    """
    p_strict = precisions["shipped_threshold"]["precision"]
    p_disputed = precisions["disputed_band"]["precision"]
    if p_strict is None or p_disputed is None:
        return {}
    band = n_permissive - n_shipped
    return {
        "from_shipped": n_shipped * p_strict,
        "from_disputed_band": band * p_disputed,
        "lower_bound_total": n_shipped * p_strict + band * p_disputed,
    }
