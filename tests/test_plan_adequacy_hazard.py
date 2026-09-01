"""
Tests for pipelines/plan_adequacy/hazard.py

The point of these is the masking property: a class that strikes late must
outrank its raw prevalence once the risk set shrinks. That is the entire
reason the module exists, so it is tested directly with a hand-built corpus
whose masking is arithmetically obvious rather than only through smoke tests.

Run: python -m pytest tests/test_plan_adequacy_hazard.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.hazard import (hazard_rank, hazard_table,
                                            mean_epl_by_class, prevalence)


def _row(cls, step, structural=False):
    return {"failure_class": cls, "failure_step": step,
            "epl": 0 if step is None else step - 1,
            "epl_is_structural": structural}


def test_prevalence_reports_every_class_including_zeros():
    rows = [_row("PROCEDURE", 2), _row("COMMITMENT", 5)]
    prev = prevalence(rows)
    assert prev["PROCEDURE"]["n"] == 1
    assert prev["INCOMPLETE"]["n"] == 0          # present, not absent
    assert abs(sum(v["pct"] for v in prev.values()) - 1.0) < 1e-9


def test_late_class_outranks_its_prevalence_once_masking_is_corrected():
    """The property the module exists for.

    18 plans fail PROCEDURE at step 1; 2 fail COMMITMENT at step 5. Raw
    prevalence says PROCEDURE is 9x more common. But only 2 plans were still
    at risk by step 5, so COMMITMENT's hazard there is 2/2 = 1.0 against
    PROCEDURE's 18/20 = 0.9 -- COMMITMENT is the more hazardous class per
    plan that actually reached it, and would have been under-ranked by a
    naive count.
    """
    rows = [_row("PROCEDURE", 1) for _ in range(18)] + \
           [_row("COMMITMENT", 5) for _ in range(2)]

    prev = prevalence(rows)
    assert prev["PROCEDURE"]["n"] > prev["COMMITMENT"]["n"]

    table = hazard_table(rows)
    assert table["at_risk"][1] == 20
    assert table["at_risk"][5] == 2                  # 18 already failed
    assert table["hazard"][("PROCEDURE", 1)] == 18 / 20
    assert table["hazard"][("COMMITMENT", 5)] == 2 / 2

    assert hazard_rank(rows)["COMMITMENT"] < hazard_rank(rows)["PROCEDURE"]


def test_pre_execution_failures_are_excluded_from_step_risk_sets():
    """A plan that never got a route was never at risk at any step. Letting
    it into the denominator would make every step-wise hazard a function of
    the no-route rate, which is a different question entirely."""
    rows = [_row("NO_PROCEDURE", None, structural=True) for _ in range(50)]
    rows += [_row("PROCEDURE", 1) for _ in range(10)]

    table = hazard_table(rows)
    assert table["k0"]["NO_PROCEDURE"] == 50
    assert table["at_risk"][1] == 10                  # not 60
    assert table["hazard"][("PROCEDURE", 1)] == 1.0


def test_pre_execution_classes_are_not_ranked_on_the_step_wise_scale():
    """k=0 events use the whole corpus as denominator; step-wise hazards use
    a shrinking risk set. Summing them ranked NO_PROCEDURE (90 plans, 0.9 of
    the corpus) BELOW PROCEDURE (10 plans, but hazard 1.0 because only 10
    were ever at risk). The two are not on a common scale, so the
    pre-execution classes report None and their prevalence is read directly.
    """
    rows = [_row("NO_PROCEDURE", None, structural=True) for _ in range(90)]
    rows += [_row("PROCEDURE", 1) for _ in range(10)]
    ranks = hazard_rank(rows)
    assert ranks["NO_PROCEDURE"] is None
    assert ranks["PROCEDURE"] == 1
    assert prevalence(rows)["NO_PROCEDURE"]["pct"] == 0.9


def test_valid_and_incomplete_plans_stay_in_the_risk_set_and_produce_no_events():
    rows = [_row("PROCEDURE", 3) for _ in range(5)]
    rows += [{"failure_class": "VALID", "failure_step": None, "epl": 6,
              "epl_is_structural": False} for _ in range(5)]
    table = hazard_table(rows)
    assert table["at_risk"][1] == 10            # censored plans are at risk
    assert "VALID" not in table["k0"]           # but never an event
    assert table["hazard"][("PROCEDURE", 3)] == 0.5


def test_mean_epl_is_none_for_structural_zeros_not_zero():
    """Averaging a definition produces a number that looks like evidence."""
    rows = [_row("NO_PROCEDURE", None, structural=True) for _ in range(3)]
    rows += [_row("COMMITMENT", 5), _row("COMMITMENT", 3)]
    means = mean_epl_by_class(rows)
    assert means["NO_PROCEDURE"] is None
    assert means["COMMITMENT"] == 3.0           # (4 + 2) / 2


def test_empty_corpus_does_not_divide_by_zero():
    assert hazard_table([])["at_risk"][1] == 0
    assert hazard_rank([]) == {}
    assert prevalence([])["PROCEDURE"]["n"] == 0
