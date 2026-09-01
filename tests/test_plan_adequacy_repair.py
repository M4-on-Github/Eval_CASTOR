"""
Tests for pipelines/plan_adequacy/repair.py

Driven with hand-written ToolCall lists against the real registry, so these
exercise the actual neutralise path through executor.py rather than a mock of
it. The two properties that matter: repairing a clean plan changes nothing
(idempotence), and repairing a real failure moves EPL by exactly the distance
to the next failure.

Run: python -m pytest tests/test_plan_adequacy_repair.py -v
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.classify import classify
from pipelines.plan_adequacy.executor import execute_plan
from pipelines.plan_adequacy.methods import RouteRegistry
from pipelines.plan_adequacy.repair import (delta_epl_by_class, repair_once,
                                            repair_to_exhaustion,
                                            transition_matrix)
from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry


def _reg():
    return ToolRegistry.load(), RouteRegistry.load()


def _scenario(**kw):
    kw.setdefault("image", "aground/test.jpg")
    return SimpleNamespace(**kw)


def _call(n, tool, text="", params=None, conditional=False,
          condition_text=None, condition_var="none"):
    return ToolCall(step_num=n, step_text=text or tool, tool=tool,
                    params=params or {}, conditional=conditional,
                    condition_text=condition_text, condition_var=condition_var)


#: A six-step aground plan with no violations and every magnitude in the TEXT
#: (specificity is read from the sentence, not the params dict).
def _clean_aground():
    return [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", text="Attach 2 tugs of 4000 shp each",
              params={"count": 2, "shp": 4000.0}),
        _call(6, "pull", text="Pull at 90 t bollard pull", params={"force_t": 90}),
    ]


def test_repairing_a_clean_plan_returns_nothing():
    """Idempotence: there is no first failure, so there is nothing to
    neutralise and no delta to claim."""
    tr, rr = _reg()
    calls = _clean_aground()
    assert classify(execute_plan(calls, "aground", _scenario(), tr, rr))["failure_step"] is None
    assert repair_once(calls, "aground", _scenario(), tr, rr) is None

    out = repair_to_exhaustion(calls, "aground", _scenario(), tr, rr)
    assert out["chain"] == []
    assert out["repairs_to_valid"] == 0


def test_repairing_a_sequence_violation_advances_epl_to_the_next_failure():
    tr, rr = _reg()
    # pull at step 1 with no freeing-force calculation: SEQUENCE_VIOLATION.
    calls = [_call(1, "pull", text="Pull at 90 t", params={"force_t": 90})] + \
            [_call(n, t) for n, t in [(2, "survey_seabed"), (3, "calculate_ground_reaction")]]
    before = classify(execute_plan(calls, "aground", _scenario(), tr, rr))
    assert before["failure_class"] == "PROCEDURE"
    assert before["epl"] == 0

    step = repair_once(calls, "aground", _scenario(), tr, rr)
    assert step["repaired_step"] == 1
    assert step["repaired_class"] == "PROCEDURE"
    assert step["delta_epl"] == step["epl_after"] - step["epl_before"]
    assert step["epl_after"] > step["epl_before"]


def test_granting_a_precondition_also_unblocks_downstream_steps():
    """The repair grants the missing fact into the world state, so a later
    step that legitimately depended on it is no longer scored against a gap
    the repair is standing in for. Without ws.grant() this test's step 2
    would still read SEQUENCE_VIOLATION and delta_epl would understate."""
    tr, rr = _reg()
    calls = [
        _call(1, "pull", text="Pull at 90 t", params={"force_t": 90}),
        _call(2, "pull", text="Pull again at 95 t", params={"force_t": 95}),
    ]
    step = repair_once(calls, "aground", _scenario(), tr, rr)
    assert step["repaired_step"] == 1
    assert step["epl_after"] >= 2          # step 2 no longer violates


def test_repairing_an_unspecified_step_is_a_commitment_repair():
    tr, rr = _reg()
    calls = _clean_aground()
    # Strip the magnitudes from the step TEXT -- params are deliberately left
    # populated, since specificity must not be satisfiable from them.
    calls[4] = _call(5, "attach_tug", text="Attach tugs",
                     params={"count": 2, "shp": 4000.0})
    calls[5] = _call(6, "pull", text="Pull hard", params={"force_t": 90})

    before = classify(execute_plan(calls, "aground", _scenario(), tr, rr))
    assert before["failure_class"] == "COMMITMENT"
    assert before["failure_step"] == 5

    step = repair_once(calls, "aground", _scenario(), tr, rr)
    assert step["repaired_class"] == "COMMITMENT"
    assert step["next_class"] == "COMMITMENT"      # step 6 is unspecified too
    assert step["delta_epl"] == 1                  # a wall, not a speed bump


def test_exhaustion_chains_repairs_and_counts_distance_to_valid():
    tr, rr = _reg()
    calls = _clean_aground()
    calls[4] = _call(5, "attach_tug", text="Attach tugs", params={"count": 2})
    calls[5] = _call(6, "pull", text="Pull hard", params={"force_t": 90})

    out = repair_to_exhaustion(calls, "aground", _scenario(), tr, rr)
    assert [c["repaired_step"] for c in out["chain"]] == [5, 6]
    assert out["class_sequence"] == ["COMMITMENT", "COMMITMENT"]
    assert out["repairs_to_valid"] == 2
    assert out["final_epl"] == 6


def test_repair_never_exceeds_the_iteration_cap():
    tr, rr = _reg()
    calls = [_call(n, "pull", text="Pull", params={"force_t": 90}) for n in range(1, 7)]
    out = repair_to_exhaustion(calls, "aground", _scenario(), tr, rr, max_repairs=3)
    assert len(out["chain"]) <= 3


def test_pre_execution_failures_are_not_repairable():
    """A plan that never matched a route has no failing STEP to neutralise;
    its remedy is measured by regeneration, not simulation."""
    tr, rr = _reg()
    calls = [_call(1, "no_match"), _call(2, "no_match")]
    result = execute_plan(calls, "aground", _scenario(), tr, rr)
    assert result.route_name is None
    assert repair_once(calls, "aground", _scenario(), tr, rr) is None


# ── aggregation helpers ──────────────────────────────────────────────────────

def _row(cls, delta, nxt):
    return {"chain": [{"repaired_class": cls, "delta_epl": delta, "next_class": nxt}]}


def test_delta_epl_by_class_uses_first_repairs_only():
    rows = [_row("COMMITMENT", 1, "COMMITMENT"), _row("COMMITMENT", 3, "VALID"),
            _row("PROCEDURE", 0, "PROCEDURE")]
    out = delta_epl_by_class(rows)
    assert out["COMMITMENT"] == {"n": 2, "mean_delta_epl": 2.0}
    assert out["PROCEDURE"]["mean_delta_epl"] == 0.0


def test_transition_matrix_exposes_the_wall_diagonal():
    rows = [_row("COMMITMENT", 1, "COMMITMENT")] * 3 + [_row("COMMITMENT", 4, "VALID")]
    tm = transition_matrix(rows)
    assert tm[("COMMITMENT", "COMMITMENT")] == 3
    assert tm[("COMMITMENT", "VALID")] == 1


def test_empty_repair_rows_do_not_crash_the_aggregators():
    assert delta_epl_by_class([]) == {}
    assert transition_matrix([]) == {}
