"""
Tests for pipelines/plan_adequacy/worldstate.py

Tests the knowledge (`establishes`) vs effects (`establishes`) split
directly, since gates.py and executor.py both depend on WorldState.knows()
reading only from established knowledge, never from ground truth -- see
worldstate.py's module docstring and salvage_simulation.md's
knowledge-vs-truth split, reused here for plan text.
Run: python -m pytest tests/test_plan_adequacy_worldstate.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry
from pipelines.plan_adequacy.worldstate import WorldState


def _call(tool, params=None):
    return ToolCall(step_num=1, step_text=tool, tool=tool, params=params or {})


def test_knows_false_before_any_call():
    ws = WorldState()
    assert not ws.knows("substrate")


def test_knows_true_after_establishing_call():
    reg = ToolRegistry.load()
    ws = WorldState()
    ws.apply(_call("survey_seabed"), reg)
    assert ws.knows("substrate")


def test_is_true_tracks_effects_separately_from_knows():
    reg = ToolRegistry.load()
    ws = WorldState()
    ws.apply(_call("pull", params={"force_t": 90}), reg)
    assert ws.is_true("vessel_pulled")
    # pull's effects do not include "substrate" -- knows() must not leak
    # from is_true()'s set.
    assert not ws.knows("substrate")


def test_missing_requires_empty_when_satisfied():
    reg = ToolRegistry.load()
    ws = WorldState()
    ws.apply(_call("survey_seabed"), reg)
    ws.apply(_call("calculate_ground_reaction"), reg)
    ws.apply(_call("calculate_freeing_force"), reg)
    assert ws.missing_requires(_call("rig_beach_gear", {"n_legs": 4}), reg) == set()


def test_missing_requires_nonempty_when_unsatisfied():
    reg = ToolRegistry.load()
    ws = WorldState()
    missing = ws.missing_requires(_call("rig_beach_gear", {"n_legs": 4}), reg)
    assert "freeing_force" in missing


def test_apply_no_match_tool_is_a_noop_not_an_error():
    reg = ToolRegistry.load()
    ws = WorldState()
    ws.apply(_call("no_match"), reg)  # must not raise
    assert ws.known_facts() == frozenset()
    assert ws.true_facts() == frozenset()


def test_missing_requires_checks_both_known_and_true_facts():
    """right_vessel requires crew_rescued, which is an EFFECT (rescue_crew),
    not a knowledge fact -- missing_requires must check both sets."""
    reg = ToolRegistry.load()
    ws = WorldState()
    ws.apply(_call("rescue_crew", {"method": "hand-over-hand"}), reg)
    ws.apply(_call("calculate_stability"), reg)
    missing = ws.missing_requires(_call("right_vessel", {"method": "parbuckle"}), reg)
    assert missing == set()
