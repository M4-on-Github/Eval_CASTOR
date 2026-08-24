"""
Tests for pipelines/plan_adequacy/methods.py

Route anchor: hand-built call sets for each of the 7 aground routes must
each be recognised as their own route -- proving the multi-path validity
model (design plan section 2) actually distinguishes legitimate alternative
routes instead of penalising a plan for not following one canonical
sequence.
Run: python -m pytest tests/test_plan_adequacy_methods.py -v
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.methods import RouteRegistry, admissible, recognise_route


def _reg():
    return RouteRegistry.load()


# ── route recognition: one call set per aground route ────────────────────────

def test_recognises_tide_refloat():
    match = recognise_route({"monitor_tide"}, "aground", _reg())
    assert match.route is not None
    assert match.route.name == "tide_refloat"


def test_recognises_tug_pull():
    match = recognise_route({"attach_tug", "pull"}, "aground", _reg())
    assert match.route.name == "tug_pull"


def test_recognises_beach_gear():
    match = recognise_route({"rig_beach_gear", "pull"}, "aground", _reg())
    assert match.route.name == "beach_gear"


def test_recognises_weight_reduction():
    match = recognise_route({"lighter_cargo", "pull"}, "aground", _reg())
    assert match.route.name == "weight_reduction"


def test_recognises_seabed_modification():
    match = recognise_route({"dredge", "pull"}, "aground", _reg())
    assert match.route.name == "seabed_modification"


def test_recognises_combined():
    match = recognise_route(
        {"lighter_cargo", "rig_beach_gear", "dredge", "pull"}, "aground", _reg()
    )
    assert match.route.name == "combined"


def test_recognises_abandon_in_place():
    match = recognise_route({"offload_fuel"}, "aground", _reg())
    assert match.route.name == "abandon_in_place"


# ── a plan choosing a valid alternative is not penalised for skipping others ─

def test_tide_refloat_plan_not_marked_off_for_missing_dredge():
    """The core validity claim: a plan that correctly waits for the tide
    must not be graded against seabed_modification's core_tools."""
    match = recognise_route({"monitor_tide"}, "aground", _reg())
    assert match.route.name == "tide_refloat"
    assert "dredge" not in match.route.core_tools


# ── shotgun-plan detection (route_coherence numerator/denominator) ──────────

def test_unmatched_tools_flags_shotgun_plan():
    # Calls tools from tug_pull AND dredge (seabed_modification) at once --
    # should recognise the higher-overlap route and flag the other as
    # unmatched (this is what route_coherence penalises).
    match = recognise_route({"attach_tug", "pull", "dredge"}, "aground", _reg())
    assert match.route is not None
    assert len(match.unmatched_tools) >= 1


# ── no match below the recognition floor ──────────────────────────────────────

def test_no_recognisable_route_for_unrelated_tools():
    match = recognise_route({"apply_foam"}, "aground", _reg())
    assert match.route is None


# ── admissibility ─────────────────────────────────────────────────────────────

def test_admissible_always_kind():
    reg = _reg()
    route = [r for r in reg.for_casualty("aground") if r.name == "weight_reduction"][0]
    scenario = SimpleNamespace()
    assert admissible(route, scenario) == "yes"


def test_admissible_scenario_field_yes():
    reg = _reg()
    route = [r for r in reg.for_casualty("capsized") if r.name == "manual_righting"][0]
    scenario = SimpleNamespace(size_category="small")
    assert admissible(route, scenario) == "yes"


def test_admissible_scenario_field_no():
    reg = _reg()
    route = [r for r in reg.for_casualty("capsized") if r.name == "manual_righting"][0]
    scenario = SimpleNamespace(size_category="large")
    assert admissible(route, scenario) == "no"


def test_admissible_unknown_when_field_missing():
    reg = _reg()
    route = [r for r in reg.for_casualty("capsized") if r.name == "manual_righting"][0]
    scenario = SimpleNamespace()  # no size_category attribute at all
    assert admissible(route, scenario) == "unknown"


def test_admissible_unknown_pending_physics():
    reg = _reg()
    route = [r for r in reg.for_casualty("aground") if r.name == "tug_pull"][0]
    scenario = SimpleNamespace()
    assert admissible(route, scenario) == "unknown"
