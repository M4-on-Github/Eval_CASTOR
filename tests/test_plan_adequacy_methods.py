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


# ---------------------------------------------------------------------------
# Recognition ties and admissibility
# ---------------------------------------------------------------------------
# capsized/manual_righting and capsized/crane_lift_right both have
# core_tools == {right_vessel}. Declaration order made manual_righting win
# every time, so crane_lift_right was unreachable and all 22 corpus plans
# saying only "right the vessel" were scored inadmissible -- including 10
# medium-vessel cases crane_lift_right allows. See admissible_over_ties.

def _scen(**kw):
    from types import SimpleNamespace
    kw.setdefault("image", "capsized/test.jpg")
    return SimpleNamespace(**kw)


def test_right_vessel_alone_records_both_tied_routes():
    from pipelines.plan_adequacy.methods import RouteRegistry, recognise_route
    reg = RouteRegistry.load()
    match = recognise_route({"right_vessel"}, "capsized", reg)
    names = {r.name for r in match.tied_routes}
    assert names == {"manual_righting", "crane_lift_right"}, names
    # route stays first-declared, so route_name/coherence are unchanged.
    assert match.route.name == "manual_righting"


def test_a_medium_vessel_righting_tie_is_ambiguous_not_inadmissible():
    """manual_righting says no, crane_lift_right says yes, and the plan said
    only "right the vessel". Recording that as a pass OR a failure invents
    information the call set does not contain."""
    from pipelines.plan_adequacy.methods import (RouteRegistry, recognise_route,
                                                 admissible_over_ties)
    reg = RouteRegistry.load()
    match = recognise_route({"right_vessel"}, "capsized", reg)
    assert admissible_over_ties(match, _scen(size_category="medium")) == "ambiguous"


def test_a_large_vessel_righting_tie_is_still_inadmissible():
    """The tie only matters when the tied routes disagree. At `large` neither
    righting route is admissible, so the finding stands however the tie falls
    -- these are the 12 genuine corpus failures that must NOT be lost."""
    from pipelines.plan_adequacy.methods import (RouteRegistry, recognise_route,
                                                 admissible_over_ties)
    reg = RouteRegistry.load()
    match = recognise_route({"right_vessel"}, "capsized", reg)
    assert admissible_over_ties(match, _scen(size_category="large")) == "no"


def test_a_small_vessel_righting_tie_is_admissible():
    from pipelines.plan_adequacy.methods import (RouteRegistry, recognise_route,
                                                 admissible_over_ties)
    reg = RouteRegistry.load()
    match = recognise_route({"right_vessel"}, "capsized", reg)
    assert admissible_over_ties(match, _scen(size_category="small")) == "yes"


def test_an_unpopulated_scenario_field_still_reads_unknown_not_ambiguous():
    """"unknown" outranks "ambiguous": if we don't know the size we can't know
    the two routes disagree. executor.py already declines to penalise it."""
    from pipelines.plan_adequacy.methods import (RouteRegistry, recognise_route,
                                                 admissible_over_ties)
    reg = RouteRegistry.load()
    match = recognise_route({"right_vessel"}, "capsized", reg)
    assert admissible_over_ties(match, _scen()) == "unknown"


def test_the_foam_tie_is_not_ambiguous_because_both_routes_agree():
    """high_expansion_foam and deck_foam_system also tie on {apply_foam}, but
    both are admissibility_kind "always", so the tie does not matter and the
    verdict is a confident "yes". Guards against over-firing ambiguity."""
    from pipelines.plan_adequacy.methods import (RouteRegistry, recognise_route,
                                                 admissible_over_ties)
    reg = RouteRegistry.load()
    match = recognise_route({"apply_foam"}, "on_fire", reg)
    assert len(match.tied_routes) >= 2
    assert admissible_over_ties(match, _scen()) == "yes"


def test_a_non_tie_still_resolves_to_a_single_route():
    """The matched-count tie-break must survive: {lighter_cargo,
    rig_beach_gear, dredge, pull} fully covers several routes but `combined`
    explains the most, so it wins outright and is NOT a tie."""
    from pipelines.plan_adequacy.methods import RouteRegistry, recognise_route
    reg = RouteRegistry.load()
    match = recognise_route({"lighter_cargo", "rig_beach_gear", "dredge", "pull"},
                            "aground", reg)
    assert len(match.tied_routes) == 1
    assert match.tied_routes[0] is match.route
