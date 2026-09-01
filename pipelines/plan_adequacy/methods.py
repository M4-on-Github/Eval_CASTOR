"""
Loader over registry/routes.json + registry/goals.json, plus route
recognition and admissibility evaluation.

Route content lives entirely in the registry (see vocab.py's module
docstring for why); this module is the evaluator over it. See the
plan-adequacy design plan section 2 for the full validity-model rationale:
a salvage goal is reachable by several genuinely valid routes, so a plan is
graded against the ONE route it recognisably instantiates, not a single
canonical sequence.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pipelines.plan_adequacy.paths import GOALS_PATH, ROUTES_PATH


@dataclass(frozen=True)
class Route:
    name: str
    casualty: str
    goal: str                  # a fact name -- the route's own completion condition
    core_tools: frozenset
    preconditions: frozenset
    admissibility: dict        # raw {"kind": ..., ...} from routes.json
    source: str = ""

    @classmethod
    def _from_json(cls, casualty: str, d: dict) -> "Route":
        return cls(
            name=d["name"],
            casualty=casualty,
            goal=d["goal"],
            core_tools=frozenset(d.get("core_tools", [])),
            preconditions=frozenset(d.get("preconditions", [])),
            admissibility=dict(d.get("admissibility", {"kind": "always"})),
            source=d.get("source", ""),
        )


@dataclass(frozen=True)
class Goal:
    casualty: str
    terminal_facts: frozenset
    description: str = ""


@dataclass(frozen=True)
class UniversalObligation:
    id: str
    requires_fact: str
    applies_to: frozenset
    except_routes: frozenset = frozenset()      # routes exempt from this obligation
    requires_scenario_field: Optional[str] = None  # only applies when this scenario attr is truthy
    note: str = ""
    source: str = ""


@dataclass
class RouteMatch:
    """Result of recognise_route(): which route (if any) a call list best
    fits, and how cleanly."""

    route: Optional[Route]
    score: float                 # 0.0-1.0, fraction of route.core_tools present
    matched_tools: frozenset     # called tools that belong to route.core_tools
    unmatched_tools: frozenset   # called action tools that do NOT belong to it
                                  # -- the numerator's complement for route_coherence


class RouteRegistry:
    """Routes + goals, loaded from registry/*.json.

    Use RouteRegistry.load(); instances are cheap to hold and reuse across a
    whole run once loaded.
    """

    #: Minimum core_tools-overlap score to count as a recognised route at
    #: all, rather than NO_RECOGNISABLE_ROUTE. [AUTHORED] -- not sourced,
    #: chosen so a plan that names exactly one of a multi-tool route's core
    #: tools (e.g. only "pull", no "attach_tug") is not credited with the
    #: whole route. See calibrate.py for where this would be tuned against
    #: real data if it turns out to misfire.
    RECOGNITION_FLOOR = 0.34

    def __init__(self, routes: dict, goals: dict, obligations: list):
        self._routes = routes            # {casualty: [Route, ...]}
        self._goals = goals              # {casualty: Goal}
        self._obligations = obligations  # [UniversalObligation, ...]

    @classmethod
    def load(cls, routes_path: Path = ROUTES_PATH, goals_path: Path = GOALS_PATH) -> "RouteRegistry":
        routes_raw = json.loads(Path(routes_path).read_text(encoding="utf-8"))
        goals_raw = json.loads(Path(goals_path).read_text(encoding="utf-8"))

        routes = {}
        for casualty, route_list in routes_raw["routes"].items():
            routes[casualty] = [Route._from_json(casualty, r) for r in route_list]

        goals = {}
        for casualty, g in goals_raw["casualties"].items():
            goals[casualty] = Goal(
                casualty=casualty,
                terminal_facts=frozenset(g.get("terminal_facts", [])),
                description=g.get("description", ""),
            )

        obligations = [
            UniversalObligation(
                id=o["id"], requires_fact=o["requires_fact"],
                applies_to=frozenset(o.get("applies_to", [])),
                except_routes=frozenset(o.get("except_routes", [])),
                requires_scenario_field=o.get("requires_scenario_field"),
                note=o.get("note", ""), source=o.get("source", ""),
            )
            for o in goals_raw.get("universal_obligations", [])
        ]

        return cls(routes, goals, obligations)

    def for_casualty(self, casualty: str) -> list:
        return list(self._routes.get(casualty, []))

    def goal_for(self, casualty: str) -> Optional[Goal]:
        return self._goals.get(casualty)

    def obligations_for(self, casualty: str) -> list:
        return [o for o in self._obligations if casualty in o.applies_to]

    def all_route_names(self, casualty: Optional[str] = None) -> set:
        if casualty is not None:
            return {r.name for r in self._routes.get(casualty, [])}
        return {r.name for routes in self._routes.values() for r in routes}

    def all_casualties(self) -> set:
        return set(self._routes.keys())


def recognise_route(called_tools: set, casualty: str, registry: RouteRegistry) -> RouteMatch:
    """Which route a plan's set of called tool names best fits.

    Score is core_tools coverage: |called ∩ core_tools| / |core_tools|. This
    is deliberately NOT a Jaccard score (which would also penalise a route
    for tools the plan called that aren't in its core_tools) -- a plan is
    allowed to call extra assessment tools without being marked off a route,
    it is only the ACTION tools outside the matched route that feed
    route_coherence separately (see unmatched_tools).

    Known limitation (see routes.json's _known_limitation): recognition is
    tool-name-only. high_expansion_foam and deck_foam_system currently share
    apply_foam as their sole core tool and are indistinguishable by this
    function; ties are broken by declaration order in routes.json (first
    route wins), which is arbitrary and documented as a phase-2 TODO once
    physics.py can disambiguate by params.type/params.space.
    """
    candidates = registry.for_casualty(casualty)
    if not candidates:
        return RouteMatch(route=None, score=0.0, matched_tools=frozenset(), unmatched_tools=frozenset(called_tools))

    # Ranking key is (coverage score, tools matched). The second element is
    # the tie-break that matters: several routes can each be FULLY covered
    # by the same call set (e.g. {lighter_cargo, rig_beach_gear, dredge,
    # pull} scores 1.0 for weight_reduction, beach_gear, seabed_modification,
    # AND combined -- each is individually satisfied). Preferring the
    # highest matched-count picks the route that explains the most of what
    # was actually called (combined, matched=4) over one that only explains
    # part of it and leaves the rest as unmatched_tools (beach_gear,
    # matched=2) -- see test_recognises_combined for the regression this
    # tie-break exists to fix.
    best = None
    best_key = (-1.0, -1)
    best_matched = frozenset()
    for route in candidates:
        if not route.core_tools:
            continue
        matched = called_tools & route.core_tools
        score = len(matched) / len(route.core_tools)
        key = (score, len(matched))
        if key > best_key:
            best, best_key, best_matched = route, key, matched
    best_score = best_key[0] if best is not None else 0.0

    if best is None or best_score < RouteRegistry.RECOGNITION_FLOOR:
        return RouteMatch(route=None, score=max(best_score, 0.0),
                           matched_tools=frozenset(), unmatched_tools=frozenset(called_tools))

    # unmatched_tools: called tools that belong to SOME route in this
    # casualty's library (i.e. are action tools, not assessment/terminal)
    # but not to the matched route. This is what route_coherence measures --
    # a shotgun plan calling tools from several routes at once.
    all_action_tools = {t for r in candidates for t in r.core_tools}
    unmatched = (called_tools & all_action_tools) - best.core_tools

    return RouteMatch(route=best, score=best_score, matched_tools=best_matched, unmatched_tools=unmatched)


#: How many tools distinctive to the foreign family a plan must call before
#: it is judged to be solving a different accident. [AUTHORED]
#:
#: This is the most consequential free parameter in the diagnosis, and it was
#: set by measurement rather than taste. Eight of the 24 routes have a SINGLE
#: core tool (aground/tide_refloat, capsized/manual_righting,
#: sunken/lift_bags_or_craft, ...), so any plan uttering that one tool scores
#: a perfect 1.0 against that route and beats almost any partial match in its
#: own family. At a threshold of 1, aground/00017 -- a recognisably aground
#: plan that happens to mention "lift" and a cofferdam -- is called a sunken
#: plan.
#:
#: Over the 330-plan corpus the threshold moves the STRATEGY_PERCEPTION class
#: from 118 plans (35.8%) at 1 to 33 (10.0%) at 2. Both numbers should be
#: reported as a sensitivity band until hand coding settles it. 2 is the
#: default because the asymmetry of the error is not symmetric: a false
#: perception finding points the whole research programme at visual grounding
#: on the strength of a shared verb, while a missed one merely lands the plan
#: in a neighbouring class.
MIN_DISTINCTIVE_FOREIGN_TOOLS = 2


def detect_perception_mismatch(called_tools: set, casualty: str,
                                registry: RouteRegistry,
                                min_distinctive: int = None) -> Optional[str]:
    """Does this plan fit some OTHER casualty's routes better than its own?

    Diagnostic overlay only -- this must never feed grading. recognise_route()
    above only ever considers registry.for_casualty(casualty), which is
    correct for scoring (a plan is graded against the routes that could
    actually resolve the casualty it was given) but means a plan describing
    an aground refloat for a sunken vessel simply matches nothing and reads
    as incoherent. That conflates two very different diagnoses: "wrote no
    recognisable procedure" and "wrote a competent procedure for the wrong
    accident". The second is a perception failure upstream of planning, and
    it needs its own class -- see the failure taxonomy in
    reports/p9/redesign.tex.

    Returns the foreign casualty name when the plan's tools fit one of its
    routes strictly better than anything in `casualty`'s own library, clear
    the recognition floor, and include at least `min_distinctive` tools that
    appear in NO route of the plan's own casualty (default
    MIN_DISTINCTIVE_FOREIGN_TOOLS -- see that constant for why this matters
    more than any other parameter here).
    """
    if min_distinctive is None:
        min_distinctive = MIN_DISTINCTIVE_FOREIGN_TOOLS
    own_routes = registry.for_casualty(casualty)
    own_best = 0.0
    for route in own_routes:
        if route.core_tools:
            own_best = max(own_best, len(called_tools & route.core_tools) / len(route.core_tools))

    # Tools that appear in ANY of this casualty's own routes. A foreign match
    # built only out of these is not evidence of anything: assessment and
    # safety tools (survey_hull, muster_personnel, calculate_stability) are
    # shared across casualty families by design, so a generic plan will
    # partially "match" every family's routes. Requiring at least one
    # foreign-DISTINCTIVE tool is what separates "this plan uses the wrong
    # family's techniques" from "this plan uses the tools everybody uses".
    own_tools = {t for r in own_routes for t in r.core_tools}

    foreign_best, foreign_casualty = 0.0, None
    for other in registry.all_casualties():
        if other == casualty:
            continue
        for route in registry.for_casualty(other):
            if not route.core_tools:
                continue
            matched = called_tools & route.core_tools
            score = len(matched) / len(route.core_tools)
            if score > foreign_best and len(matched - own_tools) >= min_distinctive:
                foreign_best, foreign_casualty = score, other

    if foreign_best < RouteRegistry.RECOGNITION_FLOOR or foreign_best <= own_best:
        return None
    return foreign_casualty


def admissible(route: Route, scenario) -> Literal["yes", "no", "unknown"]:
    """Is `route` physically admissible for `scenario`?

    "unknown" is a first-class result, not a failure: most routes' real
    admissibility needs phase-2 physics.py (F_required vs. available pull,
    depth bands, etc. -- see routes.json admissibility_kinds). executor.py
    MUST NOT treat "unknown" as ROUTE_INADMISSIBLE; it is reported
    separately, per build order section 7's "sensitivity axis, not an
    authored constant" framing.
    """
    kind = route.admissibility.get("kind", "always")
    if kind == "always":
        return "yes"
    if kind == "unknown_pending_physics":
        return "unknown"
    if kind == "scenario_field":
        field_name = route.admissibility["field"]
        allowed = set(route.admissibility.get("allowed", []))
        value = getattr(scenario, field_name, None)
        if value is None:
            return "unknown"   # scenario doesn't populate this field yet
        return "yes" if value in allowed else "no"
    return "unknown"
