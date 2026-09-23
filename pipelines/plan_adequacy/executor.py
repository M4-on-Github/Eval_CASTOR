"""
The seven-pass executor: walks an extracted (or hand-written) ToolCall list
and derives per-step verdicts plus a plan-level summary.

See the plan-adequacy design plan, section 2, for the full pass table this
implements. Numeric adequacy grading (SPECIFIED_ADEQUATE vs
SPECIFIED_INADEQUATE, the A1-A10 checks in salvage_plan_checker.md) is
PHASE 2 -- physics.py does not exist yet (build order section 7). Until then,
a step naming a concrete magnitude is marked SPECIFIED_UNGRADED rather than
guessed at; this is a deliberate placeholder, not a bug, and is called out
explicitly in every summary so it is never silently mistaken for a real
adequacy result.

Two calibration modes consume this module identically (see design plan
section 4e): executor(calls) run on hand-written gold ToolCalls is
"executor@oracle"; the same function run on LLM-extracted calls is
"executor@model". The gap between the two IS the extraction loss.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from pipelines.plan_adequacy import gates
from pipelines.plan_adequacy.methods import (RouteRegistry, admissible_over_ties,
                                             detect_perception_mismatch, recognise_route)
from pipelines.plan_adequacy.support import is_support
from pipelines.plan_adequacy.vocab import ToolRegistry
from pipelines.plan_adequacy.worldstate import WorldState

#: Matches a STANDALONE digit -- \b anchored, so it requires a non-word
#: character (or string start) immediately before the digit. Plans in this
#: corpus that DO state a magnitude write it numerically at a word boundary
#: ("2 harbor tugs", "500 tons", "10 m draft"); deliberately permissive
#: beyond that (no unit-word requirement) rather than a tight number+unit
#: grammar. See the executor.py:219-224 fix in the P9 end-to-end-pipeline
#: plan -- has_numeric_param used to be read from the model's EXTRACTED
#: params dict, which calibration showed the extractor fills with invented
#: values on a meaningful fraction of calls even after fixing the two real
#: bugs behind null_fidelity (see calibrate.py's null_fidelity metric and
#: extract.py's parse_extraction docstring). Every such invention silently
#: flipped a step from UNSPECIFIED to SPECIFIED_UNGRADED, always in the
#: direction of inflating apparent commitment. Reading straight from the
#: step text makes that flip structurally impossible, regardless of
#: extractor behavior on any given run.
#:
#: The \b anchor is load-bearing, not cosmetic: a plain r"\d" (no boundary)
#: was tried first and FALSE-POSITIVED on "CO2" -- a term this exact
#: corpus's fire-suppression tool family (release_co2) uses constantly.
#: "O" and "2" are both \w characters, so there is no boundary between them
#: and \bd correctly refuses to match the digit embedded inside "CO2",
#: while still matching genuine standalone numbers ("2 tugs", "CO2 charge
#: of 500 kg" -- the "500" still matches). Found on a recheck pass by
#: running this regex against the 338-record gold set: 6 real release_co2
#: gold steps mention "CO2" and correctly have NO expected_params, but the
#: unanchored version would have marked all 6 as a false SPECIFIED_UNGRADED.
#:
#: Known remaining limitation (accepted, not fixed): a step that SPELLS OUT
#: a quantity in words ("deploy two harbor tugs") is not detected -- only
#: numerals are. Not attempting a word-number parser is deliberate: the
#: gold-set params audit found 337/338 real steps state no magnitude at
#: all, so this only matters for the rare step that both spells out a
#: number AND has no other numeral anywhere else in the same sentence to
#: fall back on.
_DIGIT_RE = re.compile(r"\b\d")

#: Digits that are NOT an action magnitude, stripped before _DIGIT_RE runs.
#:
#: Added after a census, not a hunch. Every SPECIFIED_UNGRADED step in the
#: corpus that sits on a tool declaring a numeric parameter -- all 17 of them
#: -- was credited on a digit that had nothing to do with the action:
#:
#:   * a vessel-size or draft threshold quoted straight out of the prompt's
#:     assertion block (">50 m", ">10 m draft", "100,000 DWT"). This is by far
#:     the biggest source, and it is an artefact of the experiment itself: the
#:     assertion arms inject these phrases into the plan, so the proxy was
#:     partly measuring the PROMPT rather than the plan;
#:   * a cross-reference to another step ("proceed to step 4");
#:   * a cross-reference to an assertion ("per ON FIRE assertion #5").
#:
#: So the positive direction of the digit rule had precision 0.00 on this
#: corpus: not one step actually stated a magnitude for the action it was
#: describing. That is consistent with the independent gold-set audit finding
#: that 337 of 338 real steps state no magnitude, and it means UNSPECIFIED was
#: systematically UNDER-counted -- the error ran in the direction of
#: flattering the planner, which is the direction that matters.
#:
#: A second census over all three arms found the original seven patterns still
#: miss the same threshold quoted in prose rather than in the prompt's own
#: shorthand -- "if the vessel's draft exceeds 10 meters", "if the vessel is
#: larger than 50 meters", "its length exceeds 50 m". These are the SAME
#: assertion-block numbers, reworded by the planner, and they accounted for 12
#: of the 18 remaining credits (the corrected rate is 6/636, not 18/636). The
#: two alternatives below are anchored on a vessel ATTRIBUTE noun or on an
#: explicitly comparative "larger/longer than", never on an action verb.
#:
#: Deliberately conservative: only these patterns are stripped, all of them
#: cross-references or vessel attributes rather than action quantities. A step
#: that says "pull at 90 t" or "2 tugs of 4000 shp" still reads as specified,
#: because nothing here touches it -- and note the attribute alternative
#: requires the noun, so "pull at more than 90 t" is untouched.
_INCIDENTAL_DIGIT_RE = re.compile(
    r"[<>]\s*\d[\d,.]*\s*(?:m|meters?|metres?)\b"        # ">50 m"
    r"|\bover\s+\d[\d,.]*\s*(?:m|meters?|metres?)\b"    # "over 50 meters"
    r"|\b\d[\d,.]*\s*(?:m|meters?|metres?)\s+draft\b"   # "10 m draft"
    r"|\b\d[\d,.]*\s*DWT\b"                            # "100,000 DWT"
    r"|\bsteps?\s*#?\s*\d+"                              # "step 4"
    r"|\bassertions?\s*#?\s*\d+"                         # "assertion #5"
    r"|#\s*\d+"                                          # "#3"
    # "draft exceeds 10 meters", "length is greater than 50 m"
    r"|\b(?:draft|length|beam|displacement|LOA)\b[^.;]{0,40}?"
    r"(?:exceed(?:s|ing)?|(?:is\s+)?(?:greater|more|larger|longer)\s+than"
    r"|(?:is\s+)?(?:over|above))\s*\d[\d,.]*"
    # "is larger than 50 meters"
    r"|\b(?:is|are|appears?\s+to\s+be|seems?\s+to\s+be)\s+"
    r"(?:larger|longer|bigger|greater)\s+than\s+\d[\d,.]*"
    # "exceeds 50 meters in length" -- same threshold, attribute noun trailing
    r"|(?:exceed(?:s|ing)?|(?:greater|more|larger|longer)\s+than)\s*\d[\d,.]*\s*"
    r"(?:m|meters?|metres?)?\s*in\s+(?:length|draft|beam)\b"
    # "assertion SUNKEN-3" -- the hyphenated assertion IDs, which the "#5"
    # alternative above does not reach
    r"|\b(?:AGROUND|SUNKEN|CAPSIZED|ON[\s_-]?FIRE)[\s_-]*\d+\b"
    # "a vessel under 10 meters" -- the same size threshold, lower comparator.
    # Anchored on the noun so "pull at less than 90 t" is untouched.
    r"|\bvessels?\s+(?:under|below|less\s+than|smaller\s+than)\s*\d[\d,.]*",
    re.IGNORECASE)


def states_magnitude(step_text: str) -> bool:
    """Does this step state a magnitude for its OWN action?

    Incidental digits are stripped first -- see _INCIDENTAL_DIGIT_RE for the
    census that motivated it.
    """
    return bool(_DIGIT_RE.search(_INCIDENTAL_DIGIT_RE.sub(" ", step_text)))

#: The seven-plus verdicts -- see salvage_plan_checker.md section 3 and 3a
#: for the original six/seven-state design; NO_RECOGNISABLE_ROUTE and
#: ROUTE_INADMISSIBLE are plan-level additions from the route-scoped model
#: (design plan section 2) rather than per-step verdicts.
STEP_VERDICTS = (
    "SPECIFIED_UNGRADED",       # phase-2 placeholder -- see module docstring
    "UNSPECIFIED",
    "CONDITIONAL_UNRESOLVED",
    "SEQUENCE_VIOLATION",
    "METHOD_ERROR",
    "NO_MATCH",                 # tool == "no_match": filler/non-actionable step
    "SUPPORT",                  # no_match that is pure scaffolding -- ungraded, see support.py
)
PLAN_VERDICTS = ("NO_RECOGNISABLE_ROUTE", "ROUTE_INADMISSIBLE", "ROUTE_OK")

#: Verdicts that disqualify a step from counting as "done cleanly" --
#: shared by route_completeness and goal_reached below, so the two can't
#: silently drift apart on what counts as a problem.
BAD_VERDICTS = frozenset({"SEQUENCE_VIOLATION", "METHOD_ERROR", "CONDITIONAL_UNRESOLVED", "NO_MATCH"})


@dataclass
class StepResult:
    n: int
    text: str
    tool: str
    params: dict
    conditional: bool
    condition_text: Optional[str]
    verdict: str
    detail: str
    #: All-dimensions scan: {"no_match", "hedged", "method", "sequence",
    #: "unquantified"} -> bool. `verdict` above is the FIRST of these to fire
    #: in precedence order, so it hides every later one; these flags record
    #: what each check actually found on this step, whether or not it was
    #: reached. See execute_plan for why the two cannot disagree.
    #:
    #: Unlike `verdict`, the flags are RAW -- `repaired_steps` exemptions are
    #: applied to the verdict only. A repaired step still reports what the
    #: check found; it just isn't scored on it.
    flags: dict = field(default_factory=dict)


@dataclass
class PlanResult:
    image: str
    casualty: str
    steps: list                 # list[StepResult]
    route_name: Optional[str]
    route_score: float
    #: "yes" | "no" | "unknown" | "ambiguous" | "n/a" (no route recognised).
    #: Only "no" is penalised. "unknown" is pending physics.py; "ambiguous"
    #: means two tied routes disagreed and the call set cannot tell them
    #: apart -- see methods.admissible_over_ties.
    route_admissible: str
    route_coherence: Optional[float]
    route_completeness: Optional[float]
    not_attempted: list         # missing core_tools + missing universal obligations
    sequence_violations: list   # [(step_n, missing_fact), ...]
    unused_assessments: list    # [(step_n, tool, fact), ...] -- see execute_plan docstring
    gate_rate: int
    unresolved_gate_count: int
    self_contradictory_on_size: bool
    goal_reached: bool          # see execute_plan's goal_reached computation, below
    #: Diagnostic only -- NEVER read by any grading path. When the plan's
    #: tools fit some other casualty's routes better than its own, this is
    #: that casualty's name; else None. Feeds the STRATEGY-PERCEPTION class
    #: in classify.py, which otherwise cannot be told apart from
    #: NO-PROCEDURE. Defaulted so every existing PlanResult construction
    #: (13 test modules) keeps working unchanged.
    foreign_casualty: Optional[str] = None

    def summary(self) -> dict:
        counts = {}
        for s in self.steps:
            counts[s.verdict] = counts.get(s.verdict, 0) + 1
        return {
            "image": self.image,
            "casualty": self.casualty,
            "route_name": self.route_name,
            "route_score": round(self.route_score, 3),
            "route_admissible": self.route_admissible,
            "route_coherence": self.route_coherence,
            "route_completeness": self.route_completeness,
            "counts": counts,
            "not_attempted": list(self.not_attempted),
            "sequence_violations": [f"{n}: missing {fact}" for n, fact in self.sequence_violations],
            "unused_assessments": [f"{n} ({tool}): established '{fact}', never consumed"
                                    for n, tool, fact in self.unused_assessments],
            "gate_rate": self.gate_rate,
            "unresolved_gate_count": self.unresolved_gate_count,
            "self_contradictory_on_size": self.self_contradictory_on_size,
            "goal_reached": self.goal_reached,
            "foreign_casualty": self.foreign_casualty,
        }


def execute_plan(calls: list, casualty: str, scenario, tool_registry: ToolRegistry,
                  route_registry: RouteRegistry, plan_text: str = "",
                  repaired_steps: frozenset = frozenset()) -> PlanResult:
    """Run the seven passes over one plan's extracted (or gold) ToolCalls.

    `plan_text` is optional raw text used only for the pure-regex metrics
    (gate_rate, self_contradictory_on_size via gates.py part 1) -- pass ""
    when only hand-written ToolCalls exist and no source text is available
    (e.g. some calibration probes), those two fields will just read 0/False.

    `repaired_steps` is the counterfactual-repair hook (repair.py): step
    numbers whose failure is to be NEUTRALISED rather than graded -- the gate
    is treated as resolved, the method as fitting, the missing preconditions
    as granted, the magnitude as stated. The step then executes and the walk
    continues, so the next failure downstream becomes observable. This is
    strictly a measurement path: it answers "if this one failure vanished,
    how much further would the plan get", which is the only way to rank
    remedies. It must never be set on a scoring run, and defaults to empty so
    it cannot be reached by accident.
    """
    action_calls = [c for c in calls if c.tool != "no_match"]
    called_tool_names = {c.tool for c in action_calls if tool_registry.has(c.tool)}

    # ── Pass 1-2: route recognition + admissibility ─────────────────────────
    match = recognise_route(called_tool_names, casualty, route_registry)
    if match.route is None:
        route_name, route_score, route_adm = None, match.score, "n/a"
        route_coherence = None
        route_completeness = None
        not_attempted = ["NO_RECOGNISABLE_ROUTE"]
    else:
        route_name, route_score = match.route.name, match.score
        # Not admissible(match.route, ...): when two routes tie on recognition
        # and disagree on admissibility, match.route is whichever was declared
        # first, and scoring on it would turn JSON key order into a verdict.
        # See methods.admissible_over_ties.
        route_adm = admissible_over_ties(match, scenario)
        total_action = len(called_tool_names & (match.matched_tools | match.unmatched_tools)) or len(called_tool_names)
        route_coherence = (len(match.matched_tools) / total_action) if total_action else None
        missing_core = match.route.core_tools - called_tool_names
        not_attempted = [f"route:{t}" for t in sorted(missing_core)]
        # route_completeness is computed AFTER the per-step loop below, from
        # verdicts rather than mere presence -- see the comment there. Left
        # as None here as a placeholder to keep this branch's shape uniform.
        route_completeness = None

    # ── Pass 5: universal obligations (route-independent, but NOT
    # exemption-independent -- an obligation can be waived for a specific
    # route (C1 for tide_refloat) or gated on a scenario attribute (C6 only
    # when habitat_sensitive), per registry/goals.json. Both exemption kinds
    # were previously read from the registry but never applied here, which
    # over-penalised e.g. every tide_refloat plan for "missing" a fuel
    # offload it was never supposed to need -- see the code-review finding
    # this fixes.
    ws_probe = WorldState()
    for c in action_calls:
        ws_probe.apply(c, tool_registry)
    for ob in route_registry.obligations_for(casualty):
        if route_name is not None and route_name in ob.except_routes:
            continue
        if ob.requires_scenario_field is not None and not getattr(scenario, ob.requires_scenario_field, False):
            continue
        if not (ws_probe.knows(ob.requires_fact) or ws_probe.is_true(ob.requires_fact)):
            not_attempted.append(f"{ob.id}:{ob.requires_fact}")

    # ── Pass 3, 6, 7: walk calls in order ────────────────────────────────────
    ws = WorldState()
    step_results = []
    sequence_violations = []
    unresolved_gate_count = 0
    established_at = []   # [(step_n, tool, newly_known_fact), ...] -- see unused-assessment check below

    def _apply_and_track(call):
        before = ws.known_facts()
        ws.apply(call, tool_registry)
        for fact in ws.known_facts() - before:
            established_at.append((call.step_num, call.tool, fact))

    for c in calls:
        #: Repair exempts a step from being SCORED, not from being measured --
        #: see StepResult.flags.
        exempt = c.step_num in repaired_steps

        # SUPPORT: a step the extractor mapped to no tool that is pure
        # operational scaffolding (a safety perimeter, liaison, logistics).
        # It is not graded -- it cannot stop the plan and does not count as an
        # executed step -- and its no_match flag is False, so it never shows
        # up among a plan's errors. An unknown tool NAME is an extraction
        # failure and never qualifies. A repaired no_match lands here too:
        # the repair operator's contract for NO_MATCH is to SKIP the step,
        # and before this branch existed the executor ignored the exemption,
        # so repair re-picked the same step until it hit the iteration cap
        # and every NO_MATCH delta_epl read 0.
        if c.tool == "no_match" and (exempt or is_support(c.step_text, c.secondary_tools)):
            step_results.append(StepResult(
                n=c.step_num, text=c.step_text, tool=c.tool, params=c.params,
                conditional=c.conditional, condition_text=c.condition_text,
                verdict="SUPPORT",
                detail=("Skipped by counterfactual repair." if exempt else
                        "Operational scaffolding the registry does not model; not graded."),
                flags={"no_match": False, "hedged": False, "method": False,
                       "sequence": False, "unquantified": False},
            ))
            continue

        if c.tool == "no_match" or not tool_registry.has(c.tool):
            step_results.append(StepResult(
                n=c.step_num, text=c.step_text, tool=c.tool, params=c.params,
                conditional=c.conditional, condition_text=c.condition_text,
                verdict="NO_MATCH",
                detail="No recognised tool for this step (filler, or extraction failure).",
                flags={"no_match": True, "hedged": False, "method": False,
                       "sequence": False, "unquantified": False},
            ))
            continue

        spec = tool_registry.spec(c.tool)

        # --- all-dimensions scan ------------------------------------------
        # Every check runs on every step, here, BEFORE the precedence cascade
        # below picks exactly one of them to become the verdict.
        #
        # This is not a second implementation of the cascade and cannot drift
        # from it: the cascade reads these same flags rather than recomputing.
        # The reason it is safe to hoist them is that the cascade's `continue`s
        # never skipped _apply_and_track -- effects are applied on every path
        # except NO_MATCH, which has no tool and so no effects. The WorldState
        # trajectory therefore does not depend on which check fired first, so
        # a check evaluated up front sees exactly the world it would have seen
        # had the cascade reached it. Taking these flags in precedence order
        # reproduces `verdict` on all 1,980 steps of the CASTOR corpus, 0
        # disagreements; test_scan_flags_reproduce_the_verdict is the guard.
        #
        # Why this exists: `verdict` answers "what is the first thing wrong
        # with this step", which systematically under-counts everything low in
        # the order. Plan-level, sequencing measured 58% of plans through the
        # verdict and 88% through these flags -- 86 plans were scored clean on
        # ordering only because another check fired first.
        flags = {
            "no_match": False,
            "hedged": bool(c.conditional
                           and gates.resolve_conditional(c.condition_var, ws) == "unresolved"),
            "method": bool(not spec.is_assessment
                           and spec.family not in ("assessment", "terminal", "universal")
                           and spec.family != casualty),
            "sequence": bool(ws.missing_requires(c, tool_registry)),
            "unquantified": bool(any(t.startswith(("int", "float"))
                                     for t in spec.params.values())
                                 and not states_magnitude(c.step_text)),
        }
        # Pass 7: conditional resolution, checked before applying this call's
        # own effects (a gate can only be resolved by a PRIOR step).
        if flags["hedged"] and not exempt:
            unresolved_gate_count += 1
            step_results.append(StepResult(
                n=c.step_num, text=c.step_text, tool=c.tool, params=c.params,
                conditional=True, condition_text=c.condition_text,
                verdict="CONDITIONAL_UNRESOLVED",
                detail=(f"Condition '{c.condition_text}' (var={c.condition_var}) "
                        f"is never established by a prior step."),
                flags=flags,
            ))
            _apply_and_track(c)
            continue

        # Pass 6: method fit -- action tool called but wrong family for this
        # casualty entirely. Checked BEFORE sequencing: a tool that doesn't
        # belong here at all shouldn't be diagnosed by its (irrelevant)
        # preconditions -- e.g. rig_parbuckling in an aground plan should
        # read as METHOD_ERROR, not SEQUENCE_VIOLATION for never having
        # rescued a crew that was never capsized in the first place.
        #
        # "universal" joins "assessment"/"terminal" as a family this check
        # does not fire on. It exists because several tools were filed under
        # the casualty whose ROUTE first cited them rather than the casualty
        # they apply to, and the corpus made the consequence obvious: 24
        # steps were flagged METHOD_ERROR for rescuing the crew off a vessel
        # that was aground rather than capsized, because rescue_crew lived in
        # the capsized family. Crew rescue, pollution containment, dewatering,
        # tug assistance and sonar search are not casualty-specific methods.
        # See registry/tools.json for the five reclassified entries and, just
        # as importantly, for the ones deliberately left alone: right_vessel
        # stays capsized, because sunken plans attempting to right a wreck are
        # the STRATEGY_PERCEPTION signal and universalising it would erase the
        # finding.
        if flags["method"] and not exempt:
            step_results.append(StepResult(
                n=c.step_num, text=c.step_text, tool=c.tool, params=c.params,
                conditional=c.conditional, condition_text=c.condition_text,
                verdict="METHOD_ERROR",
                detail=f"'{c.tool}' belongs to family '{spec.family}', not '{casualty}'.",
                flags=flags,
            ))
            _apply_and_track(c)
            continue

        # Pass 3: sequencing.
        missing = ws.missing_requires(c, tool_registry)
        if missing and exempt:
            # Grant what the plan failed to establish, so downstream steps
            # that legitimately depend on it are no longer scored against a
            # gap this repair is standing in for.
            ws.grant(missing)
            missing = set()
        if missing:
            sequence_violations.append((c.step_num, ", ".join(sorted(missing))))
            step_results.append(StepResult(
                n=c.step_num, text=c.step_text, tool=c.tool, params=c.params,
                conditional=c.conditional, condition_text=c.condition_text,
                verdict="SEQUENCE_VIOLATION",
                detail=f"Requires {sorted(missing)} not yet established.",
                flags=flags,
            ))
            _apply_and_track(c)
            continue

        # Default: quantified-vs-not. Numeric adequacy grading is phase 2
        # (physics.py) -- see module docstring.
        #
        # has_numeric_param is read from the STEP TEXT, not from c.params --
        # see the _DIGIT_RE module comment. A param the model extracted is
        # never trusted on its own to prove a magnitude was actually stated;
        # this makes UNSPECIFIED immune to the extractor inventing a value.
        if flags["unquantified"] and not exempt:
            verdict, detail = "UNSPECIFIED", "No magnitude given; adequacy unverifiable."
        else:
            verdict, detail = "SPECIFIED_UNGRADED", "Magnitude present (or none required); numeric grading is phase 2."

        step_results.append(StepResult(
            n=c.step_num, text=c.step_text, tool=c.tool, params=c.params,
            conditional=c.conditional, condition_text=c.condition_text,
            verdict=verdict, detail=detail, flags=flags,
        ))
        _apply_and_track(c)

    # route_completeness: fraction of the matched route's core_tools that
    # were not just CALLED but called to a verdict that isn't itself a
    # problem (SEQUENCE_VIOLATION / METHOD_ERROR / CONDITIONAL_UNRESOLVED /
    # NO_MATCH). Deliberately NOT the same formula as route_score (core_tool
    # presence, used only for route RECOGNITION) -- a call that's present
    # but sequence-violated shouldn't count as the route step being
    # completed. Previously this field duplicated route_score exactly,
    # which is misleading in per_image.csv/summary.csv; see the code-review
    # finding this fixes.
    if match.route is not None and match.route.core_tools:
        ok_tools = {
            s.tool for s in step_results
            if s.tool in match.route.core_tools and s.verdict not in BAD_VERDICTS
        }
        route_completeness = len(ok_tools) / len(match.route.core_tools)

    # Unused-assessment check: a fact an assessment step established but
    # that nothing in the plan ever actually depended on. Answers a
    # different question than SEQUENCE_VIOLATION (which catches a fact
    # USED before it existed) -- this catches a fact CREATED and then
    # ignored: the plan performed the assessment but never let it change
    # anything downstream, e.g. "determine if it can be refloated" followed
    # by a pull step that isn't actually gated on that determination.
    # "Used" means referenced by ANY call's `requires` in the plan (any
    # position, not just later -- if a fact is required anywhere the
    # assessment had a purpose), by a resolved conditional's underlying
    # fact, or by an applicable universal obligation.
    used_facts = set()
    for c in action_calls:
        if tool_registry.has(c.tool):
            used_facts |= set(tool_registry.spec(c.tool).requires)
        if c.conditional and c.condition_var != "none":
            used_facts.add(gates.condition_var_to_fact(c.condition_var))
    for ob in route_registry.obligations_for(casualty):
        if route_name is not None and route_name in ob.except_routes:
            continue
        if ob.requires_scenario_field is not None and not getattr(scenario, ob.requires_scenario_field, False):
            continue
        used_facts.add(ob.requires_fact)

    unused_assessments = [
        (n, tool, fact) for n, tool, fact in established_at if fact not in used_facts
    ]

    gate_count = gates.gate_rate(plan_text) if plan_text else 0
    self_contra = gates.is_self_contradictory_on_size(plan_text) if plan_text else False

    # goal_reached: true iff (a) NO step in the whole plan has a BAD_VERDICT
    # -- not just the step that happens to establish the terminal fact, the
    # entire plan -- (b) the chosen route is not scenario-inadmissible (a
    # technique that reaches the terminal fact via an approach that doesn't
    # actually make sense for THIS vessel is exactly the kind of "logically
    # doesn't hold together" case that disqualifies, even though no single
    # step verdict catches it -- e.g. manual_righting reaching
    # vessel_righted cleanly, on a vessel too large for manual righting to
    # ever actually work), AND (c) the casualty's terminal_facts (goals.json)
    # were actually established by the time every step has been applied.
    # All three are required. SUPPORT steps are not bad verdicts.
    #
    # There used to be a fourth conjunct: no step UNSPECIFIED (no magnitude
    # stated). Quantities were dropped from P9's question -- see
    # classify.NON_EXECUTABLE -- so a plan is no longer held short of its
    # goal for them either; the two had to move together or a clean plan
    # would read INCOMPLETE purely for being unquantified.
    goal = route_registry.goal_for(casualty)
    no_bad_verdicts = not any(s.verdict in BAD_VERDICTS for s in step_results)
    route_ok = route_adm != "no"
    terminal_facts_met = bool(goal) and goal.terminal_facts <= (ws.known_facts() | ws.true_facts())
    goal_reached = no_bad_verdicts and route_ok and terminal_facts_met

    return PlanResult(
        image=getattr(scenario, "image", ""), casualty=casualty, steps=step_results,
        route_name=route_name, route_score=route_score, route_admissible=route_adm,
        route_coherence=route_coherence, route_completeness=route_completeness,
        not_attempted=not_attempted, sequence_violations=sequence_violations,
        unused_assessments=unused_assessments,
        gate_rate=gate_count, unresolved_gate_count=unresolved_gate_count,
        self_contradictory_on_size=self_contra,
        goal_reached=goal_reached,
        foreign_casualty=detect_perception_mismatch(called_tool_names, casualty, route_registry),
    )
