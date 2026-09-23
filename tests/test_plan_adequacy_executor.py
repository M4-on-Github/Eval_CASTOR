"""
Tests for pipelines/plan_adequacy/executor.py

Drives the seven-pass executor with hand-written ToolCall lists --
"executor@oracle" per the design plan section 4e: no model in the loop, so
these tests isolate logic bugs in methods.py / worldstate.py / executor.py
from extraction accuracy. Full annotation of the 33-plan
synthetic_calibration.jsonl into a gold ToolCall set (build order section 6
step 3, "gold set layers A-C") is separate follow-on work -- these are
smaller hand-built probes covering each pass individually.
Run: python -m pytest tests/test_plan_adequacy_executor.py -v
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.executor import BAD_VERDICTS, execute_plan
from pipelines.plan_adequacy.methods import RouteRegistry
from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry
from pipelines.plan_adequacy.worldstate import WorldState


def _reg():
    return ToolRegistry.load(), RouteRegistry.load()


def _scenario(**kw):
    kw.setdefault("image", "aground/test.jpg")
    return SimpleNamespace(**kw)


def _call(n, tool, text="", params=None, conditional=False,
          condition_text=None, condition_var="none"):
    return ToolCall(
        step_num=n, step_text=text or tool, tool=tool,
        params=params or {}, conditional=conditional,
        condition_text=condition_text, condition_var=condition_var,
    )


# ── pass 3: sequencing ────────────────────────────────────────────────────────

def test_pull_before_freeing_force_is_sequence_violation():
    tool_reg, route_reg = _reg()
    calls = [_call(1, "pull", params={"force_t": 90})]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "SEQUENCE_VIOLATION"
    assert len(result.sequence_violations) == 1


def test_pull_after_assessment_chain_is_not_a_sequence_violation():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", params={"count": 2, "shp": 4000.0}),
        _call(6, "pull", params={"force_t": 90}),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[-1].verdict != "SEQUENCE_VIOLATION"
    assert result.sequence_violations == []


# ── pass 7: conditional resolution ───────────────────────────────────────────

def test_unresolved_gate_flagged_conditional_unresolved():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "attach_tug", params={"count": 2},
              conditional=True, condition_text="if the vessel is large",
              condition_var="vessel_size"),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "CONDITIONAL_UNRESOLVED"
    assert result.unresolved_gate_count == 1


def test_gate_resolved_after_survey_hull_establishes_vessel_size():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "survey_hull"),
        _call(2, "attach_tug", params={"count": 2},
              conditional=True, condition_text="if the vessel is large",
              condition_var="vessel_size"),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[1].verdict != "CONDITIONAL_UNRESOLVED"
    assert result.unresolved_gate_count == 0


# ── pass 6: method error ─────────────────────────────────────────────────────

def test_capsized_only_tool_in_aground_plan_is_method_error():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "attach_tug", params={"count": 2}),
        _call(2, "pull", params={"force_t": 90}),
        _call(3, "rig_parbuckling", params={"n_points": 2}),  # capsized-family tool
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    verdicts = {s.tool: s.verdict for s in result.steps}
    assert verdicts["rig_parbuckling"] == "METHOD_ERROR"


# ── no_match handling ─────────────────────────────────────────────────────────

def test_no_match_tool_produces_no_match_verdict_not_a_crash():
    tool_reg, route_reg = _reg()
    calls = [_call(1, "no_match", text="Deploy divers to cut the anchor chain.")]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "NO_MATCH"
    assert result.steps[0].flags["no_match"] is True


def test_pure_scaffolding_no_match_is_support_and_not_bad():
    tool_reg, route_reg = _reg()
    calls = [_call(1, "no_match", text="Coordinate with relevant stakeholders."),
             _call(2, "no_match", text="Establish a safety perimeter using buoys.")]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert [s.verdict for s in result.steps] == ["SUPPORT", "SUPPORT"]
    assert not any(s.flags["no_match"] for s in result.steps)
    assert "SUPPORT" not in BAD_VERDICTS


def test_a_support_category_step_naming_a_real_action_stays_no_match():
    """The audit's finding: category patterns match anywhere in a step, and
    v2 steps pack several actions together. A tow or a diver inspection
    inside a 'coordinate with' step is not scaffolding."""
    tool_reg, route_reg = _reg()
    for text in ("Coordinate with the tug master to tow the vessel to port.",
                 "Establish a perimeter and deploy divers to inspect the hull.",
                 "Coordinate with engineers to determine whether to use tugs or cranes."):
        result = execute_plan([_call(1, "no_match", text=text)], "aground",
                              _scenario(), tool_reg, route_reg)
        assert result.steps[0].verdict == "NO_MATCH", text


def test_extractor_secondary_tools_veto_support():
    tool_reg, route_reg = _reg()
    call = ToolCall(step_num=1, step_text="Establish a perimeter around the vessel.",
                    tool="no_match", params={}, conditional=False, condition_text=None,
                    condition_var="none", secondary_tools=("muster_personnel",))
    result = execute_plan([call], "on_fire", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "NO_MATCH"


def test_an_unknown_tool_name_is_never_support():
    tool_reg, route_reg = _reg()
    result = execute_plan([_call(1, "summon_kraken", text="Establish a perimeter.")],
                          "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "NO_MATCH"


# ── route recognition wired end to end ───────────────────────────────────────

def test_full_tug_pull_plan_recognises_route_and_sequences_cleanly():
    # Route recognition + absence of sequencing errors only -- see
    # test_goal_reached_* below for the actual goal-achievement checks
    # (goal.terminal_facts IS checked now, via execute_plan's goal_reached).
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", params={"count": 2, "shp": 4000.0}),
        _call(6, "pull", params={"force_t": 90}),
        _call(7, "post_operation_assessment"),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.route_name == "tug_pull"
    assert result.sequence_violations == []
    assert "SEQUENCE_VIOLATION" not in {s.verdict for s in result.steps}


# ── goal_reached: no violations anywhere AND terminal_facts satisfied ──────
#
# Spec (explicit, from the user): goal_reached is true if and only if the
# plan has zero violations/errors ANYWHERE (not just on the step that
# happens to reach the terminal fact) AND the vessel was actually,
# logically salvaged (goals.json's terminal_facts established by the end).
# Both halves are independently testable and both must hold.

def test_goal_reached_true_for_a_fully_clean_plan_that_reaches_the_goal():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        # Magnitudes must appear in the STEP TEXT, not merely in the params
        # dict -- specificity is derived from what the plan actually said, so
        # that goal_reached cannot be satisfied by extractor-fabricated params.
        _call(5, "attach_tug", text="Attach 2 tugs of 4000 shp each",
              params={"count": 2, "shp": 4000.0}),
        _call(6, "pull", text="Pull at 90 t bollard pull",
              params={"force_t": 90}),                     # pull -> effects: vessel_refloated
        _call(7, "post_operation_assessment"),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.goal_reached is True


def test_goal_reached_is_not_withheld_for_an_unquantified_step():
    """Quantities were dropped from P9's question. This plan has zero
    violations, an admissible route, and establishes vessel_refloated; its
    two physical steps state no magnitude. They are still labelled
    UNSPECIFIED, but that no longer holds the plan short of its goal --
    otherwise a clean plan would read INCOMPLETE purely for being
    unquantified. (capsized/00195.jpg was the real case that first put the
    magnitude conjunct in.)"""
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", text="Attach ocean-going salvage tugs"),
        _call(6, "pull", text="Pull the vessel free with controlled force"),
        _call(7, "post_operation_assessment"),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert any(s.verdict == "UNSPECIFIED" for s in result.steps)
    assert not any(s.verdict in BAD_VERDICTS for s in result.steps)
    assert result.goal_reached is True


def test_goal_reached_false_when_terminal_fact_met_but_plan_has_a_violation_elsewhere():
    """The core of the spec: reaching the terminal fact is NOT enough on its
    own if anything else in the plan is broken -- even a violation on a
    totally unrelated step must disqualify goal_reached, not just a
    violation on the step that reaches the goal."""
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", params={"count": 2, "shp": 4000.0}),
        _call(6, "pull", params={"force_t": 90}),           # reaches vessel_refloated cleanly
        _call(7, "rig_parbuckling", params={"n_points": 2}),  # capsized-family -> METHOD_ERROR, unrelated to the goal
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert "METHOD_ERROR" in {s.verdict for s in result.steps}
    assert result.goal_reached is False


def test_goal_reached_false_when_plan_is_clean_but_never_reaches_the_goal():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", params={"count": 2, "shp": 4000.0}),
        # no pull / no monitor_tide -- nothing ever establishes vessel_refloated
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert "SEQUENCE_VIOLATION" not in {s.verdict for s in result.steps}
    assert "METHOD_ERROR" not in {s.verdict for s in result.steps}
    assert result.goal_reached is False


def test_goal_reached_false_when_route_is_scenario_inadmissible():
    """A plan can be step-by-step spotless -- zero violations, terminal
    fact genuinely reached -- and still not count if the TECHNIQUE itself
    doesn't make sense for this vessel. manual_righting is only admissible
    for size_category='small'; on a large vessel it's cleanly executed but
    logically doesn't hold together, and must disqualify goal_reached even
    though no single step verdict catches it."""
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "rescue_crew"),
        _call(2, "calculate_stability"),
        _call(3, "right_vessel"),
    ]
    result = execute_plan(calls, "capsized", _scenario(size_category="large"), tool_reg, route_reg)
    assert result.route_name == "manual_righting"
    assert result.route_admissible == "no"
    assert "SEQUENCE_VIOLATION" not in {s.verdict for s in result.steps}
    assert "METHOD_ERROR" not in {s.verdict for s in result.steps}
    assert result.goal_reached is False


def test_goal_reached_false_when_the_reaching_step_itself_is_sequence_violated():
    """pull is the ONLY tool called and it's the one that would establish
    vessel_refloated -- but it's called before calculate_freeing_force, so
    it reads SEQUENCE_VIOLATION. WorldState still applies its effects (see
    _apply_and_track's existing behavior), so terminal_facts_met alone would
    wrongly read True here -- goal_reached must catch this via the
    no-violations-anywhere half of the spec, not just terminal fact presence."""
    tool_reg, route_reg = _reg()
    calls = [_call(1, "pull", params={"force_t": 90})]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "SEQUENCE_VIOLATION"
    assert result.goal_reached is False


def test_unspecified_verdict_when_no_magnitude_given():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", params={"count": None, "shp": None}),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[-1].verdict == "UNSPECIFIED"


def test_unspecified_verdict_is_immune_to_extractor_hallucinated_params():
    """Regression anchor for the executor.py:219-224 specificity fix (P9
    end-to-end-pipeline plan, Part 1b). Calibration showed the extractor
    fabricates parameter values not stated in the step text on a meaningful
    fraction of calls, even after the two real bugs behind null_fidelity=0.0
    were fixed. Every such fabrication used to silently flip a step from
    UNSPECIFIED to SPECIFIED_UNGRADED -- always in the direction of
    inflating apparent commitment, since specificity was read from the
    extracted params dict rather than the step text. This asserts a step
    with no digit anywhere in its text stays UNSPECIFIED even when handed a
    ToolCall carrying a fabricated numeric param."""
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        # No digit anywhere in the step text -- default text is just the
        # tool name "attach_tug" -- but params claims a real magnitude, the
        # exact shape of an extractor hallucination.
        _call(5, "attach_tug", params={"count": 4, "shp": 6000.0}),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[-1].verdict == "UNSPECIFIED"


def test_specified_ungraded_verdict_when_step_text_states_a_magnitude():
    """Complement of the hallucination-immunity test above: a step whose
    TEXT does state a digit is SPECIFIED_UNGRADED, confirming the fix reads
    from step_text and isn't just permanently returning UNSPECIFIED."""
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", text="Deploy 2 harbor tugs at 4000 shp.",
              params={"count": 2, "shp": 4000.0}),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[-1].verdict == "SPECIFIED_UNGRADED"


def test_unspecified_verdict_survives_a_co2_mention_with_no_stated_mass():
    """Regression anchor for a false positive found on a recheck pass: an
    unanchored r"\\d" (an earlier version of _DIGIT_RE) matched the digit
    embedded inside "CO2" itself, wrongly marking every release_co2 step
    that mentions CO2 -- which is nearly all of them, since it's the only
    tool name for this action -- as SPECIFIED_UNGRADED even when no mass
    was actually stated. 6 real gold-set records have exactly this shape.
    _DIGIT_RE is now \\b-anchored, which requires a non-word character
    immediately before the digit -- "O" and "2" in "CO2" are both \\w
    characters so there is no boundary between them, and the anchored
    version correctly does not match."""
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "muster_personnel", text="Confirm the engine room is clear of all personnel.",
              params={"space": "engine room"}),
        _call(2, "seal_boundaries", text="Seal hatches and vents around the fire space."),
        _call(3, "activate_predischarge_alarm",
              text="Sound the pre-discharge alarm before releasing CO2 into the space."),
        _call(4, "release_co2",
              text="Activate the fixed CO2 system to discharge gas into the machinery space.",
              params={"space": "engine room", "mass_kg": None}),
    ]
    result = execute_plan(calls, "on_fire", _scenario(), tool_reg, route_reg)
    assert result.steps[-1].tool == "release_co2"
    assert result.steps[-1].verdict == "UNSPECIFIED"


# ── pass 5: universal obligation exemptions (except_routes / scenario gate) ──

def test_tide_refloat_is_exempt_from_c1_fuel_offload():
    tool_reg, route_reg = _reg()
    calls = [_call(1, "monitor_tide")]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert not any(na.startswith("C1:") for na in result.not_attempted)


def test_tug_pull_is_not_exempt_from_c1_fuel_offload():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        _call(5, "attach_tug", params={"count": 2}),
        _call(6, "pull", params={"force_t": 90}),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert any(na.startswith("C1:") for na in result.not_attempted)


def test_c6_neba_skipped_when_not_habitat_sensitive():
    tool_reg, route_reg = _reg()
    calls = [_call(1, "lift", params={"force_t": 50})]
    result = execute_plan(calls, "sunken", _scenario(habitat_sensitive=False), tool_reg, route_reg)
    assert not any(na.startswith("C6:") for na in result.not_attempted)


def test_c6_neba_required_when_habitat_sensitive():
    tool_reg, route_reg = _reg()
    calls = [_call(1, "lift", params={"force_t": 50})]
    result = execute_plan(calls, "sunken", _scenario(habitat_sensitive=True), tool_reg, route_reg)
    assert any(na.startswith("C6:") for na in result.not_attempted)


# ── test_atmosphere sub-order enforcement ────────────────────────────────────

def test_atmosphere_toxic_test_before_explosive_is_sequence_violation():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "equalize_pressure", params={"space": "hold1"}),
        _call(2, "test_atmosphere", params={"space": "hold1", "test_type": "toxic"}),
    ]
    result = execute_plan(calls, "sunken", _scenario(), tool_reg, route_reg)
    assert result.steps[1].verdict == "SEQUENCE_VIOLATION"


def test_atmosphere_correct_order_is_not_a_sequence_violation():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "equalize_pressure", params={"space": "hold1"}),
        _call(2, "test_atmosphere", params={"space": "hold1", "test_type": "explosive"}),
        _call(3, "test_atmosphere", params={"space": "hold1", "test_type": "oxygen"}),
        _call(4, "test_atmosphere", params={"space": "hold1", "test_type": "toxic"}),
        _call(5, "cut_section", params={"location": "hold1"}),
    ]
    result = execute_plan(calls, "sunken", _scenario(), tool_reg, route_reg)
    assert result.sequence_violations == []
    assert result.steps[-1].verdict != "SEQUENCE_VIOLATION"


def test_atmosphere_safe_not_known_until_all_three_tests_done():
    reg = ToolRegistry.load()
    ws = WorldState()
    ws.apply(_call(1, "test_atmosphere", params={"test_type": "explosive"}), reg)
    assert not ws.knows("atmosphere_safe")
    ws.apply(_call(2, "test_atmosphere", params={"test_type": "oxygen"}), reg)
    assert not ws.knows("atmosphere_safe")
    ws.apply(_call(3, "test_atmosphere", params={"test_type": "toxic"}), reg)
    assert ws.knows("atmosphere_safe")


# ── route_completeness is distinct from route_score ──────────────────────────

def test_route_completeness_differs_from_route_score_when_a_core_tool_call_fails():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed"),
        _call(3, "calculate_ground_reaction"),
        _call(4, "calculate_freeing_force"),
        # attach_tug and pull are both tug_pull's core_tools and both
        # present (route_score 1.0), but attach_tug is unresolved-gated so
        # its verdict is a "problem" verdict -- completeness should drop
        # below score.
        _call(5, "attach_tug", params={"count": 2}, conditional=True,
              condition_text="if large", condition_var="vessel_size"),
        _call(6, "pull", params={"force_t": 90}),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.route_name == "tug_pull"
    assert result.route_score == 1.0
    assert result.route_completeness < result.route_score


# ── unused-assessment check: fact established but never depended on ─────────

def test_survey_result_never_used_is_flagged_unused():
    """The example discussed in review: 'assess the hull and determine if
    it can be refloated' followed by a pull that doesn't actually depend on
    what was found -- the assessment was theater, not input to the plan."""
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "survey_hull"),
        # pull's real requires are freeing_force/tank_contents, not
        # hull_condition/vessel_size -- so survey_hull's findings are never
        # actually consumed by anything downstream in this plan.
        _call(2, "sound_tanks", params={"tank_ids": ["1"]}),
        _call(3, "survey_seabed"),
        _call(4, "calculate_ground_reaction"),
        _call(5, "calculate_freeing_force"),
        _call(6, "pull", params={"force_t": 90}),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    unused_facts = {fact for _, _, fact in result.unused_assessments}
    assert "hull_condition" in unused_facts
    assert "vessel_size" in unused_facts
    # substrate/ground_reaction WERE used (freeing_force requires them) --
    # must not be flagged.
    assert "substrate" not in unused_facts


def test_survey_result_used_by_a_resolved_conditional_is_not_flagged():
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "survey_hull"),
        _call(2, "attach_tug", params={"count": 2}, conditional=True,
              condition_text="if the vessel is large", condition_var="vessel_size"),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    unused_facts = {fact for _, _, fact in result.unused_assessments}
    assert "vessel_size" not in unused_facts


# ── plan-level summary shape ──────────────────────────────────────────────────

def test_summary_returns_expected_keys():
    tool_reg, route_reg = _reg()
    calls = [_call(1, "survey_hull")]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    summary = result.summary()
    for key in ("image", "casualty", "route_name", "counts", "gate_rate",
                "unresolved_gate_count", "self_contradictory_on_size"):
        assert key in summary


# ── specificity: incidental digits ───────────────────────────────────────────
# Added after a census over the corpus found that ALL 17 SPECIFIED_UNGRADED
# steps on numeric-param tools were credited on a digit belonging to something
# other than the action -- mostly a vessel-size threshold quoted out of the
# prompt's own assertion block, which meant the proxy was partly measuring the
# PROMPT rather than the plan.

def test_real_action_magnitudes_still_count_as_specified():
    from pipelines.plan_adequacy.executor import states_magnitude
    for text in ["Pull at 90 t bollard pull",
                 "Attach 2 tugs of 4000 shp each",
                 "Release the 1200 kg CO2 bank into the space",
                 "Dewater the engine room at 400 m3/h",
                 "Rig 6 parbuckling points rated 900 t each"]:
        assert states_magnitude(text) is True, text


def test_incidental_digits_do_not_count_as_a_stated_magnitude():
    from pipelines.plan_adequacy.executor import states_magnitude
    for text in ["Given its apparent size (>50 m), remove weight from the aft",
                 "If the vessel is a deep-draft vessel (>10 m draft), dredge",
                 "If the substrate is soft, proceed to step 4",
                 "Confirm pump capacity, as per ON FIRE assertion #5",
                 "Ensure tugs are rated for the vessel (100,000 DWT)",
                 "Given its likely length over 50 meters, lighter the vessel"]:
        assert states_magnitude(text) is False, text


# A second census over all three arms found the seven original patterns still
# missed the SAME assertion-block thresholds when the planner reworded them
# into prose. These twelve steps were the bulk of the remaining false credits;
# stripping them took the corpus rate from 18/636 to 2/636, of which exactly
# one is a genuine action magnitude on hand audit.

def test_reworded_vessel_size_thresholds_are_still_incidental():
    from pipelines.plan_adequacy.executor import states_magnitude
    for text in ["If the vessel's draft exceeds 10 meters, deploy tugs",
                 "If the vessel's draft is greater than 10 m, deploy tugs",
                 "If the vessel is larger than 50 meters, lighter the cargo",
                 "Lighter the vessel if it exceeds 50 meters in length",
                 "Its length exceeds 50 m, so remove weight forward",
                 "For a vessel under 10 meters, right it manually",
                 "Offload fuel, as mandated by assertion SUNKEN-3",
                 "Lighter the vessel, as required by assertion AGROUND-2"]:
        assert states_magnitude(text) is False, text


def test_new_strips_do_not_swallow_a_comparative_action_magnitude():
    """The new alternatives are anchored on a vessel-attribute noun or on
    `vessel <comparator>`, so a comparative magnitude for the ACTION survives.
    This is the over-stripping regression guard: stripping too much would
    push steps into UNSPECIFIED, which is the direction that flatters nothing
    but is still wrong."""
    from pipelines.plan_adequacy.executor import states_magnitude
    for text in ["Pull at more than 90 t",
                 "Lift with less than 900 t of rated capacity",
                 "Dredge to 5 m below the keel",
                 "Apply foam at over 4000 lpm"]:
        assert states_magnitude(text) is True, text


def test_a_vessel_size_threshold_does_not_rescue_an_otherwise_vague_step():
    """The whole failure mode in one case: the assertion block injects
    ">50 m" into the plan, and that phrase alone used to flip a step from
    UNSPECIFIED to SPECIFIED."""
    tool_reg, route_reg = _reg()
    calls = [
        _call(1, "sound_tanks", text="Sound the tanks", params={"tank_ids": ["1"]}),
        _call(2, "survey_seabed", text="Survey the seabed"),
        _call(3, "calculate_ground_reaction", text="Calculate ground reaction"),
        _call(4, "calculate_freeing_force", text="Calculate freeing force"),
        _call(5, "attach_tug",
              text="Since the vessel is >50 m, deploy ocean-going salvage tugs",
              params={"count": 2}),
        _call(6, "pull", text="Apply controlled pulling force", params={"force_t": 90}),
    ]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    by_n = {s.n: s.verdict for s in result.steps}
    assert by_n[5] == "UNSPECIFIED"
    assert by_n[6] == "UNSPECIFIED"


# ── pass 6: universal families ───────────────────────────────────────────────
# Five tools were filed under the casualty whose ROUTE first cited them rather
# than the casualty they apply to. The corpus made the cost obvious: 24 steps
# were flagged METHOD_ERROR for rescuing the crew off a vessel that happened to
# be aground rather than capsized.

def test_universal_tools_are_not_method_errors_on_any_casualty():
    tool_reg, route_reg = _reg()
    universal = ["rescue_crew", "dewater", "attach_tug", "deploy_boom", "sonar_search"]
    for casualty in ("aground", "capsized", "on_fire", "sunken"):
        for tool in universal:
            result = execute_plan([_call(1, tool, text=f"Apply {tool} with 2 units")],
                                  casualty, _scenario(), tool_reg, route_reg)
            assert result.steps[0].verdict != "METHOD_ERROR", (
                f"{tool} still reads as a method error on {casualty}")


def test_righting_a_sunken_vessel_is_still_a_method_error():
    """The load-bearing regression. right_vessel deliberately stays in the
    capsized family: sunken plans that attempt to right a wreck are the
    STRATEGY_PERCEPTION signal, and universalising it would erase the one
    finding that survives the trust filter."""
    tool_reg, route_reg = _reg()
    result = execute_plan([_call(1, "right_vessel", text="Right the vessel at 1800 t",
                                 params={"method": "parbuckling", "load_t": 1800.0})],
                          "sunken", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "METHOD_ERROR"


def test_monitor_tide_stays_aground_scoped():
    """Its source is specifically ground reaction as the tide rises, so it is
    genuinely aground-specific -- unlike the five reclassified tools."""
    tool_reg, route_reg = _reg()
    result = execute_plan([_call(1, "monitor_tide", text="Monitor the tide for 6 hours")],
                          "on_fire", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "METHOD_ERROR"


def test_every_registry_family_is_one_the_executor_understands():
    """A typo in tools.json's family field would silently make a tool a
    METHOD_ERROR on every casualty, which reads as a planner failure."""
    from pipelines.plan_adequacy.vocab import ToolRegistry
    reg = ToolRegistry.load()
    allowed = {"assessment", "terminal", "universal",
               "aground", "capsized", "on_fire", "sunken"}
    names = reg.all_tool_names()
    assert names, "registry loaded no tools"
    for name in sorted(names):
        assert reg.family(name) in allowed, f"{name}: unknown family {reg.family(name)}"


# ---------------------------------------------------------------------------
# All-dimensions scan (StepResult.flags)
# ---------------------------------------------------------------------------
# The scan exists because `verdict` is the FIRST failing check in precedence
# order, so every check below the one that fired is silently under-counted.
# Plan-level on the CASTOR corpus, sequencing measures 58% of plans through
# verdict counts and 88% through these flags.
#
# The design rests on one claim: hoisting the checks above the cascade cannot
# change what they see, because the cascade already applied effects on every
# path except NO_MATCH. These tests are that claim's guard.

def _verdict_from_flags(f, tool=None):
    """The executor's precedence order, applied to the scan flags. SUPPORT is
    the one verdict the flags cannot see (it is ungraded, so every flag is
    False), so it is recognised from the tool: a no_match step whose
    no_match flag is off."""
    if tool == "no_match" and not f["no_match"]:
        return "SUPPORT"
    if f["no_match"]:
        return "NO_MATCH"
    if f["hedged"]:
        return "CONDITIONAL_UNRESOLVED"
    if f["method"]:
        return "METHOD_ERROR"
    if f["sequence"]:
        return "SEQUENCE_VIOLATION"
    return "UNSPECIFIED" if f["unquantified"] else "SPECIFIED_UNGRADED"


def test_scan_flags_reproduce_the_verdict():
    """Taking the flags in precedence order must give back `verdict` exactly.
    If this fails, the flags and the cascade have drifted and every rate read
    off the flags is unanchored."""
    tool_reg, route_reg = _reg()
    calls = [_call(1, "survey_hull"),
             _call(2, "no_match", text="Establish a safety perimeter"),
             _call(6, "no_match", text="Deploy divers to cut the anchor chain"),
             _call(3, "rig_parbuckling", text="Rig parbuckling points"),
             _call(4, "right_vessel", text="Right the vessel"),
             _call(5, "monitor_tide", text="Monitor the tide")]
    for casualty in ("capsized", "aground", "sunken", "on_fire"):
        result = execute_plan(calls, casualty, _scenario(), tool_reg, route_reg)
        for s in result.steps:
            assert _verdict_from_flags(s.flags, s.tool) == s.verdict, (
                f"{casualty} step {s.n}: flags say "
                f"{_verdict_from_flags(s.flags, s.tool)}, verdict says {s.verdict}")


def test_scan_records_checks_the_cascade_never_reached():
    """The point of the scan: a step whose verdict is decided high in the
    order still reports what the lower checks found."""
    tool_reg, route_reg = _reg()
    # right_vessel in an aground plan is a METHOD_ERROR, which short-circuits
    # before sequencing and magnitude are ever consulted by the cascade.
    result = execute_plan([_call(1, "right_vessel", text="Right the vessel")],
                          "aground", _scenario(), tool_reg, route_reg)
    s = result.steps[0]
    assert s.verdict == "METHOD_ERROR"
    assert s.flags["method"] is True
    # ...but the plan ALSO never rescued the crew, and never stated a load.
    assert s.flags["sequence"] is True, "sequencing was masked by METHOD_ERROR"
    assert s.flags["unquantified"] is True, "magnitude was masked by METHOD_ERROR"


def test_every_step_carries_all_five_flags():
    tool_reg, route_reg = _reg()
    result = execute_plan([_call(1, "survey_hull"),
                           _call(2, "no_match", text="Liaise with the authorities")],
                          "capsized", _scenario(), tool_reg, route_reg)
    for s in result.steps:
        assert set(s.flags) == {"no_match", "hedged", "method",
                                "sequence", "unquantified"}, s.flags
