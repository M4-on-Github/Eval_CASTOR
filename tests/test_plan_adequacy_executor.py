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

from pipelines.plan_adequacy.executor import execute_plan
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
    calls = [_call(1, "no_match", text="Coordinate with relevant stakeholders.")]
    result = execute_plan(calls, "aground", _scenario(), tool_reg, route_reg)
    assert result.steps[0].verdict == "NO_MATCH"


# ── route recognition wired end to end ───────────────────────────────────────

def test_full_tug_pull_plan_recognises_route_and_sequences_cleanly():
    # NOT a goal-achievement test -- executor.py does not check
    # goal.terminal_facts yet (phase 2, see routes.json TODOs). This only
    # asserts route recognition + absence of sequencing errors.
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
