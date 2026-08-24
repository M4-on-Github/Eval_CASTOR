"""
Tests for pipelines/plan_adequacy/calibration/build_gold_layer_a.py's
_draft_tool() heuristic -- the scaffold-drafting function used to bootstrap
(never finalize) the P9 gold tool-call set.

These are regression anchors for a confirmed audit finding: an earlier
version silently promoted a keyword found anywhere in a multi-sentence step
-- including deep inside a LATER sentence, or inside an "if/unless/should/
once" conditional clause -- to be the step's primary tool, with no
confidence penalty. A full hand-audit of 238 gold-set records found this
was wrong roughly as often as it was right. Each test here is a real
example pulled from that audit, not a synthetic case.
Run: python -m pytest tests/test_plan_adequacy_gold_draft.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.calibration.build_gold_layer_a import _draft_tool


def test_keyword_in_a_later_sentence_does_not_become_primary():
    # "tugboats" only appears in sentence 2, deep in a conditional clause --
    # the step's actual first sentence is about a final inspection.
    text = (
        "Once the vessel is stabilized or the fire is under control, "
        "conduct a final inspection to ensure that all hazards have been "
        "mitigated and that the vessel is safe for any further operations. "
        "If the vessel is to be towed, coordinate with tugboats to execute "
        "the towing plan, ensuring that the vessel is moved to a safe "
        "location for repair or disposal."
    )
    tool, secondary, confidence = _draft_tool(text)
    assert tool == "post_operation_assessment"
    assert "attach_tug" in secondary


def test_meta_planning_language_is_no_match_regardless_of_mentioned_equipment():
    text = (
        "If the vessel is deemed salvageable, develop a detailed salvage "
        "plan that includes the use of cranes, pumps, and patching "
        "materials to remove water and repair structural damage."
    )
    tool, secondary, confidence = _draft_tool(text)
    assert tool == "no_match"


def test_hit_only_inside_a_conditional_clause_is_not_promoted_as_a_commitment():
    # "refloating" appears only inside the trailing "if...for potential
    # lifting or refloating" clause -- the main action (ballasting) has no
    # tool in the vocabulary. Must fall through to no_match, not be
    # silently promoted to pull -- confirmed wrong by hand audit.
    text = (
        "Begin controlled flooding or ballasting operations if the vessel "
        "is deemed stable enough to be stabilized for potential lifting or "
        "refloating."
    )
    tool, secondary, confidence = _draft_tool(text)
    assert tool == "no_match"
    assert confidence == "no_match_confident"


def test_purpose_clause_promotion_is_always_low_confidence():
    # The "Deploy X to Y" shape is genuinely ambiguous (X can be the real
    # action's resource, as here, or unrelated staging material) -- the
    # promotion path must never claim "medium" regardless of which way it
    # resolves.
    text = "Deploy pumps to remove water from the hull if flooding is detected."
    tool, secondary, confidence = _draft_tool(text)
    assert confidence == "low"


def test_fronted_condition_clause_main_verb_is_still_found():
    text = (
        "Once the fire is out, reassess hull and stability before "
        "resuming operations."
    )
    tool, secondary, confidence = _draft_tool(text)
    assert tool == "survey_hull"


def test_co2_unicode_subscript_is_recognised():
    text = (
        "Directly activate the CO₂ fixed suppression system to flood "
        "the engine room and suppress the machinery space fire."
    )
    tool, secondary, confidence = _draft_tool(text)
    assert tool == "release_co2"


def test_confirm_clear_of_personnel_before_suppression_is_muster_not_the_suppression_tool():
    # "before activating X" defers X -- the step's own commitment is the
    # personnel-clearance confirmation, not the (not-yet-activated) system.
    text = "Confirm the cargo hold is clear of all personnel before activating the foam suppression system."
    tool, secondary, confidence = _draft_tool(text)
    assert tool == "muster_personnel"
    assert "apply_foam" in secondary
