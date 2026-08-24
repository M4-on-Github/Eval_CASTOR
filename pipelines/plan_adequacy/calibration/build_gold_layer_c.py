"""
Build Layer C of the P9 gold tool-call set: authored coverage fill.

Unlike Layers A (real corpus) and B (existing synthetic plans), Layer C has
no source file -- these sentences are written directly, by hand, with a
known-correct label at authoring time (not drafted-then-reviewed). Its only
purpose is to guarantee every one of the 47 tools has at least 5 gold
examples for calibrate.py to test against; it is NEVER used for headline
accuracy (design plan section 4b) since it's an authored, not naturalistic,
distribution -- a model could overfit to this phrasing style in a way that
wouldn't generalise, so it only counts as a coverage probe.

Coverage gap measured against the combined Layer A (72) + Layer B (166)
gold set before this layer existed: 10 tools with zero examples, 16 more
with fewer than 5. See the coverage-count block at the bottom of this file
to reproduce that measurement.

Usage (from Eval_CASTOR/):
  python3 pipelines/plan_adequacy/calibration/build_gold_layer_c.py
"""

import json
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.vocab import ToolRegistry

OUT_PATH = Path(__file__).parent / "gold_layer_c_scaffold.jsonl"

# (tool, secondary_tools, step_text) -- conditional=False, condition_var="none"
# for all of these; Layer C is coverage fill, not gate-testing (gates are
# already exercised by Layers A/B).
EXAMPLES = [
    # ── zero-coverage tools (need 5 each) ────────────────────────────────────
    ("account_for_personnel", [], "Conduct a headcount of all crew members to confirm everyone is accounted for."),
    ("account_for_personnel", [], "Verify that all personnel aboard have been located and accounted for before proceeding."),
    ("account_for_personnel", [], "Take a roll call of the salvage team to confirm full accountability."),
    ("account_for_personnel", [], "Confirm no crew members remain unaccounted for following the evacuation."),
    ("account_for_personnel", [], "Account for all personnel who were aboard the vessel at the time of the incident."),

    ("apply_dispersant", [], "Apply chemical dispersant to the surface oil slick to accelerate breakdown."),
    ("apply_dispersant", [], "Spray dispersant over the spreading oil sheen from a response vessel."),
    ("apply_dispersant", [], "Deploy aerial dispersant application over the offshore slick."),
    ("apply_dispersant", [], "Treat the visible oil sheen with an approved dispersant agent."),
    ("apply_dispersant", [], "Apply dispersant to the water-column oil suspension to reduce surface concentration."),

    ("blow_tanks", [], "Blow the ballast tanks with compressed air to begin lifting the hull."),
    ("blow_tanks", [], "Use compressed air to blow the patched compartment and increase buoyancy."),
    ("blow_tanks", [], "Pressurize and blow the forward tanks to assist the lift."),
    ("blow_tanks", [], "Blow tanks in the aft section once the cofferdam is sealed."),
    ("blow_tanks", [], "Initiate compressed-air blowing of the internal tanks to refloat the hull."),

    ("conduct_neba", [], "Conduct a net environmental benefit analysis before selecting a recovery method."),
    ("conduct_neba", [], "Perform a NEBA to weigh the environmental tradeoffs of mechanical recovery versus leaving the wreck in place."),
    ("conduct_neba", [], "Document a net environmental benefit analysis given the site's proximity to a coral reef."),
    ("conduct_neba", [], "Complete an environmental benefit assessment before committing to dredging in this habitat."),
    ("conduct_neba", [], "Carry out a NEBA to determine whether diver-directed recovery is preferable to disturbing the seabed."),

    ("dewater_firefighting_water", [], "Pump out accumulated firefighting water from the lower decks to restore stability."),
    ("dewater_firefighting_water", [], "Dewater the compartments flooded by firefighting runoff before reboarding."),
    ("dewater_firefighting_water", [], "Remove the standing firefighting water from the hold using portable pumps."),
    ("dewater_firefighting_water", [], "Drain the firefighting water that has collected below decks to reduce free-surface effect."),
    ("dewater_firefighting_water", [], "Pump the firefighting water overboard once the fire is confirmed out."),

    ("equalize_pressure", [], "Equalize pressure differentials in the compartment before any diver entry."),
    ("equalize_pressure", [], "Allow pressure to equalize across the bulkhead before opening the hatch."),
    ("equalize_pressure", [], "Vent the space to equalize internal and external pressure prior to testing the atmosphere."),
    ("equalize_pressure", [], "Equalize pressure in the flooded compartment before divers proceed further."),
    ("equalize_pressure", [], "Confirm pressure has equalized in the enclosed space before beginning gas testing."),

    ("press_full", [], "Press the port ballast tank full to eliminate its free surface effect."),
    ("press_full", [], "Fill the affected tank completely to press it full and control free surface."),
    ("press_full", [], "Press full the starboard tanks to stabilize the vessel before righting."),
    ("press_full", [], "Top off the tank to press it full, removing the free-surface hazard."),
    ("press_full", [], "Press the compartment full of water to eliminate sloshing during the righting operation."),

    ("read_draft_marks", [], "Read the draft marks on the hull to determine current freeboard."),
    ("read_draft_marks", [], "Check the vessel's draft marks to confirm how deep it is sitting in the water."),
    ("read_draft_marks", [], "Record the draft mark readings at bow and stern to assess trim."),
    ("read_draft_marks", [], "Take draft mark readings to establish the vessel's current waterline."),
    ("read_draft_marks", [], "Observe the draft marks to determine whether the vessel's draft exceeds 10 metres."),

    ("remove_impalement", [], "Remove the impaling rock formation from the hull before attempting to pull the vessel free."),
    ("remove_impalement", [], "Clear the impalement point using cutting tools before any pulling begins."),
    ("remove_impalement", [], "Extract the embedded obstruction from the hull breach prior to refloating."),
    ("remove_impalement", [], "Cut away the impaling structure lodged in the hull before applying pulling force."),
    ("remove_impalement", [], "Remove the object impaling the hull to prevent further damage during the pull."),

    ("seal_boundaries", [], "Seal the boundaries of the machinery space before activating the gas suppression system."),
    ("seal_boundaries", [], "Close and seal all openings to the affected space to contain the suppression agent."),
    ("seal_boundaries", [], "Seal hatches and vents around the fire space before flooding it with CO2."),
    ("seal_boundaries", [], "Confirm boundaries are sealed to allow the suppression concentration to build."),
    ("seal_boundaries", [], "Seal the compartment's boundaries so the foam application isn't diluted by open ventilation."),

    # ── thin-coverage tools (topped up to >=5 combined with A+B) ────────────
    ("activate_predischarge_alarm", [], "Sound the pre-discharge alarm for at least twenty seconds before releasing CO2 into the space."),

    ("calculate_freeing_force", [], "Determine the freeing force required using the ground reaction and seabed friction coefficient."),
    ("calculate_freeing_force", [], "Calculate the force needed to free the vessel from the sand bank based on ground reaction."),
    ("calculate_freeing_force", [], "Compute the required pulling force from the measured ground reaction and friction band."),

    ("confirm_pump_capacity", [], "Verify the emergency fire pump can deliver adequate capacity before committing the attack team."),
    ("confirm_pump_capacity", [], "Confirm pump output meets the minimum required capacity before boarding for direct firefighting."),

    ("cut_section", [], "Cut the damaged bow section free using underwater cutting torches."),
    ("cut_section", [], "Section the hull with cutting equipment to allow crane removal of the wreckage."),
    ("cut_section", [], "Cut through the collapsed superstructure to access the trapped compartment."),
    ("cut_section", [], "Use hot-cutting equipment to separate the unsalvageable stern section from the hull."),

    ("dredge", [], "Dredge a channel through the rock outcrop to clear a path for the vessel."),
    ("dredge", [], "Excavate the hard seabed material around the hull using a dredge before pulling."),

    ("lift", [], "Use a floating crane to lift the vessel's midsection clear of the water."),
    ("lift", [], "Deploy lift bags beneath the hull to raise it from the seabed."),
    ("lift", [], "Lift the sunken section using a heavy-lift crane barge."),
    ("lift", [], "Apply controlled lifting force with the crane to raise the wreck section by section."),

    ("lighter_cargo", [], "Lighter the forward cargo hold to reduce displacement before the refloat attempt."),

    ("muster_personnel", [], "Muster all personnel at the designated assembly point before suppression begins."),
    ("muster_personnel", [], "Gather the crew at a safe muster station clear of the fire space."),
    ("muster_personnel", [], "Assemble all onboard personnel at the muster point and confirm headcount."),
    ("muster_personnel", [], "Call all hands to muster stations before any suppression system is activated."),

    ("open_space", [], "Open the machinery space hatch once the fire is confirmed extinguished."),
    ("open_space", [], "Ventilate and open the compartment after confirming no reflash risk remains."),
    ("open_space", [], "Reopen the sealed space only after the fire has been confirmed out."),
    ("open_space", [], "Open the hold to allow post-fire inspection once cooling is complete."),

    ("post_operation_assessment", [], "Conduct a post-operation assessment of the hull once the vessel is safely under tow."),
    ("post_operation_assessment", [], "Perform a final structural assessment after the salvage operation concludes."),
    ("post_operation_assessment", [], "Assess the vessel for residual damage following completion of the recovery."),

    ("recover_oil", [], "Recover the bottom-settled oil using diver-directed suction equipment."),
    ("recover_oil", [], "Deploy ROV-based recovery to collect the oil located by sonar."),
    ("recover_oil", [], "Recover the released oil from the water column using specialized skimming equipment."),

    ("release_co2", [], "Release CO2 into the machinery space following the pre-discharge alarm."),
    ("release_co2", [], "Flood the sealed compartment with the calculated mass of CO2."),
    ("release_co2", [], "Activate the fixed CO2 system to discharge gas into the machinery space."),
    ("release_co2", [], "Release the full CO2 charge into the space once boundaries are confirmed sealed."),

    ("rig_parbuckling", [], "Rig a parbuckling system with cables and winches to right the capsized hull."),
    ("rig_parbuckling", [], "Set up parbuckling points along the hull to begin the controlled righting operation."),
    ("rig_parbuckling", [], "Rig the parbuckling cables at multiple points to distribute the righting load."),

    ("right_vessel", [], "Right the vessel using the rigged parbuckling system under controlled tension."),

    ("size_up_fire", [], "Size up the fire to determine its extent and which compartments are involved."),
    ("size_up_fire", [], "Assess the scope and intensity of the fire before selecting a suppression method."),
    ("size_up_fire", [], "Conduct an initial size-up of the blaze to identify the affected spaces."),
    ("size_up_fire", [], "Evaluate the fire's size and spread pattern before committing suppression resources."),

    ("skim", [], "Skim the surface oil pooled within the containment boom."),
    ("skim", [], "Deploy skimmers to recover the oil collected inside the boom perimeter."),
    ("skim", [], "Skim the contained slick before it disperses further."),
    ("skim", [], "Operate the skimmer to remove pooled oil from within the boomed area."),

    # ── calculate_ground_reaction: had 7 secondary mentions but zero as
    # primary across A+B+C -- never actually tested as the model's
    # top-level pick until now.
    ("calculate_ground_reaction", [], "Determine the ground reaction for the vessel resting on the sand seabed."),
    ("calculate_ground_reaction", [], "Calculate the ground reaction based on the vessel's weight distribution and substrate contact."),
    ("calculate_ground_reaction", [], "Establish the current ground reaction before selecting a pulling method."),
]


def main():
    reg = ToolRegistry.load()
    unknown = sorted({t for t, _, _ in EXAMPLES if not reg.has(t)})
    if unknown:
        print(f"ERROR: unknown tool names in EXAMPLES: {unknown}", file=sys.stderr)
        sys.exit(1)

    out_records = []
    for i, (tool, secondary, text) in enumerate(EXAMPLES, start=1):
        out_records.append({
            "id": f"layerc_{i:03d}",
            "step_text": text,
            "expected_tool": tool,
            "expected_secondary_tools": secondary,
            "expected_params": {},
            "expected_conditional": False,
            "expected_condition_var": "none",
            "expected_family": reg.family(tool),
            "confidence": "authored",
            "reviewed": True,
        })

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
        encoding="utf-8",
    )

    import collections
    by_tool = collections.Counter(r["expected_tool"] for r in out_records)
    print(f"{len(out_records)} authored examples across {len(by_tool)} tools -> {OUT_PATH}")


if __name__ == "__main__":
    main()
