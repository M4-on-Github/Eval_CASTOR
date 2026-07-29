"""
check_seq.py — Deterministic SEQ (sequencing) checker for salvage plans.

SEQ is pass/fail per precondition chain:
  - Each chain defines a GUARD action that must precede a TRIGGER action.
  - If the trigger is present in the plan but the guard is not, SEQ fails for that chain.
  - If neither guard nor trigger appears, the chain is not applicable (N/A).
  - If the guard appears before or alongside the trigger, SEQ passes for that chain.

22 chains, organized by casualty type.

Usage:
    from eval.check_seq import check_seq
    result = check_seq(steps, gt_state)
    # result: {
    #   "seq_score": float,        # fraction of applicable chains that pass
    #   "chains_applicable": int,
    #   "chains_passed": int,
    #   "chains_failed": int,
    #   "failures": List[str],     # names of failed chains
    # }
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chain:
    name: str
    state: str                  # casualty type this chain applies to
    guard_kws: list[str]        # guard step keywords (any one suffices)
    trigger_kws: list[str]      # trigger step keywords (any one suffices)
    description: str = ""


# ---------------------------------------------------------------------------
# 14 SEQ chains
# ---------------------------------------------------------------------------

CHAINS: list[Chain] = [

    # AGROUND (6 chains)
    Chain(
        name="AG_FUEL_CHECK_BEFORE_WORK",
        state="aground",
        guard_kws=["fuel tank", "fuel check", "check fuel", "residual fuel", "fuel survey"],
        trigger_kws=["cutting", "welding", "torch", "hot work", "pull", "tow", "dredg"],
        description="Fuel tanks must be checked before any cutting, welding, or active movement.",
    ),
    Chain(
        name="AG_HAZMAT_BEFORE_WORK",
        state="aground",
        guard_kws=["hazmat", "cargo check", "check cargo", "dangerous goods", "cargo survey",
                   "cargo manifest", "inventory cargo"],
        trigger_kws=["cutting", "welding", "torch", "hot work", "pull", "tow", "move"],
        description="Cargo holds must be checked for hazardous materials before any active work.",
    ),
    Chain(
        name="AG_FRICTION_BEFORE_PULL",
        state="aground",
        guard_kws=["friction", "ground reaction", "seabed", "substrate", "pulling force",
                   "bollard pull", "calculate pull"],
        trigger_kws=["pull", "towing", "kedging", "winch", "beach gear"],
        description="Seabed friction / ground reaction must be calculated before pulling begins.",
    ),
    Chain(
        name="AG_LIGHTERING_BEFORE_REFLOAT",
        state="aground",
        guard_kws=["lighter", "lightering", "discharge cargo", "reduce displacement",
                   "offload cargo", "reduce draft"],
        trigger_kws=["refloat", "pull", "kedge", "tow off"],
        description="Large vessels must be lightered before refloat attempt.",
    ),
    Chain(
        name="AG_TIDE_MONITOR_BEFORE_PULL",
        state="aground",
        guard_kws=["tide gauge", "tide monitor", "tidal monitoring", "monitor tide",
                   "tidal rise", "tide rise", "ground reaction monitor", "monitor ground reaction"],
        trigger_kws=["pull", "towing", "kedging", "winch", "beach gear", "active pull"],
        description="Tide gauge monitoring must be in place before committing to active pulling.",
    ),
    Chain(
        name="AG_DREDGE_BEFORE_PULL_HARD_SUBSTRATE",
        state="aground",
        guard_kws=["dredg", "clear channel", "channel dredg", "dredge channel",
                   "dredging operation", "clear a channel"],
        trigger_kws=["pull", "towing", "kedging", "winch", "beach gear"],
        description="Channel must be dredged before pulling on hard substrate (rock/coral).",
    ),

    # CAPSIZED (5 chains)
    Chain(
        name="CAP_RESCUE_BEFORE_RIGHTING",
        state="capsized",
        guard_kws=["rescue", "crew recovery", "personnel recovery", "search survivors",
                   "rescue swimmer", "account for", "recover crew", "personnel aboard",
                   "trapped survivors"],
        trigger_kws=["right", "righting", "parbuckl", "deballast", "crane lift",
                     "upend", "refloat"],
        description="Crew rescue must precede any righting attempt.",
    ),
    Chain(
        name="CAP_STABILITY_CHECK_BEFORE_RIGHTING",
        state="capsized",
        guard_kws=["stability", "gm", "gz", "metacentric", "righting arm", "righting lever",
                   "salvage engineer", "naval architect", "stability analysis"],
        trigger_kws=["right", "righting", "parbuckl", "lift", "deballast", "crane"],
        description="Formal stability check (GM/GZ) must precede righting.",
    ),
    Chain(
        name="CAP_DEWATER_BEFORE_RIGHTING",
        state="capsized",
        guard_kws=["pump", "dewater", "drain", "remove trapped water", "remove water",
                   "submersible pump"],
        trigger_kws=["right", "righting", "parbuckl", "upend", "crane lift"],
        description="Trapped water must be pumped out before righting.",
    ),
    Chain(
        name="CAP_RIGHTING_BEFORE_SEAWORTHINESS",
        state="capsized",
        guard_kws=["right", "righting complete", "vessel upright", "hull upright",
                   "refloated", "successfully righted"],
        trigger_kws=["seaworthy", "under tow", "tow to port", "tow away", "declare safe",
                     "return to service"],
        description="Post-righting hull reassessment must precede declaring seaworthy.",
    ),
    Chain(
        name="CAP_ENGINEER_VERIFICATION_BEFORE_RIGHTING",
        state="capsized",
        guard_kws=["verify righting", "verify lift", "lift plan", "righting plan",
                   "approve plan", "engineer approv", "architect approv",
                   "verify the plan", "sign off", "plan verification"],
        trigger_kws=["right", "righting", "parbuckl", "lift", "deballast", "crane"],
        description="Salvage engineer/naval architect must verify the righting and lift plan before execution.",
    ),

    # SUNKEN (6 chains)
    Chain(
        name="SU_GAS_TEST_BEFORE_DIVE",
        state="sunken",
        guard_kws=["gas test", "explosive gas", "oxygen test", "toxic gas", "atmosphere test",
                   "confined space test", "three-gas", "3-gas", "gas check"],
        trigger_kws=["diver enter", "dive team enter", "enter enclosed", "hot work",
                     "cutting", "welding", "diver access"],
        description="Three gas tests (explosive→oxygen→toxic) must precede enclosed-space diver entry.",
    ),
    Chain(
        name="SU_SURVEY_BEFORE_RECOVERY",
        state="sunken",
        guard_kws=["dive survey", "hull survey", "survey team", "assess hull", "sonar survey",
                   "rov survey", "dive and survey", "hull assessment", "initial assessment",
                   "condition assessment"],
        trigger_kws=["raise", "lift", "recover", "patch", "cofferdam", "refloat", "crane"],
        description="Dive/survey team must assess hull before recovery attempt.",
    ),
    Chain(
        name="SU_DEFUEL_BEFORE_DISTURBANCE",
        state="sunken",
        guard_kws=["defuel", "offload fuel", "fuel removal", "pump fuel", "remove fuel",
                   "fuel offload"],
        trigger_kws=["raise", "lift", "cut", "patch", "move hull", "disturb", "refloat",
                     "crane", "cofferdam"],
        description="Fuel must be offloaded before any hull disturbance.",
    ),
    Chain(
        name="SU_CARGO_INVENTORY_BEFORE_DISTURBANCE",
        state="sunken",
        guard_kws=["cargo inventory", "inventory cargo", "cargo manifest", "cargo check",
                   "account for cargo", "unknown cargo", "cargo survey"],
        trigger_kws=["raise", "lift", "cut", "move hull", "disturb", "refloat", "crane"],
        description="Cargo must be inventoried before any hull disturbance.",
    ),
    Chain(
        name="SU_SONAR_BEFORE_RECOVERY_BOTTOM_OIL",
        state="sunken",
        guard_kws=["side-scan sonar", "electro-acoustic sonar", "acoustic sonar",
                   "sonar locate", "locate oil", "sonar survey oil", "bottom-settled oil"],
        trigger_kws=["recover oil", "oil recovery", "remove oil", "oil removal",
                     "skim oil", "oil response", "diver oil", "rov oil"],
        description="Sonar must locate bottom-settled oil before diver/ROV oil recovery.",
    ),
    Chain(
        name="SU_NEBA_BEFORE_RECOVERY_METHOD_SELECTION",
        state="sunken",
        guard_kws=["neba", "net environmental benefit", "environmental benefit analysis",
                   "environmental assessment", "habitat assessment", "neba documented",
                   "environmental analysis"],
        trigger_kws=["dredg", "raise", "lift", "recover", "refloat", "crane",
                     "recovery method", "diver-directed", "mechanical recovery"],
        description="NEBA must be documented before choosing recovery method in sensitive habitat.",
    ),

    # ON FIRE (5 chains)
    Chain(
        name="FIRE_CLEAR_BEFORE_SUPPRESSION",
        state="on_fire",
        guard_kws=["clear personnel", "evacuate", "all clear", "confirmed clear",
                   "personnel clear", "clear space", "all personnel out",
                   "evacuated", "headcount"],
        trigger_kws=["co2", "fixed suppression", "activate suppression", "flood space",
                     "deploy foam", "fixed system", "suppression system"],
        description="Space must be confirmed clear before activating fixed suppression.",
    ),
    Chain(
        name="FIRE_COOL_BEFORE_VENTILATE",
        state="on_fire",
        guard_kws=["cool boundary", "cool bulkhead", "boundary cooling", "cool adjacent",
                   "cool surrounding", "temperature check", "cool before", "cool down"],
        trigger_kws=["ventilat", "open hatch", "open door", "re-enter", "reenter",
                     "re-board", "reboard", "enter space"],
        description="Superheated spaces must be cooled before opening, ventilating, or entering.",
    ),
    Chain(
        name="FIRE_PREDISCHARGE_ALARM_BEFORE_GAS_FLOOD",
        state="on_fire",
        guard_kws=["pre-discharge alarm", "predischarge alarm", "pre-discharge signal",
                   "discharge alarm", "two-stage release", "pre-discharge",
                   "alarm sound", "warning alarm"],
        trigger_kws=["co2 flood", "flood space", "gas flood", "activate co2",
                     "release co2", "co2 release", "flood with co2"],
        description="Pre-discharge alarm must sound before CO2 gas floods the space.",
    ),
    Chain(
        name="FIRE_PUMP_CAPACITY_BEFORE_TEAM_COMMIT",
        state="on_fire",
        guard_kws=["pump capacity", "confirm pump", "verify pump", "adequate pump",
                   "pump output", "firefighting pump", "emergency pump capacity",
                   "confirm capacity"],
        trigger_kws=["deploy team", "firefighting team", "commit team", "direct attack",
                     "fighting team", "board vessel", "attack team"],
        description="Pump capacity must be confirmed adequate before committing a direct firefighting team.",
    ),
    Chain(
        name="FIRE_STABILITY_REASSESS_AFTER_FIRE_OUT",
        state="on_fire",
        guard_kws=["fire out", "fire confirmed out", "fire extinguished", "confirmed out",
                   "fire under control", "fire is out"],
        trigger_kws=["declare safe", "seaworthy", "under tow", "return to service",
                     "safe to board", "reboard", "hull reassess", "stability reassess"],
        description="Hull and stability must be reassessed after fire is confirmed out before declaring safe.",
    ),
]


# ---------------------------------------------------------------------------
# keyword matching
# ---------------------------------------------------------------------------

def _kw_present(keywords: list[str], text: str) -> bool:
    text_l = text.lower()
    return any(kw.lower() in text_l for kw in keywords)


def _find_first_step(keywords: list[str], steps: list[tuple[int, str]]) -> int | None:
    """Return the step number of the first step containing any keyword, or None."""
    for num, text in steps:
        if _kw_present(keywords, text):
            return num
    return None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def check_seq(
    steps: list[tuple[int, str]],
    gt_state: str,
) -> dict:
    """
    Evaluate SEQ for the given steps under the given gt_state.

    Args:
        steps:    List of (step_num, step_text) from parse_steps_v2.
        gt_state: Canonical casualty type string, e.g. 'aground', 'on_fire'.

    Returns:
        dict with keys:
          seq_score         float  fraction of applicable chains that pass (0.0–1.0)
          chains_applicable int
          chains_passed     int
          chains_failed     int
          failures          List[str]  names of failed chains
          na_chains         List[str]  names of N/A chains (trigger absent)
    """
    applicable_chains = [c for c in CHAINS if c.state == gt_state]

    if not steps or not applicable_chains:
        return {
            "seq_score": 1.0,
            "chains_applicable": 0,
            "chains_passed": 0,
            "chains_failed": 0,
            "failures": [],
            "na_chains": [],
        }

    full_text = " ".join(t for _, t in steps)

    passed: list[str] = []
    failed: list[str] = []
    na:     list[str] = []

    for chain in applicable_chains:
        trigger_step = _find_first_step(chain.trigger_kws, steps)
        if trigger_step is None:
            # Trigger not mentioned — chain not applicable
            na.append(chain.name)
            continue

        guard_step = _find_first_step(chain.guard_kws, steps)
        if guard_step is None:
            # Trigger present, guard absent — SEQ failure
            failed.append(chain.name)
        elif guard_step <= trigger_step:
            # Guard precedes or co-occurs with trigger — pass
            passed.append(chain.name)
        else:
            # Guard appears after trigger — also a sequencing failure
            failed.append(chain.name)

    n_applicable = len(passed) + len(failed)
    seq_score = (len(passed) / n_applicable) if n_applicable > 0 else 1.0

    return {
        "seq_score": round(seq_score, 4),
        "chains_applicable": n_applicable,
        "chains_passed": len(passed),
        "chains_failed": len(failed),
        "failures": failed,
        "na_chains": na,
    }


# ---------------------------------------------------------------------------
# CLI: test on a parsed steps JSON file  (list of [num, text] pairs)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json, sys
    data = json.load(open(sys.argv[1]))
    steps_input = [(int(r[0]), r[1]) for r in data["steps"]]
    state = data["gt_state"]
    result = check_seq(steps_input, state)
    print(json.dumps(result, indent=2))
