"""
inject.py -- S1a-1: does the checker assign the RIGHT diagnosis?

The failure profile is a set of labels. Nothing in it establishes that the
labels are correct, and a diagnostic study's only real failure mode is a wrong
diagnosis. This module builds that evidence the one way it can be built
without a second corpus: take a plan known to be clean, inject exactly ONE
defect whose class is known by construction, and check which class comes back.

Over all defects that produces a confusion matrix over failure classes, and
from it the RECALL column of the diagnosis table. A class whose recall is low
is one whose prevalence must not be reported -- the number would be measuring
the checker, not the corpus.

Why hand-written ToolCalls rather than plan text. This is the
executor-in-isolation half of the study (S1a-1): no extractor is involved, so
any misdiagnosis here is the checker's own. The companion half (S1a-2) injects
the same defects into plan TEXT and runs the full pipeline; the difference
between the two matrices is the extractor's contribution to misdiagnosis. That
decomposition is the reason for splitting the study, and it is free -- the
same defect list drives both.

The base plans below are hand-built from registry/routes.json and verified
clean by test, not assumed clean. If a registry change breaks one, the test
fails loudly rather than silently turning a "clean base" into a defect the
injection then compounds.
"""

import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.classify import classify
from pipelines.plan_adequacy.executor import execute_plan
from pipelines.plan_adequacy.methods import RouteRegistry
from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry


def _c(n, tool, text, **params):
    return ToolCall(step_num=n, step_text=text, tool=tool, params=params,
                    conditional=False, condition_text=None, condition_var="none")


def _gate(n, tool, text, var, **params):
    return ToolCall(step_num=n, step_text=text, tool=tool, params=params,
                    conditional=True, condition_text=f"if {var} permits",
                    condition_var=var)


# ── clean base plans ─────────────────────────────────────────────────────────
# One per casualty family, each instantiating a real route with its
# preconditions established in order and every magnitude stated in the step
# TEXT (specificity is read from the sentence, never from params).

def base_aground():
    return [
        _c(1, "sound_tanks", "Sound all 6 tanks and record contents.", tank_ids=["1"]),
        _c(2, "survey_seabed", "Survey the seabed over a 200 m radius."),
        _c(3, "calculate_ground_reaction", "Calculate ground reaction; expect 400 t."),
        _c(4, "calculate_freeing_force", "Calculate freeing force; expect 250 t."),
        _c(5, "attach_tug", "Attach 2 tugs of 4000 shp each.", count=2, shp=4000.0),
        _c(6, "pull", "Pull at 90 t bollard pull on the rising tide.", force_t=90.0),
    ]


def base_capsized():
    return [
        _c(1, "rescue_crew", "Recover all 14 crew to the standby vessel."),
        _c(2, "calculate_stability", "Compute GZ curve and GM of 0.4 m."),
        _c(3, "sound_tanks", "Sound all 8 tanks.", tank_ids=["1"]),
        _c(4, "rig_parbuckling", "Rig 6 parbuckling points rated 900 t each.",
           n_points=6, capacity_t=900.0),
        _c(5, "dewater", "Dewater the engine room at 400 m3/h.",
           space="engine_room", pump_capacity=400.0),
        _c(6, "right_vessel", "Right the vessel by parbuckling at 1800 t load.",
           method="parbuckling", load_t=1800.0),
    ]
# NOTE the parbuckling route is admissible for size_category in
# {medium, large} (registry/routes.json), which is why SCENARIOS pins this
# base to "large". Pinning it to "small" made the BASE itself carry a
# STRATEGY_TECHNIQUE defect, so every injection on top would have measured
# two defects at once.


def base_on_fire():
    return [
        _c(1, "muster_personnel", "Muster all 22 crew and confirm the count."),
        _c(2, "seal_boundaries", "Seal all 4 boundaries of the machinery space.",
           space="machinery"),
        _c(3, "activate_predischarge_alarm", "Sound the 30 s pre-discharge alarm.",
           space="machinery"),
        _c(4, "release_co2", "Release the 1200 kg CO2 bank into the space.",
           space="machinery", mass_kg=1200.0),
        _c(5, "boundary_cool", "Boundary cool the 2 adjacent bulkheads.",
           space="machinery", asset="bulkhead"),
        _c(6, "confirm_fire_out", "Confirm the fire is out after 6 hours of cooling."),
    ]


def base_sunken():
    # test_atmosphere establishes ONE sub-fact per call (worldstate.apply
    # special-cases it), and atmosphere_safe only follows once all three are
    # known -- so the three gas tests are three separate steps, exactly as
    # the FSS-derived registry intends.
    return [
        _c(1, "equalize_pressure", "Equalize pressure across all 3 compartments."),
        _c(2, "test_atmosphere", "Gas test 1 of 3: explosive.", test_type="explosive"),
        _c(3, "test_atmosphere", "Gas test 2 of 3: oxygen.", test_type="oxygen"),
        _c(4, "test_atmosphere", "Gas test 3 of 3: toxic.", test_type="toxic"),
        _c(5, "sound_tanks", "Sound all 5 tanks.", tank_ids=["1"]),
        _c(6, "lift", "Lift the hull at 2400 t total force with 8 lift bags.",
           method="lift_bags", force_t=2400.0),
    ]


BASES = {"aground": base_aground, "capsized": base_capsized,
         "on_fire": base_on_fire, "sunken": base_sunken}

#: Scenario fields chosen so each base's route is ADMISSIBLE -- otherwise the
#: base itself would carry a STRATEGY_TECHNIQUE defect and every injection on
#: top of it would be measuring two things at once.
SCENARIOS = {
    "aground":  dict(size_category="large", habitat_sensitive=False),
    "capsized": dict(size_category="large", habitat_sensitive=False),
    "on_fire":  dict(size_category="large", habitat_sensitive=False),
    "sunken":   dict(size_category="small", habitat_sensitive=False, depth_category="shallow"),
}

#: A tool from another family, used to inject a perception defect. Chosen to
#: be DISTINCTIVE to that family (not shared assessment vocabulary), since
#: detect_perception_mismatch requires a distinctive tool before it will call
#: a foreign match.
_FOREIGN = {
    "aground":  ("sunken",   ["patch_hull", "blow_tanks"]),
    "capsized": ("aground",  ["attach_tug", "pull"]),
    "on_fire":  ("aground",  ["attach_tug", "pull"]),
    "sunken":   ("capsized", ["rig_parbuckling", "right_vessel"]),
}


# ── defect injectors ─────────────────────────────────────────────────────────
# Each returns (calls, expected_class). One defect per plan, always.

def inject_no_procedure(calls, casualty):
    """Every action step becomes filler. Nothing matches any route, for this
    casualty or any other."""
    out = [_c(c.step_num, "no_match", c.step_text) for c in calls]
    return out, "NO_PROCEDURE"


def inject_perception(calls, casualty, tool_registry=None):
    """Replace EVERY action step with another family's distinctive tools: a
    competent procedure aimed at the wrong accident.

    Every action step, not just the last few. A first version replaced only
    the trailing two, which left the base's own route fully intact alongside
    the foreign one -- and detect_perception_mismatch correctly declined to
    call that a perception failure, because it is not one. A plan carrying
    two complete procedures is a shotgun plan, which route_coherence measures
    and this class does not. Keeping the assessments is deliberate: a
    misperceiving planner still assesses, it just assesses toward the wrong
    accident.
    """
    tool_registry = tool_registry or ToolRegistry.load()
    fam, tools = _FOREIGN[casualty]
    kept = [c for c in calls
            if tool_registry.has(c.tool)
            and tool_registry.spec(c.tool).family in ("assessment", "terminal")]
    out = [_c(i + 1, c.tool, c.step_text, **c.params) for i, c in enumerate(kept)]
    for t in tools:
        out.append(_c(len(out) + 1, t,
                      f"Apply {t.replace('_', ' ')} with 4 units rated 600 t.",
                      method="x", force_t=600.0, count=4, capacity_t=600.0,
                      n_points=4, load_t=600.0, breach="bow", material="steel"))
    return out, "STRATEGY_PERCEPTION"


def inject_method_error(calls, casualty):
    """One step calls a tool from a family that does not belong to this
    casualty at all -- technique failure caught at step level."""
    wrong = {"aground": "release_co2", "capsized": "release_co2",
             "on_fire": "rig_parbuckling", "sunken": "release_co2"}[casualty]
    out = list(calls)
    out[2] = _c(3, wrong, f"Apply {wrong.replace('_', ' ')} using 800 kg.",
                space="machinery", mass_kg=800.0, n_points=4, capacity_t=800.0)
    return out, "STRATEGY_TECHNIQUE"


def inject_sequence_violation(calls, casualty, tool_registry=None):
    """Move the last step that actually HAS preconditions to step 1, ahead of
    everything that establishes them.

    "The step with preconditions", not "the last step". Two earlier versions
    picked positionally -- second-to-last, then last -- and both produced
    still-VALID plans on some bases, because the step they moved had an empty
    `requires` and reordering it broke nothing. on_fire is the clear case: its
    terminal confirm_fire_out requires nothing, while the step that genuinely
    depends on ordering is release_co2. Selecting on the registry rather than
    on position makes the injection mean the same thing for every base, which
    is what a confusion matrix needs if its rows are to be comparable.
    """
    tool_registry = tool_registry or ToolRegistry.load()

    def _has_reqs(c):
        return tool_registry.has(c.tool) and bool(tool_registry.spec(c.tool).requires)

    idx = next((i for i in range(len(calls) - 1, -1, -1) if _has_reqs(calls[i])), None)
    if idx is None or idx == 0:
        raise ValueError(f"no reorderable step with preconditions in the {casualty} base")
    out = [calls[idx]] + [c for i, c in enumerate(calls) if i != idx]
    return [_c(i + 1, c.tool, c.step_text, **c.params) for i, c in enumerate(out)], "PROCEDURE"


def inject_unresolved_gate(calls, casualty):
    """Gate a step on a condition no prior step ever establishes."""
    out = list(calls)
    c = out[3]
    out[3] = _gate(c.step_num, c.tool, c.step_text, "hull_condition", **c.params)
    return out, "PROCEDURE"


def inject_commitment(calls, casualty):
    """Strip the magnitudes from the step TEXT while leaving params intact --
    which also verifies specificity cannot be satisfied from the params dict.
    """
    out = []
    for c in calls:
        text = "".join(ch for ch in c.step_text if not ch.isdigit())
        out.append(_c(c.step_num, c.tool, text, **c.params))
    return out, "COMMITMENT"


INJECTORS = [
    ("no_procedure", inject_no_procedure),
    ("perception", inject_perception),
    ("method_error", inject_method_error),
    ("sequence_violation", inject_sequence_violation),
    ("unresolved_gate", inject_unresolved_gate),
    ("commitment", inject_commitment),
]


# ── harness ──────────────────────────────────────────────────────────────────

def _scenario(casualty):
    return SimpleNamespace(image=f"{casualty}/inject.jpg", state=casualty,
                            **SCENARIOS[casualty])


def run_injections(tool_registry=None, route_registry=None) -> list:
    """Every (base casualty x defect) pair. Returns one record per case with
    the class expected by construction and the class actually recovered."""
    tool_registry = tool_registry or ToolRegistry.load()
    route_registry = route_registry or RouteRegistry.load()

    out = []
    for casualty, base in BASES.items():
        for defect_name, injector in INJECTORS:
            if defect_name in ("perception", "sequence_violation"):
                calls, expected = injector(base(), casualty, tool_registry)
            else:
                calls, expected = injector(base(), casualty)
            scenario = _scenario(casualty)
            plan_text = "\n".join(c.step_text for c in calls)
            result = execute_plan(calls, casualty, scenario, tool_registry,
                                  route_registry, plan_text)
            got = classify(result)
            out.append({
                "casualty": casualty, "defect": defect_name,
                "expected": expected, "got": got["failure_class"],
                "correct": got["failure_class"] == expected,
                "epl": got["epl"], "failure_step": got["failure_step"],
            })
    return out


def confusion_matrix(records: list) -> dict:
    """{(expected, got): n}."""
    return dict(Counter((r["expected"], r["got"]) for r in records))


def recall_by_class(records: list) -> dict:
    """Diagnostic recall per class: of the cases carrying a defect of class C,
    how many were labelled C.

    This is the RECALL column of the diagnosis table, and the gate on whether
    a class's prevalence is reportable at all.
    """
    tot, hit = Counter(), Counter()
    for r in records:
        tot[r["expected"]] += 1
        hit[r["expected"]] += int(r["correct"])
    return {cls: {"n": tot[cls], "recall": hit[cls] / tot[cls]} for cls in sorted(tot)}


def main():
    records = run_injections()
    print(f"{'casualty':<10}{'defect':<20}{'expected':<22}{'got':<22}ok")
    print("-" * 80)
    for r in records:
        print(f"{r['casualty']:<10}{r['defect']:<20}{r['expected']:<22}"
              f"{r['got']:<22}{'y' if r['correct'] else 'N'}")
    print("\nrecall by class:")
    for cls, v in recall_by_class(records).items():
        flag = "" if v["recall"] >= 0.7 else "   <- below 0.7: prevalence not reportable"
        print(f"  {cls:<22} {v['recall']:.2f}  (n={v['n']}){flag}")


if __name__ == "__main__":
    main()
