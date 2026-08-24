"""
Build Layer A of the P9 gold tool-call set: a stratified sample of REAL
plan steps from temp_p8_improved/*.jsonl -- the CURRENT run this project is
actually judging (see temp_p8_improved/report.md), not the earlier
exploratory p7_to_check/ corpus this script originally sampled from.

This is the true test distribution -- calibrate.py reports headline
extraction accuracy on Layer A only (design plan section 4b). Layers B
(the existing 170-step synthetic_calibration.jsonl, re-annotated) and C
(authored coverage fill for tools Layer A/B don't reach) are separate,
smaller efforts -- see the design plan and this module's __main__ block.

Usage (from Eval_CASTOR/):
  python3 pipelines/plan_adequacy/calibration/build_gold_layer_a.py

Writes pipelines/plan_adequacy/calibration/gold_layer_a_scaffold.jsonl --
one record per selected step, with expected_* fields set to a heuristic
DRAFT guess (never gold) plus a `confidence` field. Every record needs
human review before it can be used as gold_tool_calls.jsonl input; nothing
here should be trusted as ground truth on its own -- see the module
docstring's tool-keyword table, which is deliberately coarse and will
mis-hit on ambiguous phrasing (that's exactly the kind of case calibration
is supposed to catch, so leaving mis-drafts in is more useful than hiding
them).
"""

import json
import re
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))
# improved/ isn't a proper package (no __init__.py) -- mirror the sys.path
# convention run_judge_v2.py itself uses to import its sibling eval/ modules,
# rather than a dotted absolute import that wouldn't resolve.
IMPROVED_DIR = EVAL_ROOT / "pipelines" / "plan_coherence" / "improved"
sys.path.insert(0, str(IMPROVED_DIR))

from eval.parse_steps_v2 import parse_steps_v2  # noqa: E402 -- needs the sys.path insert above first
from pipelines.plan_adequacy.gates import detect_gates, guess_condition_var
from pipelines.plan_adequacy.vocab import ToolRegistry

#: Source of REAL plan text -- switched from p7_to_check/ (an older,
#: exploratory run) to temp_p8_improved/ (the CURRENT run this project is
#: actually judging -- see report.md there) per explicit direction. Only 3
#: arms here (ablation_v2 / control_v2 / standard_v2), not 4; the script
#: adapts automatically since it just globs whatever *.jsonl files exist.
SOURCE_DIR = EVAL_ROOT.parent / "temp_p8_improved"
OUT_PATH = Path(__file__).parent / "gold_layer_a_scaffold.jsonl"

#: One image per (arm, casualty) stratum, chosen deterministically by
#: smallest image filename within the casualty -- reproducible without a
#: random seed, and stable across reruns as long as the corpus doesn't
#: change. 4 arms x 4 casualties = 16 images, ~9.4 steps/plan average
#: (measured, see memory: castor-plans-have-no-magnitudes.md) -> ~150
#: steps, close to the design plan's "~120" target.
CASUALTIES = ("aground", "capsized", "sunken", "on_fire")

#: Coarse keyword -> tool heuristic for the DRAFT guess only. Deliberately
#: the same shape as the ad-hoc regex used during corpus measurement
#: earlier in this project, not a real extractor -- calibrate.py's whole
#: purpose is to replace this with a calibrated LLM.
_TOOL_KEYWORDS = {
    "survey_hull": r"hull survey|survey.*hull|inspect.*hull|hull.*inspect|assess.*hull|hull.*(condition|breach|integrity|damage|contact)",
    "sound_tanks": r"tank.*(check|sound)|fuel tanks?.*check|check.*(fuel|tank)|account.*(cargo|manifest)|cargo.*manifest",
    "read_draft_marks": r"draft mark",
    "survey_seabed": r"seabed|substrate",
    "account_for_personnel": r"account.*(personnel|crew)|personnel.*account",
    "size_up_fire": r"size.?up|assess.*fire",
    "equalize_pressure": r"equaliz",
    "test_atmosphere": r"(explosive|toxic|oxygen).*gas|gas.*test|atmosphere test|oxygen deficiency|test.*for oxygen",
    "calculate_ground_reaction": r"ground reaction",
    "calculate_freeing_force": r"freeing force|pulling force (needed|required)",
    "calculate_stability": r"metacentric|righting.?arm|stability (check|criteria)",
    "lighter_cargo": r"lighter(ing)?",
    "offload_fuel": r"fuel.*(offload|remov)|offload.*fuel",
    "dredge": r"dredg",
    "remove_impalement": r"impal",
    "rig_beach_gear": r"beach gear",
    "attach_tug": r"\btug",
    "pull": r"\bpull|refloat|free the vessel",
    "rescue_crew": r"rescue|evacuat",
    "dewater": r"dewater|pump out|pump(s|ing)?.*(remove|to remove).*water|remove water|trapped water|clear.*water.*(hull|compartment)",
    "press_full": r"press.*(full|tank)",
    "rig_parbuckling": r"parbuckl",
    "right_vessel": r"right(ing)? the vessel|controlled righting|begin.*righting",
    "patch_hull": r"patch",
    "install_cofferdam": r"cofferdam|watertight bulkhead|inflatable barrier",
    "blow_tanks": r"blow.*tank|compressed air",
    "lift": r"lift bag|crane lift|crane.and.barge|sectional removal|heavy.?lift|floating crane|crane vessel",
    "cut_section": r"\bcut\b|sectional removal",
    "sonar_search": r"sonar",
    "deploy_boom": r"\bboom",
    "skim": r"\bskim",
    "apply_dispersant": r"dispersant",
    "recover_oil": r"recover.*oil|oil.*recover",
    "conduct_neba": r"\bneba\b",
    "muster_personnel": r"personnel (clear|muster)|clear of personnel|muster|clear of all personnel",
    "seal_boundaries": r"seal.*(boundar|space)",
    "confirm_pump_capacity": r"pump capacity|pump.{0,45}capacity|capacity.{0,45}pump",
    "boundary_cool": r"boundary cool",
    "release_co2": r"\bco2\b|co₂|carbon dioxide",
    "apply_foam": r"\bfoam\b",
    "activate_predischarge_alarm": r"pre.?discharge|alarm",
    "open_space": r"open.*(space|hold)|reopen",
    "confirm_fire_out": r"fire.*(out|extinguish)|confirm.*fire",
    "dewater_firefighting_water": r"firefighting water",
    "post_operation_assessment": r"post.?(fire|operation|salvage).*assess|final inspection|post.?refloat.*assess",
    "tow": r"\btow\b",
    "monitor_tide": r"tide gauge|monitor.*tide",
}
_COMPILED = {name: re.compile(pat, re.IGNORECASE) for name, pat in _TOOL_KEYWORDS.items()}

#: Meta-planning guard: "develop a plan" / "coordinate to plan" names no
#: concrete action, regardless of what equipment gets mentioned inside the
#: plan being described (e.g. "develop a plan that includes...patching
#: materials" is not itself a patch_hull call). Checked BEFORE keyword
#: matching -- see the audit finding that "If the vessel is deemed
#: salvageable, develop a detailed salvage plan that includes..." was
#: mislabeled patch_hull off an incidental keyword deep in the description
#: of what the (not-yet-written) plan would contain.
_META_PLANNING_RE = re.compile(
    r"\b(develop|create|formulate|draft)\s+(a|an|the)\s+(detailed\s+)?"
    r"(salvage\s+|recovery\s+|rescue\s+)?plan\b",
    re.IGNORECASE,
)

#: Sentence boundary -- a period/!/? followed by whitespace. Deliberately
#: simple (no abbreviation handling); good enough for plan-step prose.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


#: Splits a step into (main_clause, rest). A step like "Deploy a tugboat
#: ... to lift the vessel ... and refloat it" names two things at once: the
#: concrete commitment (main clause -- what is actually deployed) and the
#: goal it's for (purpose clause -- why). Per the multi_action policy in
#: salvage_plan_checker.md sec.9 / design plan sec.4c, the tool named in the
#: main clause is `expected_tool`; anything only named after the boundary
#: goes to `expected_secondary_tools`.
#:
#: Two DIFFERENT kinds of boundary are recognised, and they are NOT treated
#: the same way (this distinction is the fix for a confirmed audit bug --
#: "Begin controlled flooding or ballasting operations if the vessel is
#: deemed stable enough...for potential lifting or refloating" was wrongly
#: promoted to `pull` because "refloating" appears inside the "if" clause):
#:   purpose   ("to ...", "in order to ...")  -- may be promoted to primary
#:             when the main clause is empty (the "pumps to remove water"
#:             pattern: the purpose clause names the real action).
#:   condition ("if ...", "unless ...", "should ...", "once ...") -- NEVER
#:             promoted to primary. A hit found only inside a hypothetical
#:             or conditional clause is not a commitment.
_PURPOSE_CLAUSE_RE = re.compile(
    r"\b(to|in order to|which will|so (?:as|that) to)\b", re.IGNORECASE
)
_CONDITION_CLAUSE_RE = re.compile(r"\b(if|unless|should|once)\b", re.IGNORECASE)


def _find_boundary(text: str):
    """Earliest of a purpose-clause or condition-clause marker in `text`.
    Returns (position, 'purpose'|'condition') or (None, None)."""
    candidates = []
    m = _PURPOSE_CLAUSE_RE.search(text)
    if m:
        candidates.append((m.start(), "purpose"))
    m = _CONDITION_CLAUSE_RE.search(text)
    if m:
        candidates.append((m.start(), "condition"))
    if not candidates:
        return None, None
    return min(candidates, key=lambda c: c[0])


def _hits_in(text: str) -> list:
    return [name for name, pat in _COMPILED.items() if pat.search(text)]


def _strip_fronted_clause(text: str) -> tuple:
    """Peel off a LEADING condition clause ("Once the vessel is stabilized,
    conduct...", "If the fire is out, reassess...") ending at its comma.

    This is structurally different from a trailing condition ("Begin
    flooding if the vessel is stable") -- a fronted clause's TRUE main
    clause is what comes AFTER the comma, whereas a trailing condition's
    main clause is what comes BEFORE the marker. Treating both the same way
    (as _find_boundary alone does) puts a fronted sentence's own main verb
    into `rest_text`, unreachable except via the low-confidence promotion
    paths -- confirmed audit bug on "Once the vessel is stabilized..., a
    conduct a final inspection..." Returns (fronted_clause_or_None,
    remainder) -- remainder is `text` unchanged if no fronted clause found.
    """
    m = _CONDITION_CLAUSE_RE.match(text.lstrip())
    if not m:
        return None, text
    comma_pos = text.find(",")
    if comma_pos == -1:
        return None, text
    return text[:comma_pos + 1], text[comma_pos + 1:].lstrip()


def _draft_tool(step_text: str) -> tuple:
    """Best-effort heuristic guess: (primary_tool_or_no_match, secondary_tools, confidence).

    Scoped to the FIRST SENTENCE for primary/purpose-clause determination --
    a confirmed audit bug was a keyword three sentences later (e.g.
    "...coordinate with tugboats..." in a step's third sentence) getting
    treated as if it were the first sentence's purpose clause, because the
    old version split the whole step_text on the first "to" with no
    sentence boundary at all. Later sentences still contribute to
    `secondary`, just never to `primary` directly (see the `later_hits`
    branch below) -- promoting from a sentence the main clause never even
    mentions is exactly the kind of over-confident guess this rewrite
    exists to stop making silently.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(step_text.strip()) if s]
    first = sentences[0] if sentences else step_text

    # Peel off a FRONTED condition clause ("Once X, conduct Y...") before
    # any boundary analysis -- its true main clause is what follows the
    # comma, not what the marker-position logic below would otherwise sweep
    # into `rest_text`. See _strip_fronted_clause docstring.
    fronted, first = _strip_fronted_clause(first)
    fronted_hits = _hits_in(fronted) if fronted else []

    if _META_PLANNING_RE.search(first):
        # "develop a plan that includes patching materials..." -- naming no
        # concrete action, regardless of what equipment gets mentioned
        # inside the plan being described. See _META_PLANNING_RE docstring.
        return "no_match", [], "no_match_confident"

    boundary_pos, marker_type = _find_boundary(first)
    if boundary_pos is None:
        main_text, rest_text = first, ""
    else:
        main_text, rest_text = first[:boundary_pos], first[boundary_pos:]

    main_hits = _hits_in(main_text)
    rest_hits = [h for h in _hits_in(rest_text) if h not in main_hits] if rest_text else []

    later_hits = []
    for s in sentences[1:]:
        for h in _hits_in(s):
            if h not in main_hits and h not in rest_hits and h not in later_hits:
                later_hits.append(h)

    if main_hits:
        # A real hit in the sentence's own main clause -- the reliable case.
        secondary = main_hits[1:] + rest_hits + later_hits + fronted_hits
        confidence = "medium" if len(main_hits) == 1 else "low"
        return main_hits[0], secondary, confidence

    if marker_type == "purpose" and rest_hits:
        # "Deploy pumps to remove water" -- promote, but ALWAYS low
        # confidence. This same shape ("Deploy X to Y") also covers "Deploy
        # a barge to support the tugboats", a genuine coverage gap where
        # promoting to attach_tug would be wrong -- no keyword heuristic
        # can tell these apart, so never let this path masquerade as
        # "medium" (confirmed audit finding: it was silently wrong roughly
        # as often as it was right).
        return rest_hits[0], rest_hits[1:] + later_hits + fronted_hits, "low"

    if later_hits:
        # Nothing in the first sentence at all; a later sentence names
        # something. Weaker promotion source than even the purpose-clause
        # case above -- always low.
        return later_hits[0], later_hits[1:] + rest_hits + fronted_hits, "low"

    # A hit found ONLY inside a condition clause (trailing "if X" or the
    # stripped-off fronted "Once X,") and nowhere else in the step is
    # deliberately NOT promoted to primary. Earlier this branch surfaced
    # such hits as a low-confidence guess on the theory that "surfacing is
    # better than hiding" -- but the one confirmed audit example of this
    # exact shape ("Begin controlled flooding or ballasting operations if
    # the vessel is deemed stable enough...for potential lifting or
    # refloating") was wrong: "refloating" lives only in the hypothetical
    # tail, the real action (ballasting) has no tool in the vocabulary, and
    # promoting produced a confident-looking `pull` for a genuine coverage
    # gap. A conditional clause is not a commitment; falling through to
    # no_match_confident here still gets audited (see the no_match tier's
    # own review pass), just through that queue instead of "low".

    return "no_match", [], "no_match_confident"


def _select_images(records: list) -> dict:
    """{casualty -> image} for this arm file, smallest filename per casualty."""
    by_casualty = {c: [] for c in CASUALTIES}
    for r in records:
        image = r.get("image", "")
        casualty = image.split("/")[0] if "/" in image else ""
        if casualty in by_casualty:
            by_casualty[casualty].append(image)
    return {c: sorted(imgs)[0] for c, imgs in by_casualty.items() if imgs}


def main():
    reg = ToolRegistry.load()
    out_records = []
    # answers_*.jsonl only -- SOURCE_DIR also contains judge_scores_improved.jsonl
    # (judge OUTPUT, no `text` plan field) which a bare *.jsonl glob would
    # wrongly pick up as a 4th "arm".
    files = sorted(SOURCE_DIR.glob("answers_*.jsonl"))
    if not files:
        print(f"ERROR: no jsonl files in {SOURCE_DIR}", file=sys.stderr)
        sys.exit(1)

    n_gaps = n_failed = 0
    for f in files:
        arm = f.stem
        recs = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
        by_image = {r.get("image", ""): r for r in recs}
        selected = _select_images(recs)

        for casualty, image in selected.items():
            plan_text = by_image[image].get("text", "")
            steps, parse_quality = parse_steps_v2(plan_text, source_id=f"{arm}:{image}")
            if parse_quality == "gaps":
                n_gaps += 1
            elif parse_quality == "failed":
                n_failed += 1
            gates_in_plan = detect_gates(plan_text)
            gate_spans = [g.span for g in gates_in_plan]

            for step_num, step_text in steps:
                tool_guess, secondary_guess, confidence = _draft_tool(step_text)
                step_gates = [g for g in gates_in_plan if g.condition_text[:40] in step_text[:200]]
                conditional_guess = bool(step_gates)
                condition_var_guess = step_gates[0].condition_var if step_gates else "none"

                out_records.append({
                    "arm": arm,
                    "casualty": casualty,
                    "image": image,
                    "step_num": step_num,
                    "step_text": step_text,
                    "parse_quality": parse_quality,
                    "expected_tool": tool_guess,
                    "expected_secondary_tools": secondary_guess,
                    "expected_params": {},
                    "expected_conditional": conditional_guess,
                    "expected_condition_var": condition_var_guess,
                    "expected_family": reg.family(tool_guess) if reg.has(tool_guess) else "no_match",
                    "confidence": confidence,
                    "reviewed": False,
                })

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
        encoding="utf-8",
    )
    import collections
    counts = collections.Counter(r["confidence"] for r in out_records)
    print(f"{len(out_records)} step records across {len(files)} arms x {len(CASUALTIES)} casualties -> {OUT_PATH}")
    print(f"  medium (single clean hit): {counts['medium']}")
    print(f"  no_match_confident (zero hits, likely filler): {counts['no_match_confident']}")
    print(f"  low (genuine multi-hit ambiguity -- review these): {counts['low']}")
    print(f"  parse_steps_v2 quality: {n_gaps} plans 'gaps', {n_failed} plans 'failed' "
          f"(of {len(files)*len(CASUALTIES)} selected)")


if __name__ == "__main__":
    main()
