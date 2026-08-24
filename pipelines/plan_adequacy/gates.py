"""
Decision-gate detection: the plan-adequacy metric with a validated regression
anchor (see test_plan_adequacy_gates.py).

Measured 2026-08-19 across all 110 plans in every p7_to_check/*.jsonl run file
(see memory: castor-plans-have-no-magnitudes.md): if/then decision gates per
plan discriminate arms 2.5x (ABLATION 2.39, CONTROL 4.24, IMPROVED 5.90,
self_verify 5.60), where the numeric-magnitude metric the original
salvage_plan_checker.md spec proposed measures 0.00 everywhere. This module
is the one that produced those numbers, kept regex-identical to the ad-hoc
analysis so the regression test is meaningful.

Two independent things live here:
  1. Gate DETECTION over raw plan text -- pure regex, no tool-call extraction
     needed. This is what produces gate_rate and self_contradiction_rate
     immediately, with no model in the loop (build order step 1).
  2. Gate RESOLUTION against an established WorldState -- takes a Gate found
     in an extracted step plus the knowledge tracker from worldstate.py, and
     decides CONDITIONAL_UNRESOLVED vs fall-through-to-normal-grading. This
     needs extracted tool calls (`conditional`, `condition_var`), so it is
     exercised with hand-written ToolCall lists (executor@oracle) before any
     model is involved -- see salvage_plan_checker.md section 3a.
"""

import re
from dataclasses import dataclass
from typing import Optional

# ── Part 1: pure-text gate detection ─────────────────────────────────────────

#: An if/then DECISION gate: the plan defers a choice to a condition rather
#: than committing. Kept narrower than a generic hedge-word scanner (which
#: also fires on register words like "may" used descriptively, e.g. "the
#: vessel may be a warship") -- these patterns require an explicit
#: conditional/deferral construction, matching the regex used to produce the
#: validated 2.39/4.24/5.90/5.60 numbers.
GATE_RE = re.compile(
    r"\b(if|should|unless|in the event|depending on whether|"
    r"where necessary|as (?:may be )?required)\b[^.]{0,90}",
    re.IGNORECASE,
)

#: A DEFINITE claim about vessel size, stated as fact rather than deferred.
DEFINITE_SIZE_RE = re.compile(
    r"(is|appears to be|estimated)\s+(a\s+)?(large|medium|small|medium-sized|large-sized)\b"
    r"|\b(>50m|<10m|10-50m|10–50m)",
    re.IGNORECASE,
)

#: A gate whose condition variable is vessel size specifically -- the
#: self-contradiction pattern: state a size as fact, then hedge on that same
#: fact later. "Larger vessels may ..." counts (it is the prompt's own
#: register, mirrored by the model) as well as explicit "if the vessel is
#: large" phrasing.
SIZE_GATE_RE = re.compile(
    r"(if|should)\s+(the\s+)?(vessel|ship|it)\s+(is|were|be)\s+"
    r"(large|larger|small|smaller|of\s+significant)"
    r"|larger\s+vessels?\s+may"
    r"|if\s+.{0,25}(size|draft)\s",
    re.IGNORECASE,
)

#: condition_var candidates a gate's condition text might name -- used both
#: by the (future, LLM-driven) extraction schema and by the lightweight
#: heuristic classifier below, so the two stay in the same vocabulary.
CONDITION_VARS = (
    "vessel_size", "draft", "substrate", "depth", "cargo_type", "damage", "none",
)

_CONDITION_VAR_KEYWORDS = {
    "vessel_size": re.compile(r"\b(large|larger|small|smaller|size|sized)\b", re.IGNORECASE),
    "draft": re.compile(r"\bdraft\b", re.IGNORECASE),
    "substrate": re.compile(r"\b(mud|sand|coral|rock|seabed|bottom)\b", re.IGNORECASE),
    "depth": re.compile(r"\bdepth|deep|shallow\b", re.IGNORECASE),
    "cargo_type": re.compile(r"\bcargo|hazmat|hazardous\b", re.IGNORECASE),
    "damage": re.compile(r"\bdamage[d]?|breach(ed)?\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class Gate:
    """One detected decision gate."""

    condition_text: str
    condition_var: str   # best-guess from CONDITION_VARS, "none" if unclear
    span: tuple           # (start, end) offsets into the source text


def guess_condition_var(condition_text: str) -> str:
    """Heuristic mapping from a condition clause to a CONDITION_VARS entry.
    Used as a fallback when extraction doesn't supply condition_var, and as
    the ground-truth heuristic for the pure-text detector below."""
    for var, pattern in _CONDITION_VAR_KEYWORDS.items():
        if pattern.search(condition_text):
            return var
    return "none"


def detect_gates(text: str) -> list:
    """All decision gates in a plan's raw text. Pure regex; no tool-call
    extraction required. This is what test_plan_adequacy_gates.py's
    regression anchor calls."""
    gates = []
    for m in GATE_RE.finditer(text):
        clause = m.group(0)
        gates.append(Gate(
            condition_text=clause.strip(),
            condition_var=guess_condition_var(clause),
            span=m.span(),
        ))
    return gates


def gate_rate(text: str) -> int:
    """Decision gates in one plan. A per-plan COUNT (matching how the
    regression numbers were produced: gates/plan is this summed and divided
    by len(records), not normalized per-step here)."""
    return len(GATE_RE.findall(text))


def has_definite_size_claim(text: str) -> bool:
    return bool(DEFINITE_SIZE_RE.search(text))


def has_size_gate(text: str) -> bool:
    return bool(SIZE_GATE_RE.search(text))


def is_self_contradictory_on_size(text: str) -> bool:
    """True if the plan both states a definite vessel size AND later gates
    on vessel size -- the sharpest hedging signature: the model already
    established the fact, then hedged on it anyway. Measured 10/110 on the
    IMPROVED arm."""
    return has_definite_size_claim(text) and has_size_gate(text)


# ── Part 2: gate resolution against established knowledge ───────────────────
# See salvage_plan_checker.md section 3a. Operates on an extracted step's
# `conditional`/`condition_var` fields plus a worldstate.WorldState -- both
# only exist once a ToolCall exists (hand-written for executor@oracle, or
# LLM-extracted at scale). No text regex involved past this point.


#: condition_var (the human-readable extraction-schema vocabulary in
#: CONDITION_VARS) -> the fact identifier worldstate.py / registry/tools.json
#: actually track. Kept as an explicit table rather than requiring the two
#: vocabularies to share spelling, since "depth" and "cargo_type" don't map
#: 1:1 onto any single tool's `establishes` name.
_CONDITION_VAR_TO_FACT = {
    "vessel_size": "vessel_size",
    "draft": "draft",
    "substrate": "substrate",
    "depth": "wreck_located",       # sonar_search establishes depth-relevant location knowledge
    "cargo_type": "tank_contents",  # sound_tanks is what reveals cargo/hazmat contents
    "damage": "hull_condition",     # survey_hull is what reveals damage
}


def condition_var_to_fact(condition_var: str) -> str:
    """Public accessor for _CONDITION_VAR_TO_FACT -- used by executor.py's
    unused-assessment check to know which fact a resolved gate actually
    consumed, without reaching into a private module dict."""
    return _CONDITION_VAR_TO_FACT.get(condition_var, condition_var)


def resolve_conditional(condition_var: str, world_state) -> str:
    """Classify one conditional call against what the plan has established
    so far.

    Returns "resolved" if `condition_var` is already known (the executor
    then grades the resolved branch normally as SPECIFIED_ADEQUATE /
    SPECIFIED_INADEQUATE), or "unresolved" if not (verdict
    CONDITIONAL_UNRESOLVED regardless of which branch would have been
    correct). `condition_var == "none"` always resolves -- an extraction
    that could not identify a condition variable is not penalised for the
    ambiguity, since "none" only ever appears when the step is NOT flagged
    conditional in the first place (see vocab CONDITION_VARS).
    """
    if condition_var == "none":
        return "resolved"
    fact = _CONDITION_VAR_TO_FACT.get(condition_var, condition_var)
    if world_state.knows(fact):
        return "resolved"
    return "unresolved"
