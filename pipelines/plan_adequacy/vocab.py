"""
Loader over registry/tools.json -- the P9 tool vocabulary.

All domain content (which tools exist, what facts each establishes/requires,
what world-state effects each has) lives in registry/tools.json as reviewable
data, not in this module. This file is a thin loader and query surface over
it, the same split assertion_coverage/check_assertions.py uses for
IMPROVED_assertion_registry.csv via AssertionRegistry.load().

Keeping the split matters here specifically because the registry is the
biggest content-authoring surface in this pipeline (47 tools x their fact
sets) and needs to be editable without touching Python -- see
salvage_plan_checker.md and the plan-adequacy design plan for the full
rationale.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipelines.plan_adequacy.paths import TOOLS_PATH


@dataclass(frozen=True)
class ToolCall:
    """One extracted (or hand-written) call: a plan step bound to a tool.

    This is the shared currency between extract.py (produces these from
    plan text), worldstate.py / executor.py (consume them to derive
    verdicts), and calibrate.py (compares them against gold_tool_calls.jsonl).
    Mirrors the extraction schema in salvage_plan_checker.md section 9: a
    flat object with nullable params plus the conditional/condition_var pair
    from gates.py, rather than a per-tool union type.
    """

    #: frozen=True gives this an auto-generated __hash__, but `params` below
    #: is a plain dict -- hashing an instance will raise TypeError. Nothing
    #: in this codebase hashes a ToolCall/ToolSpec today; if that ever
    #: changes (e.g. memoizing on one), address the dict field first.
    step_num: int
    step_text: str
    tool: str                      # a name in ToolRegistry, or "no_match"
    params: dict = field(default_factory=dict)   # {param_name: value|None}
    conditional: bool = False
    condition_text: Optional[str] = None
    condition_var: str = "none"    # one of gates.CONDITION_VARS
    secondary_tools: tuple = ()    # multi_action policy, see section 4c


@dataclass(frozen=True)
class ToolSpec:
    """One entry from registry/tools.json, as a typed object."""

    name: str
    family: str
    is_assessment: bool
    params: dict            # {param_name: type_str}
    establishes: frozenset  # facts this call, once made, makes knowable
    requires: frozenset     # facts that must already be known for this call to be legal
    effects: frozenset      # world-state facts this call makes true
    source: str = ""

    @classmethod
    def _from_json(cls, d: dict) -> "ToolSpec":
        return cls(
            name=d["name"],
            family=d["family"],
            is_assessment=bool(d.get("is_assessment", False)),
            params=dict(d.get("params", {})),
            establishes=frozenset(d.get("establishes", [])),
            requires=frozenset(d.get("requires", [])),
            effects=frozenset(d.get("effects", [])),
            source=d.get("source", ""),
        )


class ToolRegistry:
    """All 47 tools, loaded from registry/tools.json.

    Use ToolRegistry.load() rather than constructing directly -- the
    classmethod is what resolves the default path and does the JSON parse.
    Instances are cheap to hold once loaded; callers should load once per
    process, not per call.
    """

    def __init__(self, tools: dict, fact_universe: frozenset):
        self._tools = tools                  # {name: ToolSpec}
        self._fact_universe = fact_universe   # every fact declared in facts.list

    @classmethod
    def load(cls, path: Path = TOOLS_PATH) -> "ToolRegistry":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tools = {t["name"]: ToolSpec._from_json(t) for t in raw["tools"]}
        fact_universe = frozenset(raw.get("facts", {}).get("list", []))
        return cls(tools, fact_universe)

    def spec(self, name: str) -> ToolSpec:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def family(self, name: str) -> str:
        return self._tools[name].family

    def all_tool_names(self) -> set:
        return set(self._tools.keys())

    def assessment_tools(self) -> set:
        return {n for n, t in self._tools.items() if t.is_assessment}

    def tools_in_family(self, family: str) -> set:
        return {n for n, t in self._tools.items() if t.family == family}

    def all_established_facts(self) -> set:
        """Union of every tool's `establishes` and `effects` -- every fact
        that SOME tool can actually make known/true. Used by the
        registry-integrity test to catch a `requires`/precondition that
        references a fact no tool ever establishes (an unreachable
        precondition -- see test_plan_adequacy_vocab.py)."""
        facts = set()
        for t in self._tools.values():
            facts |= t.establishes
            facts |= t.effects
        return facts

    def fact_universe(self) -> set:
        """The full declared fact vocabulary (facts.list in tools.json),
        independent of whether anything currently establishes it."""
        return set(self._fact_universe)


# ── Compatibility facade ─────────────────────────────────────────────────────

_DEFAULT_REGISTRY = None


def _default() -> ToolRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ToolRegistry.load()
    return _DEFAULT_REGISTRY


def tool_family(name: str) -> str:
    return _default().family(name)


def assessment_tools() -> set:
    return _default().assessment_tools()


#: The extraction-schema vocabulary for `condition_var` -- see gates.py
#: CONDITION_VARS. Duplicated here (not imported) to avoid a vocab<->gates
#: circular import; gates.py's `_CONDITION_VAR_TO_FACT` mapping is the
#: authority on what each of these actually resolves to.
_CONDITION_VARS = (
    "vessel_size", "draft", "substrate", "depth", "cargo_type", "damage", "none",
)


def build_guided_json_schema(registry: "ToolRegistry" = None) -> dict:
    """The flat guided-JSON schema extract.py's guided decoding is
    constrained to -- one object per plan step. Generated from the
    registry rather than hand-duplicated, so adding a tool to
    registry/tools.json automatically appears in the `tool` enum and the
    `params` object without a second edit here.

    Deliberately a FLAT schema (one `tool` enum + one `params` object
    covering every tool's params, all nullable) rather than a per-tool
    `oneOf` union -- cheaper to compile under guided decoding and better
    supported by the XGrammar backend. See salvage_plan_checker.md sec.9
    and the design plan sec.4a (S1 flat vs S2 two-stage) for the schema
    design rationale; this schema IS "S1 flat".
    """
    reg = registry or _default()
    tool_names = sorted(reg.all_tool_names()) + ["no_match"]

    # Union of every tool's params, deduplicated. Each becomes a nullable
    # property on the shared `params` object -- the executor only reads the
    # subset relevant to whichever `tool` was actually extracted.
    param_names = set()
    for name in reg.all_tool_names():
        param_names |= set(reg.spec(name).params.keys())

    # anyOf, not the "type": [...] shorthand -- calibration against
    # glm4_32b (2026-08-24) showed null_fidelity pinned at EXACTLY 0.0
    # across two runs, unmoved by twelve explicit prompt examples of
    # correct null usage. That flatness (not "usually wrong", literally
    # ALWAYS wrong) is the signature of a structurally unreachable value,
    # not a prompt-following failure -- some guided-decoding grammar
    # compilers (this cluster runs vLLM 0.8.5) don't fully support the
    # type-array shorthand and can silently collapse it to a single type,
    # making null impossible to emit regardless of instructions. anyOf is
    # the more universally supported way to express the same union.
    _NULLABLE_STRING_OR_NUMBER = {
        "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}]
    }
    _NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}

    params_properties = {p: dict(_NULLABLE_STRING_OR_NUMBER) for p in sorted(param_names)}

    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": tool_names},
            "params": {
                "type": "object",
                "properties": params_properties,
                "additionalProperties": False,
            },
            "secondary_tools": {
                "type": "array",
                "items": {"type": "string", "enum": tool_names},
            },
            "conditional": {"type": "boolean"},
            "condition_text": dict(_NULLABLE_STRING),
            "condition_var": {"type": "string", "enum": list(_CONDITION_VARS)},
        },
        "required": ["tool", "params", "secondary_tools", "conditional",
                     "condition_text", "condition_var"],
        "additionalProperties": False,
    }
