"""
Tests for pipelines/plan_adequacy/vocab.py and registry/tools.json integrity.

The registry (registry/tools.json, registry/routes.json) is hand-edited
domain content, not code -- it needs the same dangling-reference checks a
schema would give. These tests catch the typo class that would otherwise
surface as a silent wave of false SEQUENCE_VIOLATIONs (a `requires` fact
that no tool ever establishes can never be satisfied).
Run: python -m pytest tests/test_plan_adequacy_vocab.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.paths import ROUTES_PATH, TOOLS_PATH
from pipelines.plan_adequacy.vocab import ToolRegistry


# ── loading ──────────────────────────────────────────────────────────────────

def test_registry_loads():
    reg = ToolRegistry.load()
    assert len(reg.all_tool_names()) > 0


def test_registry_has_46_documented_plus_gap_fill_tools():
    # 46 tools per salvage_plan_checker.md sec.9, plus monitor_tide (gap fill
    # for the tide_refloat route -- see tools.json's monitor_tide entry).
    reg = ToolRegistry.load()
    assert len(reg.all_tool_names()) == 47


# ── every tool belongs to exactly one family, has is_assessment set ─────────

def test_every_tool_has_a_family():
    reg = ToolRegistry.load()
    for name in reg.all_tool_names():
        assert reg.family(name), f"{name} has no family"


def test_assessment_tools_are_flagged_is_assessment():
    reg = ToolRegistry.load()
    assessment = reg.assessment_tools()
    assert assessment == reg.tools_in_family("assessment")
    assert len(assessment) >= 10


# ── registry integrity: no dangling fact references ─────────────────────────

def test_every_tool_requires_fact_is_establishable():
    """Every fact named in some tool's `requires` must be establish-able by
    at least one tool's `establishes` or `effects` -- otherwise that
    precondition can never be satisfied and the step is permanently
    unreachable."""
    reg = ToolRegistry.load()
    establishable = reg.all_established_facts()
    raw = json.loads(TOOLS_PATH.read_text(encoding="utf-8"))
    problems = []
    for t in raw["tools"]:
        for fact in t.get("requires", []):
            if fact not in establishable:
                problems.append((t["name"], fact))
    assert not problems, f"requires facts with no establishing tool: {problems}"


def test_every_route_precondition_is_establishable():
    reg = ToolRegistry.load()
    establishable = reg.all_established_facts()
    raw = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
    problems = []
    for casualty, routes in raw["routes"].items():
        for r in routes:
            for fact in r.get("preconditions", []):
                if fact not in establishable:
                    problems.append((casualty, r["name"], fact))
    assert not problems, f"route preconditions with no establishing tool: {problems}"


def test_every_route_core_tool_exists_in_tools_json():
    reg = ToolRegistry.load()
    all_tools = reg.all_tool_names()
    raw = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
    problems = []
    for casualty, routes in raw["routes"].items():
        for r in routes:
            for tool in r.get("core_tools", []):
                if tool not in all_tools:
                    problems.append((casualty, r["name"], tool))
    assert not problems, f"route core_tools referencing unknown tools: {problems}"


def test_every_tool_has_a_source_citation():
    raw = json.loads(TOOLS_PATH.read_text(encoding="utf-8"))
    missing = [t["name"] for t in raw["tools"] if not t.get("source", "").strip()]
    assert not missing, f"tools with no source field: {missing}"


def test_every_route_has_a_source_citation():
    raw = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
    missing = []
    for casualty, routes in raw["routes"].items():
        for r in routes:
            if not r.get("source", "").strip():
                missing.append((casualty, r["name"]))
    assert not missing, f"routes with no source field: {missing}"


def test_every_casualty_has_at_least_one_route():
    raw = json.loads(ROUTES_PATH.read_text(encoding="utf-8"))
    for casualty in ("aground", "capsized", "sunken", "on_fire"):
        assert casualty in raw["routes"]
        assert len(raw["routes"][casualty]) >= 1
