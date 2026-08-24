"""
Tests for pipelines/plan_adequacy/extract.py

Tests the pure prompt-building and response-parsing logic without vLLM --
_run_vllm_batch is never called here, matching the convention in
test_run_judge.py and test_salvage_extract.py (model calls replaced with
fake data, only pure logic exercised).
Run: python -m pytest tests/test_plan_adequacy_extract.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.extract import (
    _strip_wrapper_artifacts,
    build_system_prompt,
    build_user_prompt,
    parse_extraction,
)
from pipelines.plan_adequacy.vocab import ToolRegistry, build_guided_json_schema


def _reg():
    return ToolRegistry.load()


# ── prompt building ───────────────────────────────────────────────────────

def test_system_prompt_lists_every_registry_tool():
    reg = _reg()
    sp = build_system_prompt(reg)
    for name in reg.all_tool_names():
        assert name in sp, f"{name} missing from system prompt"


def test_system_prompt_includes_no_match():
    assert "no_match" in build_system_prompt(_reg())


def test_user_prompt_embeds_step_text_and_number():
    up = build_user_prompt("aground", 4, "Deploy two harbor tugs.")
    assert "Deploy two harbor tugs." in up
    assert "4" in up
    assert "aground" in up


def test_user_prompt_handles_missing_casualty():
    up = build_user_prompt("", 1, "Survey the hull.")
    assert "unknown" in up


# ── response parsing ──────────────────────────────────────────────────────

def test_parse_extraction_well_formed():
    raw = {
        "tool": "attach_tug", "params": {"count": 2, "shp": None},
        "secondary_tools": ["pull"], "conditional": False,
        "condition_text": None, "condition_var": "none",
    }
    call = parse_extraction(raw, 3, "Deploy two tugs and pull.")
    assert call.tool == "attach_tug"
    assert call.params == {"count": 2}  # None values dropped
    assert call.secondary_tools == ("pull",)


def test_parse_extraction_none_input_is_no_match():
    call = parse_extraction(None, 1, "some step")
    assert call.tool == "no_match"
    assert call.params == {}


def test_parse_extraction_missing_keys_defaults_safely():
    call = parse_extraction({"tool": "survey_hull"}, 1, "text")
    assert call.tool == "survey_hull"
    assert call.params == {}
    assert call.conditional is False
    assert call.condition_var == "none"
    assert call.secondary_tools == ()


def test_parse_extraction_malformed_params_type_is_ignored_not_raised():
    call = parse_extraction({"tool": "attach_tug", "params": "not a dict"}, 1, "text")
    assert call.params == {}


def test_parse_extraction_malformed_secondary_type_is_ignored_not_raised():
    call = parse_extraction({"tool": "attach_tug", "secondary_tools": "not a list"}, 1, "text")
    assert call.secondary_tools == ()


def test_parse_extraction_preserves_step_num_and_text():
    call = parse_extraction({"tool": "no_match"}, 7, "Coordinate with stakeholders.")
    assert call.step_num == 7
    assert call.step_text == "Coordinate with stakeholders."


# ── wrapper-artifact stripping ────────────────────────────────────────────
# Added after the calibration bake-off (2026-08-24) hit 100% parse failure
# against llama_3_3_70b with no exception anywhere in the vLLM logs --
# guided decoding is supposed to constrain the whole completion to the
# schema, but some models' native chat/tool-call formats can still wrap it.

def test_strip_wrapper_artifacts_plain_json_passthrough():
    raw = '{"tool": "attach_tug"}'
    assert _strip_wrapper_artifacts(raw) == raw


def test_strip_wrapper_artifacts_markdown_fence():
    raw = '```json\n{"tool": "attach_tug"}\n```'
    assert _strip_wrapper_artifacts(raw) == '{"tool": "attach_tug"}'


def test_strip_wrapper_artifacts_leading_prose():
    raw = 'Here is the JSON:\n{"tool": "attach_tug"}'
    assert _strip_wrapper_artifacts(raw) == '{"tool": "attach_tug"}'


def test_strip_wrapper_artifacts_trailing_prose():
    raw = '{"tool": "attach_tug"}\nLet me know if you need anything else.'
    assert _strip_wrapper_artifacts(raw) == '{"tool": "attach_tug"}'


def test_strip_wrapper_artifacts_no_braces_returns_unchanged():
    raw = 'no json here'
    assert _strip_wrapper_artifacts(raw) == raw


# ── schema/prompt consistency ─────────────────────────────────────────────

def test_schema_tool_enum_matches_system_prompt_tool_list():
    reg = _reg()
    schema = build_guided_json_schema(reg)
    schema_tools = set(schema["properties"]["tool"]["enum"])
    prompt_tools = {t for t in reg.all_tool_names() if t in build_system_prompt(reg)}
    assert schema_tools - {"no_match"} == prompt_tools
