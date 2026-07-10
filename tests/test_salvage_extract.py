"""
Tests for pipelines/salvage_analysis/extract.py (Stage 1) and the
extraction prompt files it depends on.
Run: python -m pytest tests/test_salvage_extract.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROMPTS_DIR = Path(__file__).parent.parent / "pipelines" / "salvage_analysis" / "prompts"


# ── prompt files ──────────────────────────────────────────────────────────────

def test_system_prompt_file_exists():
    p = PROMPTS_DIR / "salvage_extract_system.txt"
    assert p.exists(), f"Missing: {p}"
    assert len(p.read_text(encoding="utf-8").strip()) > 0


def test_user_prompt_file_exists():
    p = PROMPTS_DIR / "salvage_extract_user.txt"
    assert p.exists(), f"Missing: {p}"
    assert "{recovery_text}" in p.read_text(encoding="utf-8")


def test_user_prompt_formats_without_error():
    p = PROMPTS_DIR / "salvage_extract_user.txt"
    tmpl = p.read_text(encoding="utf-8")
    filled = tmpl.format(recovery_text="Call a fireboat and contain the spill.")
    assert "fireboat" in filled
    import re
    assert not re.search(r'\{[a-z_]+\}', filled), "Unfilled placeholder in template"


# ── pure functions in extract.py (no network) ─────────────────────────────────

from pipelines.salvage_analysis.extract import (
    _ELEMENTS_JSON_SCHEMA,
    _VLLM_MODEL_CONFIG,
    build_extract_prompt,
    build_output_record,
    clean_and_parse_json,
    parse_extract_result,
)


# ── vLLM model config / guided-decoding schema ────────────────────────────────

def test_elements_json_schema_matches_output_format():
    assert _ELEMENTS_JSON_SCHEMA["type"] == "object"
    assert _ELEMENTS_JSON_SCHEMA["properties"]["elements"]["type"] == "array"
    assert _ELEMENTS_JSON_SCHEMA["properties"]["elements"]["items"]["type"] == "string"
    assert _ELEMENTS_JSON_SCHEMA["required"] == ["elements"]


def test_qwen25_72b_uses_guided_json_not_a_reasoning_model():
    assert _VLLM_MODEL_CONFIG["qwen25_72b"]["guided_json"] is True
    assert _VLLM_MODEL_CONFIG["qwen25_72b"]["prefill"] is None


def test_qwen25_72b_uses_higher_temperature_than_deepseek_to_avoid_collapse():
    # Low-temperature guided decoding was observed to collapse onto
    # {"elements": []} even for inputs with real extractable content.
    assert _VLLM_MODEL_CONFIG["qwen25_72b"]["temperature"] > _VLLM_MODEL_CONFIG["deepseek_r1"]["temperature"]


def test_deepseek_r1_does_not_use_guided_json_reasoning_model():
    # Guided decoding would suppress <think> entirely if enabled here.
    assert _VLLM_MODEL_CONFIG["deepseek_r1"]["guided_json"] is False


def test_qwen25_72b_max_model_len_matches_judge_panel_awq_constraint():
    # Qwen AWQ leaves only ~165 KV cache blocks on 1 GPU -- must stay <= 2048,
    # same constraint as judge_panel/run_judge.py's _MODEL_CONFIG.
    assert _VLLM_MODEL_CONFIG["qwen25_72b"]["max_model_len"] <= 2048


def test_build_extract_prompt_embeds_recovery_text():
    prompt = build_extract_prompt("Call a fireboat and contain the spill.")
    assert "Call a fireboat and contain the spill." in prompt


def test_parse_extract_result_valid_list():
    result = parse_extract_result({"elements": ["fireboat", "containment boom"]})
    assert result == {"elements": ["fireboat", "containment boom"], "parse_ok": True}


def test_parse_extract_result_coerces_non_string_items():
    result = parse_extract_result({"elements": ["tug", 5]})
    assert result == {"elements": ["tug", "5"], "parse_ok": True}


def test_parse_extract_result_empty_list_is_valid():
    result = parse_extract_result({"elements": []})
    assert result == {"elements": [], "parse_ok": True}


def test_parse_extract_result_missing_elements_key():
    result = parse_extract_result({"other_key": "oops"})
    assert result == {"elements": [], "parse_ok": False}


def test_parse_extract_result_elements_not_a_list():
    result = parse_extract_result({"elements": "fireboat"})
    assert result == {"elements": [], "parse_ok": False}


def test_parse_extract_result_none_input_is_parse_failure():
    result = parse_extract_result(None)
    assert result == {"elements": [], "parse_ok": False}


def test_build_output_record_shape():
    rec = build_output_record("aground/00017.jpg", {"elements": ["fireboat"], "parse_ok": True})
    assert rec == {"image": "aground/00017.jpg", "raw_elements": ["fireboat"], "parse_ok": True}


def test_build_output_record_includes_raw_response_on_success_too():
    # parse_ok=True can still hide a content problem (e.g. guided decoding
    # collapsing onto an empty list for input with real content) -- raw
    # text must be logged regardless of parse_ok, not just on failure.
    rec = build_output_record("aground/00017.jpg", {"elements": ["fireboat"], "parse_ok": True}, raw_response="some raw text")
    assert rec["raw_response"] == "some raw text"


def test_build_output_record_includes_raw_response_on_failure():
    rec = build_output_record("aground/00018.jpg", {"elements": [], "parse_ok": False}, raw_response="malformed output here")
    assert rec["raw_response"] == "malformed output here"


def test_build_output_record_truncates_long_raw_response():
    long_text = "x" * 5000
    rec = build_output_record("aground/00018.jpg", {"elements": [], "parse_ok": False}, raw_response=long_text)
    assert len(rec["raw_response"]) == 2000


def test_build_output_record_no_raw_response_key_when_not_provided():
    rec = build_output_record("aground/00018.jpg", {"elements": [], "parse_ok": False})
    assert "raw_response" not in rec


# ── clean_and_parse_json (deepseek/vLLM raw response cleanup) ─────────────────

def test_clean_and_parse_json_plain_json():
    result = clean_and_parse_json('{"elements": ["fireboat"]}')
    assert result == {"elements": ["fireboat"]}


def test_clean_and_parse_json_strips_think_block():
    raw = '<think>\nreasoning about the plan\n</think>\n{"elements": ["crane"]}'
    result = clean_and_parse_json(raw)
    assert result == {"elements": ["crane"]}


def test_clean_and_parse_json_normalizes_byte_level_bpe_space_marker():
    # Some vLLM/tokenizer combos leak the raw byte-level BPE "preceding
    # space" marker (U+0120, 'Ġ') into decoded text instead of converting it
    # back to a real space -- this can corrupt JSON syntax itself, not just
    # readability, so it must be normalized before parsing.
    raw = '{"elements":Ġ["underwaterĠsalvage",Ġ"dryĠdock"]}'
    result = clean_and_parse_json(raw)
    assert result == {"elements": ["underwater salvage", "dry dock"]}


def test_clean_and_parse_json_strips_code_fence():
    raw = '```json\n{"elements": ["tug"]}\n```'
    result = clean_and_parse_json(raw)
    assert result == {"elements": ["tug"]}


def test_clean_and_parse_json_extracts_object_from_preamble_and_trailing_text():
    raw = 'Sure, here is the extraction:\n{"elements": ["diver"]}\nLet me know if you need more.'
    result = clean_and_parse_json(raw)
    assert result == {"elements": ["diver"]}


def test_clean_and_parse_json_returns_none_on_garbage():
    result = clean_and_parse_json("no json here at all")
    assert result is None


def test_clean_and_parse_json_returns_none_on_empty_string():
    result = clean_and_parse_json("")
    assert result is None
