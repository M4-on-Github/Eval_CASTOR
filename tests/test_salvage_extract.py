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
    build_extract_prompt,
    build_output_record,
    parse_extract_result,
)


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
