"""
Tests for pipelines/salvage_analysis/records.py
Run: python -m pytest tests/test_salvage_records.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.salvage_analysis.records import get_field_text

# Mimics the real full-answer shape: chain-of-thought reasoning followed by
# a trailing JSON blob with LaTeX-style backslash-escaped keys, exactly as
# seen in tempp/answers_baseline.jsonl.
FULL_ANSWER_TEXT = (
    "Step 1 - Evidence Catalog: The bow is on the beach.\n"
    "Step 10 - Output:\n"
    "{\n"
    '"description": "A vessel aground on a beach.",\n'
    '"state": "aground",\n'
    '"vessel\\_type": "fishing boat",\n'
    '"size\\_estimate": "small (10m or less)",\n'
    '"cargo": null,\n'
    '"surroundings": "sandy beach with rocks",\n'
    '"recovery\\_considerations": "Careful navigation around the rocks and reef is necessary.",\n'
    '"confidence\\_scores": {"state_confidence": 0.6, "type_confidence": 0.6, "reasoning": "low clarity"}\n'
    "}"
)

TRUNCATED_TEXT = (
    "Step 1 - Evidence Catalog: The bow is on the beach.\n"
    "Step 10 - Output:\n"
    "{\n"
    '"description": "A vessel aground on a beach.",\n'
    '"state": "aground"'
    # deliberately truncated -- no closing brace
)

NO_JSON_TEXT = "The model just rambled without ever producing a JSON block."


def test_extracts_recovery_considerations():
    result = get_field_text({"text": FULL_ANSWER_TEXT}, "recovery_considerations")
    assert result == "Careful navigation around the rocks and reef is necessary."


def test_extracts_state():
    result = get_field_text({"text": FULL_ANSWER_TEXT}, "state")
    assert result == "aground"


def test_unescapes_backslash_underscore_keys():
    result = get_field_text({"text": FULL_ANSWER_TEXT}, "vessel_type")
    assert result == "fishing boat"


def test_missing_field_returns_none():
    result = get_field_text({"text": FULL_ANSWER_TEXT}, "not_a_real_field")
    assert result is None


def test_truncated_json_returns_none_without_raising():
    result = get_field_text({"text": TRUNCATED_TEXT}, "recovery_considerations")
    assert result is None


def test_no_json_block_returns_none_without_raising():
    result = get_field_text({"text": NO_JSON_TEXT}, "recovery_considerations")
    assert result is None


def test_missing_text_key_returns_none():
    result = get_field_text({}, "recovery_considerations")
    assert result is None
