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

# Mimics the free-form (no embedded JSON) shape some runs actually produce
# (e.g. answers_qwen3vl8b_*_neutral_assertions_*.jsonl): a vessel-classification
# narrative, then a "Salvage Plan" markdown section with the real content.
FREE_FORM_TEXT = (
    "Based on the visual evidence, the vessel is aground. It is a military "
    "vessel, large, with unknown cargo, likely aground.\n\n"
    "---\n\n"
    "### Salvage Plan: Aground Naval Vessel\n\n"
    "#### 1. Initial Assessment & Safety\n"
    "- Secure the scene and establish a safety perimeter.\n"
    "- Deploy containment booms and a dive team.\n\n"
    "#### 2. Refloating Operations\n"
    "- Use tugs and beach gear to pull the vessel free."
)

FREE_FORM_TEXT_NO_HEADING = (
    "Based on the visual evidence, the vessel is aground. There is no clear "
    "plan section in this response at all, just a classification narrative."
)

# Real records were observed using a bold-only heading with no '#' at all --
# e.g. "**Salvage Plan: Aground Vessel (Rustic, Large Merchant Ship)**".
FREE_FORM_TEXT_BOLD_ONLY_HEADING = (
    "Based on the visual evidence, the vessel is aground. It is a large "
    "merchant vessel.\n\n---\n\n"
    "**Salvage Plan: Aground Vessel (Rustic, Large Merchant Ship)**\n\n"
    "**1. Initial Assessment & Safety**\n"
    "- **Site Survey**: Deploy a dive and survey team to assess stability.\n"
    "- **Tugs**: Position tugs for potential towing operations."
)

FREE_FORM_TEXT_MINIMAL_BOLD_HEADING = (
    "Based on the visual evidence, the vessel is aground.\n\n"
    "**Salvage Plan:**\n\n"
    "Tugs and beach gear will be used to refloat the vessel."
)


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


def test_truncated_json_falls_back_to_whole_text_without_raising():
    # No JSON, no heading -- whole-text last resort, not a crash.
    result = get_field_text({"text": TRUNCATED_TEXT}, "recovery_considerations")
    assert result == TRUNCATED_TEXT


def test_truncated_json_state_field_still_returns_none():
    # The whole-text fallback is scoped to recovery_considerations only.
    result = get_field_text({"text": TRUNCATED_TEXT}, "state")
    assert result is None


def test_no_json_block_falls_back_to_whole_text_without_raising():
    result = get_field_text({"text": NO_JSON_TEXT}, "recovery_considerations")
    assert result == NO_JSON_TEXT


def test_missing_text_key_returns_none():
    result = get_field_text({}, "recovery_considerations")
    assert result is None


def test_reads_merged_shard_record_directly_without_json_unwrap():
    # combine_shards.py output shape: plain top-level field keys, no `text`
    # blob to unwrap.
    merged_record = {"image": "a.jpg", "state": "aground", "recovery_considerations": "call a fireboat"}
    assert get_field_text(merged_record, "recovery_considerations") == "call a fireboat"
    assert get_field_text(merged_record, "state") == "aground"


def test_merged_record_missing_field_falls_back_to_none():
    merged_record = {"image": "a.jpg", "state": "aground"}
    assert get_field_text(merged_record, "recovery_considerations") is None


# ── free-form (no embedded JSON) fallback for recovery_considerations ────────

def test_free_form_text_falls_back_to_salvage_plan_section():
    result = get_field_text({"text": FREE_FORM_TEXT}, "recovery_considerations")
    assert result is not None
    assert "Initial Assessment" in result
    assert "tugs and beach gear" in result


def test_free_form_fallback_excludes_classification_preamble():
    # Only the planning section should be extracted -- not the vessel-type/
    # state classification narrative that precedes it.
    result = get_field_text({"text": FREE_FORM_TEXT}, "recovery_considerations")
    assert "Based on the visual evidence" not in result
    assert "military vessel" not in result


def test_free_form_fallback_does_not_apply_to_other_fields():
    # The free-form fallback is scoped to recovery_considerations only --
    # state/vessel_type etc. still require the JSON-block format.
    result = get_field_text({"text": FREE_FORM_TEXT}, "state")
    assert result is None


def test_free_form_text_without_any_plan_heading_falls_back_to_whole_text():
    # No detectable heading -- fall back to the whole text rather than
    # giving up, since some heading styles may not be anticipated. This is
    # a last resort (see module docstring on why heading-based isolation is
    # preferred: it keeps the state-classification sentence out of what the
    # extractor sees, which matters for Pipeline 6's own statistical tests).
    result = get_field_text({"text": FREE_FORM_TEXT_NO_HEADING}, "recovery_considerations")
    assert result == FREE_FORM_TEXT_NO_HEADING


def test_free_form_fallback_matches_bold_only_heading_no_hash():
    # Real records use "**Salvage Plan: ...**" with no '#' at all -- this
    # must match, not just '###'-style markdown headings.
    result = get_field_text({"text": FREE_FORM_TEXT_BOLD_ONLY_HEADING}, "recovery_considerations")
    assert result is not None
    assert "Initial Assessment" in result
    assert "Based on the visual evidence" not in result


def test_free_form_fallback_matches_minimal_bold_heading():
    result = get_field_text({"text": FREE_FORM_TEXT_MINIMAL_BOLD_HEADING}, "recovery_considerations")
    assert result == "Tugs and beach gear will be used to refloat the vessel."
