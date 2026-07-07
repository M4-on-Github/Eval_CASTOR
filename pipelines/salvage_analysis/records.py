r"""
Shared field-extraction helper for Pipeline 6 (salvage plan analysis).

Full-answer inference records store the VLM's structured output as an
embedded JSON blob at the end of a chain-of-thought `text` string, with
LaTeX-style backslash-escaped keys (e.g. "recovery\_considerations").
Reuses shared/metrics.py's existing extraction logic rather than
re-implementing it.
"""

import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.metrics import extract_json_block


def get_field_text(record: dict, field: str):
    """Return record's embedded-JSON field value, or None if extraction or
    the field lookup fails. Never raises."""
    text = record.get("text")
    if not text:
        return None
    parsed, _reason = extract_json_block(text)
    if parsed is None:
        return None
    return parsed.get(field)
