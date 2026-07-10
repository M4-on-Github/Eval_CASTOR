r"""
Shared field-extraction helper for Pipeline 6 (salvage plan analysis).

Full-answer inference records store the VLM's structured output as an
embedded JSON blob at the end of a chain-of-thought `text` string, with
LaTeX-style backslash-escaped keys (e.g. "recovery\_considerations").
Reuses shared/metrics.py's existing extraction logic rather than
re-implementing it.

combine_shards.py-merged records (from separated-into-parts shard files)
already carry each field as a plain top-level string -- no CoT/JSON
unwrapping needed -- so those are checked first.

Some runs (e.g. answers_qwen3vl8b_*_neutral_assertions_*.jsonl) produce no
embedded JSON at all -- free-form markdown instead, a vessel-classification
paragraph followed by a "Salvage Plan" section. For recovery_considerations
only, _extract_salvage_plan_section() falls back to locating that section
directly, so the extractor sees just the planning content, not the
classification preamble -- this matters beyond tidiness: the preamble
states the record's own predicted state in plain text (e.g. "the vessel is
**aground**"), and Pipeline 6's whole point is testing whether extracted
elements correlate with state. Leaking the state label into what Stage 1
reads would contaminate that exact measurement. If no heading can be found
at all, get_field_text() falls back to the whole text as a last resort
(better than silently extracting nothing) -- prefer fixing
_SALVAGE_PLAN_HEADING_RE for a new heading style over relying on this.
"""

import re
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.metrics import extract_json_block

_SALVAGE_PLAN_HEADING_RE = re.compile(
    r'^[ \t]*#{0,6}[ \t]*\**[ \t]*salvage\s+plan\b[^\n]*\n+',
    re.IGNORECASE | re.MULTILINE,
)


def _extract_salvage_plan_section(text: str):
    """Locate a "Salvage Plan" markdown heading and return everything after
    it. Returns None if no such heading is found (caller falls back to the
    whole text as a last resort -- see module docstring)."""
    match = _SALVAGE_PLAN_HEADING_RE.search(text)
    if match is None:
        return None
    section = text[match.end():].strip()
    return section or None


def get_field_text(record: dict, field: str):
    """Return record's field value, or None if extraction or the field
    lookup fails. Never raises."""
    direct = record.get(field)
    if isinstance(direct, str):
        return direct

    text = record.get("text")
    if not text:
        return None

    parsed, _reason = extract_json_block(text)
    if parsed is not None:
        value = parsed.get(field)
        if value is not None:
            return value

    if field == "recovery_considerations":
        section = _extract_salvage_plan_section(text)
        return section if section is not None else text

    return None
