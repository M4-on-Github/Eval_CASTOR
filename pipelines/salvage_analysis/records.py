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


class FieldExtractor:
    """Recovers one field from an inference record, however it was written.

    The same field arrives in three shapes depending on the prompt variant and
    how well the model complied, so they are tried in DESCENDING order of
    confidence:

      1. a top-level key on the record — already extracted upstream
      2. a key inside a JSON block in the model's text
      3. for recovery_considerations only, the prose under a "Salvage Plan"
         markdown heading

    Never raises. A field that cannot be found is None, and the caller decides
    whether that is fatal — P6 runs over many records and one malformed answer
    must not abort the analysis.

    The heading fallback is DELIBERATELY LIMITED to recovery_considerations.
    That field is long-form prose the model often writes as markdown rather
    than JSON, so a heading is a reliable marker. Applying the same fallback to
    a short field like `state` would let arbitrary prose masquerade as a value.
    """

    #: "## Salvage Plan", "**Salvage Plan**", "Salvage Plan:" and so on.
    HEADING_RE = _SALVAGE_PLAN_HEADING_RE

    #: The only field permitted the prose fallback — see the class docstring.
    PROSE_FALLBACK_FIELD = "recovery_considerations"

    @classmethod
    def salvage_plan_section(cls, text: str):
        """Text following a "Salvage Plan" heading, or None if absent."""
        match = cls.HEADING_RE.search(text)
        if match is None:
            return None
        section = text[match.end():].strip()
        return section or None

    @classmethod
    def get(cls, record: dict, field: str):
        """Return the field's value, or None. Never raises."""
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

        if field == cls.PROSE_FALLBACK_FIELD:
            # Last resort: if there is no heading either, use the whole answer.
            # Better to over-supply text to the element extractor than to drop
            # a record that plainly contains a plan.
            section = cls.salvage_plan_section(text)
            return section if section is not None else text

        return None


# ── Compatibility facade ─────────────────────────────────────────────────────

def _extract_salvage_plan_section(text: str):
    """Text after a "Salvage Plan" heading, or None. Facade."""
    return FieldExtractor.salvage_plan_section(text)


def get_field_text(record: dict, field: str):
    """Return record's field value, or None. Never raises. Facade."""
    return FieldExtractor.get(record, field)
