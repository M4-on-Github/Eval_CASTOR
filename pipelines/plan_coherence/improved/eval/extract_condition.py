"""
extract_condition.py — Extract the VLM's stated casualty type from plan text.

The VLM is prompted to begin its response with:
    "Casualty type: [aground / capsized / sunken / on fire]"

This module extracts that stated type so the pipeline can evaluate plans under
both (a) ground-truth state (from image path) and (b) model-predicted state.

Usage:
    from eval.extract_condition import extract_condition
    predicted = extract_condition(plan_text)
    # returns one of: 'aground' | 'capsized' | 'sunken' | 'on_fire' | 'unknown'
"""

import re

# Canonical state labels (match GT labels in human_gt.csv)
STATES = ["aground", "capsized", "sunken", "on_fire"]

# Aliases the model may use
_ALIASES: dict[str, str] = {
    "aground":   "aground",
    "grounded":  "aground",
    "grounding": "aground",
    "capsized":  "capsized",
    "capsizing": "capsized",
    "rolled over": "capsized",
    "rolled-over": "capsized",
    "sunken":    "sunken",
    "sunk":      "sunken",
    "submerged": "sunken",
    "underwater": "sunken",
    "on fire":   "on_fire",
    "on_fire":   "on_fire",
    "fire":      "on_fire",
    "burning":   "on_fire",
    "aflame":    "on_fire",
}

# Match the header line: "Casualty type: <value>"
_HEADER_RE = re.compile(
    r'casualty\s*type\s*[:\-]\s*(.+?)(?:\n|$)',
    re.IGNORECASE,
)


def extract_condition(text: str) -> str:
    """Return the VLM-predicted state from the plan header, or 'unknown'."""
    if not text:
        return "unknown"

    # Try the header line first
    m = _HEADER_RE.search(text)
    if m:
        raw = m.group(1).strip().lower().rstrip(".")
        # Direct alias match
        if raw in _ALIASES:
            return _ALIASES[raw]
        # Partial match against aliases
        for alias, canonical in _ALIASES.items():
            if alias in raw:
                return canonical

    # Fallback: scan first 300 chars for any state label
    snippet = text[:300].lower()
    for alias, canonical in sorted(_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in snippet:
            return canonical

    return "unknown"


# ---------------------------------------------------------------------------
# CLI: test on a file or stdin
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    print(extract_condition(text))
