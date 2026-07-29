"""
Pipeline 8 — Step parser for numbered salvage plans.

Extracts (step_num, step_text) tuples from numbered list plans produced by
CASTOR VLMs. Plans are already structured as numbered lists across all prompt
variants (IMPROVED, ABLATION, CONTROL, VISUAL_GROUNDED) — no LLM needed.

Usage:
    from pipelines.plan_coherence.parse_steps import parse_steps
    steps = parse_steps(plan_text)  # [(1, "Deploy dive team..."), (2, ...)]
"""

import re

_BOLD_RE      = re.compile(r'\*\*(.+?)\*\*')
_EM_DASH_RE   = re.compile(r'\s+[—–]\s+')
_STEP_RE      = re.compile(r'(?:^|\n)\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|\Z)', re.DOTALL)
# Fallback: **Step N: Title** blocks (title inside bold, body as bullet lines below)
_BOLD_STEP_RE = re.compile(
    r'\*\*Step\s+(\d+)[:\s]+(.*?)\*\*(.*?)(?=\*\*Step\s+\d+|\Z)',
    re.DOTALL | re.IGNORECASE,
)


def parse_steps(text: str) -> list[tuple[int, str]]:
    """Extract numbered steps from a salvage plan text.

    Handles two formats:
      Primary:  "1. Step text..."  (standard numbered list)
      Fallback: "**Step 1: Title**\\n- bullet lines..." (bold-header format)

    Returns a list of (step_num, step_text) tuples, in document order.
    step_text has bold markers stripped and internal whitespace collapsed.
    Returns [] if no numbered steps are found in either format.
    """
    if not text or not text.strip():
        return []

    # Primary: N. text
    cleaned = _BOLD_RE.sub(r'\1', text)
    cleaned = _EM_DASH_RE.sub(' — ', cleaned)
    matches = _STEP_RE.findall(cleaned)
    if matches:
        steps = []
        for num_str, body in matches:
            body = ' '.join(body.split())
            if body:
                steps.append((int(num_str), body))
        if steps:
            return steps

    # Fallback: **Step N: Title** with bullet body beneath
    matches = _BOLD_STEP_RE.findall(text)
    steps = []
    for num_str, title, body in matches:
        title = _BOLD_RE.sub(r'\1', title).strip()
        body  = _BOLD_RE.sub(r'\1', body)
        body  = _EM_DASH_RE.sub(' — ', body)
        combined = (title + ' ' + ' '.join(body.split())).strip()
        if combined:
            steps.append((int(num_str), combined))
    return steps
