"""
parse_steps_v2.py — Enhanced step parser for improved CASTOR pipeline.

Improvements over parse_steps.py:
  - Strips leading assessment / observation header before parsing
    (models often output "Casualty type: ...\nObserved conditions: ...\n\nSalvage Plan:\n1. ...")
  - Adds quality flag: 'ok' | 'gaps' | 'failed'
      ok     — steps parsed, numbering is contiguous (no gaps)
      gaps   — steps parsed but numbering has gaps (e.g. 1,2,4,5 — missing 3)
      failed — no steps found at all
  - Logs failures to stderr for manual review

Usage:
    from eval.parse_steps_v2 import parse_steps_v2
    steps, flag = parse_steps_v2(plan_text)
    # steps: List[Tuple[int, str]]
    # flag:  'ok' | 'gaps' | 'failed'
"""

import re
import sys

_BOLD_RE     = re.compile(r'\*\*(.+?)\*\*')
_EM_DASH_RE  = re.compile(r'\s+[—–]\s+')

# Primary: "1. text"
_STEP_RE = re.compile(
    r'(?:^|\n)\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|\Z)',
    re.DOTALL,
)
# Fallback: "**Step N: Title**\nbody"
_BOLD_STEP_RE = re.compile(
    r'\*\*Step\s+(\d+)[:\s]+(.*?)\*\*(.*?)(?=\*\*Step\s+\d+|\Z)',
    re.DOTALL | re.IGNORECASE,
)

# Header patterns to strip before parsing
# Matches "Casualty type: ...\nObserved conditions: ...\n\nSalvage Plan:\n"
_HEADER_RE = re.compile(
    r'^(?:.*?casualty\s*type\s*:.*?\n)?'
    r'(?:.*?observed\s*conditions?\s*:.*?\n)?'
    r'(?:.*?salvage\s*plan\s*:?\s*\n)?',
    re.IGNORECASE | re.DOTALL,
)
_PLAN_SECTION_RE = re.compile(
    r'(?:salvage\s*plan\s*:?\s*\n)(.*)',
    re.IGNORECASE | re.DOTALL,
)


def _strip_header(text: str) -> str:
    """Extract the plan body after any header block."""
    m = _PLAN_SECTION_RE.search(text)
    if m:
        return m.group(1)
    return text


def _is_contiguous(nums: list[int]) -> bool:
    if not nums:
        return True
    return all(b == a + 1 for a, b in zip(nums, nums[1:]))


def parse_steps_v2(text: str, source_id: str = "") -> tuple[list[tuple[int, str]], str]:
    """Parse numbered steps from a salvage plan.

    Args:
        text:      Raw plan text from VLM output.
        source_id: Optional identifier for logging (e.g. question_id).

    Returns:
        (steps, flag)
        steps: List of (step_num, step_text) in document order.
        flag:  'ok' | 'gaps' | 'failed'
    """
    if not text or not text.strip():
        _log_failure(source_id, "empty input")
        return [], "failed"

    body = _strip_header(text)

    # --- Primary: N. text ---
    cleaned = _BOLD_RE.sub(r'\1', body)
    cleaned = _EM_DASH_RE.sub(' — ', cleaned)
    matches = _STEP_RE.findall(cleaned)
    if matches:
        steps = []
        for num_str, step_body in matches:
            step_body = ' '.join(step_body.split())
            if step_body:
                steps.append((int(num_str), step_body))
        if steps:
            nums = [n for n, _ in steps]
            flag = "ok" if _is_contiguous(nums) else "gaps"
            if flag == "gaps":
                _log_warning(source_id, f"step number gaps: {nums}")
            return steps, flag

    # --- Fallback: **Step N: Title** ---
    matches = _BOLD_STEP_RE.findall(text)
    steps = []
    for num_str, title, body_text in matches:
        title = _BOLD_RE.sub(r'\1', title).strip()
        body_text = _BOLD_RE.sub(r'\1', body_text)
        body_text = _EM_DASH_RE.sub(' — ', body_text)
        combined = (title + ' ' + ' '.join(body_text.split())).strip()
        if combined:
            steps.append((int(num_str), combined))
    if steps:
        nums = [n for n, _ in steps]
        flag = "ok" if _is_contiguous(nums) else "gaps"
        if flag == "gaps":
            _log_warning(source_id, f"(fallback) step number gaps: {nums}")
        return steps, flag

    _log_failure(source_id, "no numbered steps found")
    return [], "failed"


def _log_failure(source_id: str, reason: str):
    print(f"[parse_steps_v2] FAILED{' (' + source_id + ')' if source_id else ''}: {reason}",
          file=sys.stderr)


def _log_warning(source_id: str, msg: str):
    print(f"[parse_steps_v2] WARN{' (' + source_id + ')' if source_id else ''}: {msg}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI: parse a single plan file for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        text = open(path).read()
    else:
        text = sys.stdin.read()
    steps, flag = parse_steps_v2(text, source_id=path or "stdin")
    print(f"Flag: {flag}  |  {len(steps)} steps")
    for n, s in steps:
        print(f"  {n}. {s[:100]}")
