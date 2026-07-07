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
