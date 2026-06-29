"""
Tests for pipelines/judge_panel/preprocess.py
Run from Eval_CASTOR/ root:  python -m pytest tests/test_preprocess.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.judge_panel.preprocess import preprocess, PROMPTS_DIR

GT = {"state": "aground", "vessel_type": "cargo ship", "size_estimate": "large", "cargo": "none"}


# ── markdown stripping ────────────────────────────────────────────────────────

def test_strips_bold():
    r = preprocess("**The vessel** is on fire", GT)
    assert "**" not in r.clean_pred
    assert "The vessel" in r.clean_pred

def test_strips_italic():
    r = preprocess("*The ship* is sinking", GT)
    assert "*" not in r.clean_pred
    assert "The ship" in r.clean_pred

def test_strips_bold_italic():
    r = preprocess("***capsized***", GT)
    assert "*" not in r.clean_pred

def test_strips_trailing_whitespace():
    r = preprocess("  hello world  ", GT)
    assert r.clean_pred == r.clean_pred.strip()


# ── numeral normalization ─────────────────────────────────────────────────────

def test_three_to_3():
    r = preprocess("I see three vessels", GT)
    assert "3" in r.clean_pred
    assert "three" not in r.clean_pred

def test_twelve_to_12():
    r = preprocess("twelve sailors on board", GT)
    assert "12" in r.clean_pred

def test_numeral_not_partial_match():
    # "rone" should NOT become "r1" — only whole-word matches
    r = preprocess("The drone is airborne", GT)
    assert "drone" in r.clean_pred

def test_twenty_to_20():
    r = preprocess("twenty meters long", GT)
    assert "20" in r.clean_pred


# ── verbosity flag ────────────────────────────────────────────────────────────

def test_verbosity_not_flagged_when_similar_length():
    short_pred = "The ship is aground near the coast."
    r = preprocess(short_pred, GT)
    assert r.verbosity_flagged is False

def test_verbosity_flagged_when_triple_length():
    gt_narrative = " ".join(str(v) for v in GT.values())  # ~30 chars
    long_pred = "x " * (len(gt_narrative) * 3)
    r = preprocess(long_pred, GT)
    assert r.verbosity_flagged is True


# ── prompt files loadable ─────────────────────────────────────────────────────

def test_system_prompt_file_exists():
    p = PROMPTS_DIR / "castor_judge_system.txt"
    assert p.exists(), f"Missing: {p}"
    assert len(p.read_text(encoding="utf-8").strip()) > 0

def test_user_prompt_file_exists():
    p = PROMPTS_DIR / "castor_judge_user.txt"
    assert p.exists(), f"Missing: {p}"
    assert "{gt_state}" in p.read_text(encoding="utf-8")
    assert "{pred_text}" in p.read_text(encoding="utf-8")

def test_user_prompt_formats_without_error():
    p = PROMPTS_DIR / "castor_judge_user.txt"
    tmpl = p.read_text(encoding="utf-8")
    filled = tmpl.format(
        gt_state="aground",
        gt_vessel_type="cargo ship",
        gt_size_estimate="large",
        gt_cargo="none",
        pred_text="The ship appears aground.",
    )
    assert "aground" in filled
    # No unresolved placeholder variables remain (literal { from JSON example is fine)
    import re
    assert not re.search(r'\{[a-z_]+\}', filled), "Unfilled placeholder in template"
