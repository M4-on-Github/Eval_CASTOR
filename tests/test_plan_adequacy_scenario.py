"""
Tests for pipelines/plan_adequacy/scenario.py

Tests the pure size-normalization and habitat-sensitivity logic without
requiring the real human_gt.csv, plus one integration check against the
real file for the en-dash/hyphen split it's known to contain.
Run: python -m pytest tests/test_plan_adequacy_scenario.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.scenario import (
    _habitat_sensitive,
    _normalize_size,
    load_scenarios,
)

EVAL_ROOT = Path(__file__).parent.parent
REAL_GT_PATH = EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv"


# ── _normalize_size ──────────────────────────────────────────────────────────

def test_normalize_size_collapses_en_dash_and_hyphen_variants():
    # human_gt.csv contains both "medium (10-50m)" (hyphen, x6) and
    # "medium (10–50m)" (en dash, x43) for the same category.
    assert _normalize_size("medium (10-50m)") == "medium"
    assert _normalize_size("medium (10–50m)") == "medium"


def test_normalize_size_large_small():
    assert _normalize_size("large (>50m)") == "large"
    assert _normalize_size("small (<10m)") == "small"


def test_normalize_size_unknown_on_empty_or_unrecognized():
    assert _normalize_size("") == "unknown"
    assert _normalize_size("some other text") == "unknown"


# ── _habitat_sensitive ────────────────────────────────────────────────────────

def test_habitat_sensitive_true_for_tanker():
    assert _habitat_sensitive("Oil Tanker", "") is True


def test_habitat_sensitive_false_for_generic_cargo():
    assert _habitat_sensitive("Cargo Ship", "") is False


# ── integration against the real CSV ─────────────────────────────────────────

def test_load_scenarios_against_real_csv_normalizes_all_size_variants():
    if not REAL_GT_PATH.exists():
        return  # skip silently if the data file isn't present in this checkout
    scenarios = load_scenarios(REAL_GT_PATH)
    assert len(scenarios) == 110
    categories = {s.size_category for s in scenarios.values()}
    # every size_estimate value in human_gt.csv must normalize to one of these
    assert categories <= {"small", "medium", "large", "unknown"}
    # both the ASCII-hyphen and en-dash "medium" spellings must have merged
    medium_count = sum(1 for s in scenarios.values() if s.size_category == "medium")
    assert medium_count == 43 + 6  # measured counts, see module docstring


def test_load_scenarios_keys_match_gt_image_format():
    if not REAL_GT_PATH.exists():
        return
    scenarios = load_scenarios(REAL_GT_PATH)
    assert "aground/00017.jpg" in scenarios
    assert scenarios["aground/00017.jpg"].state == "aground"
