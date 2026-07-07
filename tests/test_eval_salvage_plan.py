"""
Tests for pipelines/eval_salvage_plan.py (Stage 4).
Uses a synthetic contingency table with one deliberately planted association
(fireboat <-> on_fire) and one noise element with no real association, per
the plan's Task 7 verification requirement.
Run: python -m pytest tests/test_eval_salvage_plan.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from pipelines.eval_salvage_plan import (
    apply_fdr_correction,
    build_report,
    run_all_fisher_tests,
    run_omnibus_test,
    element_tests_to_dataframe,
)


def _make_synthetic_df():
    """4 states x 25 records. 'fireboat' is present in 92% of on_fire records
    and ~4% elsewhere (planted association). 'noise_item' is present on
    every 7th record regardless of state (no real association)."""
    n_per_state = 25
    states = ["aground", "capsized", "on_fire", "sunken"]
    rows = []
    idx = 0
    for state in states:
        for i in range(n_per_state):
            fireboat = (i < 23) if state == "on_fire" else (i < 1)
            noise = (idx % 7 == 0)
            rows.append({
                "image": f"img{idx}",
                "predicted_state": state,
                "gt_state": state,
                "fireboat": fireboat,
                "noise_item": noise,
                # Not meaningfully different across states -- just non-constant
                # so scipy's Kruskal-Wallis tie correction doesn't hit a
                # zero-variance divide-by-zero edge case.
                "typicality_score_pred": 0.5 + (0.01 if idx % 2 == 0 else -0.01),
                "typicality_score_gt": 0.5 + (0.01 if idx % 2 == 0 else -0.01),
            })
            idx += 1
    return pd.DataFrame(rows)


# ── run_all_fisher_tests + apply_fdr_correction ───────────────────────────────

def test_planted_association_survives_fdr_correction():
    df = _make_synthetic_df()
    tests = run_all_fisher_tests(df, elements=["fireboat", "noise_item"])
    tests = apply_fdr_correction(tests)

    fireboat_on_fire_pred = next(
        t for t in tests
        if t.element == "fireboat" and t.state == "on_fire" and t.state_source == "predicted"
    )
    assert fireboat_on_fire_pred.p_corrected < 0.05
    assert fireboat_on_fire_pred.odds_ratio > 1.0


def test_noise_element_does_not_survive_fdr_correction():
    df = _make_synthetic_df()
    tests = run_all_fisher_tests(df, elements=["fireboat", "noise_item"])
    tests = apply_fdr_correction(tests)

    noise_tests = [t for t in tests if t.element == "noise_item"]
    assert len(noise_tests) > 0
    assert all(t.p_corrected >= 0.05 for t in noise_tests)


def test_run_all_fisher_tests_covers_both_state_sources():
    df = _make_synthetic_df()
    tests = run_all_fisher_tests(df, elements=["fireboat"])
    sources = {t.state_source for t in tests}
    assert sources == {"predicted", "gt"}


def test_apply_fdr_correction_sets_p_corrected_on_every_test():
    df = _make_synthetic_df()
    tests = run_all_fisher_tests(df, elements=["fireboat", "noise_item"])
    tests = apply_fdr_correction(tests)
    assert all(t.p_corrected is not None for t in tests)


# ── run_omnibus_test ──────────────────────────────────────────────────────────

def test_run_omnibus_test_triggers_dunn_when_significant():
    df = pd.DataFrame({
        "predicted_state": ["x"] * 5 + ["y"] * 5 + ["z"] * 5,
        "typicality_score_pred": [0.9] * 5 + [0.1] * 5 + [0.5] * 5,
    })
    result = run_omnibus_test(df, "typicality_score_pred", "predicted_state")
    assert result["p_value"] < 0.05
    assert result["dunn"] is not None
    assert len(result["dunn"]) == 3


def test_run_omnibus_test_no_dunn_when_not_significant():
    df = pd.DataFrame({
        "predicted_state": ["x"] * 5 + ["y"] * 5,
        "typicality_score_pred": [0.5, 0.51, 0.49, 0.5, 0.52, 0.5, 0.49, 0.51, 0.5, 0.5],
    })
    result = run_omnibus_test(df, "typicality_score_pred", "predicted_state")
    assert result["dunn"] is None


# ── element_tests_to_dataframe / build_report ─────────────────────────────────────────

def test_element_tests_to_dataframe_has_expected_columns():
    df = _make_synthetic_df()
    tests = apply_fdr_correction(run_all_fisher_tests(df, elements=["fireboat"]))
    out = element_tests_to_dataframe(tests)
    assert list(out.columns) == ["element", "state", "state_source", "odds_ratio", "p_value", "p_corrected"]
    assert len(out) == len(tests)


def test_build_report_lists_significant_association():
    df = _make_synthetic_df()
    tests = apply_fdr_correction(run_all_fisher_tests(df, elements=["fireboat", "noise_item"]))
    omnibus_pred = run_omnibus_test(df, "typicality_score_pred", "predicted_state")
    omnibus_gt = run_omnibus_test(df, "typicality_score_gt", "gt_state")

    report = build_report(tests, omnibus_pred, omnibus_gt, "synthetic_run")

    assert "fireboat" in report
    assert "on_fire" in report
    assert "Kruskal-Wallis" in report


def test_build_report_honest_when_nothing_significant():
    tests = apply_fdr_correction([])
    df = pd.DataFrame({
        "predicted_state": ["x"] * 3 + ["y"] * 3,
        "typicality_score_pred": [0.49, 0.5, 0.51, 0.49, 0.5, 0.51],
    })
    omnibus = run_omnibus_test(df, "typicality_score_pred", "predicted_state")
    report = build_report([], omnibus, omnibus, "empty_run")
    assert "No associations survived FDR correction" in report
