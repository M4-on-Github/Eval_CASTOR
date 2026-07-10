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
    omnibus_to_dataframe,
    dunn_to_dataframe,
)
from shared.stats import ElementStateTest


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


def test_fdr_correction_applied_separately_per_state_source():
    # predicted_state and gt_state answer different questions (self-
    # templating vs. grounding-in-reality) and are corrected independently
    # -- a real predicted-source finding should not be washed out by
    # unrelated noise in the much larger gt-source family. See ADR-001.
    tests = [ElementStateTest(element="fireboat", state="on_fire", state_source="predicted",
                               odds_ratio=10.0, p_value=0.01)]
    tests += [
        ElementStateTest(element=f"noise{i}", state="aground", state_source="gt",
                          odds_ratio=1.0, p_value=0.9)
        for i in range(99)
    ]
    corrected = apply_fdr_correction(tests)
    predicted_test = next(t for t in corrected if t.state_source == "predicted")
    assert predicted_test.p_corrected == pytest.approx(0.01)


def test_fdr_correction_applied_separately_per_state_within_same_source():
    # Different states within the SAME state_source are also corrected
    # independently -- one-vs-rest groups for different states are mutually
    # exclusive (an image is aground XOR on_fire, never both), so "is there
    # a signature element for on_fire" and "...for aground" are separable
    # questions, not one shared claim needing one combined FDR budget.
    tests = [ElementStateTest(element="fireboat", state="on_fire", state_source="predicted",
                               odds_ratio=10.0, p_value=0.01)]
    tests += [
        ElementStateTest(element=f"noise{i}", state="aground", state_source="predicted",
                          odds_ratio=1.0, p_value=0.9)
        for i in range(99)
    ]
    corrected = apply_fdr_correction(tests)
    on_fire_test = next(t for t in corrected if t.state == "on_fire")
    assert on_fire_test.p_corrected == pytest.approx(0.01)


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


def test_run_omnibus_test_handles_single_group_without_crashing():
    # A Stage 1 extraction failure can leave every record in one predicted
    # state group (e.g. all UNPARSEABLE) -- kruskal() itself raises
    # ValueError on fewer than 2 groups; this must degrade gracefully
    # instead of crashing the whole Stage 4 run.
    df = pd.DataFrame({
        "predicted_state": ["UNPARSEABLE"] * 5,
        "typicality_score_pred": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    result = run_omnibus_test(df, "typicality_score_pred", "predicted_state")
    assert result["H"] is None
    assert result["p_value"] is None
    assert result["dunn"] is None


def test_run_omnibus_test_handles_zero_groups_without_crashing():
    df = pd.DataFrame({"predicted_state": pd.Series([], dtype=object), "typicality_score_pred": pd.Series([], dtype=float)})
    result = run_omnibus_test(df, "typicality_score_pred", "predicted_state")
    assert result["H"] is None
    assert result["p_value"] is None
    assert result["dunn"] is None


# ── element_tests_to_dataframe / build_report ─────────────────────────────────────────

def test_element_tests_to_dataframe_has_expected_columns():
    df = _make_synthetic_df()
    tests = apply_fdr_correction(run_all_fisher_tests(df, elements=["fireboat"]))
    out = element_tests_to_dataframe(tests)
    assert list(out.columns) == ["element", "state", "state_source", "odds_ratio", "p_value", "p_corrected"]
    assert len(out) == len(tests)


def test_element_tests_to_dataframe_sorted_by_source_then_significance():
    # Most-significant-first within each state_source track makes the CSV
    # scannable without needing to sort it in a spreadsheet first.
    tests = [
        ElementStateTest(element="a", state="s1", state_source="gt", odds_ratio=1.0, p_value=0.5, p_corrected=0.6),
        ElementStateTest(element="b", state="s1", state_source="predicted", odds_ratio=5.0, p_value=0.01, p_corrected=0.02),
        ElementStateTest(element="c", state="s2", state_source="predicted", odds_ratio=2.0, p_value=0.3, p_corrected=0.4),
    ]
    out = element_tests_to_dataframe(tests)
    assert list(out["state_source"]) == ["predicted", "predicted", "gt"]
    assert list(out["element"])[:2] == ["b", "c"]


# ── omnibus_to_dataframe / dunn_to_dataframe ──────────────────────────────────

def test_omnibus_to_dataframe_has_one_row_per_source():
    df = _make_synthetic_df()
    omnibus_pred = run_omnibus_test(df, "typicality_score_pred", "predicted_state")
    omnibus_gt = run_omnibus_test(df, "typicality_score_gt", "gt_state")
    out = omnibus_to_dataframe(omnibus_pred, omnibus_gt)
    assert list(out.columns) == ["state_source", "H", "p_value", "significant"]
    assert sorted(out["state_source"]) == ["gt", "predicted"]


def test_omnibus_to_dataframe_flags_significant_rows():
    sig = {"H": 12.0, "p_value": 0.001, "dunn": [{"group_a": "x", "group_b": "y", "p_value": 0.02}]}
    not_sig = {"H": 1.0, "p_value": 0.9, "dunn": None}
    out = omnibus_to_dataframe(sig, not_sig)
    row_pred = out[out["state_source"] == "predicted"].iloc[0]
    row_gt = out[out["state_source"] == "gt"].iloc[0]
    assert bool(row_pred["significant"]) is True
    assert bool(row_gt["significant"]) is False


def test_omnibus_to_dataframe_handles_degenerate_single_group():
    degenerate = {"H": None, "p_value": None, "dunn": None}
    out = omnibus_to_dataframe(degenerate, degenerate)
    assert out["H"].isna().all()
    assert out["significant"].tolist() == [False, False]


def test_dunn_to_dataframe_has_expected_columns_and_rows():
    sig = {"H": 12.0, "p_value": 0.001, "dunn": [
        {"group_a": "x", "group_b": "y", "p_value": 0.02},
        {"group_a": "x", "group_b": "z", "p_value": 0.5},
    ]}
    not_sig = {"H": 1.0, "p_value": 0.9, "dunn": None}
    out = dunn_to_dataframe(sig, not_sig)
    assert list(out.columns) == ["state_source", "group_a", "group_b", "p_value"]
    assert len(out) == 2
    assert (out["state_source"] == "predicted").all()


def test_dunn_to_dataframe_empty_when_nothing_significant():
    not_sig = {"H": 1.0, "p_value": 0.9, "dunn": None}
    out = dunn_to_dataframe(not_sig, not_sig)
    assert len(out) == 0
    assert list(out.columns) == ["state_source", "group_a", "group_b", "p_value"]


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


def test_build_report_handles_single_group_omnibus_without_crashing():
    # Must not raise (e.g. TypeError formatting None as .3f) when Stage 1
    # extraction failure left only one predicted-state group.
    single_group = {"H": None, "p_value": None, "dunn": None}
    report = build_report([], single_group, single_group, "degenerate_run")
    assert "not enough groups" in report.lower()
