"""
Tests for shared/stats.py — statistical primitives for Pipeline 6.
See docs/decisions/ADR-001-salvage-plan-statistical-tests.md for rationale.
Run: python -m pytest tests/test_salvage_stats.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from scipy.stats import chi2, norm

from shared.stats import (
    ElementStateTest,
    benjamini_hochberg,
    dunn_test,
    fisher_one_vs_rest,
    kruskal_wallis,
)


# ── fisher_one_vs_rest ────────────────────────────────────────────────────────

def test_fisher_one_vs_rest_known_odds_ratio():
    # 10 records with in_state=True (9 present, 1 absent),
    # 10 records with in_state=False (1 present, 9 absent)
    # -> table [[9,1],[1,9]] -> odds ratio = (9*9)/(1*1) = 81
    present  = [True] * 9 + [False] * 1 + [True] * 1 + [False] * 9
    in_state = [True] * 10 + [False] * 10

    odds_ratio, p_value = fisher_one_vs_rest(present, in_state)

    assert odds_ratio == 81.0
    assert p_value == pytest.approx(0.001093333910671372)


def test_fisher_one_vs_rest_no_association():
    # Perfectly balanced 2x2 table [[5,5],[5,5]] -> odds ratio = 1, p = 1
    present  = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
    in_state = [True] * 10 + [False] * 10

    odds_ratio, p_value = fisher_one_vs_rest(present, in_state)

    assert odds_ratio == 1.0
    assert p_value == 1.0


# ── benjamini_hochberg ────────────────────────────────────────────────────────

def test_benjamini_hochberg_known_example():
    # Textbook BH example: sorted p = [.01, .02, .03, .04, .50], m=5
    # raw adjusted = p*m/i = [.05, .05, .05, .05, .50]; already monotone.
    p_values = [0.01, 0.02, 0.03, 0.04, 0.50]

    corrected = benjamini_hochberg(p_values)

    assert corrected == pytest.approx([0.05, 0.05, 0.05, 0.05, 0.50])


def test_benjamini_hochberg_enforces_monotonicity():
    # Sorted p = [.01, .04, .03] (unsorted input) with m=3.
    # raw adjusted by rank (sorted: .01->i=1, .03->i=2, .04->i=3):
    #   .01*3/1=.03, .03*3/2=.045, .04*3/3=.04
    # monotone-from-top pass: start from largest (.04 -> .04),
    #   then .045 vs .04 -> min = .04, then .03 vs .04 -> min = .03
    # so corrected sorted = [.03, .04, .04]; original order is [.01, .04, .03]
    # -> corrected original order = [.03, .04, .04]
    p_values = [0.01, 0.04, 0.03]

    corrected = benjamini_hochberg(p_values)

    assert corrected[0] == pytest.approx(0.03)
    assert corrected[2] == pytest.approx(0.04)
    assert corrected[1] == pytest.approx(0.04)


def test_benjamini_hochberg_caps_at_one():
    p_values = [0.9, 0.95]
    corrected = benjamini_hochberg(p_values)
    assert all(p <= 1.0 for p in corrected)


# ── kruskal_wallis ────────────────────────────────────────────────────────────

def test_kruskal_wallis_matches_hand_computed_h_statistic():
    # groups with no ties: ranks are exactly 1..6
    # R1=1+2=3 (n=2), R2=3+4=7 (n=2), R3=5+6=11 (n=2), N=6
    # H = 12/(N*(N+1)) * sum(R_i^2/n_i) - 3*(N+1)
    #   = (12/42) * (4.5 + 24.5 + 60.5) - 21 = 0.285714*89.5 - 21 = 4.571428...
    groups = [[1, 2], [3, 4], [5, 6]]

    H, p_value = kruskal_wallis(groups)

    assert H == pytest.approx(4.571428571428571)
    expected_p = chi2.sf(H, df=2)
    assert p_value == pytest.approx(expected_p)


# ── dunn_test ─────────────────────────────────────────────────────────────────

def test_dunn_test_pairwise_p_values_match_hand_computed_z_scores():
    # Same groups as the kruskal_wallis test: mean ranks 1.5, 3.5, 5.5 (N=6)
    # se = sqrt((6*7/12) * (1/2+1/2)) = sqrt(3.5) = 1.8708...
    # z(a,b) = (1.5-3.5)/se = -1.0690...; z(a,c) = (1.5-5.5)/se = -2.1381...
    # z(b,c) = (3.5-5.5)/se = -1.0690...
    groups = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}

    results = dunn_test(groups)

    by_pair = {(r["group_a"], r["group_b"]): r["p_value"] for r in results}
    se = (3.5) ** 0.5

    z_ab = (1.5 - 3.5) / se
    z_ac = (1.5 - 5.5) / se
    z_bc = (3.5 - 5.5) / se

    assert by_pair[("a", "b")] == pytest.approx(2 * norm.sf(abs(z_ab)))
    assert by_pair[("a", "c")] == pytest.approx(2 * norm.sf(abs(z_ac)))
    assert by_pair[("b", "c")] == pytest.approx(2 * norm.sf(abs(z_bc)))


def test_dunn_test_returns_all_pairs():
    groups = {"aground": [1, 2, 3], "sunken": [4, 5], "on_fire": [6, 7, 8]}
    results = dunn_test(groups)
    pairs = {frozenset((r["group_a"], r["group_b"])) for r in results}
    assert pairs == {
        frozenset({"aground", "sunken"}),
        frozenset({"aground", "on_fire"}),
        frozenset({"sunken", "on_fire"}),
    }


# ── ElementStateTest dataclass ────────────────────────────────────────────────

def test_element_state_test_defaults_p_corrected_to_none():
    t = ElementStateTest(
        element="fireboat", state="on_fire", state_source="predicted",
        odds_ratio=81.0, p_value=0.0001,
    )
    assert t.p_corrected is None


def test_element_state_test_defaults_prevalence_counts_to_zero():
    # Raw prevalence fields must default so existing call sites that don't
    # pass them (e.g. hand-built tests elsewhere) keep working unmodified.
    t = ElementStateTest(
        element="fireboat", state="on_fire", state_source="predicted",
        odds_ratio=81.0, p_value=0.0001,
    )
    assert t.count_in_state == 0
    assert t.n_in_state == 0
    assert t.count_out_state == 0
    assert t.n_out_state == 0


def test_element_state_test_stores_prevalence_counts():
    t = ElementStateTest(
        element="fireboat", state="on_fire", state_source="predicted",
        odds_ratio=81.0, p_value=0.0001,
        count_in_state=8, n_in_state=16, count_out_state=2, n_out_state=94,
    )
    assert t.count_in_state == 8
    assert t.n_in_state == 16
    assert t.count_out_state == 2
    assert t.n_out_state == 94
