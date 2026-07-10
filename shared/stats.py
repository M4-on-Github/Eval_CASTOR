"""
Statistical primitives for Pipeline 6 (salvage plan templating analysis).

See docs/decisions/ADR-001-salvage-plan-statistical-tests.md for why these
specific tests were chosen over chi-squared/one-way ANOVA/Tukey HSD/t-test.
"""

from dataclasses import dataclass
from itertools import combinations

from scipy.stats import fisher_exact, kruskal, norm, rankdata


@dataclass
class ElementStateTest:
    element: str
    state: str
    state_source: str       # "predicted" or "gt"
    odds_ratio: float
    p_value: float
    p_corrected: float | None = None
    # Raw prevalence, independent of the comparative Fisher's test -- lets a
    # reader answer "how often does this actually appear in this state" (a
    # marginal frequency question) alongside "is it differentially
    # associated with this state vs. elsewhere" (what p_value tests).
    count_in_state: int = 0
    n_in_state: int = 0
    count_out_state: int = 0
    n_out_state: int = 0


def fisher_one_vs_rest(present: list, in_state: list) -> tuple:
    """present, in_state: parallel boolean lists. Returns (odds_ratio, p_value)."""
    a = sum(p and s for p, s in zip(present, in_state))
    b = sum(p and not s for p, s in zip(present, in_state))
    c = sum(not p and s for p, s in zip(present, in_state))
    d = sum(not p and not s for p, s in zip(present, in_state))
    return fisher_exact([[a, b], [c, d]])


def benjamini_hochberg(p_values: list) -> list:
    """Benjamini-Hochberg FDR correction. Returns corrected p-values in the
    same order as the input (monotone-from-the-top procedure)."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * m
    prev_min = 1.0
    for rank in range(m - 1, -1, -1):
        orig_idx, p = indexed[rank]
        adjusted = min(p * m / (rank + 1), prev_min, 1.0)
        corrected[orig_idx] = adjusted
        prev_min = adjusted
    return corrected


def kruskal_wallis(groups: list) -> tuple:
    """Thin wrapper on scipy.stats.kruskal. groups: list of lists of floats.
    Returns (H, p_value)."""
    return kruskal(*groups)


def dunn_test(groups: dict) -> list:
    """Pairwise rank-based post-hoc for Kruskal-Wallis. groups: {name: [values]}.
    Returns a list of {"group_a", "group_b", "p_value"} for every pair."""
    names = list(groups.keys())
    all_values = []
    group_slices = {}
    start = 0
    for name in names:
        vals = groups[name]
        all_values.extend(vals)
        group_slices[name] = (start, start + len(vals))
        start += len(vals)

    ranks = rankdata(all_values)
    n_total = len(all_values)

    mean_ranks = {}
    sizes = {}
    for name in names:
        lo, hi = group_slices[name]
        group_ranks = ranks[lo:hi]
        mean_ranks[name] = group_ranks.mean()
        sizes[name] = len(group_ranks)

    results = []
    for a, b in combinations(names, 2):
        n_a, n_b = sizes[a], sizes[b]
        se = ((n_total * (n_total + 1) / 12) * (1 / n_a + 1 / n_b)) ** 0.5
        z = (mean_ranks[a] - mean_ranks[b]) / se
        p_value = 2 * norm.sf(abs(z))
        results.append({"group_a": a, "group_b": b, "p_value": p_value})
    return results
