"""
Pipeline 6 Stage 4 — statistical tests + report over the Stage 3 contingency
table.

Primary test: Fisher's exact, per (element, state) pair, one-vs-rest, for
both predicted_state and gt_state, with Benjamini-Hochberg FDR correction
applied across the full combined test set. Secondary/omnibus test:
Kruskal-Wallis on the typicality score across state groups (run separately
for predicted-state and GT-state groupings), with Dunn's post-hoc when
significant. See docs/decisions/ADR-001-salvage-plan-statistical-tests.md
for why these specific tests were chosen.

Usage:
  python pipelines/eval_salvage_plan.py --run answers_baseline
  python pipelines/eval_salvage_plan.py     # processes every run found
"""

import argparse
import os
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(EVAL_ROOT))

import pandas as pd

from pipelines.salvage_analysis import contingency, paths
from pipelines.salvage_analysis.combine_shards import discover_run_names
from shared.stats import ElementStateTest, benjamini_hochberg, dunn_test, fisher_one_vs_rest, kruskal_wallis

RESULTS_IN = Path(os.environ.get("CASTOR_SALVAGE_RESULTS_DIR", paths.PLANS_TO_JUDGE_DIR))

NON_ELEMENT_COLUMNS = {
    "image", "predicted_state", "gt_state",
    "typicality_score_pred", "typicality_score_gt",
}

SIGNIFICANCE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

def run_all_fisher_tests(df: pd.DataFrame, elements: list) -> list:
    """One ElementStateTest per (element, state, source) combination, source
    in {"predicted", "gt"}, one-vs-rest. Also records raw prevalence counts
    (independent of the comparative Fisher's test) so a reader can answer
    "how often does this actually appear in this state" directly."""
    tests = []
    for source, state_col in [("predicted", "predicted_state"), ("gt", "gt_state")]:
        for state_value in sorted(df[state_col].dropna().unique()):
            in_state = (df[state_col] == state_value).tolist()
            n_in_state = sum(in_state)
            n_out_state = len(in_state) - n_in_state
            for element in elements:
                present = df[element].tolist()
                odds_ratio, p_value = fisher_one_vs_rest(present, in_state)
                count_in_state = sum(p and s for p, s in zip(present, in_state))
                count_out_state = sum(p and not s for p, s in zip(present, in_state))
                tests.append(ElementStateTest(
                    element=element, state=state_value, state_source=source,
                    odds_ratio=odds_ratio, p_value=p_value,
                    count_in_state=count_in_state, n_in_state=n_in_state,
                    count_out_state=count_out_state, n_out_state=n_out_state,
                ))
    return tests


def apply_fdr_correction(tests: list) -> list:
    """Apply Benjamini-Hochberg separately within each (state_source, state)
    pair -- e.g. "predicted, on_fire" and "predicted, aground" each get
    their own independent correction, not pooled together.

    Two reasons this is split this finely, not just by state_source:
    - predicted_state and gt_state answer different questions (does the
      plan template on the model's own guess vs. on the true state) and are
      reported as independently-labeled findings, not combined into one
      claim -- see ADR-001. Checked this pipeline's actual predicted/gt
      agreement rate (21-46% on real runs) rather than assuming the two
      tracks were correlated enough to require a combined correction.
    - Different states within the same source are mutually exclusive
      one-vs-rest groups (an image is aground XOR on_fire, never both), so
      "is there a signature element for on_fire" and "...for aground" are
      separable questions too, not one shared claim needing one combined
      FDR budget across all states."""
    if not tests:
        return tests
    for source, state in {(t.state_source, t.state) for t in tests}:
        group_tests = [t for t in tests if t.state_source == source and t.state == state]
        corrected = benjamini_hochberg([t.p_value for t in group_tests])
        for t, p_corr in zip(group_tests, corrected):
            t.p_corrected = p_corr
    return tests


def run_omnibus_test(df: pd.DataFrame, score_col: str, state_col: str) -> dict:
    """Kruskal-Wallis on score_col grouped by state_col; Dunn's post-hoc only
    when the omnibus result is significant. Returns H/p_value/dunn all None
    when fewer than 2 groups are present (e.g. a Stage 1 extraction failure
    left every record in one predicted-state bucket) -- kruskal() itself
    raises ValueError on fewer than 2 groups, and that shouldn't crash the
    whole Stage 4 run over what's ultimately a data problem, not a bug."""
    groups_dict = {
        state: df.loc[df[state_col] == state, score_col].tolist()
        for state in sorted(df[state_col].dropna().unique())
    }
    if len(groups_dict) < 2:
        return {"H": None, "p_value": None, "dunn": None}
    H, p_value = kruskal_wallis(list(groups_dict.values()))
    dunn = dunn_test(groups_dict) if p_value < SIGNIFICANCE_THRESHOLD else None
    return {"H": H, "p_value": p_value, "dunn": dunn}


_SOURCE_SORT_ORDER = {"predicted": 0, "gt": 1}


def _safe_pct(count: int, n: int) -> float:
    return count / n if n else 0.0


def element_tests_to_dataframe(tests: list) -> pd.DataFrame:
    """Sorted predicted-then-gt, most-significant-first within each track,
    so the CSV is scannable without sorting it in a spreadsheet first.
    pct_in_state/pct_out_state are raw prevalence -- independent of
    significance, they answer "how often does this actually appear in this
    state" directly, alongside the comparative p_value/odds_ratio."""
    columns = [
        "element", "state", "state_source", "odds_ratio", "p_value", "p_corrected",
        "count_in_state", "n_in_state", "pct_in_state",
        "count_out_state", "n_out_state", "pct_out_state",
    ]
    df = pd.DataFrame([
        {
            "element": t.element, "state": t.state, "state_source": t.state_source,
            "odds_ratio": t.odds_ratio, "p_value": t.p_value, "p_corrected": t.p_corrected,
            "count_in_state": t.count_in_state, "n_in_state": t.n_in_state,
            "pct_in_state": _safe_pct(t.count_in_state, t.n_in_state),
            "count_out_state": t.count_out_state, "n_out_state": t.n_out_state,
            "pct_out_state": _safe_pct(t.count_out_state, t.n_out_state),
        }
        for t in tests
    ], columns=columns)
    sort_key = df["state_source"].map(_SOURCE_SORT_ORDER).fillna(len(_SOURCE_SORT_ORDER))
    df = df.assign(_sort_key=sort_key).sort_values(
        ["_sort_key", "p_corrected"], na_position="last"
    ).drop(columns="_sort_key").reset_index(drop=True)
    return df


def identify_generic_elements(tests: list, min_overall_pct: float) -> pd.DataFrame:
    """Elements used often overall but never significantly associated with
    any single state -- boilerplate the model reaches for regardless of
    what it thinks is happening, as opposed to a real state-specific
    signature (e.g. "fireboat" for on_fire). min_overall_pct has no default
    on purpose -- same "ask first" convention as Stage 2's clustering
    threshold, since what counts as "frequent enough to flag" is a
    judgment call, not something to guess.

    An element is flagged within a state_source track if:
    - its overall prevalence in that track is >= min_overall_pct. One-vs-
      rest states are mutually exclusive, so summing count_in_state (and
      n_in_state) across all of an element's per-state tests in one track
      gives exactly its total count (and the dataset size) for that track,
      with no double-counting.
    - none of its per-state tests in that track reached significance."""
    rows = []
    for source in {t.state_source for t in tests}:
        source_tests = [t for t in tests if t.state_source == source]
        by_element = {}
        for t in source_tests:
            by_element.setdefault(t.element, []).append(t)
        for element, elem_tests in by_element.items():
            overall_count = sum(t.count_in_state for t in elem_tests)
            overall_n = sum(t.n_in_state for t in elem_tests)
            overall_pct = _safe_pct(overall_count, overall_n)
            any_significant = any(
                t.p_corrected is not None and t.p_corrected < SIGNIFICANCE_THRESHOLD
                for t in elem_tests
            )
            if overall_pct >= min_overall_pct and not any_significant:
                p_corrected_seen = [t.p_corrected for t in elem_tests if t.p_corrected is not None]
                rows.append({
                    "element": element,
                    "state_source": source,
                    "overall_count": overall_count,
                    "overall_n": overall_n,
                    "overall_pct": overall_pct,
                    "min_p_corrected_seen": min(p_corrected_seen) if p_corrected_seen else None,
                })
    columns = ["element", "state_source", "overall_count", "overall_n", "overall_pct", "min_p_corrected_seen"]
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values("overall_pct", ascending=False).reset_index(drop=True)


def omnibus_to_dataframe(omnibus_pred: dict, omnibus_gt: dict) -> pd.DataFrame:
    """One row per state_source (predicted, gt) with the Kruskal-Wallis H,
    p_value, and whether it cleared SIGNIFICANCE_THRESHOLD. H/p_value are
    NaN when there weren't enough groups to test (see run_omnibus_test)."""
    rows = []
    for source, omnibus in [("predicted", omnibus_pred), ("gt", omnibus_gt)]:
        significant = omnibus["p_value"] is not None and omnibus["p_value"] < SIGNIFICANCE_THRESHOLD
        rows.append({
            "state_source": source,
            "H": omnibus["H"],
            "p_value": omnibus["p_value"],
            "significant": significant,
        })
    return pd.DataFrame(rows, columns=["state_source", "H", "p_value", "significant"])


def dunn_to_dataframe(omnibus_pred: dict, omnibus_gt: dict) -> pd.DataFrame:
    """One row per Dunn's post-hoc pairwise comparison, across both
    state_source tracks. Empty (but correctly-columned) when neither
    omnibus test was significant -- Dunn's only runs when Kruskal-Wallis
    already cleared the bar (see run_omnibus_test)."""
    rows = []
    for source, omnibus in [("predicted", omnibus_pred), ("gt", omnibus_gt)]:
        for d in (omnibus["dunn"] or []):
            rows.append({
                "state_source": source,
                "group_a": d["group_a"],
                "group_b": d["group_b"],
                "p_value": d["p_value"],
            })
    return pd.DataFrame(rows, columns=["state_source", "group_a", "group_b", "p_value"])


def build_report(tests: list, omnibus_pred: dict, omnibus_gt: dict, run_name: str,
                  generic_df: pd.DataFrame = None) -> str:
    lines = [f"Salvage Plan Templating Analysis -- {run_name}", "=" * 70, ""]

    significant = [t for t in tests if t.p_corrected is not None and t.p_corrected < SIGNIFICANCE_THRESHOLD]
    lines.append(
        f"{len(significant)}/{len(tests)} (element, state, source) tests "
        f"significant after BH-FDR correction (p_corrected < {SIGNIFICANCE_THRESHOLD})."
    )
    lines.append("")
    if significant:
        lines.append("Significant associations (pct_in_state/pct_out_state show raw frequency, independent of significance):")
        for t in sorted(significant, key=lambda t: t.p_corrected):
            pct_in = _safe_pct(t.count_in_state, t.n_in_state)
            pct_out = _safe_pct(t.count_out_state, t.n_out_state)
            lines.append(
                f"  [{t.state_source:9s}] {t.element!r} vs {t.state}: "
                f"odds_ratio={t.odds_ratio:.2f}  p={t.p_value:.4g}  p_corrected={t.p_corrected:.4g}  "
                f"in_state={pct_in:.0%} ({t.count_in_state}/{t.n_in_state})  "
                f"out_state={pct_out:.0%} ({t.count_out_state}/{t.n_out_state})"
            )
    else:
        lines.append(
            "No associations survived FDR correction at this sample size "
            "(see ADR-001: small-n honesty -- this is an expected, valid outcome, not a failure)."
        )
    lines.append("")

    for label, omnibus, state_col in [
        ("predicted_state", omnibus_pred, "predicted_state"),
        ("gt_state", omnibus_gt, "gt_state"),
    ]:
        if omnibus["p_value"] is None:
            lines.append(f"Kruskal-Wallis on typicality score by {label}: not enough groups to test (fewer than 2 {label} values present).")
            lines.append("")
            continue
        lines.append(f"Kruskal-Wallis on typicality score by {label}: H={omnibus['H']:.3f}  p={omnibus['p_value']:.4g}")
        if omnibus["dunn"]:
            lines.append("  Significant -- Dunn's post-hoc pairwise p-values:")
            for d in omnibus["dunn"]:
                lines.append(f"    {d['group_a']} vs {d['group_b']}: p={d['p_value']:.4g}")
        lines.append("")

    lines.append("Generic / boilerplate elements (frequent overall, never significant for any single state):")
    if generic_df is not None and len(generic_df) > 0:
        for _, row in generic_df.iterrows():
            lines.append(
                f"  [{row['state_source']:9s}] {row['element']!r}: "
                f"overall={row['overall_pct']:.0%} ({row['overall_count']}/{row['overall_n']})"
            )
    else:
        lines.append("  No generic elements found at the configured --min-generic-pct threshold.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover_runs() -> list:
    return discover_run_names(RESULTS_IN)


def process_run(run_name: str, min_generic_pct: float):
    df = contingency.run(run_name)
    elements = [c for c in df.columns if c not in NON_ELEMENT_COLUMNS]

    tests = apply_fdr_correction(run_all_fisher_tests(df, elements))
    omnibus_pred = run_omnibus_test(df, "typicality_score_pred", "predicted_state")
    omnibus_gt = run_omnibus_test(df, "typicality_score_gt", "gt_state")

    paths.run_dir(run_name).mkdir(parents=True, exist_ok=True)

    tests_csv = paths.tests_path(run_name)
    element_tests_to_dataframe(tests).to_csv(tests_csv, index=False)

    omnibus_csv = paths.omnibus_path(run_name)
    omnibus_to_dataframe(omnibus_pred, omnibus_gt).to_csv(omnibus_csv, index=False)

    dunn_csv = paths.dunn_path(run_name)
    dunn_to_dataframe(omnibus_pred, omnibus_gt).to_csv(dunn_csv, index=False)

    generic_df = identify_generic_elements(tests, min_generic_pct)
    generic_csv = paths.generic_elements_path(run_name)
    generic_df.to_csv(generic_csv, index=False)

    report_file = paths.report_path(run_name)
    report_file.write_text(
        build_report(tests, omnibus_pred, omnibus_gt, run_name, generic_df=generic_df),
        encoding="utf-8",
    )

    print(f"  {run_name}: {len(tests)} tests -> {tests_csv}")
    print(f"  Omnibus -> {omnibus_csv}")
    print(f"  Dunn's -> {dunn_csv}")
    print(f"  Generic elements (>= {min_generic_pct:.0%} overall, never significant) -> {generic_csv}")
    print(f"  Report -> {report_file}")


def main():
    ap = argparse.ArgumentParser(
        description="Stage 4: statistical tests + report for salvage plan templating (Pipeline 6)"
    )
    ap.add_argument("--run", help="Single run name; omit to process every run found in RESULTS_IN")
    ap.add_argument("--min-generic-pct", type=float, required=True,
                     help="Minimum overall prevalence (0-1) for an element to be flagged as a "
                          "generic/boilerplate template in generic_elements.csv -- e.g. 0.5 means "
                          "\"present in at least half of all records, and never significant for any "
                          "single state.\" No default on purpose -- pick deliberately, same convention "
                          "as Stage 2's clustering threshold.")
    args = ap.parse_args()

    run_names = [args.run] if args.run else discover_runs()
    if not run_names:
        print(f"No runs found in {RESULTS_IN}")
        return

    for run_name in run_names:
        try:
            process_run(run_name, args.min_generic_pct)
        except FileNotFoundError as e:
            print(f"  Skipping {run_name}: {e}")


if __name__ == "__main__":
    main()
