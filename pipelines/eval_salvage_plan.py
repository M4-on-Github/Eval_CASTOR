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
    in {"predicted", "gt"}, one-vs-rest."""
    tests = []
    for source, state_col in [("predicted", "predicted_state"), ("gt", "gt_state")]:
        for state_value in sorted(df[state_col].dropna().unique()):
            in_state = (df[state_col] == state_value).tolist()
            for element in elements:
                present = df[element].tolist()
                odds_ratio, p_value = fisher_one_vs_rest(present, in_state)
                tests.append(ElementStateTest(
                    element=element, state=state_value, state_source=source,
                    odds_ratio=odds_ratio, p_value=p_value,
                ))
    return tests


def apply_fdr_correction(tests: list) -> list:
    """Apply Benjamini-Hochberg separately within each state_source track
    (predicted vs. gt). The two tracks answer different questions (does the
    plan template on the model's own guess vs. on the true state) and are
    reported as independently-labeled findings, not combined into one
    claim -- see ADR-001. Correcting them separately was chosen over a
    combined correction after checking this pipeline's actual predicted/gt
    agreement rate (21-46% on real runs, not the near-total overlap a
    combined correction would be protecting against)."""
    if not tests:
        return tests
    for source in {t.state_source for t in tests}:
        source_tests = [t for t in tests if t.state_source == source]
        corrected = benjamini_hochberg([t.p_value for t in source_tests])
        for t, p_corr in zip(source_tests, corrected):
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


def element_tests_to_dataframe(tests: list) -> pd.DataFrame:
    """Sorted predicted-then-gt, most-significant-first within each track,
    so the CSV is scannable without sorting it in a spreadsheet first."""
    df = pd.DataFrame([
        {
            "element": t.element, "state": t.state, "state_source": t.state_source,
            "odds_ratio": t.odds_ratio, "p_value": t.p_value, "p_corrected": t.p_corrected,
        }
        for t in tests
    ], columns=["element", "state", "state_source", "odds_ratio", "p_value", "p_corrected"])
    sort_key = df["state_source"].map(_SOURCE_SORT_ORDER).fillna(len(_SOURCE_SORT_ORDER))
    df = df.assign(_sort_key=sort_key).sort_values(
        ["_sort_key", "p_corrected"], na_position="last"
    ).drop(columns="_sort_key").reset_index(drop=True)
    return df


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


def build_report(tests: list, omnibus_pred: dict, omnibus_gt: dict, run_name: str) -> str:
    lines = [f"Salvage Plan Templating Analysis -- {run_name}", "=" * 70, ""]

    significant = [t for t in tests if t.p_corrected is not None and t.p_corrected < SIGNIFICANCE_THRESHOLD]
    lines.append(
        f"{len(significant)}/{len(tests)} (element, state, source) tests "
        f"significant after BH-FDR correction (p_corrected < {SIGNIFICANCE_THRESHOLD})."
    )
    lines.append("")
    if significant:
        lines.append("Significant associations:")
        for t in sorted(significant, key=lambda t: t.p_corrected):
            lines.append(
                f"  [{t.state_source:9s}] {t.element!r} vs {t.state}: "
                f"odds_ratio={t.odds_ratio:.2f}  p={t.p_value:.4g}  p_corrected={t.p_corrected:.4g}"
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

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover_runs() -> list:
    return discover_run_names(RESULTS_IN)


def process_run(run_name: str):
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

    report_file = paths.report_path(run_name)
    report_file.write_text(build_report(tests, omnibus_pred, omnibus_gt, run_name), encoding="utf-8")

    print(f"  {run_name}: {len(tests)} tests -> {tests_csv}")
    print(f"  Omnibus -> {omnibus_csv}")
    print(f"  Dunn's -> {dunn_csv}")
    print(f"  Report -> {report_file}")


def main():
    ap = argparse.ArgumentParser(
        description="Stage 4: statistical tests + report for salvage plan templating (Pipeline 6)"
    )
    ap.add_argument("--run", help="Single run name; omit to process every run found in RESULTS_IN")
    args = ap.parse_args()

    run_names = [args.run] if args.run else discover_runs()
    if not run_names:
        print(f"No runs found in {RESULTS_IN}")
        return

    for run_name in run_names:
        try:
            process_run(run_name)
        except FileNotFoundError as e:
            print(f"  Skipping {run_name}: {e}")


if __name__ == "__main__":
    main()
