"""
aggregate.py — Aggregate judge scores and produce the final report.

Reads:  results/judge_scores_improved.jsonl
Writes:
  results/summary_by_condition.csv       — mean scores per condition × track
  results/summary_by_condition_state.csv — mean scores per condition × state × track
  results/case_studies.md                — 3 case studies per condition (best/worst/flip)
  results/report.md                      — full narrative report

Usage:
    python improved/eval/aggregate.py --config improved/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return pd.DataFrame(records)


def fmt(val) -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:.3f}"


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

SCORE_COLS = ["seq_score", "method_valid", "specific", "coherence_score"]


def coerce_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda x: 1.0 if x is True else (0.0 if x is False else np.nan))


def summary_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for (condition, track), grp in df.groupby(["condition", "eval_track"]):
        row = {"condition": condition, "track": track, "n": len(grp)}
        for col in SCORE_COLS:
            row[col] = grp[col].mean()
        row["parse_failed_pct"] = (grp["parse_flag"] == "failed").mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["track", "condition"])


def summary_by_condition_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for (condition, eval_state, track), grp in df.groupby(
        ["condition", "eval_state", "eval_track"]
    ):
        row = {
            "condition": condition,
            "eval_state": eval_state,
            "track": track,
            "n": len(grp),
        }
        for col in SCORE_COLS:
            row[col] = grp[col].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["track", "eval_state", "condition"])


# ---------------------------------------------------------------------------
# case studies
# ---------------------------------------------------------------------------

def pick_cases(df: pd.DataFrame, condition: str, track: str = "gt") -> dict:
    """Pick best, worst, and a 'flip' (standard > control) case for a condition."""
    sub = df[(df["condition"] == condition) & (df["eval_track"] == track)].copy()
    if sub.empty:
        return {}

    sub = sub.sort_values("coherence_score", ascending=False)
    best  = sub.iloc[0]
    worst = sub.iloc[-1]

    # Flip: STANDARD score notably higher than CONTROL for same image
    flip = None
    if condition == "standard_v2":
        ctrl = df[(df["condition"] == "control_v2") & (df["eval_track"] == track)]
        merged = sub.merge(ctrl, on="question_id", suffixes=("_std", "_ctrl"))
        merged["delta"] = merged["coherence_score_std"] - merged["coherence_score_ctrl"]
        if not merged.empty:
            top_flip = merged.nlargest(1, "delta").iloc[0]
            flip = {
                "question_id": top_flip["question_id"],
                "std_score":   top_flip["coherence_score_std"],
                "ctrl_score":  top_flip["coherence_score_ctrl"],
                "delta":       top_flip["delta"],
            }
    return {"best": best, "worst": worst, "flip": flip}


def format_case(row, label: str) -> str:
    lines = [
        f"### {label}",
        f"- **Image**: `{row['question_id']}`",
        f"- **GT state**: {row.get('gt_state', 'N/A')}",
        f"- **Predicted state**: {row.get('predicted_state', 'N/A')}",
        f"- **Parse flag**: {row.get('parse_flag', 'N/A')}",
        f"- **SEQ score**: {fmt(row.get('seq_score'))}  "
        f"(applicable chains: {row.get('seq_chains_applicable', 0)}, "
        f"failed: {row.get('seq_chains_failed', 0)})",
        f"- **SEQ failures**: {row.get('seq_failures', [])}",
        f"- **Method valid**: {row.get('method_valid')}",
        f"- **Specific**: {row.get('specific')}",
        f"- **Coherence score**: {fmt(row.get('coherence_score'))}",
        f"- **Judge reason**: {row.get('reason', '')}",
    ]
    return "\n".join(lines)


def write_case_studies(df: pd.DataFrame, out_path: Path):
    conditions = ["standard_v2", "control_v2", "ablation_v2"]
    sections = ["# Case Studies (GT track)\n"]
    for cond in conditions:
        sections.append(f"## Condition: {cond}\n")
        cases = pick_cases(df, cond, track="gt")
        if not cases:
            sections.append("_No data._\n")
            continue
        if cases.get("best") is not None:
            sections.append(format_case(cases["best"], "Best plan"))
        if cases.get("worst") is not None:
            sections.append(format_case(cases["worst"], "Worst plan"))
        if cases.get("flip") is not None:
            flip = cases["flip"]
            sections.append(
                f"### STANDARD vs CONTROL flip\n"
                f"- **Image**: `{flip['question_id']}`\n"
                f"- STANDARD coherence: {fmt(flip['std_score'])}\n"
                f"- CONTROL coherence: {fmt(flip['ctrl_score'])}\n"
                f"- Delta: {fmt(flip['delta'])}\n"
            )
        sections.append("")

    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Case studies ->{out_path}")


# ---------------------------------------------------------------------------
# narrative report
# ---------------------------------------------------------------------------

def write_report(
    df: pd.DataFrame,
    cond_summary: pd.DataFrame,
    out_path: Path,
):
    gt = cond_summary[cond_summary["track"] == "gt"].set_index("condition")

    def row(c):
        return gt.loc[c] if c in gt.index else {}

    std  = row("standard_v2")
    ctrl = row("control_v2")
    abl  = row("ablation_v2")

    lines = [
        "# CASTOR Improved — Evaluation Report",
        "",
        "## Experiment summary",
        "",
        "Three prompt conditions tested on Qwen3-VL 8B (baseline method, 110 images each):",
        "- **STANDARD**: specific, discriminative domain assertions",
        "- **CONTROL**: vague, tautological assertions (same count)",
        "- **ABLATION**: no assertions",
        "",
        "Coherence score = 0.40 × SEQ + 0.35 × MTH + 0.25 × SPC",
        "",
        "## Results (GT track)",
        "",
        "| Condition | N | SEQ | MTH | SPC | Coherence |",
        "|---|---|---|---|---|---|",
    ]

    for label, r in [("standard_v2", std), ("control_v2", ctrl), ("ablation_v2", abl)]:
        if not isinstance(r, pd.Series) or r.empty:
            lines.append(f"| {label} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {label} | {int(r.get('n', 0))} "
            f"| {fmt(r.get('seq_score'))} "
            f"| {fmt(r.get('method_valid'))} "
            f"| {fmt(r.get('specific'))} "
            f"| {fmt(r.get('coherence_score'))} |"
        )

    lines += [
        "",
        "## Hypothesis assessment",
        "",
    ]

    # Assess hypothesis
    try:
        std_coh  = std["coherence_score"]
        ctrl_coh = ctrl["coherence_score"]
        abl_coh  = abl["coherence_score"]
        delta_sc = std_coh - ctrl_coh
        delta_sa = std_coh - abl_coh
        delta_ca = ctrl_coh - abl_coh

        lines.append(
            f"Hypothesis: STANDARD > CONTROL ≈ ABLATION\n\n"
            f"- STANDARD − CONTROL = {fmt(delta_sc)}  "
            f"({'✓ STANDARD wins' if delta_sc > 0.05 else '✗ no meaningful gap' if delta_sc < 0.01 else '~ marginal'})\n"
            f"- STANDARD − ABLATION = {fmt(delta_sa)}\n"
            f"- CONTROL − ABLATION  = {fmt(delta_ca)}  "
            f"({'~ close as expected' if abs(delta_ca) < 0.05 else '✗ CONTROL ≠ ABLATION — gap is {:.3f}'.format(abs(delta_ca))})\n"
        )
    except Exception as e:
        lines.append(f"_Could not assess hypothesis: {e}_\n")

    lines += [
        "",
        "## Parse failure rates",
        "",
        "| Condition | Parse failed % |",
        "|---|---|",
    ]
    gt2 = cond_summary[cond_summary["track"] == "gt"]
    for _, r in gt2.iterrows():
        lines.append(f"| {r['condition']} | {r['parse_failed_pct']*100:.1f}% |")

    lines += [
        "",
        "## SEQ failures by condition (GT track)",
        "",
    ]

    for cond in ["standard_v2", "control_v2", "ablation_v2"]:
        sub = df[(df["condition"] == cond) & (df["eval_track"] == "gt")]
        if sub.empty:
            continue
        all_failures: list[str] = []
        for row_failures in sub["seq_failures"]:
            if isinstance(row_failures, list):
                all_failures.extend(row_failures)
        from collections import Counter
        counts = Counter(all_failures)
        lines.append(f"**{cond}**")
        if counts:
            for chain, cnt in counts.most_common(5):
                lines.append(f"  - {chain}: {cnt} plans")
        else:
            lines.append("  - (none)")
        lines.append("")

    lines += [
        "",
        "## Per-state breakdown (GT track)",
        "",
        "| Condition | State | N | SEQ | MTH | SPC | Coherence |",
        "|---|---|---|---|---|---|---|",
    ]

    state_summary = summary_by_condition_state(df)
    gt_state = state_summary[state_summary["track"] == "gt"]
    for _, r in gt_state.iterrows():
        lines.append(
            f"| {r['condition']} | {r['eval_state']} | {int(r['n'])} "
            f"| {fmt(r.get('seq_score'))} "
            f"| {fmt(r.get('method_valid'))} "
            f"| {fmt(r.get('specific'))} "
            f"| {fmt(r.get('coherence_score'))} |"
        )

    lines += [
        "",
        "---",
        "_See case_studies.md for per-plan examples._",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report ->{out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="improved/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    import os
    pipeline_dir = Path(os.path.expandvars(cfg["paths"]["pipeline_dir"]))
    results_dir  = pipeline_dir / "results"

    scores_path = results_dir / "judge_scores_improved.jsonl"
    if not scores_path.exists():
        print(f"[ERROR] {scores_path} not found. Run run_judge_v2.py first.", file=sys.stderr)
        sys.exit(1)

    df = load_jsonl(str(scores_path))
    print(f"Loaded {len(df)} scored rows.")

    # Coerce boolean columns
    df["method_valid"] = df["method_valid"].map(
        lambda x: 1.0 if x is True else (0.0 if x is False else float("nan"))
    )
    df["specific"] = df["specific"].map(
        lambda x: 1.0 if x is True else (0.0 if x is False else float("nan"))
    )

    # Summaries
    cond_summary = summary_by_condition(df)
    cond_state_summary = summary_by_condition_state(df)

    cond_summary.to_csv(results_dir / "summary_by_condition.csv", index=False)
    cond_state_summary.to_csv(results_dir / "summary_by_condition_state.csv", index=False)
    print(f"Summaries written to {results_dir}")

    # Case studies
    write_case_studies(df, results_dir / "case_studies.md")

    # Narrative report
    write_report(df, cond_summary, results_dir / "report.md")

    print("\nDone. Key files:")
    print(f"  {results_dir}/summary_by_condition.csv")
    print(f"  {results_dir}/summary_by_condition_state.csv")
    print(f"  {results_dir}/case_studies.md")
    print(f"  {results_dir}/report.md")


if __name__ == "__main__":
    main()
