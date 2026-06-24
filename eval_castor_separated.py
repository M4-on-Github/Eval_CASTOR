"""
CASTOR Separated-Parts Evaluator
Evaluates inference runs where each field was prompted separately
(one JSONL per field: state, vessel_type, size, cargo, ...).

Records are joined across files by the 'image' key.
No JSON/CoT extraction needed — text is a direct short answer per file.

Outputs parallel to eval_castor.py for side-by-side comparison.

Run from anywhere:
    python DeGF/Eval_CASTOR/eval_castor_separated.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Reuse shared functions from eval_castor.py (same directory)
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from eval_castor import (  # noqa: E402
    VALID_STATES,
    load_ground_truth,
    normalize_state,
    normalize_size,
    vessel_jaccard,
    cargo_match,
    per_state_report,
    summary_row,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GT_PATH     = HERE / "human_ground_truth_label" / "human_gt.csv"
RESULTS_DIR = HERE.parent.parent / "results" / "separated_into_parts"
OUT_DIR     = HERE / "results"

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_separated_runs(castor_results_dir: Path) -> list:
    """Find all separated_into_parts_* subdirectories."""
    runs = []
    for subdir in sorted(castor_results_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("separated_into_parts_"):
            diffusion = "degf" in subdir.name.lower()
            runs.append((subdir, subdir.name, diffusion))
            print(f"  Discovered: {subdir.name}  diffusion={diffusion}")
    return runs

# ---------------------------------------------------------------------------
# Field loaders
# ---------------------------------------------------------------------------

def _load_jsonl_lines(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    records.append(json.loads(line, strict=False))
                except json.JSONDecodeError as e:
                    print(f"  WARNING: line {lineno} in {path.name}: {e}")
    return records


def load_field_file(subdir: Path, pattern: str) -> dict:
    """Load first JSONL matching pattern. Returns {image -> text}."""
    matches = sorted(subdir.glob(pattern))
    if not matches:
        print(f"  WARNING: no file matching '{pattern}' in {subdir.name}")
        return {}
    recs = _load_jsonl_lines(matches[0])
    return {r["image"]: r["text"] for r in recs if "image" in r and "text" in r}


def load_timing_file(subdir: Path, pattern: str) -> dict:
    """Load timing (infer_s) from first JSONL matching pattern. Returns {image -> float}."""
    matches = sorted(subdir.glob(pattern))
    if not matches:
        return {}
    recs = _load_jsonl_lines(matches[0])
    result = {}
    for r in recs:
        if "image" in r:
            result[r["image"]] = r.get("timing", {}).get("infer_s")
    return result

# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_separated_run(subdir: Path, gt_dict: dict) -> pd.DataFrame:
    states  = load_field_file(subdir, "*_p1_*_state.jsonl")
    types   = load_field_file(subdir, "*_p2_*_type.jsonl")
    sizes   = load_field_file(subdir, "*_p3_*_size.jsonl")
    cargoes = load_field_file(subdir, "*_p4_*_cargo.jsonl")
    timings = load_timing_file(subdir, "*_p1_*_state.jsonl")

    all_images = sorted(set(states) | set(types) | set(sizes) | set(cargoes))

    rows = []
    for img in all_images:
        gt = gt_dict.get(img, {})

        raw_state    = states.get(img, "")
        pred_state   = normalize_state(raw_state)
        gt_state     = gt.get("state", "")
        state_correct = (pred_state == gt_state) if pred_state != "UNPARSEABLE" else False

        pred_vessel = types.get(img, "")
        jac = vessel_jaccard(gt.get("vessel_type", ""), pred_vessel)

        pred_size_raw    = sizes.get(img, "")
        pred_size_bucket = normalize_size(pred_size_raw)
        gt_size_bucket   = normalize_size(gt.get("size_estimate", ""))
        size_correct     = (pred_size_bucket == gt_size_bucket) if pred_size_bucket != "unknown" else False

        pred_cargo = cargoes.get(img, None)
        cmatch = cargo_match(gt.get("cargo", ""), pred_cargo)

        parse_error = (pred_state == "UNPARSEABLE")
        parse_fail_reason = f"state_unparseable: {raw_state!r}" if parse_error else ""

        rows.append({
            "image":            img,
            "gt_state":         gt_state,
            "pred_state":       pred_state,
            "state_correct":    state_correct,
            # Q columns — N/A for separated format (no Q prompts)
            "gt_q1": gt.get("q1"), "gt_q2": gt.get("q2"), "gt_q3": gt.get("q3"),
            "gt_q4": gt.get("q4"), "gt_q5": gt.get("q5"),
            "pred_q1": None, "pred_q2": None, "pred_q3": None,
            "pred_q4": None, "pred_q5": None,
            "q1_correct": None, "q2_correct": None, "q3_correct": None,
            "q4_correct": None, "q5_correct": None,
            "gt_size_bucket":   gt_size_bucket,
            "pred_size_bucket": pred_size_bucket,
            "size_correct":     size_correct,
            "vessel_jaccard":   round(jac, 3),
            "cargo_match":      cmatch,
            "parse_error":      parse_error,
            "parse_fail_reason": parse_fail_reason,
            "infer_s":          timings.get(img),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth ...")
    gt = load_ground_truth(GT_PATH)
    print(f"  {len(gt)} records.")

    print(f"\nScanning {RESULTS_DIR} for separated runs ...")
    runs = discover_separated_runs(RESULTS_DIR)
    if not runs:
        print("  No separated_into_parts_* directories found. Exiting.")
        return
    print(f"  {len(runs)} run(s) found.\n")

    summary_rows = []

    for subdir, run_name, diffusion in runs:
        print(f"\n{'-'*60}")
        print(f"  Run: {run_name}")
        print(f"{'-'*60}")

        df = evaluate_separated_run(subdir, gt)
        print(f"  {len(df)} records evaluated.")

        if df.empty:
            print("  No records found — skipping (directory may not have inference outputs yet).")
            continue

        csv_out = OUT_DIR / f"eval_{run_name}.csv"
        df.to_csv(csv_out, index=False)
        print(f"  Per-entry CSV  -> {csv_out.relative_to(HERE)}")

        report = per_state_report(df, run_name)
        print(report)
        (OUT_DIR / f"eval_{run_name}_report.txt").write_text(report, encoding="utf-8")

        failures = df[df["parse_error"]][["image", "gt_state", "parse_fail_reason"]]
        if not failures.empty:
            err_path = OUT_DIR / f"parse_errors_{run_name}.txt"
            lines = [
                f"Parse failures for {run_name}  ({len(failures)}/{len(df)} records)",
                "=" * 70,
            ]
            reason_counts: dict = {}
            for reason in failures["parse_fail_reason"]:
                key = reason.split(":")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
            lines.append("Reason summary:")
            for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("Per-record detail:")
            lines.append("-" * 70)
            for _, row in failures.iterrows():
                lines.append(f"[{row['gt_state']:8s}]  {row['image']}")
                lines.append(f"           {row['parse_fail_reason']}")
            err_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  Parse errors   -> {err_path.relative_to(HERE)}")

        summary_rows.append(summary_row(df, run_name, diffusion, "direct"))

    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "eval_summary_separated.csv"
    try:
        summary_df.to_csv(summary_csv, index=False)
    except PermissionError:
        print(f"\n  WARNING: Could not write {summary_csv.name} — close it in Excel and re-run.")

    SEP = "=" * 100
    print(f"\n{SEP}")
    print("  HOLISTIC SUMMARY (SEPARATED FORMAT)")
    print(SEP)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}".format)

    key_cols = [
        "run", "diffusion", "prompt_style",
        "n_parsed", "parse_fail_%", "state_acc_%",
        "acc_aground_%", "acc_capsized_%", "acc_on_fire_%", "acc_sunken_%",
        "macro_f1_%",
        "size_bucket_acc_%", "mean_vessel_jaccard",
        "mean_infer_s", "median_infer_s",
    ]
    key_cols = [c for c in key_cols if c in summary_df.columns]
    print(summary_df[key_cols].to_string(index=False))
    print(f"\nSummary CSV -> {summary_csv.relative_to(HERE)}")


if __name__ == "__main__":
    main()
