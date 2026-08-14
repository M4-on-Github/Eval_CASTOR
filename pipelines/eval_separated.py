"""
CASTOR Separated-Parts Evaluator — Pipeline 4

Evaluates inference runs where each field was prompted separately
(one JSONL per field: state, vessel_type, size, cargo, ...).

Records are joined across files by the 'image' key.
No JSON/CoT extraction needed — text is a direct short answer per file.

Output -> results/p4_separated/

Run from anywhere:
    python pipelines/eval_separated.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

EVAL_ROOT   = Path(__file__).parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_ground_truth, used_diffusion
from shared.metrics import (
    VALID_STATES, normalize_state, normalize_size,
    vessel_jaccard, cargo_match, per_state_report, confusion_matrix_report, summary_row,
)

GT_PATH     = EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv"
RESULTS_IN  = EVAL_ROOT.parent / "results" / "separated_into_parts"
OUT_DIR     = EVAL_ROOT / "results" / "p4_separated"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class SeparatedRunDiscovery:
    """Finds separated-field runs, which are DIRECTORIES rather than files.

    P4 evaluates the multi-turn prompt format, where each field is asked as its
    own question, so one run is a directory of per-field JSONL files instead of
    a single answers file. That is why this discovery differs from P1's.

    Both accepted prefixes are load-bearing. Early runs were written with a
    misspelling ("separeted_"), and those directories still exist in the
    results tree — dropping the typo would silently make historical runs
    invisible rather than raising, so the tolerance is kept deliberately.
    """

    #: Correct spelling, then the historical misspelling. Order is cosmetic.
    PREFIXES = ("separated_into_parts_", "separeted_into_parts_")

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    @classmethod
    def is_run_dir(cls, path: Path) -> bool:
        return path.is_dir() and path.name.startswith(cls.PREFIXES)

    def discover(self) -> list:
        """Return [(dir_path, dir_name, used_diffusion), ...], sorted."""
        if not self.results_dir.exists():
            print(f"  Directory not found: {self.results_dir}")
            return []
        runs = []
        for subdir in sorted(self.results_dir.iterdir()):
            if self.is_run_dir(subdir):
                # Shared with eval_castor.py so both pipelines classify a run
                # the same way; a disagreement would not raise, it would just
                # make their tables inconsistent.
                diffusion = used_diffusion(subdir.name)
                runs.append((subdir, subdir.name, diffusion))
                print(f"  Discovered: {subdir.name}  diffusion={diffusion}")
        return runs


def discover_runs(results_dir: Path) -> list:
    """Find separated-field runs. Facade over SeparatedRunDiscovery."""
    return SeparatedRunDiscovery(results_dir).discover()


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


def load_field_file(subdir: Path, *patterns: str) -> dict:
    """Load first JSONL matching any of the patterns. Returns {image -> text}."""
    for pattern in patterns:
        matches = sorted(subdir.glob(pattern))
        if matches:
            recs = _load_jsonl_lines(matches[0])
            return {r["image"]: r["text"] for r in recs if "image" in r and "text" in r}
    print(f"  WARNING: no file matching {patterns} in {subdir.name}")
    return {}


def load_timing_file(subdir: Path, *patterns: str) -> dict:
    for pattern in patterns:
        matches = sorted(subdir.glob(pattern))
        if matches:
            recs = _load_jsonl_lines(matches[0])
            result = {}
            for r in recs:
                if "image" in r:
                    result[r["image"]] = r.get("timing", {}).get("infer_s")
            return result
    return {}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_run(subdir: Path, gt_dict: dict) -> pd.DataFrame:
    # Two naming conventions:
    #   LLaVA: answers_baseline_p1_1_state.jsonl
    #   QWEN:  answers_qwen3vl8b_baseline_1_state_j932.jsonl
    states  = load_field_file(subdir, "*_p1_*_state.jsonl", "*_1_state*.jsonl")
    types   = load_field_file(subdir, "*_p2_*_type.jsonl",  "*_2_type*.jsonl")
    sizes   = load_field_file(subdir, "*_p3_*_size.jsonl",  "*_3_size*.jsonl")
    cargoes = load_field_file(subdir, "*_p4_*_cargo.jsonl", "*_4_cargo*.jsonl")
    timings = load_timing_file(subdir, "*_p1_*_state.jsonl", "*_1_state*.jsonl")

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

    print(f"\nScanning {RESULTS_IN} ...")
    runs = discover_runs(RESULTS_IN)
    if not runs:
        print("  No separated_into_parts_* directories found. Exiting.")
        return
    print(f"  {len(runs)} run(s) found.\n")

    summary_rows = []
    confusion_matrices = []
    all_dfs = []

    for subdir, run_name, diffusion in runs:
        print(f"\n{'-'*60}")
        print(f"  Run: {run_name}")
        print(f"{'-'*60}")

        df = evaluate_run(subdir, gt)
        print(f"  {len(df)} records evaluated.")

        if df.empty:
            print("  No records found — skipping.")
            continue

        csv_out = OUT_DIR / f"eval_{run_name}.csv"
        df.to_csv(csv_out, index=False)
        print(f"  Per-entry CSV  -> {csv_out.relative_to(EVAL_ROOT)}")

        report = per_state_report(df, run_name)
        print(report)
        (OUT_DIR / f"eval_{run_name}_report.txt").write_text(report, encoding="utf-8")

        cm = confusion_matrix_report(df, run_name)
        print(cm)
        (OUT_DIR / f"eval_{run_name}_confusion.txt").write_text(cm, encoding="utf-8")
        print(f"  Confusion matrix -> {(OUT_DIR / f'eval_{run_name}_confusion.txt').relative_to(EVAL_ROOT)}")
        confusion_matrices.append(cm)
        all_dfs.append(df)

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
            print(f"  Parse errors   -> {err_path.relative_to(EVAL_ROOT)}")

        summary_rows.append(summary_row(df, run_name, diffusion, "direct"))

    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "eval_summary_separated.csv"
    try:
        summary_df.to_csv(summary_csv, index=False)
    except PermissionError:
        print(f"\n  WARNING: Could not write {summary_csv.name}")

    SEP = "=" * 100
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}".format)
    key_cols = [
        "run", "diffusion", "prompt_style",
        "n_parsed", "parse_fail_%", "state_acc_%",
        "acc_aground_%", "acc_capsized_%", "acc_on_fire_%", "acc_sunken_%",
        "macro_f1_%", "size_bucket_acc_%", "mean_vessel_jaccard",
        "mean_infer_s", "median_infer_s",
    ]
    key_cols = [c for c in key_cols if c in summary_df.columns]
    table = summary_df[key_cols].to_string(index=False)
    print(f"\n{SEP}")
    print("  HOLISTIC SUMMARY (Pipeline 4 — Separated Format)")
    print(SEP)
    print(table)
    combined_cm = confusion_matrix_report(
        pd.concat(all_dfs, ignore_index=True), "ALL RUNS COMBINED"
    ) if all_dfs else ""
    summary_report = OUT_DIR / "eval_summary_separated_report.txt"
    summary_report.write_text(
        "\n\n".join(confusion_matrices) + ("\n\n" + combined_cm if combined_cm else ""),
        encoding="utf-8",
    )
    print(f"Summary CSV    -> {summary_csv.relative_to(EVAL_ROOT)}")
    print(f"Summary report -> {summary_report.relative_to(EVAL_ROOT)}")


if __name__ == "__main__":
    main()
