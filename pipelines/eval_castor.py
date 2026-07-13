"""
CASTOR Evaluator — Pipeline 1 (regex) and Pipeline 2 (LLM-extract)

Pipeline 1 — regex extraction:
    python pipelines/eval_castor.py
    Output -> results/p1_regex/eval_summary.csv

Pipeline 2 — Gemma-extracted fields (run extract_gemma.py first):
    python pipelines/extract_gemma.py
    python pipelines/eval_castor.py --pre-parsed
    Output -> results/p2_llm_extract/eval_summary.csv

The two modes write to SEPARATE output directories so neither overwrites the other.
Run from anywhere (paths are relative to this file).
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

EVAL_ROOT   = Path(__file__).parent.parent          # Eval_CASTOR/
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders  import load_ground_truth, load_run, load_pre_parsed, _safe_str
from shared.metrics  import (
    VALID_STATES, extract_json_block, normalize_state, extract_q_answers,
    normalize_size, vessel_jaccard, cargo_match, gemma_val,
    per_state_report, confusion_matrix_report, summary_row,
)

GT_PATH     = EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv"
RESULTS_IN  = EVAL_ROOT.parent / "results" / "castor_results"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(results_dir: Path) -> list:
    runs = []
    for path in sorted(results_dir.glob("*.jsonl")):
        fname     = path.name
        diffusion = "degf" in path.stem.lower()
        prompt_style = "direct"
        try:
            import json
            with open(path, encoding="utf-8") as f:
                first_rec = json.loads(f.readline())
            text = first_rec.get("text", "")
            if re.search(r'Step\s*[12]', text, re.IGNORECASE):
                prompt_style = "cot"
        except Exception:
            try:
                from shared.loaders import _load_slash_n_jsonl
                recs = _load_slash_n_jsonl(path)
                if recs and re.search(r'Step\s*[12]', recs[0].get("text", ""), re.IGNORECASE):
                    prompt_style = "cot"
            except Exception:
                pass
        runs.append((fname, diffusion, prompt_style))
        print(f"  Discovered: {fname}  diffusion={diffusion}  prompt_style={prompt_style}")
    return runs


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_run(records: list, gt_dict: dict, prompt_style: str,
                 pre_parsed_dict: dict = None) -> pd.DataFrame:
    rows = []
    for rec in records:
        img  = rec["image"]
        text = rec.get("text", "")
        gt   = gt_dict.get(img, {})

        if pre_parsed_dict and img in pre_parsed_dict:
            pp = pre_parsed_dict[img]
            parsed = {
                "state":         gemma_val(pp.get("state")),
                "vessel_type":   gemma_val(pp.get("vessel_type")),
                "size_estimate": gemma_val(pp.get("size_estimate")),
                "cargo":         gemma_val(pp.get("cargo")),
            }
            parse_fail_reason = ""
            gemma_qs = {f"q{i}": gemma_val(pp.get(f"q{i}")) for i in range(1, 6)}
        else:
            parsed, parse_fail_reason = extract_json_block(text)
            gemma_qs = None

        parse_error    = parsed is None
        pred_state     = normalize_state(parsed.get("state") if parsed else None)
        pred_vessel    = _safe_str(parsed.get("vessel_type") if parsed else None)
        pred_size_raw  = _safe_str(parsed.get("size_estimate") if parsed else None)
        pred_cargo_raw = parsed.get("cargo") if parsed else None

        pred_qs = gemma_qs if gemma_qs is not None else extract_q_answers(text, prompt_style)

        gt_state         = gt.get("state", "")
        gt_size_bucket   = normalize_size(gt.get("size_estimate", ""))
        pred_size_bucket = normalize_size(pred_size_raw)
        state_correct    = (pred_state == gt_state) if pred_state != "UNPARSEABLE" else False

        q_correct = {}
        for i in range(1, 6):
            key = f"q{i}"
            pq  = pred_qs[key]
            q_correct[key] = None if pq is None else (pq == gt.get(key, ""))

        jac    = vessel_jaccard(gt.get("vessel_type", ""), pred_vessel)
        cmatch = cargo_match(gt.get("cargo", ""), pred_cargo_raw)

        rows.append({
            "image":            img,
            "gt_state":         gt_state,
            "pred_state":       pred_state,
            "state_correct":    state_correct,
            "gt_q1":  gt.get("q1"), "gt_q2":  gt.get("q2"), "gt_q3":  gt.get("q3"),
            "gt_q4":  gt.get("q4"), "gt_q5":  gt.get("q5"),
            "pred_q1": pred_qs["q1"], "pred_q2": pred_qs["q2"], "pred_q3": pred_qs["q3"],
            "pred_q4": pred_qs["q4"], "pred_q5": pred_qs["q5"],
            "q1_correct": q_correct["q1"], "q2_correct": q_correct["q2"],
            "q3_correct": q_correct["q3"], "q4_correct": q_correct["q4"],
            "q5_correct": q_correct["q5"],
            "gt_size_bucket":   gt_size_bucket,
            "pred_size_bucket": pred_size_bucket,
            "size_correct":     (pred_size_bucket == gt_size_bucket) if pred_size_bucket != "unknown" else False,
            "vessel_jaccard":   round(jac, 3),
            "cargo_match":      cmatch,
            "parse_error":        parse_error,
            "parse_fail_reason":  parse_fail_reason if parse_error else "",
            "infer_s":            rec.get("timing", {}).get("infer_s"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CASTOR inference against human GT (Pipeline 1 or 2)."
    )
    parser.add_argument(
        "--pre-parsed", action="store_true",
        help="Pipeline 2: load Gemma-extracted fields from results/p2_llm_extract/extracted/",
    )
    args = parser.parse_args()

    if args.pre_parsed:
        OUT_DIR   = EVAL_ROOT / "results" / "p2_llm_extract"
        GEMMA_DIR = OUT_DIR / "extracted"
        pipeline  = "Pipeline 2 (LLM extract + eval)"
    else:
        OUT_DIR   = EVAL_ROOT / "results" / "p1_regex"
        GEMMA_DIR = None
        pipeline  = "Pipeline 1 (regex only)"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline : {pipeline}")
    print(f"Output   : {OUT_DIR.relative_to(EVAL_ROOT)}")

    print("\nLoading ground truth ...")
    gt = load_ground_truth(GT_PATH)
    print(f"  {len(gt)} records.")

    print(f"\nScanning {RESULTS_IN} ...")
    runs = discover_runs(RESULTS_IN)
    if not runs:
        print("  No JSONL files found. Exiting.")
        return
    print(f"  {len(runs)} run(s) found.\n")

    summary_rows = []
    confusion_matrices = []
    all_dfs = []

    for fname, diffusion, prompt_style in runs:
        run_name = fname.replace(".jsonl", "")
        print(f"\n{'-'*60}")
        print(f"  Run: {run_name}")
        print(f"{'-'*60}")

        records = load_run(RESULTS_IN / fname)
        print(f"  {len(records)} inference records.")

        pre_parsed_dict = None
        if args.pre_parsed and GEMMA_DIR:
            pp_path = GEMMA_DIR / f"{run_name}_gemma.jsonl"
            if pp_path.exists():
                pre_parsed_dict = load_pre_parsed(pp_path)
                print(f"  Pre-parsed: {len(pre_parsed_dict)} Gemma records loaded.")
            else:
                print(f"  Pre-parsed: {pp_path.name} not found — using regex fallback.")

        df = evaluate_run(records, gt, prompt_style, pre_parsed_dict)

        if df.empty:
            print("  No records evaluated — skipping.")
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

        summary_rows.append(summary_row(df, run_name, diffusion, prompt_style))

    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "eval_summary.csv"
    try:
        summary_df.to_csv(summary_csv, index=False)
    except PermissionError:
        print(f"\n  WARNING: Could not write {summary_csv.name} — close it in Excel and re-run.")

    SEP = "=" * 100
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}".format)
    key_cols = [
        "run", "diffusion", "prompt_style",
        "n_parsed", "parse_fail_%", "state_acc_%",
        "acc_aground_%", "acc_capsized_%", "acc_on_fire_%", "acc_sunken_%",
        "macro_f1_%",
        "q1_acc_%", "q2_acc_%", "q3_acc_%", "q4_acc_%", "q5_acc_%",
        "size_bucket_acc_%", "mean_vessel_jaccard",
        "mean_infer_s", "median_infer_s",
    ]
    key_cols = [c for c in key_cols if c in summary_df.columns]
    table = summary_df[key_cols].to_string(index=False)
    print(f"\n{SEP}")
    print(f"  HOLISTIC SUMMARY  [{pipeline}]")
    print(SEP)
    print(table)
    combined_cm = confusion_matrix_report(
        pd.concat(all_dfs, ignore_index=True), "ALL RUNS COMBINED"
    ) if all_dfs else ""
    summary_report = OUT_DIR / "eval_summary_report.txt"
    summary_report.write_text(
        "\n\n".join(confusion_matrices) + ("\n\n" + combined_cm if combined_cm else ""),
        encoding="utf-8",
    )
    print(f"Summary CSV    -> {summary_csv.relative_to(EVAL_ROOT)}")
    print(f"Summary report -> {summary_report.relative_to(EVAL_ROOT)}")


if __name__ == "__main__":
    main()
