"""
Pipeline 8 — panel aggregation for plan coherence.

Merges 5 per-model judge CSVs into consensus per (image, step), computes
n_invalid and majority_invalid, and writes _per_step.csv, _per_image.csv,
_summary.csv, and appends to eval_summary_coherence.csv.

_summary.csv includes Fleiss' κ (inter-rater reliability) and per-judge
invalid rates + majority-agreement fractions to flag outlier judges.

Usage:
  python aggregate_coherence.py --run answers_baseline \\
      --dir results/p8_plan_coherence/
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

JUDGE_MODELS = [
    "deepseek_r1_32b",
    "glm4_32b",
    "llama_3_3_70b",
    "phi4_14b",
    "gemma4_31b",
]

MAJORITY_THRESHOLD = 3  # n_invalid >= this → majority_invalid = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_judge_csv(path: Path) -> dict[tuple[str, int], dict]:
    """Load a per-model judge CSV. Returns {(image, step_num) -> row}."""
    result = {}
    if not path.exists():
        return result
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["image"], int(row["step_num"]))
            result[key] = row
    return result


def _safe_mean(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 3) if vals else None


def _fleiss_kappa(per_step_rows: list[dict]) -> float | None:
    """
    Fleiss' κ over binary judge votes (valid=1, invalid=0).
    Skips steps where any judge vote is missing or 'error'.
    Returns None if fewer than 2 fully-rated steps.
    """
    n = len(JUDGE_MODELS)
    counts = []
    for row in per_step_rows:
        votes = [row.get(f"{m}_valid", "") for m in JUDGE_MODELS]
        if any(v not in ("0", "1") for v in votes):
            continue
        n_inv = sum(1 for v in votes if v == "0")
        counts.append((n - n_inv, n_inv))  # (n_valid, n_invalid)

    N = len(counts)
    if N < 2:
        return None

    P_bar = sum(nv * (nv - 1) + ni * (ni - 1) for nv, ni in counts) / (N * n * (n - 1))
    total = N * n
    p_valid   = sum(nv for nv, _ in counts) / total
    p_invalid = sum(ni for _, ni in counts) / total
    P_e = p_valid ** 2 + p_invalid ** 2
    if P_e >= 1.0:
        return None
    return round((P_bar - P_e) / (1 - P_e), 4)


def _judge_stats(per_step_rows: list[dict]) -> dict[str, dict]:
    """Per-judge invalid rate and fraction of steps agreeing with majority vote."""
    stats = {}
    for model in JUDGE_MODELS:
        invalid_flags, agree_flags = [], []
        for row in per_step_rows:
            v = row.get(f"{model}_valid", "")
            if v not in ("0", "1"):
                continue
            invalid_flags.append(v == "0")
            agree_flags.append((v == "0") == (row["majority_invalid"] == "1"))
        stats[model] = {
            "invalid_rate":        round(sum(invalid_flags) / len(invalid_flags), 3) if invalid_flags else None,
            "majority_agreement":  round(sum(agree_flags)  / len(agree_flags),   3) if agree_flags  else None,
        }
    return stats


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate_run(run_name: str, run_dir: Path):
    # ── Load all judge CSVs ───────────────────────────────────────────────────
    judge_data: dict[str, dict] = {}
    for model in JUDGE_MODELS:
        p = run_dir / f"{run_name}_{model}.csv"
        if p.exists():
            judge_data[model] = _load_judge_csv(p)
            print(f"  Loaded {model}: {len(judge_data[model])} step rows")
        else:
            print(f"  WARNING: missing judge file {p.name}")
            judge_data[model] = {}

    # ── Union of all (image, step_num) keys ──────────────────────────────────
    all_keys: set[tuple[str, int]] = set()
    for data in judge_data.values():
        all_keys |= data.keys()

    if not all_keys:
        print("  ERROR: no judge data found — nothing to aggregate.")
        return

    # ── Build per-step rows ───────────────────────────────────────────────────
    # Collect gt_state and step_text from first available judge
    key_meta: dict[tuple[str, int], dict] = {}
    for key in all_keys:
        for model in JUDGE_MODELS:
            if key in judge_data[model]:
                key_meta[key] = judge_data[model][key]
                break

    per_step_rows = []
    for key in sorted(all_keys):
        image, step_num = key
        meta = key_meta[key]

        # Count n_invalid: judges that said "0" (explicitly invalid)
        # "1" = valid, "0" = invalid, "error" or "" = not counted
        n_invalid = sum(
            1 for model in JUDGE_MODELS
            if judge_data[model].get(key, {}).get("valid") == "0"
        )
        majority_invalid = n_invalid >= MAJORITY_THRESHOLD

        row = {
            "image":            image,
            "gt_state":         meta.get("gt_state", ""),
            "step_num":         step_num,
            "step_text":        meta.get("step_text", ""),
            "n_invalid":        n_invalid,
            "majority_invalid": "1" if majority_invalid else "0",
        }
        for model in JUDGE_MODELS:
            judge_row = judge_data[model].get(key, {})
            row[f"{model}_valid"]  = judge_row.get("valid", "")
            row[f"{model}_reason"] = judge_row.get("reason", "")

        per_step_rows.append(row)

    # ── _per_step.csv ─────────────────────────────────────────────────────────
    per_step_path = run_dir / f"{run_name}_per_step.csv"
    step_fieldnames = (
        ["image", "gt_state", "step_num", "step_text", "n_invalid", "majority_invalid"]
        + [f"{m}_valid"  for m in JUDGE_MODELS]
        + [f"{m}_reason" for m in JUDGE_MODELS]
    )
    with open(per_step_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=step_fieldnames)
        w.writeheader()
        w.writerows(per_step_rows)
    print(f"  Per-step  → {per_step_path}  ({len(per_step_rows)} rows)")

    # ── _per_image.csv ────────────────────────────────────────────────────────
    image_steps: dict[str, list[dict]] = defaultdict(list)
    for row in per_step_rows:
        image_steps[row["image"]].append(row)

    per_image_rows = []
    for image in sorted(image_steps):
        steps = sorted(image_steps[image], key=lambda r: int(r["step_num"]))
        n_steps           = len(steps)
        n_maj_invalid     = sum(1 for s in steps if s["majority_invalid"] == "1")
        coherence_pct     = round((n_steps - n_maj_invalid) / n_steps, 3) if n_steps else None
        first_fail        = next(
            (s["step_num"] for s in steps if s["majority_invalid"] == "1"), ""
        )
        gt_state = steps[0]["gt_state"] if steps else ""
        per_image_rows.append({
            "image":                      image,
            "gt_state":                   gt_state,
            "n_steps":                    n_steps,
            "n_majority_invalid":         n_maj_invalid,
            "coherence_pct":              coherence_pct,
            "first_majority_invalid_step": first_fail,
        })

    per_image_path = run_dir / f"{run_name}_per_image.csv"
    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)
    print(f"  Per-image → {per_image_path}  ({len(per_image_rows)} rows)")

    # ── _summary.csv + eval_summary_coherence.csv ─────────────────────────────
    valid_coh = [float(r["coherence_pct"]) for r in per_image_rows
                 if r["coherence_pct"] not in ("", None)]
    pct_fully = sum(1 for r in per_image_rows if r["n_majority_invalid"] == 0) / len(per_image_rows)
    first_fails = [int(r["first_majority_invalid_step"]) for r in per_image_rows
                   if r["first_majority_invalid_step"] not in ("", None)]

    def _state_mean(field, state):
        vals = [float(r[field]) for r in per_image_rows
                if r["gt_state"] == state and r.get(field) not in ("", None)]
        return _safe_mean(vals)

    kappa = _fleiss_kappa(per_step_rows)
    jstats = _judge_stats(per_step_rows)

    summary = {
        "run":                       run_name,
        "n_images":                  len(per_image_rows),
        "mean_coherence_pct":        _safe_mean(valid_coh),
        "pct_fully_coherent":        round(pct_fully, 3),
        "mean_first_invalid_step":   _safe_mean([float(v) for v in first_fails]),
        "coherence_aground":         _state_mean("coherence_pct", "aground"),
        "coherence_capsized":        _state_mean("coherence_pct", "capsized"),
        "coherence_on_fire":         _state_mean("coherence_pct", "on_fire"),
        "coherence_sunken":          _state_mean("coherence_pct", "sunken"),
        # inter-rater reliability
        "fleiss_kappa":              kappa,
        # per-judge diagnostics
        **{f"{m}_invalid_rate":       jstats[m]["invalid_rate"]       for m in JUDGE_MODELS},
        **{f"{m}_majority_agreement": jstats[m]["majority_agreement"] for m in JUDGE_MODELS},
    }

    summary_path = run_dir / f"{run_name}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(f"  Summary   → {summary_path}")

    cumulative_path = run_dir.parent / "eval_summary_coherence.csv"
    write_header = not cumulative_path.exists()
    with open(cumulative_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)

    print(f"\n  mean_coherence={summary['mean_coherence_pct']}  "
          f"pct_fully_coherent={summary['pct_fully_coherent']}  "
          f"mean_first_fail_step={summary['mean_first_invalid_step']}  "
          f"fleiss_kappa={summary['fleiss_kappa']}")
    for m in JUDGE_MODELS:
        print(f"    {m}: invalid_rate={jstats[m]['invalid_rate']}  "
              f"majority_agreement={jstats[m]['majority_agreement']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="P8: aggregate 5 judge CSVs into plan coherence consensus"
    )
    ap.add_argument("--run", required=True, help="Run name (stem of inference JSONL)")
    ap.add_argument("--dir", required=True, type=Path,
                    help="Directory containing per-model judge CSVs")
    args = ap.parse_args()

    run_dir = args.dir / args.run
    aggregate_run(args.run, run_dir)


if __name__ == "__main__":
    main()
