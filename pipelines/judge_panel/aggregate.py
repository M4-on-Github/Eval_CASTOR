"""
Pipeline 5 — panel aggregation.

Merges three per-model judge JONLs into a consensus record per image,
computes mean score, std, and flags high-disagreement cases.

Usage:
  python aggregate.py --run answers_baseline --dir ../../results/p5_judge/answers_baseline/
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

JUDGE_MODELS = ["gptoss_120b", "deepseek_r1", "qwen25_72b"]
STD_FLAG_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

def load_judge_jsonl(path: Path) -> dict:
    """Load a per-model judge JSONL. Returns {image -> record}."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "image" in rec:
                    result[rec["image"]] = rec
            except json.JSONDecodeError:
                pass
    return result


def compute_consensus(image: str, gt_state: str, pred_text: str,
                      verbosity_flagged: bool,
                      scores: dict, rationales: dict,
                      hallucinations: dict) -> dict:
    """Compute panel consensus from per-model scores.

    Args:
        scores:        {model -> score_int_or_None}
        rationales:    {model -> rationale_str}
        hallucinations:{model -> [str, ...]}

    Returns a consensus record dict.
    """
    valid = [s for s in scores.values() if s is not None]

    if not valid:
        mean_score = None
        score_std  = None
        status     = "parse_error"
    else:
        mean_score = round(mean(valid), 3)
        score_std  = round(stdev(valid), 3) if len(valid) > 1 else 0.0
        status     = "flagged_for_review" if score_std > STD_FLAG_THRESHOLD else "consensus"

    hallucination_union = list({
        h
        for model_hallus in hallucinations.values()
        for h in (model_hallus or [])
    })

    return {
        "image":              image,
        "gt_state":           gt_state,
        "pred_text":          pred_text,
        "verbosity_flagged":  verbosity_flagged,
        "scores":             scores,
        "rationales":         rationales,
        "hallucinations":     hallucinations,
        "mean_score":         mean_score,
        "score_std":          score_std,
        "consensus_status":   status,
        "hallucination_union": hallucination_union,
    }


def aggregate_run(run_name: str, judge_dir: Path) -> tuple:
    """Merge three judge JONLs and write consensus + flagged outputs.

    Returns (consensus_path, flagged_path).
    """
    # Load all three judge files
    judge_data = {}
    for model in JUDGE_MODELS:
        p = judge_dir / f"{run_name}_{model}.jsonl"
        if p.exists():
            judge_data[model] = load_judge_jsonl(p)
        else:
            print(f"  WARNING: missing judge file {p.name}")
            judge_data[model] = {}

    # Union of all images seen across all judges
    all_images = sorted({
        img for model_data in judge_data.values() for img in model_data
    })

    consensus_records = []
    for img in all_images:
        # Use first available record for shared fields
        ref = next(
            (judge_data[m][img] for m in JUDGE_MODELS if img in judge_data[m]),
            {}
        )
        gt_state         = ref.get("gt_state", "")
        pred_text        = ref.get("pred_text", "")
        verbosity_flagged = ref.get("verbosity_flagged", False)

        scores = {m: judge_data[m].get(img, {}).get("score") for m in JUDGE_MODELS}
        rationales = {m: judge_data[m].get(img, {}).get("rationale", "") for m in JUDGE_MODELS}
        hallus = {m: judge_data[m].get(img, {}).get("hallucinations", []) for m in JUDGE_MODELS}

        consensus_records.append(
            compute_consensus(img, gt_state, pred_text, verbosity_flagged, scores, rationales, hallus)
        )

    consensus_path = judge_dir / f"{run_name}_consensus.jsonl"
    flagged_path   = judge_dir / f"{run_name}_flagged.jsonl"

    flagged = [r for r in consensus_records if r["consensus_status"] == "flagged_for_review"]

    consensus_path.write_text(
        "\n".join(json.dumps(r) for r in consensus_records) + "\n",
        encoding="utf-8",
    )
    flagged_path.write_text(
        "\n".join(json.dumps(r) for r in flagged) + "\n",
        encoding="utf-8",
    )

    n_consensus = sum(1 for r in consensus_records if r["consensus_status"] == "consensus")
    n_flagged   = len(flagged)
    n_error     = sum(1 for r in consensus_records if r["consensus_status"] == "parse_error")
    print(f"  {len(consensus_records)} records: consensus={n_consensus}  flagged={n_flagged}  parse_error={n_error}")
    print(f"  -> {consensus_path.name}")
    print(f"  -> {flagged_path.name}")

    return consensus_path, flagged_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate three judge JONLs into consensus.")
    ap.add_argument("--run",  required=True, help="Run name (stem of inference JSONL)")
    ap.add_argument("--dir",  required=True, type=Path, help="Directory containing judge JONLs")
    args = ap.parse_args()

    aggregate_run(args.run, args.dir)


if __name__ == "__main__":
    main()
