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
from typing import Optional

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

JUDGE_MODELS = ["deepseek_r1_32b", "glm4_32b", "selene_mini_8b"]
STD_FLAG_THRESHOLD = 0.6
FIELD_KEYS = ["state_correct", "vessel_type_correct", "size_correct", "cargo_correct"]


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

def _record_id(rec: dict) -> str:
    """Stable composite key: image||model_tag||method||prompt_stem.
    Falls back to image alone for old records that lack the extra fields."""
    img  = rec.get("image", "")
    mt   = rec.get("model_tag", "")
    meth = rec.get("method", "")
    ps   = rec.get("prompt_stem", "")
    return f"{img}||{mt}||{meth}||{ps}"


def load_judge_jsonl(path: Path) -> dict:
    """Load a per-model judge JSONL. Returns {record_id -> record}."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "image" in rec:
                    result[_record_id(rec)] = rec
            except json.JSONDecodeError:
                pass
    return result


def compute_consensus(image: str, gt_state: str, pred_text: str,
                      verbosity_flagged: bool,
                      scores: dict, rationales: dict,
                      hallucinations: dict,
                      field_votes: Optional[dict] = None,
                      model_tag: str = "", method: str = "",
                      prompt_stem: str = "") -> dict:
    """Compute panel consensus from per-model scores.

    Args:
        scores:      {model -> score_int_or_None}
        rationales:  {model -> rationale_str}
        hallucinations: {model -> [str, ...]}
        field_votes: {field_key -> {model -> bool_or_None}}

    Returns a consensus record dict.
    """
    valid = [s for s in scores.values() if s is not None]

    if not valid:
        mean_score = None
        score_std  = None
        status     = "parse_error"
        verdict    = "no_score"
    else:
        mean_score = round(mean(valid), 3)
        score_std  = round(stdev(valid), 3) if len(valid) > 1 else 0.0
        status     = "flagged_for_review" if score_std > STD_FLAG_THRESHOLD else "consensus"
        verdict    = "accurate" if mean_score >= 2.5 else "inaccurate"

    hallucination_union = list({
        h
        for model_hallus in hallucinations.values()
        for h in (model_hallus or [])
    })

    # Per-field majority vote: True if ≥ ceil(n/2 + 1) judges voted True
    field_consensus = {}
    if field_votes:
        for fk in FIELD_KEYS:
            votes = [v for v in field_votes.get(fk, {}).values() if v is not None]
            if votes:
                field_consensus[fk] = sum(1 for v in votes if v) >= (len(votes) / 2 + 0.5)
            else:
                field_consensus[fk] = None

    rec = {
        "record_id":           f"{image}||{model_tag}||{method}||{prompt_stem}",
        "image":               image,
        "model_tag":           model_tag,
        "method":              method,
        "prompt_stem":         prompt_stem,
        "gt_state":            gt_state,
        "pred_text":           pred_text,
        "verbosity_flagged":   verbosity_flagged,
        "scores":              scores,
        "rationales":          rationales,
        "hallucinations":      hallucinations,
        "mean_score":          mean_score,
        "score_std":           score_std,
        "consensus_status":    status,
        "judge_verdict":       verdict,
        "hallucination_union": hallucination_union,
    }
    if field_consensus:
        rec["field_votes"]     = field_votes
        rec["field_consensus"] = field_consensus
    return rec


def aggregate_run(run_name: str, judge_dir: Path) -> tuple:
    """Merge five judge JONLs and write consensus + flagged outputs.

    Returns (consensus_path, flagged_path).
    """
    # Load all five judge files
    judge_data = {}
    for model in JUDGE_MODELS:
        p = judge_dir / f"{run_name}_{model}.jsonl"
        if p.exists():
            judge_data[model] = load_judge_jsonl(p)
        else:
            print(f"  WARNING: missing judge file {p.name}")
            judge_data[model] = {}

    # Union of all record_ids seen across all judges
    all_record_ids = sorted({
        rid for model_data in judge_data.values() for rid in model_data
    })

    consensus_records = []
    for rid in all_record_ids:
        # Use first available record for shared fields
        ref = next(
            (judge_data[m][rid] for m in JUDGE_MODELS if rid in judge_data[m]),
            {}
        )
        image             = ref.get("image", rid.split("||")[0])
        model_tag         = ref.get("model_tag", "")
        method            = ref.get("method", "")
        prompt_stem       = ref.get("prompt_stem", "")
        gt_state          = ref.get("gt_state", "")
        pred_text         = ref.get("pred_text", "")
        verbosity_flagged = ref.get("verbosity_flagged", False)

        scores     = {m: judge_data[m].get(rid, {}).get("score")       for m in JUDGE_MODELS}
        rationales = {m: judge_data[m].get(rid, {}).get("rationale", "") for m in JUDGE_MODELS}
        hallus     = {m: judge_data[m].get(rid, {}).get("hallucinations", []) for m in JUDGE_MODELS}

        field_votes = {
            fk: {m: judge_data[m].get(rid, {}).get(fk) for m in JUDGE_MODELS}
            for fk in FIELD_KEYS
        }
        has_fields = any(
            v is not None
            for fv in field_votes.values()
            for v in fv.values()
        )

        consensus_records.append(
            compute_consensus(
                image, gt_state, pred_text, verbosity_flagged,
                scores, rationales, hallus,
                field_votes=field_votes if has_fields else None,
                model_tag=model_tag, method=method, prompt_stem=prompt_stem,
            )
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
