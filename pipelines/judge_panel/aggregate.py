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

    Keying on image alone would be wrong: one image appears once per
    (model, method, prompt) combination and each needs its own verdict, so a
    bare-image key would keep only the last and silently discard the rest.

    Missing fields become empty SEGMENTS, not a shorter key — an old record
    lacking the extra fields keys as "image||||||", which will not match a
    lookup by bare image path. There is no fallback to image alone.
    """
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


class PanelVote:
    """Turns three judge models' opinions into one verdict.

    Everything downstream is computed from these records — summary.csv
    accuracies, per-image tiers, the regex-judge kappa — so the thresholds here
    move every judged number in the project.

    Three decisions are encoded, and each distinguishes cases that look alike:

    NO SCORE IS NOT A LOW SCORE. When every judge fails to parse, mean_score is
    None and the verdict is "no_score", never "inaccurate". Collapsing them
    would count a formatting failure as a model error and quietly depress the
    reported accuracy.

    DISAGREEMENT IS REPORTED, NOT RESOLVED. A standard deviation above
    STD_FLAG_THRESHOLD marks the record "flagged_for_review" rather than
    discarding it or picking a winner. The panel disagreeing IS the finding on
    contested images.

    A TIED FIELD VOTE MEANS "NOT ESTABLISHED". The majority threshold is
    len/2 + 0.5, so 1-of-2 does not carry. A field no judge could assess is
    None, which is distinct from False — conflating them would count
    unassessable fields as failures.
    """

    #: mean score at or above this counts as accurate (scores are 1-3)
    ACCURATE_THRESHOLD = 2.5

    def __init__(self, scores: dict, std_flag_threshold: float = None):
        self.scores = scores
        self.std_flag_threshold = (STD_FLAG_THRESHOLD if std_flag_threshold is None
                                   else std_flag_threshold)

    @property
    def valid_scores(self) -> list:
        """Scores from judges that actually parsed."""
        return [s for s in self.scores.values() if s is not None]

    @property
    def all_failed(self) -> bool:
        return not self.valid_scores

    def summarise(self) -> tuple:
        """Return (mean_score, score_std, consensus_status, judge_verdict)."""
        valid = self.valid_scores
        if not valid:
            return None, None, "parse_error", "no_score"

        mean_score = round(mean(valid), 3)
        # stdev() of a single sample raises, so one surviving judge is
        # reported as zero spread rather than as an error.
        score_std = round(stdev(valid), 3) if len(valid) > 1 else 0.0
        status = ("flagged_for_review" if score_std > self.std_flag_threshold
                  else "consensus")
        verdict = "accurate" if mean_score >= self.ACCURATE_THRESHOLD else "inaccurate"
        return mean_score, score_std, status, verdict

    @staticmethod
    def union_hallucinations(hallucinations: dict) -> list:
        """Every distinct hallucination any judge reported.

        Union rather than intersection: one judge spotting a fabricated detail
        is evidence it is there, and requiring agreement would discard most
        findings.
        """
        return list({
            h
            for model_hallus in hallucinations.values()
            for h in (model_hallus or [])
        })

    @classmethod
    def field_majority(cls, votes: dict) -> Optional[bool]:
        """Majority verdict for one field, or None if no judge assessed it."""
        cast = [v for v in votes.values() if v is not None]
        if not cast:
            return None
        return sum(1 for v in cast if v) >= (len(cast) / 2 + 0.5)


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

    Returns a consensus record dict. See PanelVote for the thresholds and why
    each distinguishes the cases it does.
    """
    vote = PanelVote(scores)
    mean_score, score_std, status, verdict = vote.summarise()

    hallucination_union = vote.union_hallucinations(hallucinations)

    # Per-field majority vote: True if ≥ ceil(n/2 + 1) judges voted True
    field_consensus = {}
    if field_votes:
        for fk in FIELD_KEYS:
            field_consensus[fk] = PanelVote.field_majority(field_votes.get(fk, {}))

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
