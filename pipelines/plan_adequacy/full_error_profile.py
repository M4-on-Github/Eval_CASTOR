"""
full_error_profile.py -- what error TYPES does each plan actually exhibit,
across all six steps, not just the first one.

Every other P9 report (classify.py, hazard.py, diagnose.py) deliberately
reports only the FIRST failure per plan -- justified because later verdicts
are often downstream of an earlier one and not independent evidence (see
classify.py's docstring: ~73-75% of sequence violations occur after an
earlier NO_MATCH/METHOD_ERROR). That convention is correct for estimating
prevalence without double-counting cascades, but it throws away a real
question: for a single plan, how many DIFFERENT kinds of problems does it
actually have, and which kinds tend to co-occur?

This module answers that instead, reusing the exact same standard-name
mapping used throughout reports/recap and reports/p9. No new inference --
reads per_image.csv / per_step.csv already on disk.

Two structural facts kept visible rather than dropped, unlike the
first-failure taxonomy:
  - is_cascade is reported alongside each step-level error, so a reader can
    tell which "extra" errors are contamination-suspect (downstream of an
    earlier NO_MATCH/METHOD_ERROR) vs standalone.
  - Plans that never got a route (NO_PROCEDURE) or matched a foreign one
    (STRATEGY_PERCEPTION) never entered step-level execution at all -- their
    profile is that one structural label alone, not an empty step scan.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

RUNS = {
    "None":     "answers_qwen3vl8b_baseline_ablation_v2_improved",
    "Vague":    "answers_qwen3vl8b_baseline_control_v2_improved",
    "Specific": "answers_qwen3vl8b_baseline_standard_v2_improved",
}

#: Standard plain names, matching reports/recap/recap01.tex,
#: reports/p9/p9_recap.tex, reports/recap/slides.tex.
STANDARD_NAME = {
    "STRATEGY_PERCEPTION": "Solves the wrong emergency",
    "NO_PROCEDURE":        "No recognizable plan",
    "STRATEGY_TECHNIQUE":  "Wrong technique for this vessel",
    "PROCEDURE":           "Right approach, done wrong",
    "COMMITMENT":          "No amount/quantity given",
    "INCOMPLETE":          "Ran cleanly all the way through (incomplete)",
    "VALID":               "Ran cleanly all the way through (valid)",
}

#: Step-level verdict -> the standard name of the class it belongs to.
#: SPECIFIED_UNGRADED is deliberately absent -- it is a clean step, not an
#: error type -- and so are SUPPORT (ungraded scaffolding, support.py) and
#: UNSPECIFIED (quantities were dropped from P9's question, see
#: classify.NON_EXECUTABLE).
VERDICT_TO_STANDARD_NAME = {
    "NO_MATCH":               STANDARD_NAME["PROCEDURE"],
    "SEQUENCE_VIOLATION":     STANDARD_NAME["PROCEDURE"],
    "CONDITIONAL_UNRESOLVED": STANDARD_NAME["PROCEDURE"],
    "METHOD_ERROR":           STANDARD_NAME["STRATEGY_TECHNIQUE"],
}


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_profiles(base_dir: Path, runs: dict = RUNS) -> list:
    """One dict per plan: {image, arm, casualty, first_failure_class,
    error_types: [...], error_events: [{step, verdict, standard_name,
    is_cascade}, ...]}. error_types is the DISTINCT set (order-preserving,
    first-seen order) of standard names appearing anywhere in the plan's
    six steps -- a plan can and often does carry more than one."""
    profiles = []
    for arm, run in runs.items():
        per_image = _load_csv(base_dir / run / "per_image.csv")
        per_step = _load_csv(base_dir / run / "per_step.csv")
        steps_by_image = defaultdict(list)
        for row in per_step:
            steps_by_image[row["image"]].append(row)

        for img in per_image:
            profile = {
                "image": img["image"], "arm": arm, "casualty": img["casualty"],
                "first_failure_class": img["failure_class"],
                "error_types": [], "error_events": [],
            }
            if img["epl_is_structural"] == "True":
                # Never entered step-level execution -- the one structural
                # label IS the whole profile, not an empty step scan.
                profile["error_types"] = [STANDARD_NAME[img["failure_class"]]]
                profiles.append(profile)
                continue

            seen = []
            for step in sorted(steps_by_image[img["image"]], key=lambda r: int(r["step_num"])):
                verdict = step["verdict"]
                name = VERDICT_TO_STANDARD_NAME.get(verdict)
                if name is None:
                    continue  # a clean, unquantified, or SUPPORT step
                profile["error_events"].append({
                    "step": int(step["step_num"]), "verdict": verdict,
                    "standard_name": name, "is_cascade": step["is_cascade"] == "True",
                })
                if name not in seen:
                    seen.append(name)
            profile["error_types"] = seen
            profiles.append(profile)
    return profiles


def summarize(profiles: list) -> dict:
    """{n_error_types: n_plans}, {error_type: n_plans exhibiting it anywhere},
    and the most common co-occurring PAIRS of error types in one plan."""
    by_count = Counter(len(p["error_types"]) for p in profiles)
    by_type_anywhere = Counter()
    for p in profiles:
        for t in p["error_types"]:
            by_type_anywhere[t] += 1
    pair_counts = Counter()
    for p in profiles:
        types = sorted(p["error_types"])
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                pair_counts[(types[i], types[j])] += 1
    return {"by_count": dict(by_count), "by_type_anywhere": dict(by_type_anywhere),
            "pair_counts": pair_counts}


def main():
    ap = argparse.ArgumentParser(description="Full per-plan error-type profile (not first-failure-only)")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None, help="write per-plan profiles as CSV here")
    args = ap.parse_args()

    profiles = build_profiles(Path(args.dir))
    n = len(profiles)
    summary = summarize(profiles)

    print(f"{n} plans total\n")
    print("How many DISTINCT error types does a single plan exhibit (not just first-failure):")
    for k in sorted(summary["by_count"]):
        c = summary["by_count"][k]
        print(f"  {k} type(s): {c:4d} plans ({c/n*100:.1f}%)")

    print(f"\nHow often each error type appears ANYWHERE in a plan (not just as the first failure):")
    for t, c in sorted(summary["by_type_anywhere"].items(), key=lambda kv: -kv[1]):
        print(f"  {t:45s} {c:4d} plans ({c/n*100:.1f}%)")

    print(f"\nMost common PAIRS of error types co-occurring in the same plan:")
    for (a, b), c in summary["pair_counts"].most_common(10):
        print(f"  {a:35s} + {b:35s} {c:4d} plans")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "arm", "casualty", "first_failure_class",
                       "n_error_types", "error_types"])
            for p in profiles:
                w.writerow([p["image"], p["arm"], p["casualty"], p["first_failure_class"],
                           len(p["error_types"]), " | ".join(p["error_types"])])
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
