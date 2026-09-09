"""
precondition_position_probe.py -- Phase 1 of the revised Direction-4 protocol.

Question: does SEQUENCE_VIOLATION risk (a step's `missing_requires` check
failing -- see worldstate.py) rise with step position, controlling for
prompt condition and casualty? No new inference: reuses per_image.csv /
per_step.csv already on disk. This module only reads and analyses; it
does not touch aggregate.py's outputs.

Person-period construction follows hazard.py's own risk-set convention
exactly (see that module's docstring): a plan enters the risk set only if
its EPL is not structural (epl_is_structural == False -- it got a route
and an admissible technique), and exits at its first failure, whatever
class that is. Any first failure that is NOT a SEQUENCE_VIOLATION is a
competing risk: the person-period for that step is still included (the
plan really was at risk at that step) but coded event=0, and the plan
contributes no further steps. This is the standard discrete-time
competing-risks construction (person-period logistic regression), and it
lets step, prompt condition, and casualty enter as ordinary covariates.

RESULT (2026-09-09, Eval_CASTOR@f8a4bb4, corpus run ids 40a95ad2e5d4 /
1f38bab7f474 / 034c2339991c -- see reports/p9/precondition_position_phase1.txt
for the full recorded output):

Step-as-linear-trend gives OR=1.58/step (95% CI [1.15, 2.18], p=0.005),
which reads as a clean monotonic climb -- but does NOT survive the
categorical refit: the adjusted per-step pattern is a sharp SPIKE at step
3 (b=+1.92, p<0.001) with step 2 actually *below* step 1 (b=-1.73,
p=0.02), not a smooth increase. Steps 5-6 have collapsed risk sets (5 and
1 plans respectively) and are uninterpretable.

A spike concentrated at one step number is at least as well explained by
"step 3 is where a specific class of precondition-gated action textually
tends to land in these plans' typical structure" (a content confound --
salvage plans have a genre-typical shape: assess, prepare, THEN the first
active precondition-gated action) as by "risk climbs with how much the
model has already generated." Step position and typical action-content
are confounded by construction in observational plan text and this
design cannot separate them. That separation is exactly what Phase 3
(truncate-and-continue at fixed content, varying only prefix length) is
for -- this module motivates and aims Phase 3, it does not resolve the
question on its own.

Usage:
    python -m pipelines.plan_adequacy.precondition_position_probe \\
        --dir /path/to/results \\
        --out reports/p9/precondition_position_phase1.txt
"""
import argparse
import csv
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

MAX_STEPS = 6

#: run-name -> prompt-condition label, matching the naming used throughout
#: reports/p9/p9_diagnosis.tex and reports/recap/recap01.tex.
RUNS = {
    "None":     "answers_qwen3vl8b_baseline_ablation_v2_improved",
    "Vague":    "answers_qwen3vl8b_baseline_control_v2_improved",
    "Specific": "answers_qwen3vl8b_baseline_standard_v2_improved",
}

NON_EXECUTABLE = {"NO_MATCH", "SEQUENCE_VIOLATION", "METHOD_ERROR",
                  "CONDITIONAL_UNRESOLVED", "UNSPECIFIED"}


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_person_periods(base_dir: Path, runs: dict = RUNS, max_steps: int = MAX_STEPS):
    """[{image, arm, casualty, step, event_seqviol}] -- one row per
    (plan, step) the plan was actually at risk for. Raises AssertionError
    if a plan's per_step verdicts are inconsistent with its recorded
    failure_step (a corpus-integrity check, not a soft warning: if this
    ever fires, per_image.csv and per_step.csv have drifted apart and
    every number downstream of this module is untrustworthy until fixed).
    """
    rows = []
    n_plans = n_excluded = 0
    for arm, run in runs.items():
        per_image = _load_csv(base_dir / run / "per_image.csv")
        per_step = _load_csv(base_dir / run / "per_step.csv")
        step_by_key = {(r["image"], int(r["step_num"])): r for r in per_step}

        for img in per_image:
            n_plans += 1
            if img["epl_is_structural"] == "True":
                n_excluded += 1
                continue

            failure_step = int(img["failure_step"]) if img["failure_step"] else None
            last_k = failure_step if failure_step is not None else max_steps

            for k in range(1, last_k + 1):
                step_row = step_by_key.get((img["image"], k))
                assert step_row is not None, f"missing step row {img['image']} k={k}"
                if k < (failure_step or max_steps + 1):
                    assert step_row["verdict"] not in NON_EXECUTABLE, (
                        f"non-executable verdict before recorded failure_step: "
                        f"{arm}/{img['image']} k={k} verdict={step_row['verdict']}")
                event = int(k == failure_step and step_row["verdict"] == "SEQUENCE_VIOLATION")
                rows.append({"image": img["image"], "arm": arm,
                            "casualty": img["casualty"], "step": k,
                            "event_seqviol": event})
    return rows, {"n_plans": n_plans, "n_excluded_structural": n_excluded}


def fit_models(rows: list):
    """Returns (linear_model, categorical_model) GEE fits. Report the
    categorical one as primary -- see module docstring on why the linear
    trend is misleading here."""
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    df["cluster"] = df["arm"] + "|" + df["image"]

    linear = smf.gee(
        "event_seqviol ~ step + C(arm, Treatment('None')) + C(casualty)",
        groups="cluster", data=df, family=sm.families.Binomial(),
    ).fit()
    categorical = smf.gee(
        "event_seqviol ~ C(step) + C(arm, Treatment('None')) + C(casualty)",
        groups="cluster", data=df, family=sm.families.Binomial(),
    ).fit()
    return df, linear, categorical


def main():
    ap = argparse.ArgumentParser(description="Phase 1: precondition-violation position probe")
    ap.add_argument("--dir", required=True, help="results base directory")
    ap.add_argument("--out", default=None, help="write full text report here")
    args = ap.parse_args()

    rows, counts = build_person_periods(Path(args.dir))
    df, linear, categorical = fit_models(rows)

    import numpy as np
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p(f"plans total={counts['n_plans']}  excluded (structural, never at risk)="
      f"{counts['n_excluded_structural']}")
    p(f"person-periods={len(df)}  events={df['event_seqviol'].sum()}")
    p("")
    p("raw events/count by step (unadjusted, descriptive only):")
    p(str(df.groupby("step")["event_seqviol"].agg(["sum", "count"])))
    p("")
    p("=== Linear-in-step model (MISLEADING as a headline number -- see below) ===")
    p(str(linear.summary()))
    coef, se = linear.params["step"], linear.bse["step"]
    p(f"step OR = {np.exp(coef):.3f}  95% CI [{np.exp(coef-1.96*se):.3f}, "
      f"{np.exp(coef+1.96*se):.3f}]  p={linear.pvalues['step']:.4f}")
    p("")
    p("=== Categorical-step model (PRIMARY -- shows the actual adjusted shape) ===")
    p(str(categorical.summary()))
    p("")
    p("CONCLUSION: adjusted risk is a spike at step 3, not a monotonic climb "
      "(step 2 < step 1; step 3 >> step 1, p<.001; steps 5-6 uninterpretable, "
      "risk sets of 5 and 1). The linear model's significant positive OR is "
      "an artifact of forcing a straight line through a step-1-to-step-3 jump. "
      "Consistent with a content confound (certain action types textually land "
      "at step 3 in these plans) rather than a generation-depth effect -- "
      "motivates Phase 3, does not substitute for it.")

    if args.out:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
