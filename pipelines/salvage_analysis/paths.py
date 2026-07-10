"""
Pipeline 6 output-path convention.

Every run gets its own subdirectory under results/p6_salvage_plan/ holding
all four stages' artifacts (no more per-stage filename suffixes) so runs
never collide and `ls results/p6_salvage_plan/<run_name>/` shows everything
for that run in one place.

    results/p6_salvage_plan/
      answers_baseline/
        raw_elements.jsonl     <- Stage 1
        elements.json          <- Stage 2
        contingency.csv        <- Stage 3
        tests.csv              <- Stage 4: Fisher's exact, per (element, state, source)
        omnibus.csv            <- Stage 4: Kruskal-Wallis, per state_source
        dunn.csv               <- Stage 4: Dunn's pairwise post-hoc (only rows where omnibus was significant)
        report.txt             <- Stage 4: human-readable summary of the above
      answers_degf/
        ...
"""

from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
BASE_OUT_DIR = EVAL_ROOT / "results" / "p6_salvage_plan"

# Default staging directory: drop the full-answer run JSONLs you want P6 to
# judge here. Separate from the shared results/castor_results/ that P1-P4
# read, since P6 is meant to run over a deliberately-curated subset, not
# every experimental variant sitting in the shared results directory.
PLANS_TO_JUDGE_DIR = EVAL_ROOT / "p6_plans_to_judge"


def run_dir(run_name: str) -> Path:
    return BASE_OUT_DIR / run_name


def raw_elements_path(run_name: str) -> Path:
    return run_dir(run_name) / "raw_elements.jsonl"


def elements_path(run_name: str) -> Path:
    return run_dir(run_name) / "elements.json"


def contingency_path(run_name: str) -> Path:
    return run_dir(run_name) / "contingency.csv"


def tests_path(run_name: str) -> Path:
    return run_dir(run_name) / "tests.csv"


def omnibus_path(run_name: str) -> Path:
    return run_dir(run_name) / "omnibus.csv"


def dunn_path(run_name: str) -> Path:
    return run_dir(run_name) / "dunn.csv"


def report_path(run_name: str) -> Path:
    return run_dir(run_name) / "report.txt"
