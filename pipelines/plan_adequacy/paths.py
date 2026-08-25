"""
Pipeline 9 output-path convention.

Every run gets its own subdirectory under results/p9_plan_adequacy/ holding
both stages' artifacts, mirroring pipelines/salvage_analysis/paths.py:

    results/p9_plan_adequacy/
      answers_baseline/
        tool_calls.jsonl     <- Stage 1 (extract.py): one record per step
        per_step.csv         <- Stage 2 (run_executor.py + aggregate.py)
        per_image.csv        <- Stage 2
        summary.csv          <- Stage 2
        report.md            <- Stage 2 (report.py)
        case_studies.md       <- Stage 2 (report.py) -- record-level evidence
      answers_degf/
        ...
    eval_summary_adequacy.csv  <- cumulative, one row appended per run (CUMULATIVE_SUMMARY_PATH)

Inbox for run JSONLs is pipelines/plan_adequacy/inbox/, nested inside the
package rather than a new top-level Eval_CASTOR/p9_.../ directory -- P9 is
new enough (unlike the legacy p7_to_check/ p8_to_check/, which stay put)
that everything specific to it lives under one pipeline directory instead of
adding another root-level folder. tests/ and containers/ still follow the
existing repo-wide convention (test_plan_adequacy_*.py next to every other
pipeline's tests, plan_adequacy_*_job.sh next to every other pipeline's job
scripts) since pytest/sbatch discovery for every other pipeline depends on
that, and results/ likewise stays the shared gitignored output root.
"""

from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
BASE_OUT_DIR = EVAL_ROOT / "results" / "p9_plan_adequacy"
PLAN_ADEQUACY_CHECK_DIR = Path(__file__).parent / "inbox"

#: registry/ files, resolved relative to this module so callers never
#: hardcode a path that breaks if the package moves.
REGISTRY_DIR = Path(__file__).parent / "registry"
TOOLS_PATH = REGISTRY_DIR / "tools.json"
ROUTES_PATH = REGISTRY_DIR / "routes.json"
GOALS_PATH = REGISTRY_DIR / "goals.json"
CHECKS_PATH = REGISTRY_DIR / "checks.csv"

#: calibration/ files
CALIBRATION_DIR = Path(__file__).parent / "calibration"
GOLD_TOOL_CALLS_PATH = CALIBRATION_DIR / "gold_tool_calls.jsonl"
ROBUSTNESS_TEMPLATES_PATH = CALIBRATION_DIR / "robustness_templates.json"

#: existing calibration set (166 judgment-labeled steps), reused for
#: executor@oracle validation -- see synthetic_calibration.jsonl in the plan.
SYNTHETIC_CALIBRATION_PATH = EVAL_ROOT / "p8_to_check" / "synthetic_calibration.jsonl"

#: Cross-run comparison, appended to across every run the same way
#: aggregate_coherence.py appends to eval_summary_coherence.csv (P8's
#: cumulative file sits at run_dir.parent, which for P9 is exactly
#: BASE_OUT_DIR since every run's RunPaths.dir is BASE_OUT_DIR/run_name).
CUMULATIVE_SUMMARY_PATH = BASE_OUT_DIR / "eval_summary_adequacy.csv"


class RunPaths:
    """Every file P9 writes for one run.

    Two stages -- extraction (GPU) then execution+aggregation (CPU) -- read
    and write across a SLURM job boundary, so these filenames are the
    contract between them, not merely output locations. See
    pipelines/salvage_analysis/paths.py:RunPaths for the pattern this mirrors.
    """

    #: Stage 1 (extract.py) output: one record per (image, step) with the
    #: extracted tool call.
    TOOL_CALLS = "tool_calls.jsonl"
    #: Stage 2 (executor.py + aggregate.py) outputs.
    PER_STEP = "per_step.csv"
    PER_IMAGE = "per_image.csv"
    SUMMARY = "summary.csv"
    #: Stage 2 (report.py) narrative outputs -- see the P9 end-to-end-
    #: pipeline plan, Part 1 ("report.py") and Part 1c ("example mining").
    #: report.md mirrors pipelines/plan_coherence/improved/eval/aggregate.py's
    #: report.md; case_studies.md mirrors that same module's
    #: write_case_studies() output -- the record-level evidence a report.md
    #: claim can be traced back to.
    REPORT = "report.md"
    CASE_STUDIES = "case_studies.md"

    def __init__(self, run_name: str, base_dir: Path = None):
        self.run_name = run_name
        self.base_dir = base_dir or BASE_OUT_DIR

    @property
    def dir(self) -> Path:
        """Directory holding every artefact for this run."""
        return self.base_dir / self.run_name

    def _in_run(self, filename: str) -> Path:
        return self.dir / filename

    @property
    def tool_calls(self) -> Path:
        return self._in_run(self.TOOL_CALLS)

    @property
    def per_step(self) -> Path:
        return self._in_run(self.PER_STEP)

    @property
    def per_image(self) -> Path:
        return self._in_run(self.PER_IMAGE)

    @property
    def summary(self) -> Path:
        return self._in_run(self.SUMMARY)

    @property
    def report(self) -> Path:
        return self._in_run(self.REPORT)

    @property
    def case_studies(self) -> Path:
        return self._in_run(self.CASE_STUDIES)


# ── Compatibility facade ─────────────────────────────────────────────────────
# The stage scripts call these directly.

def run_dir(run_name: str) -> Path:
    return RunPaths(run_name).dir


def tool_calls_path(run_name: str) -> Path:
    return RunPaths(run_name).tool_calls


def per_step_path(run_name: str) -> Path:
    return RunPaths(run_name).per_step


def per_image_path(run_name: str) -> Path:
    return RunPaths(run_name).per_image


def summary_path(run_name: str) -> Path:
    return RunPaths(run_name).summary


def report_path(run_name: str) -> Path:
    return RunPaths(run_name).report


def case_studies_path(run_name: str) -> Path:
    return RunPaths(run_name).case_studies
