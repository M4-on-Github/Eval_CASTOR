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
        generic_elements.csv   <- Stage 4: frequent-but-never-significant elements (boilerplate, not a state signature)
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


class RunPaths:
    """Every file P6 writes for one run.

    The pipeline has four stages that each read the previous stage's output —
    extract, normalize, contingency, then the statistical tests — so these
    filenames are the contract BETWEEN stages, not merely output locations. A
    stage writing one name while the next reads another fails as a missing
    file at the far end of a long run, far from the mistake.

    Collecting them here means the contract is stated once. The filenames are
    class attributes so a stage can refer to the name symbolically rather than
    repeating a literal.

    Instances are cheap and hold only the run name:

        p = RunPaths("answers_baseline")
        p.raw_elements     -> .../answers_baseline/raw_elements.jsonl
        p.tests            -> .../answers_baseline/tests.csv
    """

    #: Stage 1 output: one record per image, elements as the model phrased them.
    RAW_ELEMENTS = "raw_elements.jsonl"
    #: Stage 2 output: phrasings clustered to canonical elements.
    ELEMENTS = "elements.json"
    #: Stage 3 output: element presence per image against state.
    CONTINGENCY = "contingency.csv"
    #: Stage 4 outputs: per-element tests, omnibus, post-hoc, and the summary.
    TESTS = "tests.csv"
    OMNIBUS = "omnibus.csv"
    DUNN = "dunn.csv"
    GENERIC_ELEMENTS = "generic_elements.csv"
    REPORT = "report.txt"

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
    def raw_elements(self) -> Path:
        return self._in_run(self.RAW_ELEMENTS)

    @property
    def elements(self) -> Path:
        return self._in_run(self.ELEMENTS)

    @property
    def contingency(self) -> Path:
        return self._in_run(self.CONTINGENCY)

    @property
    def tests(self) -> Path:
        return self._in_run(self.TESTS)

    @property
    def omnibus(self) -> Path:
        return self._in_run(self.OMNIBUS)

    @property
    def dunn(self) -> Path:
        return self._in_run(self.DUNN)

    @property
    def generic_elements(self) -> Path:
        return self._in_run(self.GENERIC_ELEMENTS)

    @property
    def report(self) -> Path:
        return self._in_run(self.REPORT)


# ── Compatibility facade ─────────────────────────────────────────────────────
# The stage scripts call these directly.

def run_dir(run_name: str) -> Path:
    return RunPaths(run_name).dir


def raw_elements_path(run_name: str) -> Path:
    return RunPaths(run_name).raw_elements


def elements_path(run_name: str) -> Path:
    return RunPaths(run_name).elements


def contingency_path(run_name: str) -> Path:
    return RunPaths(run_name).contingency


def tests_path(run_name: str) -> Path:
    return RunPaths(run_name).tests


def omnibus_path(run_name: str) -> Path:
    return RunPaths(run_name).omnibus


def dunn_path(run_name: str) -> Path:
    return RunPaths(run_name).dunn


def generic_elements_path(run_name: str) -> Path:
    return RunPaths(run_name).generic_elements


def report_path(run_name: str) -> Path:
    return RunPaths(run_name).report
