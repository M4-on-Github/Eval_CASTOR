"""
Tests for pipelines/salvage_analysis/paths.py -- the per-run output
directory convention for Pipeline 6.
Run: python -m pytest tests/test_salvage_paths.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.salvage_analysis import paths


def test_run_dir_is_named_subdirectory_of_base_out_dir():
    d = paths.run_dir("answers_baseline")
    assert d == paths.BASE_OUT_DIR / "answers_baseline"


def test_stage_paths_live_inside_the_run_dir():
    run_name = "answers_degf"
    run_dir = paths.run_dir(run_name)
    assert paths.raw_elements_path(run_name) == run_dir / "raw_elements.jsonl"
    assert paths.elements_path(run_name) == run_dir / "elements.json"
    assert paths.contingency_path(run_name) == run_dir / "contingency.csv"
    assert paths.tests_path(run_name) == run_dir / "tests.csv"
    assert paths.omnibus_path(run_name) == run_dir / "omnibus.csv"
    assert paths.dunn_path(run_name) == run_dir / "dunn.csv"
    assert paths.generic_elements_path(run_name) == run_dir / "generic_elements.csv"
    assert paths.report_path(run_name) == run_dir / "report.txt"


def test_different_runs_get_different_directories():
    assert paths.run_dir("answers_baseline") != paths.run_dir("answers_degf")


def test_plans_to_judge_dir_is_under_eval_root():
    assert paths.PLANS_TO_JUDGE_DIR == paths.EVAL_ROOT / "p6_plans_to_judge"
