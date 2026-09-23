"""
Tests for pipelines/plan_adequacy/diagnose.py -- the two all-errors-mode
additions: the per-arm all-errors table and the SUPPORT / NO_MATCH split of
unmapped steps.

Run: python -m pytest tests/test_plan_adequacy_diagnose.py -v
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.classify import ALL_ERRORS_FIELDS, ERROR_TYPES
from pipelines.plan_adequacy.diagnose import (_short_names, all_errors_table,
                                              no_match_breakdown)


def _row(**kw):
    """A per_image.csv row as csv.DictReader returns it: every value a str."""
    row = {f: "0" for f in ALL_ERRORS_FIELDS}
    row.update({k: str(v) for k, v in kw.items()})
    return row


def test_all_errors_table_has_one_row_per_error_type_and_reads_pct_of_plans():
    rows = {"arm": [_row(n_err_sequence=3, n_err_sequence_independent=0),
                    _row(n_err_sequence=1, n_err_sequence_independent=1),
                    _row(err_perception=1),
                    _row()]}
    table = {r["error_type"]: r for r in all_errors_table(rows)}
    assert list(table) == list(ERROR_TYPES)
    assert table["sequence"]["arm_pct"] == 50.0          # 2 of 4 plans
    assert table["sequence"]["arm_indep_pct"] == 25.0    # only one independent
    assert table["sequence"]["arm_steps"] == 4
    assert table["perception"]["arm_pct"] == 25.0
    assert table["perception"]["arm_steps"] == "--"      # plan-level: no steps
    assert table["method"]["arm_indep_pct"] == "--"      # not state-dependent


def test_all_errors_table_keeps_arms_separate():
    rows = {"a": [_row(n_err_hedged=1, n_err_hedged_independent=1)], "b": [_row()]}
    hedged = next(r for r in all_errors_table(rows) if r["error_type"] == "hedged")
    assert (hedged["a_pct"], hedged["b_pct"]) == (100.0, 0.0)


def test_no_match_breakdown_splits_support_from_genuine_no_match(tmp_path):
    path = tmp_path / "per_step.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["verdict", "no_match_category"])
        w.writeheader()
        w.writerows([
            {"verdict": "SUPPORT", "no_match_category": "site_control"},
            {"verdict": "SUPPORT", "no_match_category": "site_control"},
            {"verdict": "NO_MATCH", "no_match_category": "site_control"},
            {"verdict": "NO_MATCH", "no_match_category": "other"},
            {"verdict": "SEQUENCE_VIOLATION", "no_match_category": ""},
        ])
    gap = no_match_breakdown([path])
    assert gap["total"] == {"SUPPORT": 2, "NO_MATCH": 2}
    assert gap["counts"]["SUPPORT"] == {"site_control": 2}
    assert gap["counts"]["NO_MATCH"] == {"site_control": 1, "other": 1}


def test_arm_headers_strip_what_every_run_name_shares():
    runs = ["answers_q_baseline_ablation_v2_improved",
            "answers_q_baseline_standard_v2_improved"]
    assert _short_names(runs) == ["ablation", "standard"]
