"""
Tests for pipelines/plan_adequacy/report.py

Pure: synthetic per_step rows, calibration "scored_records" rows, and
summary dicts -- no CSV/model I/O except where tmp_path is used to check
the actual written files round-trip.
Run: python -m pytest tests/test_plan_adequacy_report.py -v
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.report import (
    build_narrative,
    build_results_table,
    fmt,
    pick_confusion_examples,
    pick_step_examples,
    write_case_studies,
    write_report,
)


def _step_row(image, step_num, verdict, tool="attach_tug", text="step text"):
    return {"image": image, "step_num": str(step_num), "step_text": text,
            "tool": tool, "verdict": verdict, "detail": "d", "casualty": "aground"}


def _confusion_row(gold_id, gold_tool, predicted_tool, correct=False, step_text="x"):
    return {"gold_id": gold_id, "gold_tool": gold_tool, "predicted_tool": predicted_tool,
            "tool_correct": correct, "step_text": step_text,
            "gold_params": {}, "predicted_params": {}}


# ── fmt ───────────────────────────────────────────────────────────────────

def test_fmt_none_is_na():
    assert fmt(None) == "N/A"


def test_fmt_empty_string_is_na():
    assert fmt("") == "N/A"


def test_fmt_number_is_three_decimals():
    assert fmt(0.5) == "0.500"


def test_fmt_non_numeric_string_passes_through():
    assert fmt("tug_pull") == "tug_pull"


# ── pick_step_examples: determinism ─────────────────────────────────────

def test_pick_step_examples_filters_by_verdict():
    rows = [
        _step_row("a/1.jpg", 1, "SEQUENCE_VIOLATION"),
        _step_row("a/1.jpg", 2, "UNSPECIFIED"),
    ]
    picked = pick_step_examples(rows, "UNSPECIFIED", n=5)
    assert len(picked) == 1
    assert picked[0]["verdict"] == "UNSPECIFIED"


def test_pick_step_examples_respects_n():
    rows = [_step_row("a/1.jpg", i, "METHOD_ERROR") for i in range(1, 6)]
    picked = pick_step_examples(rows, "METHOD_ERROR", n=2)
    assert len(picked) == 2


def test_pick_step_examples_is_order_independent():
    """Regression anchor for Part 1c's determinism requirement: same rows
    in, same cases out, regardless of input order -- this is what makes the
    examples reportable rather than anecdotal."""
    rows = [_step_row(f"a/{i}.jpg", 1, "METHOD_ERROR") for i in range(10)]
    shuffled = rows[:]
    random.Random(42).shuffle(shuffled)

    picked_a = pick_step_examples(rows, "METHOD_ERROR", n=3)
    picked_b = pick_step_examples(shuffled, "METHOD_ERROR", n=3)
    assert picked_a == picked_b


# ── pick_confusion_examples: determinism + selection rule ──────────────

def test_pick_confusion_examples_only_uses_wrong_predictions():
    rows = [
        _confusion_row("g1", "attach_tug", "attach_tug", correct=True),
        _confusion_row("g2", "attach_tug", "pull", correct=False),
    ]
    picked = pick_confusion_examples(rows, n_pairs=5)
    assert len(picked) == 1
    assert picked[0]["gold_id"] == "g2"


def test_pick_confusion_examples_ranks_by_frequency_descending():
    rows = (
        [_confusion_row(f"a{i}", "survey_hull", "sound_tanks", correct=False) for i in range(3)]
        + [_confusion_row("b1", "pull", "attach_tug", correct=False)]
    )
    picked = pick_confusion_examples(rows, n_pairs=1, per_pair=1)
    assert len(picked) == 1
    assert picked[0]["gold_tool"] == "survey_hull"  # the 3x pair, not the 1x pair


def test_pick_confusion_examples_is_order_independent():
    rows = (
        [_confusion_row(f"a{i}", "survey_hull", "sound_tanks", correct=False) for i in range(3)]
        + [_confusion_row("b1", "pull", "attach_tug", correct=False)]
        + [_confusion_row("c1", "release_co2", "muster_personnel", correct=False)]
    )
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)
    picked_a = pick_confusion_examples(rows, n_pairs=3, per_pair=1)
    picked_b = pick_confusion_examples(shuffled, n_pairs=3, per_pair=1)
    assert picked_a == picked_b


# ── build_narrative ──────────────────────────────────────────────────────

def test_build_narrative_reports_unspecified_rate():
    summary = {"mean_n_SPECIFIED_UNGRADED": "2", "mean_n_UNSPECIFIED": "2",
               "mean_n_CONDITIONAL_UNRESOLVED": "0", "mean_n_SEQUENCE_VIOLATION": "0",
               "mean_n_METHOD_ERROR": "0", "mean_n_NO_MATCH": "0"}
    lines = build_narrative(summary)
    assert any("UNSPECIFIED rate" in l for l in lines)
    assert any("0.500" in l for l in lines)


def test_build_narrative_degrades_gracefully_on_empty_summary():
    lines = build_narrative({})
    assert any("Could not assess" in l or "_" in l for l in lines)
    assert len(lines) >= 1


# ── build_results_table ──────────────────────────────────────────────────

def test_build_results_table_has_one_row_per_run():
    rows = [{"run": "arm_a", "n_images": "10", "pct_route_recognised": "0.8",
             "mean_n_UNSPECIFIED": "1.5", "mean_gate_rate": "0.2",
             "mean_route_coherence": "0.9"},
            {"run": "arm_b", "n_images": "12"}]
    table = build_results_table(rows)
    body_lines = [l for l in table if l.startswith("| arm")]
    assert len(body_lines) == 2
    assert "arm_a" in table[2]


# ── write_case_studies / write_report: file round-trip ──────────────────

def test_write_case_studies_creates_file_with_verdict_sections(tmp_path):
    per_step = [_step_row("a/1.jpg", 1, "SEQUENCE_VIOLATION")]
    out = tmp_path / "case_studies.md"
    write_case_studies(per_step, [], out)
    text = out.read_text(encoding="utf-8")
    assert "SEQUENCE_VIOLATION" in text
    assert "a/1.jpg" in text


def test_write_case_studies_handles_no_calibration_records(tmp_path):
    out = tmp_path / "case_studies.md"
    write_case_studies([], [], out)
    text = out.read_text(encoding="utf-8")
    assert "No calibration records available" in text


def test_write_report_creates_file_referencing_case_studies(tmp_path):
    summary_rows = [{"run": "answers_baseline", "n_images": "5",
                      "pct_route_recognised": "1.0", "mean_n_UNSPECIFIED": "0.5",
                      "mean_n_SPECIFIED_UNGRADED": "3", "mean_n_CONDITIONAL_UNRESOLVED": "0",
                      "mean_n_SEQUENCE_VIOLATION": "0", "mean_n_METHOD_ERROR": "0",
                      "mean_n_NO_MATCH": "0", "mean_gate_rate": "0.1",
                      "mean_route_coherence": "0.9"}]
    out = tmp_path / "report.md"
    write_report(summary_rows, out)
    text = out.read_text(encoding="utf-8")
    assert "answers_baseline" in text
    assert "case_studies.md" in text


def test_write_report_handles_no_summary_rows(tmp_path):
    out = tmp_path / "report.md"
    write_report([], out)
    text = out.read_text(encoding="utf-8")
    assert "No summary rows available" in text


def test_write_report_discloses_it_only_narrates_the_last_arm_when_multiple_rows(tmp_path):
    """Regression anchor (recheck pass): write_report used to silently
    narrate only the last summary row when given several (e.g. from the
    cumulative cross-arm CSV), with no indication the other arms' findings
    were skipped. It must say so explicitly rather than implying a
    cross-arm comparison that isn't actually computed."""
    rows = [
        {"run": "arm_a", "n_images": "5"},
        {"run": "arm_b", "n_images": "5"},
    ]
    out = tmp_path / "report.md"
    write_report(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "only the most recent" in text
    assert "arm_b" in text.split("Findings assessment")[1]
