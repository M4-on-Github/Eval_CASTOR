"""
Tests for pipelines/plan_adequacy/extract.py's run() -- the Stage 1 CLI
orchestration (resume logic, scenario lookup, batch construction, output
shape). extract_steps() itself is monkeypatched to a fake that never
touches vLLM, matching the repo convention (test_plan_adequacy_calibrate.py's
header: run_calibration()/vLLM is not exercised in local tests) -- this file
only adds coverage for the NEW orchestration around it.
Run: python -m pytest tests/test_plan_adequacy_extract_run.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pipelines.plan_adequacy.extract as extract_mod
from pipelines.plan_adequacy.paths import RunPaths
from pipelines.plan_adequacy.vocab import ToolCall

_GT_HEADER = "image,state,vessel_type,,cargo,error,q1,q2,q3,q4,q5,size_estimate\n"


def _write_gt(tmp_path, rows):
    path = tmp_path / "human_gt.csv"
    lines = [_GT_HEADER]
    for image, state, vessel_type in rows:
        lines.append(f"{image},{state},{vessel_type},,,,yes,no,no,no,no,medium (10-50m)\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_answers(tmp_path, records):
    path = tmp_path / "answers_baseline.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _fake_extract_steps(monkeypatch, calls_by_index=None):
    """Replace extract.extract_steps with a deterministic fake: one
    no_match ToolCall per input step, in order, so run()'s zip alignment
    between batch_meta and the returned calls can be checked precisely."""
    def fake(steps, model_dir, registry, max_model_len=4096, max_tokens=256):
        return [ToolCall(step_num=n, step_text=t, tool="attach_tug", params={})
                for n, t, _casualty in steps]
    monkeypatch.setattr(extract_mod, "extract_steps", fake)


def test_run_writes_one_row_per_step_across_all_pending_images(tmp_path, monkeypatch):
    _fake_extract_steps(monkeypatch)
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship"),
                               ("aground/2.jpg", "aground", "Tanker")])
    answers = _write_answers(tmp_path, [
        {"image": "aground/1.jpg", "text": "1. Sound the tanks.\n2. Pull with force."},
        {"image": "aground/2.jpg", "text": "1. Survey the seabed."},
    ])
    out_dir = tmp_path / "out"
    extract_mod.run(answers, out_dir, "fake_model", "/fake/model/dir", gt_path=gt)

    tc_path = RunPaths("answers_baseline", base_dir=out_dir).tool_calls
    rows = [json.loads(l) for l in tc_path.open(encoding="utf-8")]
    assert len(rows) == 3
    assert {r["image"] for r in rows} == {"aground/1.jpg", "aground/2.jpg"}


def test_run_writes_model_key_onto_every_row(tmp_path, monkeypatch):
    """Regression anchor (recheck pass): tool_calls.jsonl previously had no
    field recording which model produced an extraction -- run() accepted
    model_key but never wrote it. Provenance matters once more than one
    model has been tried against the same run name."""
    _fake_extract_steps(monkeypatch)
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship")])
    answers = _write_answers(tmp_path, [
        {"image": "aground/1.jpg", "text": "1. Sound the tanks."},
    ])
    out_dir = tmp_path / "out"
    extract_mod.run(answers, out_dir, "glm4_32b", "/fake/model/dir", gt_path=gt)

    tc_path = RunPaths("answers_baseline", base_dir=out_dir).tool_calls
    rows = [json.loads(l) for l in tc_path.open(encoding="utf-8")]
    assert rows[0]["model"] == "glm4_32b"


def test_run_writes_ground_truth_casualty_onto_every_row(tmp_path, monkeypatch):
    _fake_extract_steps(monkeypatch)
    gt = _write_gt(tmp_path, [("capsized/1.jpg", "capsized", "Ferry")])
    answers = _write_answers(tmp_path, [
        {"image": "capsized/1.jpg", "text": "1. Rig parbuckling gear."},
    ])
    out_dir = tmp_path / "out"
    extract_mod.run(answers, out_dir, "fake_model", "/fake/model/dir", gt_path=gt)

    tc_path = RunPaths("answers_baseline", base_dir=out_dir).tool_calls
    rows = [json.loads(l) for l in tc_path.open(encoding="utf-8")]
    assert rows[0]["casualty"] == "capsized"


def test_run_skips_image_with_no_ground_truth(tmp_path, monkeypatch):
    _fake_extract_steps(monkeypatch)
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship")])
    answers = _write_answers(tmp_path, [
        {"image": "aground/1.jpg", "text": "1. Sound the tanks."},
        {"image": "aground/UNKNOWN.jpg", "text": "1. Sound the tanks."},
    ])
    out_dir = tmp_path / "out"
    extract_mod.run(answers, out_dir, "fake_model", "/fake/model/dir", gt_path=gt)

    tc_path = RunPaths("answers_baseline", base_dir=out_dir).tool_calls
    rows = [json.loads(l) for l in tc_path.open(encoding="utf-8")]
    assert {r["image"] for r in rows} == {"aground/1.jpg"}


def test_run_skips_plan_with_no_parseable_steps(tmp_path, monkeypatch):
    _fake_extract_steps(monkeypatch)
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship")])
    answers = _write_answers(tmp_path, [
        {"image": "aground/1.jpg", "text": ""},
    ])
    out_dir = tmp_path / "out"
    tc_path = extract_mod.run(answers, out_dir, "fake_model", "/fake/model/dir", gt_path=gt)
    assert not tc_path.exists() or tc_path.read_text(encoding="utf-8").strip() == ""


def test_run_is_resume_safe_skips_already_processed_images(tmp_path, monkeypatch):
    """Regression anchor: a second run() call over the same output directory
    must not re-extract an image already present in tool_calls.jsonl --
    same convention as run_coherence_judge.py's done_images handling."""
    gt = _write_gt(tmp_path, [("aground/1.jpg", "aground", "Cargo Ship"),
                               ("aground/2.jpg", "aground", "Tanker")])
    answers = _write_answers(tmp_path, [
        {"image": "aground/1.jpg", "text": "1. Sound the tanks."},
        {"image": "aground/2.jpg", "text": "1. Survey the seabed."},
    ])
    out_dir = tmp_path / "out"

    calls_seen = []

    def counting_fake(steps, model_dir, registry, max_model_len=4096, max_tokens=256):
        calls_seen.append(list(steps))
        return [ToolCall(step_num=n, step_text=t, tool="attach_tug", params={})
                for n, t, _casualty in steps]
    monkeypatch.setattr(extract_mod, "extract_steps", counting_fake)

    extract_mod.run(answers, out_dir, "fake_model", "/fake/model/dir", gt_path=gt)
    assert len(calls_seen) == 1
    assert len(calls_seen[0]) == 2  # both images' single steps

    # Second run over the same output dir: nothing new to extract.
    extract_mod.run(answers, out_dir, "fake_model", "/fake/model/dir", gt_path=gt)
    assert len(calls_seen) == 1  # extract_steps was NOT called again

    tc_path = RunPaths("answers_baseline", base_dir=out_dir).tool_calls
    rows = [json.loads(l) for l in tc_path.open(encoding="utf-8")]
    assert len(rows) == 2  # no duplicate rows from the second run
