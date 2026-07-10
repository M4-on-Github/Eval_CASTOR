"""
Tests for pipelines/salvage_analysis/combine_shards.py -- combining
per-field shard JSONLs (answers_<base>_<N>_<field>_j<job>.jsonl) into one
merged-record JSONL per run, same convention as tempp/group_answers.py.
Run: python -m pytest tests/test_salvage_combine_shards.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.salvage_analysis.combine_shards import (
    combine_run,
    discover_run_names,
    discover_shard_groups,
    merge_run,
    resolve_input_path,
)


def _write_jsonl(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_discover_shard_groups_matches_naming_convention(tmp_path):
    _write_jsonl(tmp_path / "answers_baseline_1_state_j100.jsonl", [{"image": "a.jpg", "text": "aground"}])
    _write_jsonl(tmp_path / "answers_baseline_7_rescuePlan_j100.jsonl", [{"image": "a.jpg", "text": "tow it"}])
    _write_jsonl(tmp_path / "answers_baseline_promptv1.jsonl", [{"image": "a.jpg", "text": "unrelated"}])

    groups = discover_shard_groups(tmp_path)

    assert "answers_baseline_j100" in groups
    assert set(groups["answers_baseline_j100"].keys()) == {"state", "recovery_considerations"}


def test_discover_shard_groups_empty_when_no_shards(tmp_path):
    _write_jsonl(tmp_path / "answers_baseline.jsonl", [{"image": "a.jpg", "text": "whole answer"}])
    groups = discover_shard_groups(tmp_path)
    assert groups == {}


def test_merge_run_joins_by_image_key(tmp_path):
    state_path = tmp_path / "answers_x_1_state_j1.jsonl"
    recovery_path = tmp_path / "answers_x_7_rescuePlan_j1.jsonl"
    _write_jsonl(state_path, [{"image": "a.jpg", "text": "aground"}, {"image": "b.jpg", "text": "sunken"}])
    _write_jsonl(recovery_path, [{"image": "a.jpg", "text": "call a fireboat"}])

    merged = merge_run("answers_x_j1", {"state": state_path, "recovery_considerations": recovery_path})

    by_image = {rec["image"]: rec for rec in merged}
    assert by_image["a.jpg"]["state"] == "aground"
    assert by_image["a.jpg"]["recovery_considerations"] == "call a fireboat"
    assert by_image["b.jpg"]["state"] == "sunken"
    assert "recovery_considerations" not in by_image["b.jpg"]


def test_combine_run_writes_output_and_returns_path(tmp_path):
    _write_jsonl(tmp_path / "answers_baseline_1_state_j100.jsonl", [{"image": "a.jpg", "text": "aground"}])
    _write_jsonl(tmp_path / "answers_baseline_7_rescuePlan_j100.jsonl", [{"image": "a.jpg", "text": "tow it"}])
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = combine_run(tmp_path, "answers_baseline", out_dir)

    assert result is not None
    assert result.exists()
    lines = result.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["image"] == "a.jpg"
    assert rec["state"] == "aground"
    assert rec["recovery_considerations"] == "tow it"


def test_combine_run_returns_none_when_no_shards_match(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = combine_run(tmp_path, "answers_nonexistent", out_dir)
    assert result is None


# ── resolve_input_path ────────────────────────────────────────────────────────

def test_resolve_input_path_prefers_explicit_input(tmp_path):
    explicit = tmp_path / "explicit.jsonl"
    explicit.write_text("{}\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = resolve_input_path("answers_baseline", explicit, tmp_path, out_dir)
    assert result == explicit


def test_resolve_input_path_raises_when_explicit_input_missing(tmp_path):
    missing = tmp_path / "missing.jsonl"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    try:
        resolve_input_path("answers_baseline", missing, tmp_path, out_dir)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_resolve_input_path_finds_direct_full_answer_file(tmp_path):
    direct = tmp_path / "answers_baseline.jsonl"
    direct.write_text("{}\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = resolve_input_path("answers_baseline", None, tmp_path, out_dir)
    assert result == direct


def test_resolve_input_path_auto_combines_shards_when_no_direct_file(tmp_path):
    _write_jsonl(tmp_path / "answers_baseline_1_state_j100.jsonl", [{"image": "a.jpg", "text": "aground"}])
    _write_jsonl(tmp_path / "answers_baseline_7_rescuePlan_j100.jsonl", [{"image": "a.jpg", "text": "tow it"}])
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = resolve_input_path("answers_baseline", None, tmp_path, out_dir)

    assert result == out_dir / "answers_baseline_combined.jsonl"
    assert result.exists()


def test_resolve_input_path_reuses_existing_combined_file(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    combined = out_dir / "answers_baseline_combined.jsonl"
    combined.write_text('{"image": "a.jpg", "state": "aground"}\n', encoding="utf-8")
    # No shard files present in search_dir at all -- must reuse the existing
    # combined file rather than fail.
    result = resolve_input_path("answers_baseline", None, tmp_path, out_dir)
    assert result == combined


def test_resolve_input_path_raises_when_nothing_found(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    try:
        resolve_input_path("answers_ghost", None, tmp_path, out_dir)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


# ── discover_run_names ────────────────────────────────────────────────────────

def test_discover_run_names_finds_full_answer_files(tmp_path):
    _write_jsonl(tmp_path / "answers_baseline.jsonl", [{"image": "a.jpg"}])
    _write_jsonl(tmp_path / "answers_degf.jsonl", [{"image": "a.jpg"}])

    assert discover_run_names(tmp_path) == ["answers_baseline", "answers_degf"]


def test_discover_run_names_treats_every_jsonl_as_its_own_run(tmp_path):
    # No shard-detection heuristic -- every .jsonl in the directory is its
    # own run, even ones that happen to match the per-field shard naming
    # convention. Two files with the same "_<N>_<field>_j<job>" shape can
    # have entirely unrelated job IDs and no real sibling shards to combine
    # with, so guessing "this looks like a shard" and silently skipping it
    # was actively wrong more often than it was right.
    _write_jsonl(tmp_path / "answers_baseline.jsonl", [{"image": "a.jpg"}])
    _write_jsonl(tmp_path / "answers_baseline_1_state_j100.jsonl", [{"image": "a.jpg"}])
    _write_jsonl(tmp_path / "answers_baseline_7_rescuePlan_j100.jsonl", [{"image": "a.jpg"}])

    assert discover_run_names(tmp_path) == [
        "answers_baseline",
        "answers_baseline_1_state_j100",
        "answers_baseline_7_rescuePlan_j100",
    ]


def test_discover_run_names_empty_when_directory_missing(tmp_path):
    assert discover_run_names(tmp_path / "does_not_exist") == []
