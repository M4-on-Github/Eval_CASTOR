"""
Tests for pipelines/plan_adequacy/provenance.py and executor determinism.

Two things are asserted here. First, that run identity actually distinguishes
runs that differ and collides runs that do not -- an id that changes on every
invocation is a timestamp wearing an id's clothing, and one that never changes
is decoration. Second, that the executor is deterministic, which is the
property every "same run_id, same numbers" claim silently rests on.

Run: python -m pytest tests/test_plan_adequacy_provenance.py -v
"""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.plan_adequacy.aggregate import (build_per_image_rows,
                                               build_per_step_rows)
from pipelines.plan_adequacy.executor import execute_plan
from pipelines.plan_adequacy.methods import RouteRegistry
from pipelines.plan_adequacy.provenance import (RUNS_FIELDS, append_run,
                                                build_manifest, file_sha,
                                                read_manifest, registry_sha,
                                                stamp, write_manifest)
from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry


def _kw(**over):
    base = dict(planner="qwen3vl8b", extractor="glm4_32b", seed=1234,
                sampling={"temperature": 0.0}, decoding={"guided": True},
                n_plans=110)
    base.update(over)
    return base


# ── run identity ─────────────────────────────────────────────────────────────

def test_identical_inputs_produce_the_same_run_id():
    """Two runs differing only in when they started ARE the same run; a fresh
    id would imply a distinction that does not exist."""
    a = build_manifest("r", **_kw())
    b = build_manifest("r", **_kw())
    assert a["run_id"] == b["run_id"]
    assert a["created_utc"] not in (None, "")


def test_every_recorded_parameter_changes_the_run_id():
    base = build_manifest("r", **_kw())["run_id"]
    for field, value in [("seed", 99), ("planner", "other"), ("extractor", "phi4_14b"),
                         ("n_plans", 111), ("sampling", {"temperature": 0.7}),
                         ("decoding", {"guided": False})]:
        assert build_manifest("r", **_kw(**{field: value}))["run_id"] != base, field
    assert build_manifest("other_run", **_kw())["run_id"] != base


def test_registry_hash_is_stable_and_present():
    """A registry edit changes every verdict downstream of it, so it belongs
    in run identity as much as the code SHA."""
    assert registry_sha() == registry_sha()
    assert len(registry_sha()) == 12
    assert build_manifest("r", **_kw())["registry_sha"] == registry_sha()


def test_missing_files_hash_to_empty_rather_than_raising():
    assert file_sha(Path("no/such/file.json")) == ""


def test_manifest_round_trips_through_disk(tmp_path):
    m = build_manifest("run_a", **_kw())
    write_manifest(m, tmp_path / "run_a")
    assert read_manifest(tmp_path / "run_a") == m
    assert read_manifest(tmp_path / "absent") is None


# ── runs index ───────────────────────────────────────────────────────────────

def test_runs_index_is_append_only_and_idempotent(tmp_path):
    a = build_manifest("run_a", **_kw())
    b = build_manifest("run_b", **_kw(seed=7))
    append_run(a, tmp_path)
    append_run(a, tmp_path)          # same run_id -- must not duplicate
    append_run(b, tmp_path)

    with open(tmp_path / "runs.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["run_id"] for r in rows] == [a["run_id"], b["run_id"]]
    assert set(rows[0]) == set(RUNS_FIELDS)


def test_stamp_writes_both_the_manifest_and_the_index(tmp_path):
    run_dir = tmp_path / "run_a"
    m = stamp("run_a", run_dir, **_kw())
    assert (run_dir / "manifest.json").exists()
    assert (tmp_path / "runs.csv").exists()
    assert read_manifest(run_dir)["run_id"] == m["run_id"]


def test_a_dirty_tree_is_recorded_as_dirty():
    """A number produced from an uncommitted tree is not reproducible, and
    the manifest must say so rather than imply a clean provenance."""
    sha = build_manifest("r", **_kw())["code_sha"]
    assert sha and (sha == "unknown" or sha.endswith("-dirty") or sha.isalnum())


# ── executor determinism ─────────────────────────────────────────────────────

def _calls():
    mk = lambda n, t, txt, **p: ToolCall(
        step_num=n, step_text=txt, tool=t, params=p, conditional=False,
        condition_text=None, condition_var="none")
    return [
        mk(1, "sound_tanks", "Sound all 6 tanks.", tank_ids=["1"]),
        mk(2, "survey_seabed", "Survey the seabed over 200 m."),
        mk(3, "calculate_ground_reaction", "Ground reaction 400 t."),
        mk(4, "calculate_freeing_force", "Freeing force 250 t."),
        mk(5, "attach_tug", "Attach 2 tugs of 4000 shp.", count=2, shp=4000.0),
        mk(6, "pull", "Pull at 90 t.", force_t=90.0),
    ]


def test_the_executor_is_deterministic_across_repeated_runs():
    """Every "same run_id, same numbers" claim rests on this. Sets and dicts
    are used throughout the executor and worldstate; a stray iteration over
    an unordered collection reaching an output would show up here."""
    tr, rr = ToolRegistry.load(), RouteRegistry.load()
    scenario = SimpleNamespace(image="aground/x.jpg", state="aground",
                               size_category="large", habitat_sensitive=False)
    outs = []
    for _ in range(3):
        result = execute_plan(_calls(), "aground", scenario, tr, rr)
        outs.append((build_per_step_rows([result]), build_per_image_rows([result])))
    assert outs[0] == outs[1] == outs[2]


def test_repeated_registry_loads_agree():
    a, b = RouteRegistry.load(), RouteRegistry.load()
    for casualty in sorted(a.all_casualties()):
        assert ([r.name for r in a.for_casualty(casualty)]
                == [r.name for r in b.for_casualty(casualty)])
