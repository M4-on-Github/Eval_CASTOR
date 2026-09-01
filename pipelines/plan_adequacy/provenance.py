"""
provenance.py -- so that no number ships that cannot name its run.

This exists because of a specific incident, not as a general principle. A
published weekly report carried null_fidelity = 0.896 while the final
calibration run had produced 0.8406; the figure had been read from an earlier
run whose output looked identical on disk. Nothing in the artefacts made the
two distinguishable, so nothing caught it until the raw JSON was reread by
hand. Provenance is the fix.

The design is one manifest per run plus one append-only index of every run
ever executed. Rows in per_step.csv / per_image.csv carry ONLY the run_id --
the manifest holds everything else, so widening the recorded context later
costs a manifest field rather than a schema migration on 1,980 rows.

    manifest.json     one per run directory: run_id, code SHA, registry
                      hashes, model ids + decoding params + seeds, prompt
                      hashes, input hash, UTC timestamp
    runs.csv          append-only, one row per run, at BASE_OUT_DIR

A run_id is derived from the manifest's own content, not from a counter or a
timestamp, so two runs with genuinely identical inputs, code, and parameters
collide by construction -- which is the correct behaviour: they ARE the same
run, and a fresh id would imply a distinction that does not exist. Anything
that actually differs, including a registry edit or a changed seed, produces a
different id.
"""

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.paths import (BASE_OUT_DIR, GOALS_PATH,
                                           ROUTES_PATH, TOOLS_PATH)

MANIFEST = "manifest.json"
RUNS_INDEX = "runs.csv"

RUNS_FIELDS = ["run_id", "run_name", "created_utc", "code_sha", "registry_sha",
               "planner", "extractor", "seed", "n_plans", "note"]


def file_sha(path, length: int = 12) -> str:
    """Truncated SHA-256 of a file's bytes, or "" when absent.

    Truncated because these are identifiers to eyeball and paste into a
    caption, not cryptographic commitments -- 12 hex characters is ample to
    distinguish the handful of registry versions a project like this
    produces, and a full digest in a table nobody reads is a full digest
    nobody checks.
    """
    path = Path(path)
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def registry_sha() -> str:
    """One hash over all three registry files.

    A registry edit changes every verdict downstream of it, so it belongs in
    the run identity exactly as much as the code SHA does. Hashing the three
    together rather than separately means a run row stays one column wide and
    still changes whenever any of them does.
    """
    h = hashlib.sha256()
    for p in (TOOLS_PATH, ROUTES_PATH, GOALS_PATH):
        h.update(file_sha(p, 64).encode())
    return h.hexdigest()[:12]


def code_sha() -> str:
    """git HEAD of Eval_CASTOR, with a -dirty suffix when the tree has
    uncommitted changes. The suffix matters more than the SHA: a number
    produced from an uncommitted tree is not reproducible, and the manifest
    should say so rather than imply a clean provenance it does not have."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=EVAL_ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=EVAL_ROOT, capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def build_manifest(run_name: str, *, planner: str = "", extractor: str = "",
                   seed=None, sampling: dict = None, decoding: dict = None,
                   prompts=(), input_path=None, n_plans=None,
                   note: str = "") -> dict:
    """Everything needed to reproduce one run, plus a content-derived id."""
    manifest = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_sha": code_sha(),
        "registry_sha": registry_sha(),
        "planner": planner,
        "extractor": extractor,
        "seed": seed,
        "sampling": dict(sampling or {}),
        "decoding": dict(decoding or {}),
        "prompt_shas": {str(Path(p).name): file_sha(p) for p in prompts},
        "input_sha": file_sha(input_path) if input_path else "",
        "n_plans": n_plans,
        "note": note,
    }
    # created_utc is excluded from the identity: two runs differing only in
    # when they were started are the same run, and letting the clock into the
    # hash would defeat the collision property the id exists for.
    ident = {k: v for k, v in manifest.items() if k != "created_utc"}
    manifest["run_id"] = hashlib.sha256(
        json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return manifest


def write_manifest(manifest: dict, run_dir) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")
    return path


def read_manifest(run_dir):
    path = Path(run_dir) / MANIFEST
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_run(manifest: dict, base_dir=None) -> Path:
    """Append to the runs index, unless this exact run_id is already there.

    Append-only and idempotent: re-running the identical pipeline should not
    grow the index, but nothing already written is ever rewritten. The index
    is the thing that makes "which run produced this figure" answerable
    months later, so losing a row is worse than carrying a duplicate.
    """
    base_dir = Path(base_dir or BASE_OUT_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / RUNS_INDEX

    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            if any(r.get("run_id") == manifest["run_id"] for r in csv.DictReader(f)):
                return path

    row = {k: manifest.get(k, "") for k in RUNS_FIELDS}
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return path


def stamp(run_name: str, run_dir, **kw) -> dict:
    """Build, write, and index a manifest in one call. Returns it, so a
    caller can put the run_id straight into whatever it is about to write."""
    manifest = build_manifest(run_name, **kw)
    write_manifest(manifest, run_dir)
    append_run(manifest, Path(run_dir).parent)
    return manifest
