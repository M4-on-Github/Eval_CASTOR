"""
Combines per-field shard JSONLs into one merged-record JSONL per run, so
Pipeline 6 can consume separated-into-parts inference output the same way
it consumes full-answer output. Ports the discovery/merge logic from
tempp/group_answers.py; the only difference is the output record shape has
plain top-level field keys (state, recovery_considerations, ...) rather than
a CoT-wrapped `text` blob -- records.get_field_text() reads both shapes.

Usage (library only -- called automatically by extract.py/contingency.py
when the plain <run_name>.jsonl isn't found):
    from pipelines.salvage_analysis.combine_shards import combine_run
    combined_path = combine_run(search_dir, run_name, out_dir)
"""

import json
import re
from pathlib import Path

SHARD_RE = re.compile(r"^(?P<base>.+)_(?P<num>[1-7])_(?P<field>[A-Za-z]+)_j(?P<job>\d+)\.jsonl$")

FIELD_KEY = {
    "1": "state",
    "2": "vessel_type",
    "3": "size_estimate",
    "4": "cargo",
    "5": "environmental_conditions",
    "6": "limitations",
    "7": "recovery_considerations",
}


def _load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: {path.name}:{lineno}: {e}")
    return records


def discover_run_names(directory: Path) -> list:
    """Every *.jsonl file directly in `directory` (e.g. dropped into
    p6_plans_to_judge/) is treated as its own run -- no shard-detection
    heuristic. Two files can coincidentally match the per-field shard naming
    convention (SHARD_RE) while having unrelated job IDs and no real sibling
    shards to combine with, so guessing "this looks like a shard" and
    silently excluding it from discovery was wrong more often than it
    helped. resolve_input_path()'s auto-combine (via combine_run) is still
    available for the case where you deliberately request a run name that
    has real sibling shards on disk -- this function just no longer
    pre-filters what counts as a run."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.jsonl"))


def discover_shard_groups(directory: Path) -> dict:
    """Returns {run_key: {field_key: Path}} for every shard group found in
    directory. Files that don't match the shard naming convention are
    ignored."""
    groups = {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        m = SHARD_RE.match(path.name)
        if not m:
            continue
        run_key = f"{m.group('base')}_j{m.group('job')}"
        field_key = FIELD_KEY[m.group("num")]
        groups.setdefault(run_key, {})[field_key] = path
    return groups


def merge_run(run_key: str, field_files: dict) -> list:
    """Joins each field shard's records by image key into one record per
    image, with plain top-level field keys."""
    per_image = {}
    order = []

    for field_key, path in field_files.items():
        for rec in _load_jsonl(path):
            image = rec.get("image")
            if image is None:
                continue
            if image not in per_image:
                per_image[image] = {"image": image, "run_name": run_key}
                order.append(image)
            per_image[image][field_key] = rec.get("text")

    return [per_image[image] for image in order]


def resolve_input_path(run_name: str, input_path: Path, search_dir: Path, out_dir: Path) -> Path:
    """Resolves the Stage 1/3 input file for a run, in priority order:
    1. `input_path` if explicitly given (must exist).
    2. `<search_dir>/<run_name>.jsonl` (a full-answer file already in place).
    3. `<out_dir>/<run_name>_combined.jsonl`, reused as-is if it already
       exists from a prior auto-combine.
    4. Auto-combine: discover separated-into-parts shards for `run_name`
       under `search_dir` and merge them, writing the result to (3).
    Raises FileNotFoundError if none of the above apply."""
    if input_path is not None:
        if not Path(input_path).exists():
            raise FileNotFoundError(input_path)
        return Path(input_path)

    search_dir = Path(search_dir)
    direct = search_dir / f"{run_name}.jsonl"
    if direct.exists():
        return direct

    out_dir = Path(out_dir)
    combined_path = out_dir / f"{run_name}_combined.jsonl"
    if combined_path.exists():
        return combined_path

    combined = combine_run(search_dir, run_name, out_dir)
    if combined is not None:
        print(f"  No full-answer JSONL found for '{run_name}' -- combined separated-part shards -> {combined}")
        return combined

    raise FileNotFoundError(
        f"No input found for run '{run_name}': no {direct}, no combined file, "
        f"and no separated-part shards under {search_dir}"
    )


def combine_run(directory: Path, run_name: str, out_dir: Path) -> Path:
    """Finds every shard group under `directory` whose run_key starts with
    `run_name`, merges them, and writes one combined JSONL to
    `out_dir/<run_name>_combined.jsonl`. Returns the output path, or None if
    no matching shards were found."""
    groups = discover_shard_groups(directory)
    matching = {k: v for k, v in groups.items() if k.startswith(run_name)}
    if not matching:
        return None

    merged = []
    for run_key, field_files in sorted(matching.items()):
        merged.extend(merge_run(run_key, field_files))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_combined.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


def main():
    """Prints one full-answer run name per line, for submit_salvage.sh to
    loop over every file in p6_plans_to_judge/ without a --run argument."""
    import argparse
    ap = argparse.ArgumentParser(description="List full-answer run names in a directory")
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    for name in discover_run_names(args.dir):
        print(name)


if __name__ == "__main__":
    main()
