"""
Shared data loaders for CASTOR evaluation.
Handles JSONL inference files (including ministral /n-separator format),
ground-truth CSV, and Gemma pre-parsed files.
"""

import json
from pathlib import Path

import pandas as pd


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def load_ground_truth(csv_path: Path) -> dict:
    """Returns {image_path -> GT field dict}."""
    df = pd.read_csv(csv_path, header=0)
    df = df.rename(columns={c: "_unused" for c in df.columns if str(c).startswith("Unnamed")})
    gt = {}
    for _, row in df.iterrows():
        img = str(row["image"]).strip()
        gt[img] = {
            "state":         _safe_str(row.get("state")),
            "vessel_type":   _safe_str(row.get("vessel_type")),
            "cargo":         _safe_str(row.get("cargo")),
            "q1":            _safe_str(row.get("q1")).lower(),
            "q2":            _safe_str(row.get("q2")).lower(),
            "q3":            _safe_str(row.get("q3")).lower(),
            "q4":            _safe_str(row.get("q4")).lower(),
            "q5":            _safe_str(row.get("q5")).lower(),
            "size_estimate": _safe_str(row.get("size_estimate")),
        }
    return gt


def _load_slash_n_jsonl(jsonl_path: Path) -> list:
    """Fallback for ministral-style files that use literal /n as the record separator.
    The entire file is one line; records are delimited by }/n{ with /n inside strings
    standing in for actual newlines.
    """
    with open(jsonl_path, encoding="utf-8") as f:
        content = f.read().rstrip()
    if content.endswith("/n"):
        content = content[:-2]
    segments = content.split("}/n{")
    if len(segments) <= 1:
        return []
    records = []
    for i, seg in enumerate(segments):
        if i == 0:
            piece = seg + "}"
        elif i < len(segments) - 1:
            piece = "{" + seg + "}"
        else:
            piece = "{" + seg
        piece = piece.replace("/n", r"\n")
        try:
            records.append(json.loads(piece))
        except json.JSONDecodeError as e:
            print(f"  WARNING /n-seg {i} in {jsonl_path.name}: {e}")
    if records:
        print(f"  Using /n-separator format: {len(records)} records.")
    return records


def load_run(jsonl_path: Path) -> list:
    """Load inference JSONL; falls back to /n-separator parser if standard parsing yields 0."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    records.append(json.loads(line, strict=False))
                except json.JSONDecodeError as e:
                    print(f"  WARNING: could not parse line {lineno} in {jsonl_path.name}: {e}")
    if not records:
        records = _load_slash_n_jsonl(jsonl_path)
    return records


def load_pre_parsed(path: Path) -> dict:
    """Load a Gemma-extracted JSONL. Returns {image -> record} for gemma_parse_ok=True only."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("gemma_parse_ok") and "image" in rec:
                result[rec["image"]] = rec
    return result


def read_jsonl(path: Path):
    """Generator: yields parsed records from a JSONL file, silently skipping bad lines."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass
