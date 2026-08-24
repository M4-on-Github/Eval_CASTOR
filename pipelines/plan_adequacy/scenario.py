"""
human_gt.csv -> ScenarioParams.

Built on top of shared.loaders.load_ground_truth() rather than reading the
CSV directly, so the Unnamed-column handling and NaN-safety already fixed
there is not duplicated here.

Known data issues in human_gt.csv, both handled below (see the plan-adequacy
design plan, section 5 "Reuse", for the measurements backing these):
  - size_estimate has an en-dash/hyphen split: "medium (10-50m)" (ASCII
    hyphen, x6) vs "medium (10–50m)" (en dash, x43) are the same
    category. Both normalize to the same `size_category`.
  - `cargo` is populated in only 3/110 rows and `error` is entirely empty;
    both are carried through as raw strings but never relied on for a
    scenario decision.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shared.loaders import load_ground_truth

EVAL_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_GT_PATH = EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv"

#: Habitat-sensitive vessel/cargo keywords -- authored, feeds C6 (NEBA) via
#: goals.json. Not corpus-sourced; a coarse proxy until a real habitat
#: layer exists. See salvage_plan_checker.md scenario parameters (S4).
_HABITAT_KEYWORDS = re.compile(
    r"\b(tanker|oil|fuel|chemical|hazmat|hazardous)\b", re.IGNORECASE
)

_SIZE_PATTERNS = [
    ("small", re.compile(r"small", re.IGNORECASE)),
    ("medium", re.compile(r"medium", re.IGNORECASE)),
    ("large", re.compile(r"large", re.IGNORECASE)),
]


@dataclass(frozen=True)
class ScenarioParams:
    """Per-image scenario, derived from human_gt.csv.

    This is scenario authoring (vessel identity/size), not dynamics
    authoring (physics constants) -- identical across arms, so it cannot
    bias an arm comparison. See salvage_plan_checker.md section 4.
    """

    image: str
    state: str
    vessel_type: str
    size_category: str          # "small" | "medium" | "large" | "unknown"
    size_estimate_raw: str
    cargo_raw: str
    habitat_sensitive: bool
    q1: str
    q2: str
    q3: str
    q4: str
    q5: str


def _normalize_size(size_estimate: str) -> str:
    """Collapse the en-dash/hyphen variants of size_estimate to one category."""
    for category, pattern in _SIZE_PATTERNS:
        if pattern.search(size_estimate or ""):
            return category
    return "unknown"


def _habitat_sensitive(vessel_type: str, cargo: str) -> bool:
    """Coarse authored proxy for habitat sensitivity -- see module docstring."""
    text = f"{vessel_type} {cargo}"
    return bool(_HABITAT_KEYWORDS.search(text))


def load_scenarios(gt_path: Optional[Path] = None) -> dict:
    """Return {image -> ScenarioParams}, keyed exactly as
    shared.loaders.load_ground_truth() keys its dict (e.g. "aground/00017.jpg")."""
    gt = load_ground_truth(gt_path or _DEFAULT_GT_PATH)

    scenarios = {}
    for image, fields in gt.items():
        size_raw = fields.get("size_estimate", "")
        vessel_type = fields.get("vessel_type", "")
        cargo = fields.get("cargo", "")
        scenarios[image] = ScenarioParams(
            image=image,
            state=fields.get("state", ""),
            vessel_type=vessel_type,
            size_category=_normalize_size(size_raw),
            size_estimate_raw=size_raw,
            cargo_raw=cargo,
            habitat_sensitive=_habitat_sensitive(vessel_type, cargo),
            q1=fields.get("q1", ""),
            q2=fields.get("q2", ""),
            q3=fields.get("q3", ""),
            q4=fields.get("q4", ""),
            q5=fields.get("q5", ""),
        )
    return scenarios
