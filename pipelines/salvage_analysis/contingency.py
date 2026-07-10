"""
Pipeline 6 Stage 3 — build the per-image contingency table.

Joins the full-answer inference run (predicted state), ground truth (GT
state), and Stage 1+2's canonical element sets into one row per image, plus
a "typicality score" (Jaccard similarity to the modal element set for that
image's state) computed separately for predicted-state and GT-state
groupings — see ADR-001 and SPEC_salvage_analysis.md for why two separate
groupings are tested rather than one.

Usage:
  python pipelines/salvage_analysis/contingency.py --run answers_baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

import pandas as pd

from shared.loaders import load_ground_truth, load_run
from shared.metrics import normalize_state
from pipelines.salvage_analysis import paths
from pipelines.salvage_analysis.combine_shards import resolve_input_path
from pipelines.salvage_analysis.records import get_field_text

RESULTS_IN = Path(os.environ.get("CASTOR_SALVAGE_RESULTS_DIR", paths.PLANS_TO_JUDGE_DIR))
GT_PATH = EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv"


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

def build_element_sets(raw_elements_by_image: dict, raw_to_canonical: dict) -> dict:
    """Map each image's raw extracted phrases to canonical elements via
    Stage 2's mapping. Unmapped phrases are silently dropped."""
    result = {}
    for image, raw_phrases in raw_elements_by_image.items():
        canonical = {raw_to_canonical[p] for p in raw_phrases if p in raw_to_canonical}
        result[image] = canonical
    return result


def modal_element_set(element_sets: dict, grouping: dict, target_group: str) -> set:
    """The set of elements present in a strict majority (>50%) of images
    whose grouping[image] == target_group."""
    images = [img for img, group in grouping.items() if group == target_group]
    if not images:
        return set()

    all_elements = set()
    for img in images:
        all_elements |= element_sets.get(img, set())

    modal = set()
    for element in all_elements:
        count = sum(1 for img in images if element in element_sets.get(img, set()))
        if count / len(images) > 0.5:
            modal.add(element)
    return modal


def typicality_score(element_set: set, modal_set: set) -> float:
    """Jaccard similarity between a record's element set and its state's
    modal set. Both empty -> 1.0 (perfect match: nothing typical, nothing said)."""
    if not element_set and not modal_set:
        return 1.0
    union = element_set | modal_set
    intersection = element_set & modal_set
    return len(intersection) / len(union)


def build_contingency_table(images: list, predicted_state: dict, gt_state: dict,
                             element_sets: dict, elements: list) -> pd.DataFrame:
    """Build the full Stage 3 per-image table."""
    modal_pred = {
        state: modal_element_set(element_sets, predicted_state, state)
        for state in set(predicted_state.values())
    }
    modal_gt = {
        state: modal_element_set(element_sets, gt_state, state)
        for state in set(gt_state.values())
    }

    rows = []
    for img in images:
        pred_s = predicted_state.get(img)
        gt_s = gt_state.get(img)
        img_elements = element_sets.get(img, set())

        row = {
            "image": img,
            "predicted_state": pred_s,
            "gt_state": gt_s,
        }
        for element in elements:
            row[element] = element in img_elements

        row["typicality_score_pred"] = typicality_score(img_elements, modal_pred.get(pred_s, set()))
        row["typicality_score_gt"] = typicality_score(img_elements, modal_gt.get(gt_s, set()))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(run_name: str, input_path: Path = None, gt_path: Path = GT_PATH) -> pd.DataFrame:
    input_path = resolve_input_path(run_name, input_path, RESULTS_IN, paths.run_dir(run_name))
    raw_elements_path = paths.raw_elements_path(run_name)
    elements_map_path = paths.elements_path(run_name)

    records = load_run(input_path)
    gt = load_ground_truth(gt_path)

    predicted_state = {}
    for rec in records:
        image = rec.get("image", "")
        raw_state = get_field_text(rec, "state") or ""
        predicted_state[image] = normalize_state(raw_state)
    gt_state = {img: fields.get("state", "") for img, fields in gt.items()}

    raw_elements_by_image = {}
    with open(raw_elements_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                raw_elements_by_image[rec["image"]] = rec.get("raw_elements", [])

    elements_map = json.loads(elements_map_path.read_text(encoding="utf-8"))
    raw_to_canonical = elements_map["raw_to_canonical"]

    element_sets = build_element_sets(raw_elements_by_image, raw_to_canonical)
    elements = sorted(set(raw_to_canonical.values()))
    images = list(raw_elements_by_image.keys())

    df = build_contingency_table(images, predicted_state, gt_state, element_sets, elements)

    out_path = paths.contingency_path(run_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  {len(df)} images -> {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Stage 3: build the per-image contingency table (Pipeline 6)"
    )
    ap.add_argument("--run", required=True)
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--gt", type=Path, default=GT_PATH)
    args = ap.parse_args()

    run(args.run, args.input, args.gt)


if __name__ == "__main__":
    main()
