"""
Tests for pipelines/salvage_analysis/contingency.py (Stage 3).
All expected values are hand-computed against a small fixed toy dataset.
Run: python -m pytest tests/test_salvage_contingency.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from pipelines.salvage_analysis.contingency import (
    build_contingency_table,
    build_element_sets,
    modal_element_set,
    typicality_score,
)


# ── build_element_sets ────────────────────────────────────────────────────────

def test_build_element_sets_maps_raw_phrases_to_canonical():
    raw_elements_by_image = {
        "img1": ["call a fireboat", "crane"],
        "img2": ["dispatch fireboat"],
    }
    raw_to_canonical = {
        "call a fireboat": "fireboat",
        "dispatch fireboat": "fireboat",
        "crane": "crane",
    }

    result = build_element_sets(raw_elements_by_image, raw_to_canonical)

    assert result == {"img1": {"fireboat", "crane"}, "img2": {"fireboat"}}


def test_build_element_sets_skips_unmapped_phrases():
    raw_elements_by_image = {"img1": ["some unrecognized phrase"]}
    raw_to_canonical = {"fireboat": "fireboat"}

    result = build_element_sets(raw_elements_by_image, raw_to_canonical)

    assert result == {"img1": set()}


# ── modal_element_set ─────────────────────────────────────────────────────────

def test_modal_element_set_strict_majority():
    # on_fire group: img1={fireboat}, img2={fireboat,crane}
    # fireboat: 2/2=100% -> included; crane: 1/2=50% -> excluded (not >50%)
    element_sets = {"img1": {"fireboat"}, "img2": {"fireboat", "crane"}}
    grouping = {"img1": "on_fire", "img2": "on_fire"}

    modal = modal_element_set(element_sets, grouping, "on_fire")

    assert modal == {"fireboat"}


def test_modal_element_set_empty_when_no_majority():
    # aground group: img3={crane}, img4={}
    # crane: 1/2=50% -> excluded (not strictly >50%)
    element_sets = {"img3": {"crane"}, "img4": set()}
    grouping = {"img3": "aground", "img4": "aground"}

    modal = modal_element_set(element_sets, grouping, "aground")

    assert modal == set()


# ── typicality_score ──────────────────────────────────────────────────────────

def test_typicality_score_perfect_match():
    assert typicality_score({"fireboat"}, {"fireboat"}) == 1.0


def test_typicality_score_partial_overlap():
    # intersection={fireboat}=1, union={fireboat,crane}=2 -> 0.5
    assert typicality_score({"fireboat", "crane"}, {"fireboat"}) == pytest.approx(0.5)


def test_typicality_score_no_overlap():
    assert typicality_score({"crane"}, set()) == 0.0


def test_typicality_score_both_empty_is_perfect_match():
    assert typicality_score(set(), set()) == 1.0


# ── build_contingency_table (full toy dataset, hand-computed) ────────────────

def test_build_contingency_table_hand_computed_toy_dataset():
    # 6 images, predicted_state groups: on_fire={img1,img2}, aground={img3,img4},
    # sunken={img5,img6}. gt_state groups differ from predicted for img2 (see
    # below) so pred/gt typicality scores are verified to diverge.
    images = ["img1", "img2", "img3", "img4", "img5", "img6"]
    predicted_state = {
        "img1": "on_fire", "img2": "on_fire",
        "img3": "aground", "img4": "aground",
        "img5": "sunken",  "img6": "sunken",
    }
    gt_state = {
        "img1": "on_fire", "img2": "aground",
        "img3": "aground", "img4": "aground",
        "img5": "sunken",  "img6": "sunken",
    }
    element_sets = {
        "img1": {"fireboat"},
        "img2": {"fireboat", "crane"},
        "img3": {"crane"},
        "img4": set(),
        "img5": {"fireboat"},
        "img6": set(),
    }

    df = build_contingency_table(
        images, predicted_state, gt_state, element_sets, elements=["fireboat", "crane"],
    )

    df = df.set_index("image")

    # Boolean element columns
    assert df.loc["img1", "fireboat"] == True
    assert df.loc["img1", "crane"] == False
    assert df.loc["img2", "fireboat"] == True
    assert df.loc["img2", "crane"] == True
    assert df.loc["img4", "fireboat"] == False
    assert df.loc["img4", "crane"] == False

    # Predicted-state modal sets: on_fire->{fireboat}, aground->{}, sunken->{}
    expected_pred = {"img1": 1.0, "img2": 0.5, "img3": 0.0, "img4": 1.0, "img5": 0.0, "img6": 1.0}
    for img, expected in expected_pred.items():
        assert df.loc[img, "typicality_score_pred"] == pytest.approx(expected), img

    # GT-state modal sets: on_fire(img1 only)->{fireboat},
    # aground(img2,img3,img4)->{crane} (2/3>50%), sunken->{}
    expected_gt = {"img1": 1.0, "img2": 0.5, "img3": 1.0, "img4": 0.0, "img5": 0.0, "img6": 1.0}
    for img, expected in expected_gt.items():
        assert df.loc[img, "typicality_score_gt"] == pytest.approx(expected), img

    assert list(df["predicted_state"]) == [predicted_state[i] for i in images]
    assert list(df["gt_state"]) == [gt_state[i] for i in images]
