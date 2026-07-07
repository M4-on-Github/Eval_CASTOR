"""
Tests for pipelines/salvage_analysis/normalize.py (Stage 2).
Only the pure clustering function is tested here — no real embeddings are
fetched, per this repo's convention of not mocking network calls.
Run: python -m pytest tests/test_salvage_normalize.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.salvage_analysis.normalize import cluster_phrases


def test_near_identical_vectors_merge_into_same_canonical_label():
    # "fireboat" and "fire boat" point in nearly the same direction
    # (cosine distance ~0.00005); "crane" is orthogonal (cosine distance 1.0).
    phrase_to_vector = {
        "fireboat":  [1.0, 0.0],
        "fire boat": [0.99, 0.01],
        "crane":     [0.0, 1.0],
    }

    mapping = cluster_phrases(phrase_to_vector, threshold=0.3)

    assert mapping["fireboat"] == mapping["fire boat"]
    assert mapping["crane"] != mapping["fireboat"]


def test_far_apart_vectors_stay_separate_at_small_threshold():
    phrase_to_vector = {
        "fireboat": [1.0, 0.0],
        "crane":    [0.0, 1.0],
        "divers":   [-1.0, 0.0],
    }

    mapping = cluster_phrases(phrase_to_vector, threshold=0.1)

    assert len(set(mapping.values())) == 3


def test_large_threshold_merges_everything():
    phrase_to_vector = {
        "fireboat": [1.0, 0.0],
        "crane":    [0.0, 1.0],
        "divers":   [-1.0, 0.0],
    }

    mapping = cluster_phrases(phrase_to_vector, threshold=2.0)

    assert len(set(mapping.values())) == 1


def test_single_phrase_maps_to_itself_without_crashing():
    phrase_to_vector = {"fireboat": [1.0, 0.0]}

    mapping = cluster_phrases(phrase_to_vector, threshold=0.3)

    assert mapping == {"fireboat": "fireboat"}


def test_empty_input_returns_empty_mapping():
    assert cluster_phrases({}, threshold=0.3) == {}


def test_canonical_label_is_one_of_the_original_phrases():
    phrase_to_vector = {
        "fireboat":  [1.0, 0.0],
        "fire boat": [0.99, 0.01],
    }
    mapping = cluster_phrases(phrase_to_vector, threshold=0.3)
    canonical = mapping["fireboat"]
    assert canonical in phrase_to_vector
