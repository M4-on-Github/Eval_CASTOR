"""
Tests for pipelines/judge_panel/aggregate.py
Run: python -m pytest tests/test_aggregate.py -v
"""
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.judge_panel.aggregate import (
    load_judge_jsonl,
    compute_consensus,
    aggregate_run,
    STD_FLAG_THRESHOLD,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_record(image, gt_state, score, hallucinations=None, parse_ok=None):
    if parse_ok is None:
        parse_ok = score is not None
    return {
        "image": image,
        "gt_state": gt_state,
        "pred_text": "some text",
        "verbosity_flagged": False,
        "score": score,
        "rationale": "rationale text",
        "hallucinations": hallucinations or [],
        "parse_ok": parse_ok,
        "elapsed_s": 1.0,
    }

def _write_jsonl(path: Path, records: list):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


# ── load_judge_jsonl ──────────────────────────────────────────────────────────

def test_load_returns_dict_keyed_by_image():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "judge.jsonl"
        recs = [_make_record("img/a.jpg", "aground", 3),
                _make_record("img/b.jpg", "capsized", 2)]
        _write_jsonl(p, recs)
        result = load_judge_jsonl(p)
        assert "img/a.jpg" in result
        assert result["img/a.jpg"]["score"] == 3

def test_load_skips_blank_lines():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "judge.jsonl"
        p.write_text('{"image":"a","score":1,"gt_state":"sunken","parse_ok":true}\n\n', encoding="utf-8")
        result = load_judge_jsonl(p)
        assert len(result) == 1


# ── compute_consensus ────────────────────────────────────────────────────────

def test_consensus_all_agree():
    scores = {"gptoss_120b": 3, "deepseek_r1": 3, "qwen25_72b": 3}
    c = compute_consensus("img/a.jpg", "aground", "text", False, scores, {}, {})
    assert c["mean_score"] == 3.0
    assert c["score_std"] == 0.0
    assert c["consensus_status"] == "consensus"

def test_consensus_high_disagreement_flagged():
    scores = {"gptoss_120b": 1, "deepseek_r1": 3, "qwen25_72b": 1}
    c = compute_consensus("img/a.jpg", "aground", "text", False, scores, {}, {})
    assert c["consensus_status"] == "flagged_for_review"
    assert c["score_std"] > STD_FLAG_THRESHOLD

def test_consensus_all_null_is_parse_error():
    scores = {"gptoss_120b": None, "deepseek_r1": None, "qwen25_72b": None}
    c = compute_consensus("img/a.jpg", "aground", "text", False, scores, {}, {})
    assert c["mean_score"] is None
    assert c["consensus_status"] == "parse_error"

def test_consensus_partial_null_excluded_from_mean():
    scores = {"gptoss_120b": None, "deepseek_r1": 2, "qwen25_72b": 2}
    c = compute_consensus("img/a.jpg", "aground", "text", False, scores, {}, {})
    assert c["mean_score"] == 2.0

def test_consensus_hallucination_union():
    scores = {"gptoss_120b": 2, "deepseek_r1": 2, "qwen25_72b": 2}
    hallus = {
        "gptoss_120b": ["smoke"],
        "deepseek_r1": ["fire", "smoke"],
        "qwen25_72b":  [],
    }
    c = compute_consensus("img/a.jpg", "aground", "text", False, scores, {}, hallus)
    union = set(c["hallucination_union"])
    assert union == {"smoke", "fire"}

def test_consensus_record_has_image_key():
    scores = {"gptoss_120b": 3, "deepseek_r1": 3, "qwen25_72b": 3}
    c = compute_consensus("img/x.jpg", "sunken", "text", False, scores, {}, {})
    assert c["image"] == "img/x.jpg"


# ── aggregate_run (integration) ───────────────────────────────────────────────

def _make_three_judge_jsonls(tmp_dir: Path, run_name: str, score_sets: list[dict]):
    """score_sets: list of 5 dicts {model: score} indexed by sample."""
    images = [f"img/{i:03d}.jpg" for i in range(5)]
    gt_states = ["aground", "capsized", "on_fire", "sunken", "aground"]
    models = ["gptoss_120b", "deepseek_r1", "qwen25_72b"]
    paths = {}
    for model in models:
        recs = [
            _make_record(images[i], gt_states[i], score_sets[i][model])
            for i in range(5)
        ]
        p = tmp_dir / f"{run_name}_{model}.jsonl"
        _write_jsonl(p, recs)
        paths[model] = p
    return images, gt_states, paths

def test_aggregate_produces_consensus_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        score_sets = [{"gptoss_120b": 3, "deepseek_r1": 3, "qwen25_72b": 3}] * 5
        images, _, _ = _make_three_judge_jsonls(d, "myrun", score_sets)
        consensus_path, flagged_path = aggregate_run("myrun", d)
        assert consensus_path.exists()
        lines = [l for l in consensus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 5

def test_aggregate_flagged_jsonl_contains_disagreements():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        score_sets = [
            {"gptoss_120b": 1, "deepseek_r1": 3, "qwen25_72b": 1},  # flagged
            {"gptoss_120b": 3, "deepseek_r1": 3, "qwen25_72b": 3},  # consensus
            {"gptoss_120b": 2, "deepseek_r1": 2, "qwen25_72b": 2},  # consensus
            {"gptoss_120b": 1, "deepseek_r1": 3, "qwen25_72b": 1},  # flagged
            {"gptoss_120b": 3, "deepseek_r1": 3, "qwen25_72b": 2},  # consensus (std=0.47)
        ]
        _make_three_judge_jsonls(d, "myrun", score_sets)
        _, flagged_path = aggregate_run("myrun", d)
        assert flagged_path.exists()
        flagged = [json.loads(l) for l in flagged_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(flagged) == 2
        assert all(r["consensus_status"] == "flagged_for_review" for r in flagged)

def test_aggregate_all_null_scores_parse_error_status():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        score_sets = [{"gptoss_120b": None, "deepseek_r1": None, "qwen25_72b": None}] * 5
        _make_three_judge_jsonls(d, "myrun", score_sets)
        consensus_path, _ = aggregate_run("myrun", d)
        records = [json.loads(l) for l in consensus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert all(r["consensus_status"] == "parse_error" for r in records)

def test_aggregate_count_matches_input():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        score_sets = [{"gptoss_120b": 2, "deepseek_r1": 3, "qwen25_72b": 2}] * 5
        _make_three_judge_jsonls(d, "myrun", score_sets)
        consensus_path, _ = aggregate_run("myrun", d)
        lines = [l for l in consensus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 5
