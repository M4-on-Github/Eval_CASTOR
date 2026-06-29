"""
Tests for pipelines/judge_panel/run_judge.py

Tests the pure logic (prompt construction, response parsing, record building)
without requiring a running model. Model calls are replaced with a fake backend.
Run: python -m pytest tests/test_run_judge.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.judge_panel.run_judge import (
    build_user_prompt,
    parse_judge_response,
    build_output_record,
    VALID_SCORES,
)

GT = {"state": "aground", "vessel_type": "cargo ship", "size_estimate": "large", "cargo": "none"}
PRED_TEXT = "The vessel appears to be aground near the shoreline."


# ── build_user_prompt ─────────────────────────────────────────────────────────

def test_build_user_prompt_contains_gt_fields():
    prompt = build_user_prompt(GT, PRED_TEXT)
    assert "aground" in prompt
    assert "cargo ship" in prompt
    assert "large" in prompt
    assert "none" in prompt

def test_build_user_prompt_contains_pred_text():
    prompt = build_user_prompt(GT, PRED_TEXT)
    assert PRED_TEXT in prompt

def test_build_user_prompt_no_unresolved_placeholders():
    import re
    prompt = build_user_prompt(GT, PRED_TEXT)
    assert not re.search(r'\{[a-z_]+\}', prompt)


# ── parse_judge_response ──────────────────────────────────────────────────────

def test_parse_valid_response():
    raw = json.dumps({
        "visual_alignment_rationale": "Matches GT.",
        "hallucinations_detected": [],
        "final_score": 3,
    })
    result = parse_judge_response(raw)
    assert result["score"] == 3
    assert result["parse_ok"] is True
    assert result["hallucinations"] == []
    assert "Matches GT." in result["rationale"]

def test_parse_score_out_of_range_returns_null():
    raw = json.dumps({
        "visual_alignment_rationale": "x",
        "hallucinations_detected": [],
        "final_score": 5,
    })
    result = parse_judge_response(raw)
    assert result["score"] is None
    assert result["parse_ok"] is False

def test_parse_missing_final_score():
    raw = json.dumps({"visual_alignment_rationale": "x", "hallucinations_detected": []})
    result = parse_judge_response(raw)
    assert result["score"] is None
    assert result["parse_ok"] is False

def test_parse_malformed_json():
    result = parse_judge_response("not json at all")
    assert result["score"] is None
    assert result["parse_ok"] is False
    assert "raw_response" in result

def test_parse_with_markdown_fence():
    raw = "```json\n" + json.dumps({
        "visual_alignment_rationale": "ok",
        "hallucinations_detected": ["smoke"],
        "final_score": 2,
    }) + "\n```"
    result = parse_judge_response(raw)
    assert result["score"] == 2
    assert result["hallucinations"] == ["smoke"]

def test_parse_hallucinations_always_list():
    # Model returns null instead of []
    raw = json.dumps({
        "visual_alignment_rationale": "ok",
        "hallucinations_detected": None,
        "final_score": 1,
    })
    result = parse_judge_response(raw)
    assert isinstance(result["hallucinations"], list)

def test_valid_scores_constant():
    assert VALID_SCORES == {1, 2, 3}


# ── build_output_record ───────────────────────────────────────────────────────

def test_build_output_record_has_required_keys():
    parse_result = {"score": 3, "rationale": "ok", "hallucinations": [], "parse_ok": True}
    rec = build_output_record(
        image="img/001.jpg",
        gt_state="aground",
        pred_text=PRED_TEXT,
        verbosity_flagged=False,
        judge_model="qwen25_72b",
        parse_result=parse_result,
        elapsed_s=1.5,
    )
    for key in ("image", "gt_state", "pred_text", "verbosity_flagged",
                "judge_model", "score", "rationale", "hallucinations",
                "parse_ok", "elapsed_s"):
        assert key in rec, f"Missing key: {key}"

def test_build_output_record_image_always_present():
    parse_result = {"score": None, "rationale": "", "hallucinations": [], "parse_ok": False, "raw_response": "bad"}
    rec = build_output_record("img/002.jpg", "capsized", PRED_TEXT, False, "deepseek_r1", parse_result, 0.5)
    assert rec["image"] == "img/002.jpg"
    assert rec["score"] is None
