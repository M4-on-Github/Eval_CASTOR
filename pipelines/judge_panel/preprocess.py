"""
Pre-processing for Pipeline 5 judge inputs.

Strips markdown formatting from VLM outputs, normalizes English numerals,
and flags responses that are disproportionately verbose vs. the ground truth.
"""

import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

_NUMERAL_RE = re.compile(
    r'\b(' + '|'.join(_WORD_TO_NUM) + r')\b',
    re.IGNORECASE,
)

_MARKDOWN_RE = re.compile(r'\*{1,3}')


@dataclass
class PreprocessResult:
    clean_pred: str
    clean_gt: dict
    verbosity_flagged: bool


def _strip_markdown(text: str) -> str:
    return _MARKDOWN_RE.sub('', text).strip()


def _normalize_numerals(text: str) -> str:
    def _replace(m):
        return _WORD_TO_NUM[m.group(0).lower()]
    return _NUMERAL_RE.sub(_replace, text)


def preprocess(pred_text: str, gt_fields: dict) -> PreprocessResult:
    """Clean pred_text and check verbosity against gt_fields.

    Args:
        pred_text:  Raw VLM output string.
        gt_fields:  Dict of GT field values (state, vessel_type, size_estimate, cargo).

    Returns:
        PreprocessResult with clean_pred, clean_gt, and verbosity_flagged.
    """
    clean_pred = _normalize_numerals(_strip_markdown(pred_text))

    gt_narrative = " ".join(str(v) for v in gt_fields.values())
    verbosity_flagged = len(clean_pred) > 2 * len(gt_narrative)

    return PreprocessResult(
        clean_pred=clean_pred,
        clean_gt=dict(gt_fields),
        verbosity_flagged=verbosity_flagged,
    )
