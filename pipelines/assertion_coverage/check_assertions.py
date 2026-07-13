"""
Pipeline 7 — Assertion Coverage Analysis.

For each VLM salvage plan, determines which domain-specific assertions from
IMPROVED_assertion_registry.csv are covered (via Selene 8B LLM, one call per
assertion per image) and which wrong-casualty assertions appear (keyword scan).

Coverage  : LLM binary judgment per assertion — robust to paraphrase.
Contamination: keyword regex on wrong-type assertions — reliable for high-disc
               technical terms that wouldn't appear in a correct plan by chance.

One vLLM call per (image, assertion) pair: ~3000 calls total, ~5-10 min on
Selene 8B AWQ with vLLM prefix-cache reuse across same-image calls.

Usage (inside Apptainer container via assertion_coverage_job.sh):
  python3 check_assertions.py \\
      --input   /data/$USER/castor_results/answers_baseline.jsonl \\
      --model-dir /data/$USER/selene-1-mini-llama-3.1-8b-awq \\
      --out     /data/$USER/castor_results/p7_assertion_coverage/ \\
      [--limit N]
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_ground_truth, load_run

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REGISTRY_PATH = EVAL_ROOT / "all_prompts" / "IMPROVED_assertion_registry.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# casualty_type values that apply to ALL images regardless of gt_state
UNIVERSAL_TYPES = {"resources", "cross-cutting"}

# Discriminativeness weights for weighted coverage score
DISC_WEIGHT = {"high": 3, "medium": 2, "low": 1}

# vLLM model config for Selene 8B AWQ
_SELENE_MAX_MODEL_LEN = 4096
_SELENE_MAX_TOKENS    = 16    # only needs to output {"covered": true/false}

_COVERAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "boolean"},
    },
    "required": ["covered"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You are a maritime salvage domain expert evaluating an AI-generated salvage plan. "
    "Determine whether the plan explicitly addresses a specific technical concept. "
    "Answer only in the JSON format specified."
)


def _user_prompt(plan_text: str, assertion_text: str) -> str:
    return (
        f"SALVAGE PLAN:\n{plan_text.strip()}\n\n"
        f"Does this plan explicitly address or include the following concept?\n"
        f'"{assertion_text}"\n\n'
        f'Respond with JSON: {{"covered": true}} or {{"covered": false}}'
    )


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_registry(path: Path = REGISTRY_PATH) -> list[dict]:
    """Load assertion registry CSV. Returns list of assertion dicts."""
    assertions = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["keywords"] = [
                kw.strip().lower()
                for kw in row["checkable_keyword"].split("/")
                if kw.strip()
            ]
            assertions.append(row)
    return assertions


def relevant_assertions(assertions: list[dict], gt_state: str) -> list[dict]:
    """Assertions to check for an image: state-specific + universal types."""
    return [
        a for a in assertions
        if a["casualty_type"] == gt_state or a["casualty_type"] in UNIVERSAL_TYPES
    ]


def contamination_assertions(assertions: list[dict], gt_state: str) -> list[dict]:
    """Wrong-casualty assertions to scan for contamination (keyword only)."""
    wrong_states = {"aground", "capsized", "sunken", "on_fire"} - {gt_state}
    return [a for a in assertions if a["casualty_type"] in wrong_states]


# ---------------------------------------------------------------------------
# Keyword contamination scan (no LLM — high-disc terms are domain-unique)
# ---------------------------------------------------------------------------

def keyword_hit(text: str, keywords: list[str]) -> bool:
    """True if any keyword appears as a word/phrase in text (case-insensitive)."""
    tl = text.lower()
    for kw in keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, tl):
            return True
    return False


def scan_contamination(plan_text: str, wrong_assertions: list[dict]) -> tuple[list[str], int]:
    """Return (list of contaminating assertion IDs, count)."""
    hits = [a["id"] for a in wrong_assertions if keyword_hit(plan_text, a["keywords"])]
    return hits, len(hits)


# ---------------------------------------------------------------------------
# vLLM batch inference
# ---------------------------------------------------------------------------

def run_vllm_batch(prompts: list[str], model_dir: str) -> list[bool | None]:
    """
    Run all (image, assertion) prompts through Selene 8B in one batched pass.
    Returns list of booleans (or None on parse failure) aligned with prompts.
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    print(f"  [vLLM] Loading Selene from {model_dir} ...")
    llm = LLM(
        model=model_dir,
        dtype="auto",
        max_model_len=_SELENE_MAX_MODEL_LEN,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,   # crucial: reuses KV across same-image calls
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=_SELENE_MAX_TOKENS,
        guided_decoding=GuidedDecodingParams(json=_COVERAGE_SCHEMA),
    )

    conversations = [
        [{"role": "system", "content": _SYSTEM_PROMPT},
         {"role": "user",   "content": p}]
        for p in prompts
    ]

    print(f"  [vLLM] Scoring {len(conversations)} (image, assertion) pairs ...")
    t0 = time.perf_counter()
    outputs = llm.chat(conversations, sampling_params=params)
    elapsed = time.perf_counter() - t0
    print(f"  [vLLM] Done in {elapsed:.1f}s ({elapsed/len(outputs):.2f}s/call avg)")

    results = []
    for o in outputs:
        raw = o.outputs[0].text.strip()
        try:
            parsed = json.loads(raw)
            results.append(bool(parsed.get("covered")))
        except (json.JSONDecodeError, AttributeError):
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(input_path: Path, gt_path: Path, out_dir: Path,
        model_dir: str, limit: int | None):

    run_name = input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_path    = out_dir / f"{run_name}_per_image.csv"
    per_assert_path   = out_dir / f"{run_name}_per_assertion.csv"
    summary_path      = out_dir / f"{run_name}_summary.csv"

    assertions = load_registry()
    all_ids    = [a["id"] for a in assertions]

    gt      = load_ground_truth(gt_path)
    records = load_run(input_path)
    if limit:
        records = records[:limit]

    # ── Resume: skip already-processed images ─────────────────────────────────
    done_images: set[str] = set()
    existing_rows: list[dict] = []
    if per_image_path.exists():
        with open(per_image_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_images.add(row["image"])
                existing_rows.append(row)
        if done_images:
            print(f"  Resume: {len(done_images)} images already processed.")

    pending = [r for r in records if r.get("image", "") not in done_images]
    if not pending:
        print("  All records already processed — nothing to do.")
        return

    print(f"  Processing {len(pending)} images with {len(assertions)} assertions each ...")

    # ── Build flat prompt list for vLLM batching ──────────────────────────────
    # Order: for each image, all relevant assertions in registry order.
    # vLLM prefix cache kicks in because consecutive prompts share the same
    # plan-text prefix (same image = same system + plan text up to assertion).
    batch_meta: list[tuple[str, str]] = []   # (image, assertion_id)
    batch_prompts: list[str]          = []

    image_to_record: dict[str, dict] = {}
    image_to_gt: dict[str, dict]     = {}
    image_to_rel_ids: dict[str, list[str]] = {}
    image_to_contam: dict[str, tuple[list[str], int]] = {}

    for rec in pending:
        image     = rec.get("image", "")
        plan_text = rec.get("text", "")
        gt_fields = gt.get(image, {})
        gt_state  = gt_fields.get("state", "")

        image_to_record[image] = rec
        image_to_gt[image]     = gt_fields

        rel  = relevant_assertions(assertions, gt_state)
        wrong = contamination_assertions(assertions, gt_state)

        image_to_rel_ids[image]  = [a["id"] for a in rel]
        image_to_contam[image]   = scan_contamination(plan_text, wrong)

        for a in rel:
            batch_meta.append((image, a["id"]))
            batch_prompts.append(_user_prompt(plan_text, a["assertion_text"]))

    # ── Single vLLM pass ──────────────────────────────────────────────────────
    llm_results = run_vllm_batch(batch_prompts, model_dir)

    # Map results back: {image -> {assertion_id -> bool|None}}
    image_coverage: dict[str, dict[str, bool | None]] = {
        rec.get("image", ""): {} for rec in pending
    }
    parse_errors = 0
    for (image, aid), result in zip(batch_meta, llm_results):
        image_coverage[image][aid] = result
        if result is None:
            parse_errors += 1

    if parse_errors:
        print(f"  WARNING: {parse_errors} LLM parse failures (None recorded)")

    # ── Build per-image rows ───────────────────────────────────────────────────
    new_rows: list[dict] = []
    for rec in pending:
        image    = rec.get("image", "")
        gt_state = image_to_gt[image].get("state", "")
        coverage = image_coverage[image]
        rel_ids  = image_to_rel_ids[image]
        contam_list, contam_count = image_to_contam[image]

        # Coverage stats
        valid_scores = [v for v in coverage.values() if v is not None]
        n_covered    = sum(1 for v in valid_scores if v)
        n_relevant   = len(rel_ids)
        coverage_pct = round(n_covered / n_relevant, 3) if n_relevant else None

        # High-discriminative subset
        rel_assertions = [a for a in assertions if a["id"] in rel_ids]
        high_ids  = [a["id"] for a in rel_assertions if a["discriminative"] == "high"]
        high_cov  = sum(1 for aid in high_ids if coverage.get(aid))
        high_pct  = round(high_cov / len(high_ids), 3) if high_ids else None

        # Weighted score
        weights_total   = sum(DISC_WEIGHT.get(a["discriminative"], 1) for a in rel_assertions)
        weights_covered = sum(
            DISC_WEIGHT.get(a["discriminative"], 1)
            for a in rel_assertions if coverage.get(a["id"])
        )
        weighted_score = round(weights_covered / weights_total, 3) if weights_total else None

        row: dict = {
            "image":          image,
            "gt_state":       gt_state,
            "coverage_pct":   coverage_pct,
            "high_disc_pct":  high_pct,
            "weighted_score": weighted_score,
            "n_covered":      n_covered,
            "n_relevant":     n_relevant,
            "contam_count":   contam_count,
            "contam_list":    "|".join(contam_list),
        }
        # One boolean column per assertion (None → "" for CSV)
        for aid in all_ids:
            v = coverage.get(aid)
            if aid not in rel_ids:
                row[aid] = ""       # not applicable
            elif v is None:
                row[aid] = "error"
            else:
                row[aid] = "1" if v else "0"

        new_rows.append(row)

    # ── Write per-image CSV (append to existing) ──────────────────────────────
    all_rows = existing_rows + new_rows
    fieldnames = (
        ["image", "gt_state", "coverage_pct", "high_disc_pct", "weighted_score",
         "n_covered", "n_relevant", "contam_count", "contam_list"]
        + all_ids
    )
    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Per-image  → {per_image_path}")

    # ── Per-assertion summary ─────────────────────────────────────────────────
    assertion_rows = []
    for a in assertions:
        aid = a["id"]
        relevant_image_rows = [r for r in all_rows if r.get(aid) != ""]
        n_rel  = len(relevant_image_rows)
        n_cov  = sum(1 for r in relevant_image_rows if r.get(aid) == "1")
        n_err  = sum(1 for r in relevant_image_rows if r.get(aid) == "error")
        cov_pct = round(n_cov / n_rel, 3) if n_rel else None
        assertion_rows.append({
            "id":              aid,
            "casualty_type":   a["casualty_type"],
            "discriminative":  a["discriminative"],
            "assertion_text":  a["assertion_text"],
            "checkable_keyword": a["checkable_keyword"],
            "n_relevant_images": n_rel,
            "n_covered":       n_cov,
            "n_parse_error":   n_err,
            "coverage_pct":    cov_pct,
        })
    with open(per_assert_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(assertion_rows[0].keys()))
        writer.writeheader()
        writer.writerows(assertion_rows)
    print(f"  Per-assertion → {per_assert_path}")

    # ── Run-level summary ─────────────────────────────────────────────────────
    def _state_mean(rows, field, state):
        vals = [float(r[field]) for r in rows
                if r.get("gt_state") == state and r.get(field) not in ("", None, "error")]
        return round(sum(vals) / len(vals), 3) if vals else None

    valid_cov = [float(r["coverage_pct"]) for r in all_rows if r.get("coverage_pct") not in ("", None)]
    valid_hd  = [float(r["high_disc_pct"]) for r in all_rows if r.get("high_disc_pct") not in ("", None)]
    valid_ws  = [float(r["weighted_score"]) for r in all_rows if r.get("weighted_score") not in ("", None)]
    contam_counts = [int(r["contam_count"]) for r in all_rows if r.get("contam_count") != ""]

    summary = {
        "run":                    run_name,
        "n_images":               len(all_rows),
        "overall_coverage_pct":   round(sum(valid_cov) / len(valid_cov), 3) if valid_cov else None,
        "high_disc_coverage_pct": round(sum(valid_hd)  / len(valid_hd),  3) if valid_hd  else None,
        "weighted_score":         round(sum(valid_ws)  / len(valid_ws),  3) if valid_ws  else None,
        "mean_contam_count":      round(sum(contam_counts) / len(contam_counts), 3) if contam_counts else None,
        "coverage_aground":       _state_mean(all_rows, "coverage_pct", "aground"),
        "coverage_capsized":      _state_mean(all_rows, "coverage_pct", "capsized"),
        "coverage_on_fire":       _state_mean(all_rows, "coverage_pct", "on_fire"),
        "coverage_sunken":        _state_mean(all_rows, "coverage_pct", "sunken"),
        "high_disc_aground":      _state_mean(all_rows, "high_disc_pct", "aground"),
        "high_disc_capsized":     _state_mean(all_rows, "high_disc_pct", "capsized"),
        "high_disc_on_fire":      _state_mean(all_rows, "high_disc_pct", "on_fire"),
        "high_disc_sunken":       _state_mean(all_rows, "high_disc_pct", "sunken"),
    }

    # Append to shared summary CSV (one row per run)
    summary_csv = out_dir.parent / "eval_summary_assertion.csv"
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"  Summary    → {summary_path}")

    print(f"\n  overall_coverage={summary['overall_coverage_pct']}  "
          f"high_disc={summary['high_disc_coverage_pct']}  "
          f"weighted={summary['weighted_score']}  "
          f"mean_contam={summary['mean_contam_count']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline 7: assertion coverage analysis over CASTOR salvage plans."
    )
    ap.add_argument("--input",     required=True, type=Path,
                    help="Inference JSONL (text field = salvage plan)")
    ap.add_argument("--model-dir", required=True,
                    help="Path to Selene-1-Mini-8B-AWQ weights")
    ap.add_argument("--out",       required=True, type=Path,
                    help="Output directory (per-image and per-assertion CSVs written here)")
    ap.add_argument("--gt",        type=Path,
                    default=EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv",
                    help="Ground truth CSV (default: human_gt.csv)")
    ap.add_argument("--limit",     type=int, default=None,
                    help="Process only first N images (smoke test)")
    args = ap.parse_args()

    run(
        input_path=args.input,
        gt_path=args.gt,
        out_dir=args.out / args.input.stem,
        model_dir=args.model_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
