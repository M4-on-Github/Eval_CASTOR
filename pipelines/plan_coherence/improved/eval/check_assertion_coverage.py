"""
check_assertion_coverage.py — Assertion coverage checker for the improved CASTOR pipeline.

Stage 1.5: runs after inference, before the judge. For each plan, asks Selene 8B whether
the plan addresses each domain assertion (STANDARD or CONTROL reference set). Dual-track:
evaluated under both GT state and VLM-predicted state.

Reference sets per state:
  STANDARD  — 8 state-specific assertions (parsed from assertions/standard_v2.txt)
               + state-specific RESOURCES sub-assertions (R_AG_*, R_CA_*, R_SU_*, R_OF_*)
               + state-specific CREW sub-assertions (C_AG_*, C_CA_*, C_SU_*, C_OF_*)
  CONTROL   — 8 state-specific assertions (parsed from assertions/control_v2.txt)
               + one vague resource assertion per state (CR_AG / CR_CA / CR_SU / CR_OF)
               + one universal crew assertion (CC_ALL — same text for all states)

Metrics per (image, condition, reference_set, track) row:
  recall    = n_covered / n_relevant
  precision = n_covered / (n_covered + contam_count)  [None if denom = 0]
  f1        = harmonic mean of recall and precision    [None if either is None]

Contamination: keyword scan for wrong-state-exclusive terms relative to the reference state.

Outputs (in results/):
  coverage_per_image.csv   — one row per (image × condition × ref_set × track)
  coverage_summary.csv     — mean ± SD per (condition × ref_set × track) across images

Resumable: skips (image, condition, reference_set, track) rows already in coverage_per_image.csv.

Usage:
    python improved/eval/check_assertion_coverage.py --config improved/config.yaml
    python improved/eval/check_assertion_coverage.py --config improved/config.yaml \\
        --condition standard_v2 --ref standard --limit 10
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# parents[1] = improved/ dir → "from eval.X" resolves to improved/eval/X.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.extract_condition import extract_condition

# ---------------------------------------------------------------------------
# Canonical state labels
# ---------------------------------------------------------------------------

STATES = ["aground", "capsized", "sunken", "on_fire"]

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a maritime salvage domain expert. "
    "You will be given a salvage plan and a specific concept. "
    "Determine whether the plan explicitly addresses or includes that concept. "
    "Respond with JSON only."
)

USER_TEMPLATE = (
    "SALVAGE PLAN:\n{plan}\n\n"
    "Does this plan explicitly address or include the following concept?\n"
    '"{assertion}"\n\n'
    'Respond with JSON: {{"covered": true}} or {{"covered": false}}'
)

# ---------------------------------------------------------------------------
# Hardcoded STANDARD sub-assertions (RESOURCES + CREW, state-specific)
# ---------------------------------------------------------------------------

STANDARD_RESOURCE_ASSERTIONS: dict[str, list[tuple[str, str]]] = {
    "aground": [
        ("R_AG_1", "The plan specifies harbor tugs or ocean-going salvage tugs for the pulling or towing operation"),
        ("R_AG_2", "The plan specifies beach gear or anchor-wire-chain system for controlled pulling"),
        ("R_AG_3", "The plan uses barges for cargo lightering to reduce vessel displacement"),
    ],
    "capsized": [
        ("R_CA_1", "The plan specifies floating cranes for the righting or lifting operation"),
        ("R_CA_2", "The plan specifies submersible pumps to remove trapped water before righting"),
    ],
    "sunken": [
        ("R_SU_1", "The plan specifies floating cranes or barges for lifting or sectional removal of the wreck"),
        ("R_SU_2", "The plan specifies cofferdams or patching materials for hull repair before refloating"),
        ("R_SU_3", "The plan deploys containment booms and skimmers for surface oil management"),
        ("R_SU_4", "The plan uses side-scan sonar or electro-acoustic sonar for hull survey or submerged oil location"),
    ],
    "on_fire": [
        ("R_OF_1", "The plan specifies fireboats with external monitors for boundary cooling of spaces adjacent to the fire only"),
        ("R_OF_2", "The plan specifies alcohol-resistant (AR) foam concentrate if polar-solvent cargo is present or suspected"),
    ],
}

STANDARD_CREW_ASSERTIONS: dict[str, list[tuple[str, str]]] = {
    "aground": [
        ("C_AG_1", "The plan assigns a salvage engineer or naval architect to calculate ground reaction and plan the refloat"),
        ("C_AG_2", "The plan assigns a hazmat or ordnance specialist to check cargo before any cutting or movement"),
    ],
    "capsized": [
        ("C_CA_1", "The plan assigns rescue swimmers or boat crew for personnel recovery before any righting attempt"),
        ("C_CA_2", "The plan assigns a salvage engineer or naval architect to verify the GZ/GM stability plan before righting"),
        ("C_CA_3", "The plan assigns a dive and survey team for hull condition assessment after righting"),
    ],
    "sunken": [
        ("C_SU_1", "The plan assigns a dive and survey team for gas testing and hull integrity assessment before any recovery"),
        ("C_SU_2", "The plan assigns a hazmat or ordnance specialist to assess unknown cargo before hull disturbance"),
    ],
    "on_fire": [
        ("C_OF_1", "The plan assigns a firefighting team for suppression and boundary cooling operations"),
        ("C_OF_2", "The plan assigns rescue swimmers or boat crew to evacuate all personnel before suppression begins"),
    ],
}

# ---------------------------------------------------------------------------
# Hardcoded CONTROL sub-assertions (vague resource + universal crew)
# ---------------------------------------------------------------------------

CONTROL_RESOURCE_ASSERTIONS: dict[str, list[tuple[str, str]]] = {
    "aground":  [("CR_AG", "The plan mentions tugboats or equipment for moving or towing the vessel")],
    "capsized": [("CR_CA", "The plan mentions cranes or pumps for lifting or removing water")],
    "sunken":   [("CR_SU", "The plan mentions cranes, barges, booms, or patching materials for recovery operations")],
    "on_fire":  [("CR_OF", "The plan mentions fireboats or extinguishing agents for firefighting")],
}

# CC_ALL applies to every state in the CONTROL reference set
CONTROL_CREW_UNIVERSAL: list[tuple[str, str]] = [
    ("CC_ALL", "The plan mentions divers, engineers, firefighters, or rescue personnel as appropriate to the scenario"),
]

# ---------------------------------------------------------------------------
# Contamination keywords (wrong-state-exclusive terminology)
# ---------------------------------------------------------------------------

CONTAM_KEYWORDS: dict[str, list[str]] = {
    "aground":  ["beach gear", "anchor-wire-chain", "ground reaction", "tide gauge",
                 "seabed friction", "lightering", "neutral loading point"],
    "capsized": ["parbuckling", "righting arm", "GZ curve", "metacentric height",
                 "upside-down refloating"],
    "sunken":   ["cofferdam", "side-scan sonar", "electro-acoustic sonar", "NEBA",
                 "gas test sequence", "dispersant"],
    "on_fire":  ["AR foam", "alcohol-resistant foam", "pre-discharge alarm",
                 "CO2 flooding", "backdraft", "fireboat monitor"],
}

# ---------------------------------------------------------------------------
# Canonical assertion ID lists (defines CSV column order)
# ---------------------------------------------------------------------------

_STANDARD_IDS: list[str] = (
    [f"AG_{i}" for i in range(1, 9)]
    + [f"CA_{i}" for i in range(1, 9)]
    + [f"SU_{i}" for i in range(1, 9)]
    + [f"OF_{i}" for i in range(1, 9)]
    + [f"R_AG_{i}" for i in range(1, 4)]
    + [f"R_CA_{i}" for i in range(1, 3)]
    + [f"R_SU_{i}" for i in range(1, 5)]
    + [f"R_OF_{i}" for i in range(1, 3)]
    + [f"C_AG_{i}" for i in range(1, 3)]
    + [f"C_CA_{i}" for i in range(1, 4)]
    + [f"C_SU_{i}" for i in range(1, 3)]
    + [f"C_OF_{i}" for i in range(1, 3)]
)

_CONTROL_IDS: list[str] = (
    [f"AG_ctrl_{i}" for i in range(1, 9)]
    + [f"CA_ctrl_{i}" for i in range(1, 9)]
    + [f"SU_ctrl_{i}" for i in range(1, 9)]
    + [f"OF_ctrl_{i}" for i in range(1, 9)]
    + ["CR_AG", "CR_CA", "CR_SU", "CR_OF", "CC_ALL"]
)

ALL_ASSERTION_IDS = _STANDARD_IDS + _CONTROL_IDS

PER_IMAGE_FIELDS = (
    ["image", "condition", "reference_set", "track",
     "gt_state", "predicted_state",
     "n_relevant", "n_covered", "contam_count", "contam_list",
     "recall", "precision", "f1"]
    + ALL_ASSERTION_IDS
)

SUMMARY_FIELDS = [
    "condition", "reference_set", "track", "n_images",
    "mean_recall", "sd_recall",
    "mean_precision", "sd_precision",
    "mean_f1", "sd_f1",
    "mean_contam_count",
    "recall_aground", "recall_capsized", "recall_sunken", "recall_on_fire",
]

# ---------------------------------------------------------------------------
# Assertion loading
# ---------------------------------------------------------------------------

def _parse_state_blocks(
    txt_path: Path,
    prefix_map: dict[str, str],
) -> dict[str, list[tuple[str, str]]]:
    """
    Parse [STATE] blocks from a standard_v2.txt or control_v2.txt file.
    Stops collecting when [RESOURCES] or [CREW] is encountered.
    Returns {state: [(assertion_id, assertion_text), ...]}
    """
    BLOCK_MAP = {
        "[AGROUND]": "aground", "[CAPSIZED]": "capsized",
        "[SUNKEN]": "sunken", "[ON_FIRE]": "on_fire",
    }
    STOP_BLOCKS = {"[RESOURCES]", "[CREW]"}

    result: dict[str, list[tuple[str, str]]] = {s: [] for s in STATES}
    counters: dict[str, int] = defaultdict(int)
    current: str | None = None

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tag = stripped.upper().replace(" ", "_")
        if tag in BLOCK_MAP:
            current = BLOCK_MAP[tag]
            continue
        if tag in STOP_BLOCKS:
            current = None
            continue
        if current is None:
            continue
        counters[current] += 1
        aid = f"{prefix_map[current]}_{counters[current]}"
        result[current].append((aid, stripped))

    return result


def build_standard_assertions(assertions_dir: Path) -> dict[str, list[tuple[str, str]]]:
    prefix_map = {"aground": "AG", "capsized": "CA", "sunken": "SU", "on_fire": "OF"}
    state = _parse_state_blocks(assertions_dir / "standard_v2.txt", prefix_map)
    return {
        s: state[s] + STANDARD_RESOURCE_ASSERTIONS.get(s, []) + STANDARD_CREW_ASSERTIONS.get(s, [])
        for s in STATES
    }


def build_control_assertions(assertions_dir: Path) -> dict[str, list[tuple[str, str]]]:
    prefix_map = {
        "aground": "AG_ctrl", "capsized": "CA_ctrl",
        "sunken": "SU_ctrl", "on_fire": "OF_ctrl",
    }
    state = _parse_state_blocks(assertions_dir / "control_v2.txt", prefix_map)
    return {
        s: state[s] + CONTROL_RESOURCE_ASSERTIONS.get(s, []) + CONTROL_CREW_UNIVERSAL
        for s in STATES
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state_from_path(path_str: str) -> str:
    """Infer GT state from image path prefix (e.g. 'aground/img.jpg' → 'aground')."""
    for part in Path(path_str).parts:
        pl = part.lower()
        if pl in STATES:
            return pl
        if pl == "on fire":
            return "on_fire"
    return "unknown"


def load_gt(gt_csv: str) -> dict[str, str]:
    df = pd.read_csv(gt_csv)
    return {
        str(row.get("image", "")).strip(): str(row.get("state", "unknown")).strip().lower()
        for _, row in df.iterrows()
        if str(row.get("image", "")).strip()
    }


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def scan_contamination(plan_text: str, reference_state: str) -> tuple[int, list[str]]:
    """Keyword scan for wrong-state terminology relative to the reference state."""
    plan_lower = plan_text.lower()
    found: list[str] = []
    for state, keywords in CONTAM_KEYWORDS.items():
        if state == reference_state:
            continue
        for kw in keywords:
            if kw.lower() in plan_lower and kw not in found:
                found.append(kw)
    return len(found), sorted(found)


def _safe_metrics(
    n_covered: int, n_relevant: int, contam_count: int
) -> tuple[float | None, float | None, float | None]:
    recall = n_covered / n_relevant if n_relevant > 0 else None
    denom_p = n_covered + contam_count
    precision = (n_covered / denom_p) if denom_p > 0 else None
    if recall is not None and precision is not None:
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        f1 = None
    return recall, precision, f1


def load_done_keys(per_image_path: Path) -> set[tuple[str, str, str, str]]:
    if not per_image_path.exists():
        return set()
    done: set[tuple[str, str, str, str]] = set()
    with open(per_image_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["image"], row["condition"], row["reference_set"], row["track"]))
    return done

# ---------------------------------------------------------------------------
# Core evaluation (one condition per call)
# ---------------------------------------------------------------------------

def evaluate_condition(
    condition: str,
    records: list[dict],
    gt_map: dict[str, str],
    standard_assertions: dict[str, list[tuple[str, str]]],
    control_assertions: dict[str, list[tuple[str, str]]],
    llm: LLM,
    sampling: SamplingParams,
    done_keys: set[tuple[str, str, str, str]],
    active_refs: list[str],
    limit: int | None,
) -> list[dict]:
    """
    Evaluate one condition's records. Returns per-image CSV row dicts.
    Sends one llm.chat() call for all prompts in this condition, ordered by
    image for prefix-cache efficiency.
    """
    prompts: list[list[dict]] = []
    prompt_meta: list[tuple[tuple, str]] = []  # (unit_key, assertion_id)
    unit_data: dict[tuple, dict] = {}
    no_llm_keys: list[tuple] = []

    img_count = 0

    for rec in records:
        if limit is not None and img_count >= limit:
            break

        image = rec.get("image", "")
        plan_text = rec.get("text", "")
        qid = rec.get("question_id", image)

        gt_state = (
            gt_map.get(qid)
            or gt_map.get(image)
            or _state_from_path(qid or image)
        )
        predicted_state = extract_condition(plan_text)

        for track, eval_state in [("gt", gt_state), ("predicted", predicted_state)]:
            for ref_set in active_refs:
                key = (image, condition, ref_set, track)
                if key in done_keys:
                    continue

                unit_data[key] = {
                    "image": image, "condition": condition,
                    "reference_set": ref_set, "track": track,
                    "gt_state": gt_state, "predicted_state": predicted_state,
                    "eval_state": eval_state, "plan_text": plan_text,
                }

                assertions_dict = (
                    standard_assertions if ref_set == "standard" else control_assertions
                )
                applicable = assertions_dict.get(eval_state, []) if eval_state in STATES else []

                if not applicable:
                    no_llm_keys.append(key)
                    continue

                for (aid, atext) in applicable:
                    prompts.append([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TEMPLATE.format(
                            plan=plan_text, assertion=atext,
                        )},
                    ])
                    prompt_meta.append((key, aid))

        img_count += 1

    print(f"  [{condition}] {len(prompts)} prompts across {img_count} images "
          f"({len(no_llm_keys)} no-LLM rows for unknown state)")

    # Single batched LLM call for this condition
    covered: dict[tuple, dict[str, bool]] = defaultdict(dict)
    if prompts:
        outputs = llm.chat(prompts, sampling_params=sampling)
        for (key, aid), out in zip(prompt_meta, outputs):
            raw = out.outputs[0].text.strip()
            try:
                covered[key][aid] = bool(json.loads(raw).get("covered", False))
            except Exception:
                covered[key][aid] = False  # parse error → treat as not covered

    # Build CSV rows
    rows: list[dict] = []

    for key in unit_data:
        d = unit_data[key]
        ref_set = d["reference_set"]
        eval_state = d["eval_state"]
        plan_text = d["plan_text"]

        assertions_dict = (
            standard_assertions if ref_set == "standard" else control_assertions
        )
        applicable = assertions_dict.get(eval_state, []) if eval_state in STATES else []
        applicable_ids = [aid for (aid, _) in applicable]

        row: dict = {f: "" for f in PER_IMAGE_FIELDS}
        row.update({
            "image": d["image"], "condition": condition,
            "reference_set": ref_set, "track": d["track"],
            "gt_state": d["gt_state"], "predicted_state": d["predicted_state"],
        })

        if key in no_llm_keys:
            # Unknown predicted state → write NaN row (no LLM calls)
            row["n_relevant"] = 0
            rows.append(row)
            continue

        covered_this = covered.get(key, {})
        n_covered = sum(1 for aid in applicable_ids if covered_this.get(aid, False))
        n_relevant = len(applicable_ids)

        # Contamination: scan against eval_state as reference
        contam_count, contam_list = scan_contamination(plan_text, eval_state)
        recall, precision, f1 = _safe_metrics(n_covered, n_relevant, contam_count)

        row.update({
            "n_relevant": n_relevant,
            "n_covered": n_covered,
            "contam_count": contam_count,
            "contam_list": "|".join(contam_list),
            "recall": f"{recall:.4f}" if recall is not None else "",
            "precision": f"{precision:.4f}" if precision is not None else "",
            "f1": f"{f1:.4f}" if f1 is not None else "",
        })

        # Per-assertion columns: 1/0 for applicable, "" for inapplicable
        for aid in applicable_ids:
            row[aid] = "1" if covered_this.get(aid, False) else "0"

        rows.append(row)

    return rows

# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def append_per_image_rows(out_path: Path, rows: list[dict]):
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def compute_and_write_summary(per_image_path: Path, summary_path: Path):
    df = pd.read_csv(per_image_path)

    def _mean(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    def _sd(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    def _state_mean_recall(grp: pd.DataFrame, state: str) -> float:
        sub = grp[grp["gt_state"] == state]["recall"]
        return _mean(sub)

    rows: list[dict] = []
    for (condition, ref_set, track), grp in df.groupby(
        ["condition", "reference_set", "track"]
    ):
        rows.append({
            "condition": condition,
            "reference_set": ref_set,
            "track": track,
            "n_images": len(grp),
            "mean_recall":       round(_mean(grp["recall"]), 4),
            "sd_recall":         round(_sd(grp["recall"]), 4),
            "mean_precision":    round(_mean(grp["precision"]), 4),
            "sd_precision":      round(_sd(grp["precision"]), 4),
            "mean_f1":           round(_mean(grp["f1"]), 4),
            "sd_f1":             round(_sd(grp["f1"]), 4),
            "mean_contam_count": round(_mean(grp["contam_count"]), 4),
            "recall_aground":    round(_state_mean_recall(grp, "aground"), 4),
            "recall_capsized":   round(_state_mean_recall(grp, "capsized"), 4),
            "recall_sunken":     round(_state_mean_recall(grp, "sunken"), 4),
            "recall_on_fire":    round(_state_mean_recall(grp, "on_fire"), 4),
        })

    summary_df = pd.DataFrame(rows, columns=SUMMARY_FIELDS)
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary -> {summary_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="P7-equivalent assertion coverage for the improved pipeline."
    )
    parser.add_argument("--config", default="improved/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--condition", choices=["standard_v2", "control_v2", "ablation_v2"],
                        default=None,
                        help="Run only this condition (default: all three)")
    parser.add_argument("--ref", choices=["standard", "control"],
                        default=None,
                        help="Evaluate against only this reference set (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N images per condition for smoke testing")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pipeline_dir    = Path(os.path.expandvars(cfg["paths"]["pipeline_dir"]))
    results_dir     = pipeline_dir / "results"
    assertions_dir  = pipeline_dir / "assertions"
    gt_csv          = os.path.expandvars(cfg["paths"]["gt_csv"])
    user_models_dir = Path(os.path.expandvars(cfg["paths"]["user_models_dir"]))
    selene_path     = user_models_dir / cfg["models"]["selene_dir"]

    results_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = results_dir / "coverage_per_image.csv"
    summary_path   = results_dir / "coverage_summary.csv"

    # Load assertions
    print("Loading assertions...")
    standard_assertions = build_standard_assertions(assertions_dir)
    control_assertions  = build_control_assertions(assertions_dir)
    for state in STATES:
        n_std = len(standard_assertions[state])
        n_ctl = len(control_assertions[state])
        print(f"  {state}: {n_std} standard / {n_ctl} control assertions")

    # Load GT
    gt_map: dict[str, str] = {}
    if Path(gt_csv).exists():
        gt_map = load_gt(gt_csv)
        print(f"GT map loaded: {len(gt_map)} entries from {gt_csv}")
    else:
        print(f"[WARN] GT CSV not found: {gt_csv} — will infer state from image path prefix",
              file=sys.stderr)

    # Determine scope
    conditions = (
        [args.condition] if args.condition
        else ["standard_v2", "control_v2", "ablation_v2"]
    )
    active_refs = [args.ref] if args.ref else ["standard", "control"]

    # Resumability
    done_keys = load_done_keys(per_image_path)
    print(f"Resuming: {len(done_keys)} (image, condition, ref_set, track) rows already done")

    # Validate model path
    if not selene_path.exists():
        print(f"[ERROR] Selene model not found: {selene_path}", file=sys.stderr)
        sys.exit(1)

    # Load vLLM (Selene 8B AWQ)
    print(f"\nLoading Selene from {selene_path} ...")
    llm = LLM(
        model=str(selene_path),
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        enable_prefix_caching=True,
        enforce_eager=True,   # skip CUDA graph capture — lower peak CPU RAM
        tensor_parallel_size=1,
        dtype="float16",
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        guided_decoding=GuidedDecodingParams(
            json={
                "type": "object",
                "properties": {"covered": {"type": "boolean"}},
                "required": ["covered"],
            }
        ),
    )

    # Evaluate one condition at a time (each → one llm.chat() call)
    for condition in conditions:
        jsonl_path = (
            results_dir / f"answers_qwen3vl8b_baseline_{condition}_improved.jsonl"
        )
        if not jsonl_path.exists():
            print(f"\n[SKIP] {jsonl_path.name} not found — run inference first")
            continue

        records = load_jsonl(jsonl_path)
        print(f"\n[{condition}] {len(records)} records loaded")

        rows = evaluate_condition(
            condition=condition,
            records=records,
            gt_map=gt_map,
            standard_assertions=standard_assertions,
            control_assertions=control_assertions,
            llm=llm,
            sampling=sampling,
            done_keys=done_keys,
            active_refs=active_refs,
            limit=args.limit,
        )

        if rows:
            append_per_image_rows(per_image_path, rows)
            for r in rows:
                done_keys.add((
                    r["image"], r["condition"], r["reference_set"], r["track"]
                ))
            print(f"  Wrote {len(rows)} rows -> {per_image_path}")
        else:
            print(f"  [SKIP] all rows already done for {condition}")

    # Recompute summary from full CSV
    if per_image_path.exists():
        compute_and_write_summary(per_image_path, summary_path)

    print("\nDone.")
    print(f"  {per_image_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
