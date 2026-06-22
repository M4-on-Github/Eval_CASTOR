"""
CASTOR LLM Inference Evaluator
Evaluates 4 JSONL runs against human ground-truth labels.
Outputs: per-entry CSV, per-state text report, holistic summary CSV.

Run from anywhere:
    python DeGF/Eval_CASTOR/eval_castor.py
"""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

# ---------------------------------------------------------------------------
# Paths  (all relative to this file so the script is portable)
# ---------------------------------------------------------------------------
HERE        = Path(__file__).parent                          # DeGF/Eval_CASTOR/
GT_PATH     = HERE / "human_ground_truth_label" / "human_gt.csv"
RESULTS_DIR = HERE.parent.parent / "results" / "castor_results"
OUT_DIR     = HERE / "results"

VALID_STATES = ["aground", "capsized", "on_fire", "sunken"]


def discover_runs(results_dir: Path) -> list:
    """Scan results_dir for all *.jsonl files and infer metadata.

    diffusion  : True if 'degf' appears in the filename stem.
    prompt_style: 'cot' if the first record's text contains Step 1/Step 2
                  markers; 'direct' otherwise.
    """
    runs = []
    for path in sorted(results_dir.glob("*.jsonl")):
        fname     = path.name
        diffusion = "degf" in path.stem.lower()

        prompt_style = "direct"
        try:
            with open(path, encoding="utf-8") as f:
                first_rec = json.loads(f.readline())
            text = first_rec.get("text", "")
            if re.search(r'Step\s*[12]\s*[^\n]*\n', text, re.IGNORECASE):
                prompt_style = "cot"
        except Exception:
            pass

        runs.append((fname, diffusion, prompt_style))
        print(f"  Discovered: {fname}  diffusion={diffusion}  prompt_style={prompt_style}")

    return runs

STATE_MAP = {
    "aground":   "aground",
    "grounded":  "aground",
    "beached":   "aground",
    "sunken":    "sunken",
    "sinking":   "sunken",
    "capsized":  "capsized",
    "on_fire":   "on_fire",
    "on fire":   "on_fire",
    "good":      "good",
    "floating":  "good",
    "undamaged": "good",
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: Path) -> dict:
    """Returns dict {image_path -> GT field dict}."""
    df = pd.read_csv(csv_path, header=0)
    df = df.rename(columns={c: "_unused" for c in df.columns if str(c).startswith("Unnamed")})

    gt = {}
    for _, row in df.iterrows():
        img = str(row["image"]).strip()
        gt[img] = {
            "state":        str(row["state"]).strip(),
            "vessel_type":  _safe_str(row.get("vessel_type")),
            "cargo":        _safe_str(row.get("cargo")),
            "q1": _safe_str(row.get("q1")).lower(),
            "q2": _safe_str(row.get("q2")).lower(),
            "q3": _safe_str(row.get("q3")).lower(),
            "q4": _safe_str(row.get("q4")).lower(),
            "q5": _safe_str(row.get("q5")).lower(),
            "size_estimate": _safe_str(row.get("size_estimate")),
        }
    return gt


def load_run(jsonl_path: Path) -> list:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Retry with strict=False to tolerate raw control characters
                try:
                    records.append(json.loads(line, strict=False))
                except json.JSONDecodeError as e:
                    print(f"  WARNING: could not parse line {lineno} in {jsonl_path.name}: {e}")
    return records


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

_UNESCAPE = re.compile(r'\\([_\-/])')

def _unescape(text: str) -> str:
    return _UNESCAPE.sub(r'\1', text)


def extract_json_block(text: str):
    """Return (parsed_dict | None, failure_reason_str).

    Scans brace positions in reverse. Prefers the outermost block that
    contains a 'state' key (skipping nested helpers like confidence_scores).
    Falls back to any valid dict if none have 'state'.
    On total failure, failure_reason_str describes why.
    """
    text = _unescape(text)
    positions = [m.start() for m in re.finditer(r'\{', text)]

    if not positions:
        return None, "no_braces: output text contains no '{' character"

    errors = []       # per-position error notes
    fallback = None
    fallback_pos = None

    for pos in reversed(positions):
        chunk = text[pos:]
        depth, end = 0, -1
        for i, ch in enumerate(chunk):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            errors.append(f"pos={pos}: unmatched braces (no closing '}}')")
            continue

        try:
            parsed = json.loads(chunk[: end + 1])
            if not isinstance(parsed, dict):
                errors.append(f"pos={pos}: parsed but top level is {type(parsed).__name__}, not dict")
                continue
            if 'state' in parsed:
                return parsed, ""           # success
            # valid dict but no 'state' — keep as fallback, note keys seen
            if fallback is None:
                fallback = parsed
                fallback_pos = pos
            errors.append(f"pos={pos}: valid JSON but missing 'state' key (keys={list(parsed.keys())})")
        except json.JSONDecodeError as e:
            # Capture up to 80 chars around the error site for context
            snip_start = max(0, e.pos - 40)
            snip = chunk[snip_start: e.pos + 40].replace('\n', ' ')
            errors.append(f"pos={pos}: JSONDecodeError at char {e.pos} — {e.msg!r} | near: ...{snip!r}...")

    if fallback is not None:
        return fallback, (
            f"no_state_key: best JSON (pos={fallback_pos}) lacks 'state' "
            f"(keys={list(fallback.keys())})"
        )

    summary = "; ".join(errors[:5])   # cap to avoid huge strings
    if len(errors) > 5:
        summary += f" ... (+{len(errors)-5} more)"
    return None, f"all_attempts_failed [{len(errors)} blocks tried]: {summary}"


def normalize_state(raw) -> str:
    """Map raw model state to canonical value or UNPARSEABLE."""
    if raw is None:
        return "UNPARSEABLE"
    clean = re.sub(r'\\+', '', str(raw)).strip().lower().replace('_', ' ')
    if clean in STATE_MAP:
        return STATE_MAP[clean]
    if '|' not in clean and '/' not in clean:
        for key, val in STATE_MAP.items():
            if key in clean:
                return val
    return "UNPARSEABLE"


def extract_q_answers(text: str, prompt_style: str) -> dict:
    """Extract Q1-Q5 answers from chain-of-thought text.
    Returns {q1..q5: 'yes'/'no'/None}. None = N/A (direct files or not found).
    """
    result = {f"q{i}": None for i in range(1, 6)}
    if prompt_style != "cot":
        return result

    # Isolate Step 2 section to avoid the Step 3 consistency-check echo
    m = re.search(
        r'Step\s*2\s*[^\n]*\n.*?(?=Step\s*3\s*[^\n]*\n|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    section = m.group(0) if m else text[:int(len(text) * 0.6)]

    # Pattern A - explicit Q label: "Q1 = yes", "Q1: ... yes", "* Q1 = no"
    for qnum, answer in re.findall(r'Q(\d)\s*[=:][^\n]*(yes|no)', section, re.IGNORECASE):
        key = f"q{qnum}"
        if key in result and result[key] is None:
            result[key] = answer.lower()

    # Pattern B - numbered list without Q label: "1. Yes" / "1. No"
    if any(v is None for v in result.values()):
        for qnum, answer in re.findall(r'^\s*(\d)\.\s*(yes|no)\b', section, re.IGNORECASE | re.MULTILINE):
            key = f"q{qnum}"
            if key in result and result[key] is None:
                result[key] = answer.lower()

    # Pattern C - Q anywhere in line (fallback): "Q1 ... No."
    if any(v is None for v in result.values()):
        for qnum, answer in re.findall(r'Q(\d)[^\n]{0,120}?(yes|no)', section, re.IGNORECASE):
            key = f"q{qnum}"
            if key in result and result[key] is None:
                result[key] = answer.lower()

    return result


def normalize_size(text: str) -> str:
    """Bucket free-text size estimate to small / medium / large / unknown."""
    t = text.lower() if text else ""
    if re.search(r'\bsmall\b', t):
        return "small"
    if re.search(r'\bmedium\b', t):
        return "medium"
    if re.search(r'\blarge\b', t):
        return "large"
    return "unknown"


def vessel_jaccard(gt_text: str, pred_text: str) -> float:
    """Jaccard similarity on lower-case alpha token sets."""
    def tok(s):
        return set(re.findall(r'[a-z]+', s.lower())) if s else set()
    a, b = tok(gt_text), tok(pred_text)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cargo_match(gt_cargo: str, pred_cargo) -> str:
    null_vals = {"", "nan", "none", "null"}
    gt_has   = gt_cargo.lower() not in null_vals
    pc       = str(pred_cargo).lower() if pred_cargo is not None else ""
    pred_has = pc not in null_vals
    if gt_has and pred_has:
        return "both_present"
    if not gt_has and not pred_has:
        return "both_absent"
    return "mismatch"

# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_run(records: list, gt_dict: dict, prompt_style: str) -> pd.DataFrame:
    rows = []
    for rec in records:
        img  = rec["image"]
        text = rec["text"]
        gt   = gt_dict.get(img, {})

        parsed, parse_fail_reason = extract_json_block(text)
        parse_error = parsed is None

        pred_state     = normalize_state(parsed.get("state") if parsed else None)
        pred_vessel    = _safe_str(parsed.get("vessel_type") if parsed else None)
        pred_size_raw  = _safe_str(parsed.get("size_estimate") if parsed else None)
        pred_cargo_raw = parsed.get("cargo") if parsed else None

        pred_qs = extract_q_answers(text, prompt_style)

        gt_state         = gt.get("state", "")
        gt_size_bucket   = normalize_size(gt.get("size_estimate", ""))
        pred_size_bucket = normalize_size(pred_size_raw)

        state_correct = (pred_state == gt_state) if pred_state != "UNPARSEABLE" else False

        q_correct = {}
        for i in range(1, 6):
            key  = f"q{i}"
            gt_q = gt.get(key, "")
            pq   = pred_qs[key]
            q_correct[key] = None if pq is None else (pq == gt_q)

        jac    = vessel_jaccard(gt.get("vessel_type", ""), pred_vessel)
        cmatch = cargo_match(gt.get("cargo", ""), pred_cargo_raw)

        rows.append({
            "image":            img,
            "gt_state":         gt_state,
            "pred_state":       pred_state,
            "state_correct":    state_correct,
            "gt_q1":            gt.get("q1"), "gt_q2": gt.get("q2"), "gt_q3": gt.get("q3"),
            "gt_q4":            gt.get("q4"), "gt_q5": gt.get("q5"),
            "pred_q1":          pred_qs["q1"], "pred_q2": pred_qs["q2"], "pred_q3": pred_qs["q3"],
            "pred_q4":          pred_qs["q4"], "pred_q5": pred_qs["q5"],
            "q1_correct":       q_correct["q1"], "q2_correct": q_correct["q2"],
            "q3_correct":       q_correct["q3"], "q4_correct": q_correct["q4"],
            "q5_correct":       q_correct["q5"],
            "gt_size_bucket":   gt_size_bucket,
            "pred_size_bucket": pred_size_bucket,
            "size_correct":     (pred_size_bucket == gt_size_bucket) if pred_size_bucket != "unknown" else False,
            "vessel_jaccard":   round(jac, 3),
            "cargo_match":      cmatch,
            "parse_error":        parse_error,
            "parse_fail_reason":  parse_fail_reason if parse_error else "",
            "infer_s":            rec.get("timing", {}).get("infer_s"),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def per_state_report(df: pd.DataFrame, run_name: str) -> str:
    SEP = "=" * 62
    lines = [f"\n{SEP}", f"  PER-STATE REPORT: {run_name}", SEP]

    for state in VALID_STATES:
        sub = df[df["gt_state"] == state]
        if sub.empty:
            continue
        n      = len(sub)
        n_ok   = int(sub["state_correct"].sum())
        n_fail = int(sub["parse_error"].sum())
        acc    = n_ok / n * 100

        pred_dist = sub["pred_state"].value_counts().to_dict()

        lines.append(f"\n  [{state.upper()}]  n={n}  state_acc={acc:.1f}%  parse_fail={n_fail}")
        lines.append(f"    Predicted distribution: {pred_dist}")

        q_parts = []
        for q in ["q1", "q2", "q3", "q4", "q5"]:
            valid = sub[f"{q}_correct"].dropna()
            if len(valid) > 0:
                q_parts.append(f"{q}={valid.mean()*100:.0f}%")
        if q_parts:
            lines.append(f"    Q-accuracy: {', '.join(q_parts)}")
        else:
            lines.append("    Q-accuracy: N/A (direct prompt)")

        lines.append(
            f"    size_bucket_acc={sub['size_correct'].mean()*100:.1f}%  "
            f"vessel_jaccard={sub['vessel_jaccard'].mean():.3f}"
        )

    lines.append("\n  --- Classification Report (parsed records only) ---")
    parsed = df[~df["parse_error"]]
    if len(parsed) > 0:
        y_true = parsed["gt_state"].tolist()
        y_pred = [p if p in VALID_STATES else "UNPARSEABLE" for p in parsed["pred_state"]]
        lines.append(classification_report(y_true, y_pred, labels=VALID_STATES, zero_division=0))
    else:
        lines.append("  No parsed records.")

    return "\n".join(lines)


def summary_row(df: pd.DataFrame, run_name: str, diffusion: bool, prompt_style: str) -> dict:
    n        = len(df)
    n_parsed = int((~df["parse_error"]).sum())
    n_fail   = n - n_parsed

    parsed = df[~df["parse_error"]]
    y_true = parsed["gt_state"].tolist()
    y_pred = [p if p in VALID_STATES else "UNPARSEABLE" for p in parsed["pred_state"]]

    macro_f1, per_class_acc = None, {}
    if y_true:
        try:
            macro_f1 = round(
                f1_score(y_true, y_pred, labels=VALID_STATES, average="macro", zero_division=0) * 100, 1
            )
        except Exception:
            pass
        for state in VALID_STATES:
            sub = df[df["gt_state"] == state]
            per_class_acc[state] = round(sub["state_correct"].mean() * 100, 1) if len(sub) else None

    q_accs = {}
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        valid = df[f"{q}_correct"].dropna()
        q_accs[q] = round(valid.mean() * 100, 1) if len(valid) > 0 else None

    timing = df["infer_s"].dropna()
    return {
        "run":               run_name,
        "diffusion":         diffusion,
        "prompt_style":      prompt_style,
        "n_records":         n,
        "n_parsed":          n_parsed,
        "parse_fail_%":      round(n_fail / n * 100, 1),
        "state_acc_%":       round(df["state_correct"].mean() * 100, 1),
        "acc_aground_%":     per_class_acc.get("aground"),
        "acc_capsized_%":    per_class_acc.get("capsized"),
        "acc_on_fire_%":     per_class_acc.get("on_fire"),
        "acc_sunken_%":      per_class_acc.get("sunken"),
        "macro_f1_%":        macro_f1,
        "q1_acc_%":          q_accs["q1"],
        "q2_acc_%":          q_accs["q2"],
        "q3_acc_%":          q_accs["q3"],
        "q4_acc_%":          q_accs["q4"],
        "q5_acc_%":          q_accs["q5"],
        "size_bucket_acc_%": round(df["size_correct"].mean() * 100, 1),
        "mean_vessel_jaccard": round(df["vessel_jaccard"].mean(), 3),
        "mean_infer_s":      round(timing.mean(), 2) if len(timing) else None,
        "median_infer_s":    round(timing.median(), 2) if len(timing) else None,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth ...")
    gt = load_ground_truth(GT_PATH)
    print(f"  {len(gt)} records.")

    print(f"\nScanning {RESULTS_DIR} ...")
    runs = discover_runs(RESULTS_DIR)
    if not runs:
        print("  No JSONL files found. Exiting.")
        return
    print(f"  {len(runs)} run(s) found.\n")

    summary_rows = []

    for fname, diffusion, prompt_style in runs:
        run_name = fname.replace(".jsonl", "")
        print(f"\n{'-'*60}")
        print(f"  Run: {run_name}")
        print(f"{'-'*60}")

        records = load_run(RESULTS_DIR / fname)
        print(f"  {len(records)} inference records.")

        df = evaluate_run(records, gt, prompt_style)

        csv_out = OUT_DIR / f"eval_{run_name}.csv"
        df.to_csv(csv_out, index=False)
        print(f"  Per-entry CSV  -> {csv_out.relative_to(HERE)}")

        report = per_state_report(df, run_name)
        print(report)
        (OUT_DIR / f"eval_{run_name}_report.txt").write_text(report, encoding="utf-8")

        # Parse-failure detail file
        failures = df[df["parse_error"]][["image", "gt_state", "parse_fail_reason"]]
        if not failures.empty:
            err_path = OUT_DIR / f"parse_errors_{run_name}.txt"
            lines = [
                f"Parse failures for {run_name}  ({len(failures)}/{len(df)} records)",
                "=" * 70,
            ]
            # Group by reason prefix for a quick summary
            reason_counts: dict = {}
            for reason in failures["parse_fail_reason"]:
                key = reason.split(":")[0]
                reason_counts[key] = reason_counts.get(key, 0) + 1
            lines.append("Reason summary:")
            for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("Per-record detail:")
            lines.append("-" * 70)
            for _, row in failures.iterrows():
                lines.append(f"[{row['gt_state']:8s}]  {row['image']}")
                lines.append(f"           {row['parse_fail_reason']}")
            err_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  Parse errors   -> {err_path.relative_to(HERE)}")

        summary_rows.append(summary_row(df, run_name, diffusion, prompt_style))

    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "eval_summary.csv"
    try:
        summary_df.to_csv(summary_csv, index=False)
    except PermissionError:
        print(f"\n  WARNING: Could not write {summary_csv.name} — close it if open in Excel, then re-run.")

    SEP = "=" * 100
    print(f"\n{SEP}")
    print("  HOLISTIC SUMMARY")
    print(SEP)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.1f}".format)

    key_cols = [
        "run", "diffusion", "prompt_style",
        "n_parsed", "parse_fail_%", "state_acc_%",
        "acc_aground_%", "acc_capsized_%", "acc_on_fire_%", "acc_sunken_%",
        "macro_f1_%",
        "q1_acc_%", "q2_acc_%", "q3_acc_%", "q4_acc_%", "q5_acc_%",
        "size_bucket_acc_%", "mean_vessel_jaccard",
        "mean_infer_s", "median_infer_s",
    ]
    key_cols = [c for c in key_cols if c in summary_df.columns]
    print(summary_df[key_cols].to_string(index=False))
    print(f"\nSummary CSV -> {summary_csv.relative_to(HERE)}")


if __name__ == "__main__":
    main()
