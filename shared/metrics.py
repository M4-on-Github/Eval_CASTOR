"""
Shared metric and extraction functions for CASTOR evaluation.
Used by all four evaluation pipelines.
"""

import json
import re

import pandas as pd
from sklearn.metrics import classification_report, f1_score

VALID_STATES = ["aground", "capsized", "on_fire", "sunken"]

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

_UNESCAPE = re.compile(r'\\([_\-/])')


def _safe_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _unescape(text: str) -> str:
    return _UNESCAPE.sub(r'\1', text)


# ---------------------------------------------------------------------------
# Extraction helpers (regex-based, used by Pipeline 1)
# ---------------------------------------------------------------------------

def extract_json_block(text: str):
    """Return (parsed_dict | None, failure_reason_str).

    Scans brace positions in reverse. Prefers the outermost block containing a
    'state' key; falls back to any valid dict; returns None on total failure.
    """
    text = _unescape(text)
    positions = [m.start() for m in re.finditer(r'\{', text)]
    if not positions:
        return None, "no_braces: output text contains no '{' character"

    errors, fallback, fallback_pos = [], None, None

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
            errors.append(f"pos={pos}: unmatched braces")
            continue

        try:
            parsed = json_loads_safe(chunk[:end + 1])
            if not isinstance(parsed, dict):
                errors.append(f"pos={pos}: top level is {type(parsed).__name__}")
                continue
            if 'state' in parsed:
                return parsed, ""
            if fallback is None:
                fallback = parsed
                fallback_pos = pos
            errors.append(f"pos={pos}: no 'state' key (keys={list(parsed.keys())})")
        except Exception as e:
            errors.append(f"pos={pos}: {e}")

    if fallback is not None:
        return fallback, f"no_state_key: best JSON (pos={fallback_pos}) lacks 'state'"

    summary = "; ".join(errors[:5])
    if len(errors) > 5:
        summary += f" ... (+{len(errors)-5} more)"
    return None, f"all_attempts_failed [{len(errors)} blocks tried]: {summary}"


def json_loads_safe(s: str):
    """json.loads with strict=False retry."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(s, strict=False)


def normalize_state(raw) -> str:
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
    """Extract Q1-Q5 yes/no answers from CoT text. Returns {q1..q5: str|None}."""
    result = {f"q{i}": None for i in range(1, 6)}
    if prompt_style != "cot":
        return result

    m = re.search(
        r'Step\s*2\s*[^\n]*\n.*?(?=Step\s*3\s*[^\n]*\n|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    section = m.group(0) if m else text[:int(len(text) * 0.6)]

    for qnum, answer in re.findall(r'Q(\d)\s*[=:][^\n]*(yes|no)', section, re.IGNORECASE):
        key = f"q{qnum}"
        if key in result and result[key] is None:
            result[key] = answer.lower()

    if any(v is None for v in result.values()):
        for qnum, answer in re.findall(r'^\s*(\d)\.\s*(yes|no)\b', section, re.IGNORECASE | re.MULTILINE):
            key = f"q{qnum}"
            if key in result and result[key] is None:
                result[key] = answer.lower()

    if any(v is None for v in result.values()):
        for qnum, answer in re.findall(r'Q(\d)[^\n]{0,120}?(yes|no)', section, re.IGNORECASE):
            key = f"q{qnum}"
            if key in result and result[key] is None:
                result[key] = answer.lower()

    return result


def normalize_size(text: str) -> str:
    t = text.lower() if text else ""
    if re.search(r'\bsmall\b', t):
        return "small"
    if re.search(r'\bmedium\b', t):
        return "medium"
    if re.search(r'\blarge\b', t):
        return "large"
    return "unknown"


def vessel_jaccard(gt_text: str, pred_text: str) -> float:
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


def gemma_val(v):
    """Translate Gemma UNKNOWN sentinel → None so downstream treats it as absent."""
    if v is None:
        return None
    s = str(v).strip()
    return None if s.upper() == "UNKNOWN" else s


# ---------------------------------------------------------------------------
# Reporting
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
        lines.append(f"    Q-accuracy: {', '.join(q_parts) if q_parts else 'N/A (direct prompt)'}")
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


def confusion_matrix_report(df: pd.DataFrame, run_name: str) -> str:
    """
    Returns a formatted string with:
      1. Overall confusion matrix (counts + row %)
      2. Per-folder (per GT class) breakdown with ASCII bar charts
    """
    # Columns = every label the model actually produced, in a stable order:
    #   VALID_STATES (defined order) → any other predicted labels (sorted) → UNPARSEABLE last
    pred = df["pred_state"].fillna("UNPARSEABLE")
    extra = sorted(pred.unique().tolist())
    LABELS = (
        [s for s in VALID_STATES if s in extra]
        + [s for s in extra if s not in VALID_STATES and s != "UNPARSEABLE"]
        + (["UNPARSEABLE"] if "UNPARSEABLE" in extra else [])
    )

    SEP  = "=" * max(72, 12 + 13 * len(LABELS))
    SEP2 = "-" * max(72, 12 + 13 * len(LABELS))

    # Build count dict: counts[gt_state][pred_label]
    counts = {}
    for gt in VALID_STATES:
        mask = df["gt_state"] == gt
        vc   = pred[mask].value_counts()
        counts[gt] = {lbl: int(vc.get(lbl, 0)) for lbl in LABELS}

    row_totals = {gt: sum(counts[gt].values()) for gt in VALID_STATES}
    col_totals = {lbl: sum(counts[gt][lbl] for gt in VALID_STATES) for lbl in LABELS}
    grand_total = sum(row_totals.values())

    # ── Overall matrix ────────────────────────────────────────────────────────
    LW, CW = 12, 13   # label width, column width (CW >= len("UNPARSEABLE")+2)
    lines = [f"\n{SEP}", f"  CONFUSION MATRIX: {run_name}", SEP]
    lines.append("\n  Rows = Ground Truth      Columns = Predicted\n")

    # Header row
    hdr = f"{'':>{LW}}" + "".join(f"{lbl:>{CW}}" for lbl in LABELS) + f"{'Total':>{CW}}"
    lines.append("  " + hdr)
    lines.append("  " + SEP2)

    # Count rows
    for gt in VALID_STATES:
        row = f"{gt:>{LW}}" + "".join(f"{counts[gt][lbl]:>{CW}}" for lbl in LABELS) + f"{row_totals[gt]:>{CW}}"
        lines.append("  " + row)

    # Column totals
    lines.append("  " + SEP2)
    tot = f"{'Total':>{LW}}" + "".join(f"{col_totals[lbl]:>{CW}}" for lbl in LABELS) + f"{grand_total:>{CW}}"
    lines.append("  " + tot)

    # Row-% table
    lines.append("\n  Row %  (what each GT class was predicted as):\n")
    lines.append("  " + f"{'':>{LW}}" + "".join(f"{lbl:>{CW}}" for lbl in LABELS))
    lines.append("  " + SEP2)
    for gt in VALID_STATES:
        n = row_totals[gt]
        pcts = []
        for lbl in LABELS:
            if n > 0:
                pcts.append(f"{counts[gt][lbl]/n*100:>{CW-1}.1f}%")
            else:
                pcts.append(f"{'N/A':>{CW}}")
        lines.append("  " + f"{gt:>{LW}}" + "".join(pcts))

    # ── Per-folder detail ─────────────────────────────────────────────────────
    lines.append(f"\n{SEP}")
    lines.append("  PER-FOLDER DETAIL")
    lines.append(SEP)

    BAR_W = 32
    for gt in VALID_STATES:
        n       = row_totals[gt]
        n_right = counts[gt].get(gt, 0)  # 0 if that label never appeared as a prediction
        acc     = n_right / n * 100 if n > 0 else 0.0
        lines.append(f"\n  [{gt.upper()}]  n={n}  acc={acc:.1f}%")
        for lbl in LABELS:
            c      = counts[gt][lbl]
            pct    = c / n * 100 if n > 0 else 0.0
            filled = round(pct / 100 * BAR_W)
            bar    = "#" * filled + "." * (BAR_W - filled)
            mark   = " <- correct" if lbl == gt else ""
            lines.append(f"    {lbl:<13}: {c:>4}  ({pct:>5.1f}%)  [{bar}]{mark}")

    lines.append("")
    return "\n".join(lines)


def panel_score_summary(consensus_path, run_name: str) -> dict:
    """Summarize a consensus JSONL from Pipeline 5 into one CSV row."""
    import json as _json
    records = []
    with open(consensus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(_json.loads(line))
                except _json.JSONDecodeError:
                    pass
    if not records:
        return {"run": run_name}

    valid = [r for r in records if r.get("mean_score") is not None]
    n_flagged = sum(1 for r in records if r.get("consensus_status") == "flagged_for_review")
    n_error   = sum(1 for r in records if r.get("consensus_status") == "parse_error")

    per_class = {}
    for state in VALID_STATES:
        subset = [r["mean_score"] for r in valid if r.get("gt_state") == state]
        per_class[state] = round(sum(subset) / len(subset), 3) if subset else None

    return {
        "run":              run_name,
        "n_records":        len(records),
        "n_valid":          len(valid),
        "n_flagged":        n_flagged,
        "flagged_%":        round(n_flagged / len(records) * 100, 1) if records else None,
        "n_parse_error":    n_error,
        "mean_score":       round(sum(r["mean_score"] for r in valid) / len(valid), 3) if valid else None,
        "mean_score_aground":  per_class.get("aground"),
        "mean_score_capsized": per_class.get("capsized"),
        "mean_score_on_fire":  per_class.get("on_fire"),
        "mean_score_sunken":   per_class.get("sunken"),
    }


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
