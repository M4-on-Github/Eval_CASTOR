"""
CASTOR LLM Reference Judge — Pipeline 3

Uses an Ollama LLM as a semantic reference judge: evaluates each inference output
against human GT field-by-field, with natural-language reasoning.

Advantages over regex eval:
  - State near-misses  ("the vessel is grounded" → aground)
  - Semantic vessel_type ("cargo ship" ≈ "freighter")
  - Semantic size  ("very large" → large bucket)
  - Richer cargo matching
  - Flexible yes/no extraction

Modes:
  Full-text (default)  — judge reads raw inference output + GT
  Extracted (--pre-parsed) — judge reads Gemma-extracted fields + GT (faster)
  Eval-only (--eval-only)  — recompute metrics from existing verdict JSONL

Output -> results/p3_llm_judge/

Run from anywhere:
    python pipelines/judge_castor.py
    python pipelines/judge_castor.py --pre-parsed
    python pipelines/judge_castor.py --eval-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

EVAL_ROOT  = Path(__file__).parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_ground_truth, load_run, read_jsonl
from shared.metrics import VALID_STATES
from shared.ollama  import call_ollama

GT_PATH    = EVAL_ROOT / "human_ground_truth_label" / "human_gt.csv"
RESULTS_IN = EVAL_ROOT.parent / "results" / "castor_results"
GEMMA_DIR  = EVAL_ROOT / "results" / "p2_llm_extract" / "extracted"
OUT_DIR    = EVAL_ROOT / "results" / "p3_llm_judge"

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
MODEL      = os.environ.get("CASTOR_JUDGE_MODEL", "gemma4:31b-cloud")

JUDGE_FIELDS = ["state", "vessel_type", "size_estimate", "cargo", "q1", "q2", "q3", "q4", "q5"]
_TEXT_WINDOW = 3000

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYS_FULL = """\
You are a strict reference judge for a maritime disaster classification task.
Given GROUND TRUTH (human-annotated) and a MODEL RESPONSE (raw inference text),
judge whether the model correctly identified each field.

Rules:
- state: must clearly match one of [aground, capsized, on_fire, sunken, good].
  Accept near-synonyms (grounded=aground, sinking=sunken). Reject ambiguous.
- vessel_type: correct if semantically equivalent (cargo ship ≈ freighter).
- size_estimate: correct if maps to the same bucket [small/medium/large].
- cargo: correct if GT and model agree on presence/absence and type.
- q1-q5: binary yes/no only. No answer = false.

Return JSON with keys: state, vessel_type, size_estimate, cargo, q1, q2, q3, q4, q5
Each value: {"correct": bool, "reason": "one sentence"}. No markdown."""

_USER_FULL = """\
GROUND TRUTH:
  state: {gt_state}  vessel_type: {gt_vessel_type}  size_estimate: {gt_size}
  cargo: {gt_cargo}  q1: {gt_q1}  q2: {gt_q2}  q3: {gt_q3}  q4: {gt_q4}  q5: {gt_q5}

MODEL RESPONSE (last {window} chars):
---
{excerpt}
---
Judge each field and return the JSON verdict."""

_SYS_EXTR = """\
You are a strict reference judge for a maritime disaster classification task.
Given GROUND TRUTH and EXTRACTED PREDICTIONS from a model, judge each prediction.

Rules:
- state: must clearly match one of [aground, capsized, on_fire, sunken, good].
  UNKNOWN = incorrect.
- vessel_type: correct if semantically equivalent.
- size_estimate: correct if same bucket [small/medium/large].
- cargo: correct if presence/absence and type broadly match.
- q1-q5: UNKNOWN = incorrect.

Return JSON with keys: state, vessel_type, size_estimate, cargo, q1, q2, q3, q4, q5
Each value: {"correct": bool, "reason": "one sentence"}. No markdown."""

_USER_EXTR = """\
GROUND TRUTH:
  state: {gt_state}  vessel_type: {gt_vessel_type}  size_estimate: {gt_size}
  cargo: {gt_cargo}  q1: {gt_q1}  q2: {gt_q2}  q3: {gt_q3}  q4: {gt_q4}  q5: {gt_q5}

EXTRACTED PREDICTIONS:
  state: {pred_state}  vessel_type: {pred_vessel}  size_estimate: {pred_size}
  cargo: {pred_cargo}  q1: {pred_q1}  q2: {pred_q2}  q3: {pred_q3}  q4: {pred_q4}  q5: {pred_q5}

Judge each field and return the JSON verdict."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class JudgePromptBuilder:
    """Builds the (system, user) prompt pair sent to the semantic judge.

    Two modes, matching the two P3 entry points:

      full       the model's raw answer is shown to the judge
      extracted  Gemma-extracted fields are shown instead of prose

    Both interpolate the same ground truth, so the judge always compares
    against the identical reference regardless of what it is shown.
    """

    #: Characters of model output shown to the judge in full mode.
    TEXT_WINDOW = _TEXT_WINDOW

    @staticmethod
    def normalise_value(v) -> str:
        """Render an extracted field for the prompt.

        None becomes "UNKNOWN", as does the literal string in any case, so the
        judge sees one consistent token for "the model did not say". An EMPTY
        string is left alone — that means the field was absent from the
        extraction, which is a different thing from the model declining.
        """
        if v is None:
            return "UNKNOWN"
        s = str(v).strip()
        return "UNKNOWN" if s.upper() == "UNKNOWN" else s

    @classmethod
    def _gt_fields(cls, gt: dict) -> dict:
        return dict(
            gt_state=gt.get("state", ""), gt_vessel_type=gt.get("vessel_type", ""),
            gt_size=gt.get("size_estimate", ""), gt_cargo=gt.get("cargo", ""),
            gt_q1=gt.get("q1", ""), gt_q2=gt.get("q2", ""),
            gt_q3=gt.get("q3", ""), gt_q4=gt.get("q4", ""), gt_q5=gt.get("q5", ""),
        )

    @classmethod
    def excerpt(cls, text: str) -> str:
        """Trim long output to the window shown to the judge.

        Keeps the TAIL, not the head. Chain-of-thought answers put the JSON
        verdict LAST, so truncating from the front would discard exactly the
        content being judged and the judge would score the reasoning preamble.
        """
        return text[-cls.TEXT_WINDOW:] if len(text) > cls.TEXT_WINDOW else text

    @classmethod
    def build_full(cls, text: str, gt: dict) -> tuple:
        user = _USER_FULL.format(window=cls.TEXT_WINDOW,
                                 excerpt=cls.excerpt(text),
                                 **cls._gt_fields(gt))
        return _SYS_FULL, user

    @classmethod
    def build_extracted(cls, gemma_rec: dict, gt: dict) -> tuple:
        gv = cls.normalise_value
        user = _USER_EXTR.format(
            pred_state=gv(gemma_rec.get("state")),
            pred_vessel=gv(gemma_rec.get("vessel_type")),
            pred_size=gv(gemma_rec.get("size_estimate")),
            pred_cargo=gv(gemma_rec.get("cargo")),
            pred_q1=gv(gemma_rec.get("q1")), pred_q2=gv(gemma_rec.get("q2")),
            pred_q3=gv(gemma_rec.get("q3")), pred_q4=gv(gemma_rec.get("q4")),
            pred_q5=gv(gemma_rec.get("q5")),
            **cls._gt_fields(gt))
        return _SYS_EXTR, user


def _gv(v) -> str:
    """Normalise an extracted field. Facade over JudgePromptBuilder."""
    return JudgePromptBuilder.normalise_value(v)


def build_prompt_full(text: str, gt: dict) -> tuple:
    """Prompt showing the model's raw answer. Facade."""
    return JudgePromptBuilder.build_full(text, gt)


def build_prompt_extracted(gemma_rec: dict, gt: dict) -> tuple:
    """Prompt showing Gemma-extracted fields. Facade."""
    return JudgePromptBuilder.build_extracted(gemma_rec, gt)


def load_gemma_parsed(path: Path) -> dict:
    result = {}
    for rec in read_jsonl(path):
        if rec.get("gemma_parse_ok") and "image" in rec:
            result[rec["image"]] = rec
    return result


def load_existing_verdicts(path: Path) -> set:
    done = set()
    for rec in read_jsonl(path):
        if rec.get("judge_ok") and "image" in rec:
            done.add(rec["image"])
    return done


class VerdictUnpacker:
    """Flattens the judge's JSON reply into per-field flags and reasons.

    The judge is asked for {"state": {"correct": bool, "reason": str}, ...} but
    does not always comply, so three shapes are accepted per field: the full
    dict, a bare bool, or anything else.

    CAUTION — an unusable verdict is recorded as correct=False, which is
    indistinguishable in the output from the judge saying the answer was wrong.
    A judge FORMATTING failure therefore depresses the reported accuracy rather
    than being excluded from it.

    That is a real limitation, not an oversight to work around silently: the
    reason string preserves the offending value ("unexpected: ..."), so such
    records remain findable after the fact. Compare PanelVote in
    judge_panel/aggregate.py, which does keep the distinction by emitting a
    "no_score" verdict — the two pipelines differ here deliberately, because P3
    has a single judge and no second opinion to fall back on.

    Every field always appears in the output so downstream frames have a
    uniform shape.
    """

    FIELDS = JUDGE_FIELDS

    @classmethod
    def unpack(cls, parsed: dict) -> dict:
        result = {}
        for field in cls.FIELDS:
            val = parsed.get(field)
            if isinstance(val, dict):
                result[field] = bool(val.get("correct", False))
                result[f"{field}_reason"] = str(val.get("reason", ""))
            elif isinstance(val, bool):
                result[field] = val
                result[f"{field}_reason"] = ""
            else:
                result[field] = False
                result[f"{field}_reason"] = f"unexpected: {val!r}"
        return result


def unpack_verdict(parsed: dict) -> dict:
    """Flatten a judge reply. Facade over VerdictUnpacker."""
    return VerdictUnpacker.unpack(parsed)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_run(jsonl_path: Path, gt_dict: dict, use_extracted: bool,
                model: str, url: str) -> Path:
    run_name = jsonl_path.stem
    out_path = OUT_DIR / "verdicts" / f"{run_name}_judge.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  Run  : {run_name}")
    print(f"  Mode : {'extracted (Gemma)' if use_extracted else 'full-text'}")
    print(f"  Model: {model}")
    print(f"{'='*62}")

    records = load_run(jsonl_path)
    if not records:
        print("  No records — skipping.")
        return out_path

    gemma_dict = {}
    if use_extracted:
        gp = GEMMA_DIR / f"{run_name}_gemma.jsonl"
        if gp.exists():
            gemma_dict = load_gemma_parsed(gp)
            print(f"  Gemma pre-parsed: {len(gemma_dict)} records.")
        else:
            print(f"  WARNING: {gp.name} not found — falling back to full-text mode.")
            use_extracted = False

    done = load_existing_verdicts(out_path)
    if done:
        print(f"  Resuming: {len(done)}/{len(records)} already judged.")

    if out_path.exists() and done:
        kept = [r for r in read_jsonl(out_path) if r.get("judge_ok")]
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")

    options = {"temperature": 0, "num_predict": 1024}

    with open(out_path, "a", encoding="utf-8") as out_f:
        for idx, rec in enumerate(records, 1):
            img  = rec.get("image", "")
            text = rec.get("text", "")
            gt   = gt_dict.get(img, {})

            if img in done:
                continue

            if use_extracted and img in gemma_dict:
                system, user = build_prompt_extracted(gemma_dict[img], gt)
            else:
                system, user = build_prompt_full(text, gt)

            parsed, raw, elapsed = call_ollama(system, user, model=model,
                                               url=url, options=options)

            if parsed is not None:
                verdict  = unpack_verdict(parsed)
                out_rec  = {"image": img, "judge_ok": True,
                            "judge_infer_s": round(elapsed, 3),
                            "gt_state": gt.get("state", "")}
                out_rec.update(verdict)
                state_ok = verdict.get("state", False)
                print(f"  {idx}/{len(records)}  {img}  "
                      f"state={'OK' if state_ok else 'WRONG'}  ({elapsed:.1f}s)")
            else:
                out_rec = {"image": img, "judge_ok": False,
                           "judge_raw": raw[:400], "judge_infer_s": round(elapsed, 3),
                           "gt_state": gt.get("state", "")}
                print(f"  {idx}/{len(records)}  {img}  -> FAILED  ({elapsed:.1f}s)")

            out_f.write(json.dumps(out_rec) + "\n")
            out_f.flush()
            done.add(img)

    n_ok   = sum(1 for r in read_jsonl(out_path) if r.get("judge_ok"))
    n_fail = sum(1 for r in read_jsonl(out_path) if not r.get("judge_ok"))
    print(f"\n  Done: {n_ok} judged, {n_fail} failed -> {out_path.relative_to(EVAL_ROOT)}")
    return out_path


# ---------------------------------------------------------------------------
# Metrics from verdicts
# ---------------------------------------------------------------------------

def evaluate_verdicts(out_path: Path, gt_dict: dict) -> pd.DataFrame:
    rows = []
    for rec in read_jsonl(out_path):
        if not rec.get("judge_ok"):
            continue
        img = rec.get("image", "")
        gt  = gt_dict.get(img, {})
        row = {"image": img, "gt_state": rec.get("gt_state", gt.get("state", ""))}
        for field in JUDGE_FIELDS:
            row[f"judge_{field}"] = rec.get(field, False)
        row["judge_all_correct"] = all(rec.get(f, False) for f in JUDGE_FIELDS)
        rows.append(row)
    return pd.DataFrame(rows)


def judge_summary_row(df: pd.DataFrame, run_name: str, diffusion: bool, n_records: int) -> dict:
    n = len(df)
    row = {
        "run":          run_name,
        "diffusion":    diffusion,
        "n_records":    n_records,
        "n_judged":     n,
        "judge_fail_%": round((n_records - n) / n_records * 100, 1) if n_records else None,
    }
    for field in JUDGE_FIELDS:
        col = f"judge_{field}"
        if col in df.columns:
            row[f"{col}_%"] = round(df[col].mean() * 100, 1) if n > 0 else None
    if "judge_all_correct" in df.columns:
        row["judge_all_fields_%"] = round(df["judge_all_correct"].mean() * 100, 1) if n > 0 else None
    for state in VALID_STATES:
        sub = df[df["gt_state"] == state]
        col = "judge_state"
        if col in df.columns:
            row[f"judge_acc_{state}_%"] = round(sub[col].mean() * 100, 1) if len(sub) > 0 else None
    return row


def judge_report(df: pd.DataFrame, run_name: str) -> str:
    SEP = "=" * 62
    lines = [f"\n{SEP}", f"  JUDGE REPORT: {run_name}", SEP]
    for state in VALID_STATES:
        sub = df[df["gt_state"] == state]
        if sub.empty:
            continue
        n = len(sub)
        n_ok = int(sub["judge_state"].sum()) if "judge_state" in sub.columns else 0
        lines.append(f"\n  [{state.upper()}]  n={n}  judge_state_acc={n_ok/n*100:.1f}%")
        for field in ["vessel_type", "size_estimate", "cargo"]:
            col = f"judge_{field}"
            if col in sub.columns:
                lines.append(f"    {field}: {sub[col].mean()*100:.1f}% correct")
        q_parts = [f"q{i}={sub[f'judge_q{i}'].mean()*100:.0f}%"
                   for i in range(1, 6) if f"judge_q{i}" in sub.columns]
        if q_parts:
            lines.append(f"    Q-accuracy: {', '.join(q_parts)}")
    n_all = len(df)
    if n_all > 0 and "judge_all_correct" in df.columns:
        lines.append(f"\n  All fields correct: {df['judge_all_correct'].mean()*100:.1f}%  ({n_all} records)")
    return "\n".join(lines)


def judge_class_breakdown(df: pd.DataFrame, run_name: str) -> str:
    """Per-GT-class correct/incorrect table for P3.

    P3 verdicts store True/False for each field — the judge does not record
    the actual predicted label, so a full confusion matrix (GT x pred) is not
    possible. This table is the closest equivalent: for each GT class, how many
    images did the judge mark state-correct vs state-incorrect.
    """
    SEP  = "=" * 72
    SEP2 = "-" * 72
    BAR_W = 32

    lines = [f"\n{SEP}", f"  JUDGE CLASS BREAKDOWN: {run_name}", SEP]
    lines.append("\n  Note: P3 verdicts are correct/incorrect only — predicted")
    lines.append("  labels are not recorded, so a full confusion matrix is not available.\n")

    # Summary table
    LW, CW = 12, 12
    lines.append(
        "  " + f"{'':>{LW}}" +
        f"{'n':>{CW}}" + f"{'correct':>{CW}}" + f"{'incorrect':>{CW}}" + f"{'acc':>{CW}}"
    )
    lines.append("  " + SEP2)

    total_n = total_ok = 0
    for state in VALID_STATES:
        sub = df[df["gt_state"] == state]
        n    = len(sub)
        n_ok = int(sub["judge_state"].sum()) if "judge_state" in sub.columns and n > 0 else 0
        acc  = n_ok / n * 100 if n > 0 else 0.0
        total_n  += n
        total_ok += n_ok
        lines.append(
            "  " + f"{state:>{LW}}" +
            f"{n:>{CW}}" + f"{n_ok:>{CW}}" + f"{n - n_ok:>{CW}}" +
            f"{acc:>{CW-1}.1f}%"
        )

    lines.append("  " + SEP2)
    overall_acc = total_ok / total_n * 100 if total_n > 0 else 0.0
    lines.append(
        "  " + f"{'Total':>{LW}}" +
        f"{total_n:>{CW}}" + f"{total_ok:>{CW}}" + f"{total_n - total_ok:>{CW}}" +
        f"{overall_acc:>{CW-1}.1f}%"
    )

    # Per-folder bars
    lines.append(f"\n{SEP}")
    lines.append("  PER-FOLDER DETAIL")
    lines.append(SEP)

    for state in VALID_STATES:
        sub  = df[df["gt_state"] == state]
        n    = len(sub)
        if n == 0:
            continue
        n_ok   = int(sub["judge_state"].sum()) if "judge_state" in sub.columns else 0
        n_fail = n - n_ok
        acc    = n_ok / n * 100

        filled_ok   = round(acc / 100 * BAR_W)
        filled_fail = BAR_W - filled_ok
        bar_ok   = "#" * filled_ok   + "." * filled_fail
        bar_fail = "#" * filled_fail + "." * filled_ok

        lines.append(f"\n  [{state.upper()}]  n={n}  acc={acc:.1f}%")
        lines.append(f"    correct   : {n_ok:>4}  ({acc:>5.1f}%)  [{bar_ok}]")
        lines.append(f"    incorrect : {n_fail:>4}  ({100-acc:>5.1f}%)  [{bar_fail}]")

        # Per-field breakdown for this class
        for field in ["vessel_type", "size_estimate", "cargo"] + [f"q{i}" for i in range(1, 6)]:
            col = f"judge_{field}"
            if col not in sub.columns:
                continue
            f_ok  = int(sub[col].sum())
            f_pct = f_ok / n * 100
            f_filled = round(f_pct / 100 * BAR_W)
            f_bar = "#" * f_filled + "." * (BAR_W - f_filled)
            lines.append(f"    {field:<15}: {f_ok:>4}  ({f_pct:>5.1f}%)  [{f_bar}]")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CASTOR LLM Reference Judge (Pipeline 3)."
    )
    parser.add_argument("--runs",       nargs="+", metavar="FILE")
    parser.add_argument("--model",      default=MODEL)
    parser.add_argument("--url",        default=OLLAMA_URL)
    parser.add_argument("--pre-parsed", action="store_true",
                        help="Use Gemma-extracted fields as judge input (faster)")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Recompute metrics from existing verdict files only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading ground truth ...")
    gt = load_ground_truth(GT_PATH)
    print(f"  {len(gt)} records.")

    runs_paths = sorted(RESULTS_IN.glob("*.jsonl"))
    if args.runs:
        runs_paths = [p for p in runs_paths if p.name in args.runs]

    summary_rows = []

    for run_path in runs_paths:
        run_name  = run_path.stem
        diffusion = "degf" in run_name.lower()
        out_path  = OUT_DIR / "verdicts" / f"{run_name}_judge.jsonl"

        if args.eval_only:
            if not out_path.exists():
                print(f"  SKIP {run_name}: no verdict file.")
                continue
            n_records = sum(1 for _ in load_run(run_path))
        else:
            records   = load_run(run_path)
            n_records = len(records)
            out_path  = process_run(run_path, gt, args.pre_parsed, args.model, args.url)

        df = evaluate_verdicts(out_path, gt)
        if df.empty:
            print(f"  No judged records for {run_name} — skipping metrics.")
            continue

        csv_out = OUT_DIR / f"judge_eval_{run_name}.csv"
        df.to_csv(csv_out, index=False)
        print(f"  Per-entry CSV -> {csv_out.relative_to(EVAL_ROOT)}")

        report_txt = judge_report(df, run_name)
        print(report_txt)
        (OUT_DIR / f"judge_report_{run_name}.txt").write_text(report_txt, encoding="utf-8")

        breakdown = judge_class_breakdown(df, run_name)
        print(breakdown)
        (OUT_DIR / f"judge_breakdown_{run_name}.txt").write_text(breakdown, encoding="utf-8")
        print(f"  Class breakdown -> {(OUT_DIR / f'judge_breakdown_{run_name}.txt').relative_to(EVAL_ROOT)}")

        summary_rows.append(judge_summary_row(df, run_name, diffusion, n_records))

    if not summary_rows:
        print("\nNo summary rows.")
        return

    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "judge_summary.csv"
    try:
        summary_df.to_csv(summary_csv, index=False)
    except PermissionError:
        print(f"\n  WARNING: Could not write {summary_csv.name}")

    SEP = "=" * 110
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.1f}".format)
    key_cols = [
        "run", "diffusion", "n_judged", "judge_fail_%",
        "judge_state_%",
        "judge_acc_aground_%", "judge_acc_capsized_%",
        "judge_acc_on_fire_%", "judge_acc_sunken_%",
        "judge_vessel_type_%", "judge_size_estimate_%", "judge_cargo_%",
        "judge_q1_%", "judge_q2_%", "judge_q3_%", "judge_q4_%", "judge_q5_%",
        "judge_all_fields_%",
    ]
    key_cols = [c for c in key_cols if c in summary_df.columns]
    table = summary_df[key_cols].to_string(index=False)
    report_lines = [f"\n{SEP}", "  JUDGE SUMMARY (Pipeline 3)", SEP, "", table, ""]
    report_str = "\n".join(report_lines)
    print(report_str)
    summary_report = OUT_DIR / "judge_summary_report.txt"
    summary_report.write_text(report_str, encoding="utf-8")
    print(f"Summary CSV    -> {summary_csv.relative_to(EVAL_ROOT)}")
    print(f"Summary report -> {summary_report.relative_to(EVAL_ROOT)}")


if __name__ == "__main__":
    main()
