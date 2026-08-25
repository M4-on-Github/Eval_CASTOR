"""
Step -> ToolCall extraction. LLM stage, cluster-only (vLLM guided JSON).

This module is shared between calibrate.py (runs this against the gold set
across several candidate models to pick one) and the real pipeline once a
model is chosen -- calibration must exercise the exact same prompt-building
and parsing code that ships, or a calibration pass would prove nothing
about what actually runs. See design plan section 4e.

Prompt building and response parsing are pure functions (tested without a
model, see tests/test_plan_adequacy_extract.py). The vLLM call itself is
isolated in _run_vllm_batch with the import INSIDE the function -- mirrors
pipelines/salvage_analysis/extract.py:321 and pipelines/plan_coherence/
run_coherence_judge.py:201, so every other module here still imports on a
machine with no vLLM installed.

Usage (inside Apptainer via containers/plan_adequacy_stage1_job.sh):
  python3 extract.py \\
      --model      glm4_32b \\
      --model-dir  /data/$USER/glm-4-32b-0414-gptq \\
      --input      pipelines/plan_adequacy/inbox/answers_baseline.jsonl \\
      --out        results/p9_plan_adequacy/ \\
      [--gt PATH] [--limit N]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.paths import RunPaths
from pipelines.plan_adequacy.scenario import load_scenarios
from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry, build_guided_json_schema
from shared.loaders import load_run

PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_BASE = (PROMPTS_DIR / "adequacy_extract_system.txt").read_text(encoding="utf-8")
_USER_TEMPLATE = (PROMPTS_DIR / "adequacy_extract_user.txt").read_text(encoding="utf-8")


def build_system_prompt(registry: ToolRegistry) -> str:
    """Static instructions (adequacy_extract_system.txt) plus a dynamically
    generated tool reference list -- generated from the registry, not
    hand-duplicated, so tools.json stays the single source of truth (see
    vocab.build_guided_json_schema's docstring for the same rationale).

    Includes each tool's `establishes`/`effects` facts alongside family and
    params -- added after calibration (2026-08-24, 3-model bake-off) found
    a reproducible confusion cluster across BOTH usable models:
    survey_seabed and sonar_search steps both got misread as survey_hull.
    The family-only line gave no way to tell them apart ("assessment" for
    both survey_hull and survey_seabed); `establishes` already distinguishes
    them in the registry (hull_condition/vessel_size vs substrate) but was
    never surfaced to the model. No new content authored -- this reuses
    data tools.json already has, same "single source of truth" rationale
    as the rest of this function."""
    lines = ["Allowed tools (name: family -- params -- establishes/effects):"]
    for name in sorted(registry.all_tool_names()):
        spec = registry.spec(name)
        params = ", ".join(spec.params.keys()) or "none"
        facts = ", ".join(sorted(spec.establishes | spec.effects)) or "none"
        lines.append(f"  {name}: {spec.family} -- params: {params} -- establishes/effects: {facts}")
    lines.append("  no_match: (use when the step names no concrete action from this list)")
    return _SYSTEM_PROMPT_BASE + "\n" + "\n".join(lines)


def build_user_prompt(casualty: str, step_num: int, step_text: str) -> str:
    return _USER_TEMPLATE.format(casualty=casualty or "unknown", step_num=step_num, step_text=step_text)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_extraction(raw: Optional[dict], step_num: int, step_text: str) -> ToolCall:
    """Turn one guided-JSON response into a ToolCall. Never raises -- a
    parse failure (raw is None, or missing keys) becomes a "no_match" call
    so a single bad response doesn't abort a batch, matching the
    resume/never-raise convention used across every other pipeline stage
    in this repo (e.g. OllamaClient's docstring, run_coherence_judge.py's
    parse_errors counter)."""
    if not raw or not isinstance(raw, dict):
        return ToolCall(step_num=step_num, step_text=step_text, tool="no_match")

    tool = raw.get("tool", "no_match")
    params = raw.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    secondary = raw.get("secondary_tools") or []
    if not isinstance(secondary, list):
        secondary = []

    return ToolCall(
        step_num=step_num,
        step_text=step_text,
        tool=tool,
        # Explicit None values are kept, not dropped -- three consecutive
        # calibration runs against glm4_32b (2026-08-24) read null_fidelity
        # as EXACTLY 0.0 with no movement across a prompt fix and a schema
        # fix, which turned out to be because a correctly-nulled param
        # never survived to here: `if v is not None` silently deleted the
        # very keys the null-fidelity metric exists to check, so the
        # scorer could never see a correct null, only ever a real value
        # (via a key gold didn't have either) or nothing at all. Params
        # the model never mentions at all are still absent from this dict,
        # same as before -- only an EXPLICIT null now survives.
        params=dict(params),
        conditional=bool(raw.get("conditional", False)),
        condition_text=raw.get("condition_text"),
        condition_var=raw.get("condition_var") or "none",
        secondary_tools=tuple(secondary),
    )


# ---------------------------------------------------------------------------
# vLLM batch inference (cluster only)
# ---------------------------------------------------------------------------

def _run_vllm_batch(prompts: list, model_dir: str, schema: dict,
                     max_model_len: int, max_tokens: int) -> list:
    """Run all (system, user) prompt pairs through one vLLM instance.
    Returns list of (parsed_dict|None) aligned with `prompts`. Mirrors the
    guided-decoding setup in run_coherence_judge.py:199-244, including the
    vLLM 0.12 GuidedDecodingParams -> StructuredOutputsParams fallback."""
    from vllm import LLM, SamplingParams

    try:
        from vllm.sampling_params import GuidedDecodingParams
        guided_kwargs = {"guided_decoding": GuidedDecodingParams(json=schema)}
    except ImportError:
        from vllm.sampling_params import StructuredOutputsParams
        guided_kwargs = {"structured_outputs": StructuredOutputsParams(json=schema)}

    print(f"  [vLLM] Loading model from {model_dir} ...")
    llm = LLM(
        model=model_dir,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,  # system prompt (tool list) is identical across all calls
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, **guided_kwargs)

    conversations = [
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for system, user in prompts
    ]
    print(f"  [vLLM] Extracting {len(conversations)} steps ...")
    t0 = time.perf_counter()
    outputs = llm.chat(conversations, sampling_params=params)
    elapsed = time.perf_counter() - t0
    print(f"  [vLLM] Done in {elapsed:.1f}s ({elapsed/max(len(outputs),1):.2f}s/call avg)")

    results = []
    for o in outputs:
        raw_text = o.outputs[0].text.strip()
        cleaned = _strip_wrapper_artifacts(raw_text)
        try:
            results.append((json.loads(cleaned), raw_text))
        except (json.JSONDecodeError, AttributeError) as e:
            # Keep the raw text on every failure, not just a sample -- the
            # calibration bake-off against llama_3_3_70b (2026-08-24) hit
            # 100% parse failure with zero exceptions anywhere in the vLLM
            # logs (guided decoding was active, no crash, no timeout) and
            # there was nothing to inspect afterward because nothing had
            # been captured. Whatever the actual cause turns out to be,
            # not saving raw_text made it undiagnosable after the fact.
            results.append((None, raw_text))
    return results


#: Guided decoding is supposed to constrain the ENTIRE completion to the
#: schema, so wrapper text shouldn't normally appear -- but some models'
#: native chat templates (Llama 3.1+'s built-in tool-calling format is the
#: leading suspect for the 100% parse-failure case above) can still inject
#: markdown fences or a tag/preamble around an otherwise schema-valid
#: object. Mirrors the stripping shared/ollama.py:call_ollama already does
#: for the same reason on the Ollama side of this repo.
_CODE_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)


def _strip_wrapper_artifacts(text: str) -> str:
    stripped = _CODE_FENCE_RE.sub('', text).strip()
    # A JSON object nested in leading/trailing prose -- take the outermost
    # {...} span rather than the whole string.
    start, end = stripped.find('{'), stripped.rfind('}')
    if start != -1 and end != -1 and end > start:
        return stripped[start:end + 1]
    return stripped


def extract_steps(steps: list, model_dir: str, registry: ToolRegistry,
                   max_model_len: int = 6144, max_tokens: int = 256) -> list:
    """Extract a list of (step_num, step_text, casualty) into ToolCalls, in
    order. The single production entrypoint -- extract.py's CLI (run()
    below) calls this once across EVERY step from EVERY plan in a run, not
    once per plan.

    `steps` is (step_num, step_text, casualty) tuples -- the exact shape
    calibrate.py's gold_to_calls() already produces (calibrate.py:69-74),
    so a real run and the calibration bake-off pass the identical tuple
    shape into the identical prompt-building code. casualty travels PER
    STEP, not once for the whole call, specifically so one production run
    covering all four casualty states -- or all N plans in a batch -- loads
    vLLM exactly ONCE (_run_vllm_batch's LLM(...) construction is the
    expensive part, not the per-prompt inference) rather than once per plan.
    An earlier version of this function took one shared `casualty: str` for
    its whole `steps` list, which only ever worked correctly when called
    once per single-casualty plan -- it was never actually called anywhere
    in the codebase (calibrate.py's run_calibration() has always built
    prompts inline instead, for exactly this per-record-casualty reason).
    """
    system = build_system_prompt(registry)
    prompts = [(system, build_user_prompt(casualty, n, t)) for n, t, casualty in steps]
    schema = build_guided_json_schema(registry)
    parsed_and_raw = _run_vllm_batch(prompts, model_dir, schema, max_model_len, max_tokens)
    return [
        parse_extraction(parsed, n, t)
        for (n, t, _casualty), (parsed, _raw) in zip(steps, parsed_and_raw)
    ]


# ---------------------------------------------------------------------------
# Production CLI -- Stage 1
# ---------------------------------------------------------------------------

def run(input_path: Path, out_dir: Path, model_key: str, model_dir: str,
        gt_path: Optional[Path] = None, max_model_len: int = 6144,
        max_tokens: int = 256, limit: Optional[int] = None) -> Path:
    """Extract every step of every plan in input_path into tool_calls.jsonl.
    Resume-safe (skip images already present in the output file), same
    convention as run_coherence_judge.py's run() -- see its `done_images`
    handling. Returns the tool_calls.jsonl path written to.
    """
    # improved/eval/ has no __init__.py (implicit namespace package) but a
    # plain dotted import resolves fine since EVAL_ROOT is already on
    # sys.path from this module's own top -- same convention every other
    # `from pipelines...` import in this file relies on. Imported here
    # (not at module top) purely so extract.py's pure functions/CLI parsing
    # still import cleanly even if parse_steps_v2.py ever moves.
    from pipelines.plan_coherence.improved.eval.parse_steps_v2 import parse_steps_v2

    run_name = input_path.stem
    paths = RunPaths(run_name, base_dir=out_dir)
    paths.dir.mkdir(parents=True, exist_ok=True)

    done_images = set()
    if paths.tool_calls.exists():
        with open(paths.tool_calls, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_images.add(json.loads(line).get("image"))
                except json.JSONDecodeError:
                    continue
        if done_images:
            print(f"  Resume: {len(done_images)} images already processed.")

    scenarios = load_scenarios(gt_path)
    records = load_run(input_path)
    if limit:
        records = records[:limit]

    pending = [r for r in records if r.get("image", "") not in done_images]
    if not pending:
        print("  All records already processed -- nothing to do.")
        return paths.tool_calls

    # ── Build the flat (step_num, step_text, casualty) list across EVERY
    # pending plan -- one vLLM load for the whole run, not one per plan
    # (see extract_steps's docstring for why this matters).
    batch_meta = []   # (image, step_num, step_text, secondary placeholder)
    batch_steps = []  # (step_num, step_text, casualty) for extract_steps
    skipped_no_steps = 0
    skipped_no_scenario = 0

    for rec in pending:
        image = rec.get("image", "")
        plan_text = rec.get("text", "")
        scenario = scenarios.get(image)
        if scenario is None:
            skipped_no_scenario += 1
            print(f"  WARNING: no ground-truth scenario for {image} -- skipping")
            continue
        steps, _flag = parse_steps_v2(plan_text, source_id=image)
        if not steps:
            skipped_no_steps += 1
            print(f"  WARNING: no steps parsed for {image} -- skipping")
            continue
        for step_num, step_text in steps:
            batch_meta.append((image, step_num, step_text))
            batch_steps.append((step_num, step_text, scenario.state))

    if not batch_steps:
        print("  Nothing extractable -- no output written.")
        return paths.tool_calls

    print(f"  Extracting {len(batch_steps)} steps across "
          f"{len({m[0] for m in batch_meta})} images "
          f"({skipped_no_steps} skipped: no steps parsed; "
          f"{skipped_no_scenario} skipped: no ground truth) ...")

    registry = ToolRegistry.load()
    calls = extract_steps(batch_steps, model_dir, registry, max_model_len, max_tokens)

    with open(paths.tool_calls, "a", encoding="utf-8") as f:
        for (image, _step_num, _step_text), call in zip(batch_meta, calls):
            row = {
                "image": image,
                "casualty": scenarios[image].state,
                "model": model_key,
                "step_num": call.step_num,
                "step_text": call.step_text,
                "tool": call.tool,
                "params": call.params,
                "conditional": call.conditional,
                "condition_text": call.condition_text,
                "condition_var": call.condition_var,
                "secondary_tools": list(call.secondary_tools),
            }
            f.write(json.dumps(row) + "\n")

    print(f"  Tool calls written to {paths.tool_calls}")
    return paths.tool_calls


def main():
    ap = argparse.ArgumentParser(description="P9 Stage 1: extract steps into tool calls")
    ap.add_argument("--model", required=True, help="Model key (for logging only)")
    ap.add_argument("--model-dir", required=True, help="Absolute path to model weights")
    ap.add_argument("--input", required=True, type=Path, help="Answer JSONL to extract from")
    ap.add_argument("--out", required=True, type=Path, help="Base output directory (paths.BASE_OUT_DIR-style)")
    ap.add_argument("--gt", type=Path, default=None, help="human_gt.csv path (default: scenario.py's default)")
    ap.add_argument("--max-model-len", type=int, default=6144,
                     help="Bumped from 4096 after the survey-tool disambiguation prompt fix pushed "
                          "the system prompt to ~3800 of a 4096 budget -- see calibrate.py's matching flag.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: process only the first N records")
    args = ap.parse_args()

    run(args.input, args.out, args.model, args.model_dir, args.gt,
        args.max_model_len, args.max_tokens, args.limit)


if __name__ == "__main__":
    main()
