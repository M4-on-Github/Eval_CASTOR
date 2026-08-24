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
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_adequacy.vocab import ToolCall, ToolRegistry, build_guided_json_schema

PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_BASE = (PROMPTS_DIR / "adequacy_extract_system.txt").read_text(encoding="utf-8")
_USER_TEMPLATE = (PROMPTS_DIR / "adequacy_extract_user.txt").read_text(encoding="utf-8")


def build_system_prompt(registry: ToolRegistry) -> str:
    """Static instructions (adequacy_extract_system.txt) plus a dynamically
    generated tool reference list -- generated from the registry, not
    hand-duplicated, so tools.json stays the single source of truth (see
    vocab.build_guided_json_schema's docstring for the same rationale)."""
    lines = ["Allowed tools (name: family -- params):"]
    for name in sorted(registry.all_tool_names()):
        spec = registry.spec(name)
        params = ", ".join(spec.params.keys()) or "none"
        lines.append(f"  {name}: {spec.family} -- params: {params}")
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


def extract_steps(steps: list, casualty: str, model_dir: str, registry: ToolRegistry,
                   max_model_len: int = 4096, max_tokens: int = 256) -> list:
    """Extract a list of (step_num, step_text) into ToolCalls, in order.
    The single production/calibration entrypoint -- both extract.py's
    eventual CLI and calibrate.py's bake-off call this, so they exercise
    identical code."""
    system = build_system_prompt(registry)
    prompts = [(system, build_user_prompt(casualty, n, t)) for n, t in steps]
    schema = build_guided_json_schema(registry)
    parsed_and_raw = _run_vllm_batch(prompts, model_dir, schema, max_model_len, max_tokens)
    return [
        parse_extraction(parsed, n, t)
        for (n, t), (parsed, _raw) in zip(steps, parsed_and_raw)
    ]
