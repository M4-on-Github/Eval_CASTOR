"""
Pipeline 6 Stage 1 — LLM element extraction from salvage plans.

For each record's recovery_considerations text, asks an LLM to extract the
concrete salvage entities/actions mentioned, open-vocabulary (no fixed
taxonomy). Writes a checkpoint JSONL of {image, raw_elements} that Stage 2
(normalize.py) clusters into canonical categories.

Two backends:
  --backend ollama (default) -- local dev, HTTP calls to a locally-running
    Ollama server (needs OLLAMA_HOST reachable).
  --backend vllm -- cluster runs, no Ollama available there. Loads deepseek_r1
    (or another local HF checkpoint) directly via vLLM and extracts all
    pending records in one batch pass, same pattern as
    judge_panel/run_judge.py's _run_vllm_batch.

Usage:
  python pipelines/salvage_analysis/extract.py --run answers_baseline
  python pipelines/salvage_analysis/extract.py --run answers_baseline --backend vllm \\
      --model-dir /data/$USER/deepseek-r1-distill-llama-70b-awq
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from shared.loaders import load_run
from shared.ollama import call_ollama
from pipelines.salvage_analysis import paths
from pipelines.salvage_analysis.combine_shards import resolve_input_path
from pipelines.salvage_analysis.records import get_field_text

PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT = (PROMPTS_DIR / "salvage_extract_system.txt").read_text(encoding="utf-8")
_USER_TEMPLATE = (PROMPTS_DIR / "salvage_extract_user.txt").read_text(encoding="utf-8")

RESULTS_IN = Path(os.environ.get("CASTOR_SALVAGE_RESULTS_DIR", paths.PLANS_TO_JUDGE_DIR))

DEFAULT_MODEL = os.environ.get("CASTOR_SALVAGE_MODEL", "gemma4:31b-cloud")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"

# Forcing an empty think block (via assistant prefill) to skip deepseek_r1's
# CoT produced ~0% real extractions in production (100% of "successes" were
# trivial empty-list parses, not genuine extraction -- see
# docs/decisions/ or the Pipeline 6 build history for the raw counts).
# Letting it think normally works with how the model was actually trained;
# clean_and_parse_json already strips the resulting <think>...</think> block
# before parsing, so no prefill/no-prefill code path is needed elsewhere.
# max_tokens=2048 bounds the thinking budget -- 3072 cut inference time by
# ~10x for a real but partial improvement (still 47-75% parse failures);
# 2048 trades some of that improvement back for a shorter, more predictable
# runtime, while still leaving ~2048 tokens of the 4096 max_model_len for
# the (short) prompt.
#
# qwen25_72b is the recommended default: not a reasoning model (no <think>
# block), so guided JSON decoding (see _ELEMENTS_JSON_SCHEMA) can constrain
# its output to valid JSON from the first token -- structurally impossible
# to produce the malformed-JSON/repetition-loop/rule-recitation failures
# deepseek_r1 hit. Same max_model_len<=2048 cap as judge_panel/run_judge.py's
# _MODEL_CONFIG (Qwen AWQ leaves only ~165 KV cache blocks on 1 GPU).
# Known caveat: some CASTOR runs use qwen3vl8b as the VLM under evaluation,
# so a Qwen extractor shares a model family with that VLM (not with LLaVA-
# based runs) -- a possible confound for cross-family comparisons. Accepted
# as a documented limitation rather than standing up a third model family.
_VLLM_MODEL_CONFIG = {
    "deepseek_r1": {
        "max_model_len": 4096,
        "max_tokens": 2048,
        "prefill": None,
        "guided_json": False,
        "temperature": 0.1,
    },
    "qwen25_72b": {
        "max_model_len": 2048,
        "max_tokens": 512,
        "prefill": None,
        "guided_json": True,
        # Higher than deepseek_r1's 0.1 -- low-temperature guided decoding
        # collapsed onto {"elements": []} even for inputs with real
        # extractable content (no working space before the schema
        # constraint kicks in makes the "safe" empty completion too likely
        # at low temperature). See _run_vllm_batch docstring.
        "temperature": 0.4,
    },
    # Diagnostic variant: same model, guided decoding OFF, falls back to
    # clean_and_parse_json's prompt-based parsing. guided_json=True
    # deterministically returned {"elements": []} (byte-for-byte identical
    # across 20 different inputs) regardless of temperature or
    # repetition_penalty -- this isolates whether guided decoding itself is
    # the cause before assuming it's a sampling-parameter problem.
    "qwen25_72b_plain": {
        "max_model_len": 2048,
        "max_tokens": 512,
        "prefill": None,
        "guided_json": False,
        "temperature": 0.4,
    },
}

# Matches OUTPUT FORMAT in salvage_extract_system.txt.
_ELEMENTS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["elements"],
    "additionalProperties": False,
}

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_FENCE_RE = re.compile(r'^```(?:json)?\s*', re.MULTILINE)
_FENCE_END_RE = re.compile(r'\s*```\s*$')
_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


# ---------------------------------------------------------------------------
# Public helpers (tested directly)
# ---------------------------------------------------------------------------

def build_extract_prompt(recovery_text: str) -> str:
    """Fill the user template with the recovery-plan text to extract from."""
    return _USER_TEMPLATE.format(recovery_text=recovery_text)


def clean_and_parse_json(raw: str):
    """Strip <think> blocks, code fences, and stray byte-level BPE space
    markers from a raw LLM response and parse the JSON object inside.
    Returns None (never raises) if nothing parses."""
    if not raw:
        return None
    raw = raw.replace('Ġ', ' ')
    cleaned = _THINK_RE.sub('', raw).strip()
    cleaned = _FENCE_RE.sub('', cleaned)
    cleaned = _FENCE_END_RE.sub('', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def parse_extract_result(parsed: dict) -> dict:
    """Normalize a parsed response dict (or None on failure) into
    {elements: list[str], parse_ok: bool}."""
    if parsed is None:
        return {"elements": [], "parse_ok": False}
    elements = parsed.get("elements")
    if not isinstance(elements, list):
        return {"elements": [], "parse_ok": False}
    return {"elements": [str(e) for e in elements], "parse_ok": True}


_RAW_RESPONSE_MAX_CHARS = 2000


def build_output_record(image: str, result: dict, raw_response: str = None) -> dict:
    """Assemble the Stage 1 checkpoint record for one image. raw_response
    (truncated) is logged regardless of parse_ok -- parse_ok=True can still
    hide a content problem (e.g. guided decoding collapsing onto an empty
    list for input with real extractable content), so success alone isn't
    enough signal to skip keeping the raw text."""
    record = {
        "image": image,
        "raw_elements": result["elements"],
        "parse_ok": result["parse_ok"],
    }
    if raw_response is not None:
        record["raw_response"] = raw_response[:_RAW_RESPONSE_MAX_CHARS]
    return record


# ---------------------------------------------------------------------------
# Resume support (shared by both backends)
# ---------------------------------------------------------------------------

def _resume_done_images(out_path: Path) -> dict:
    """Returns {image: json_line} for previously successful records, rewriting
    out_path to drop any failed ones so they get retried on this run."""
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if r.get("parse_ok"):
                            done[r["image"]] = line
                    except json.JSONDecodeError:
                        pass
        if done:
            with open(out_path, "w", encoding="utf-8") as f:
                for l in done.values():
                    f.write(l + "\n")
        else:
            out_path.unlink()
        print(f"  Resume: {len(done)} already extracted.")
    return done


# ---------------------------------------------------------------------------
# Ollama backend (local dev)
# ---------------------------------------------------------------------------

def run(input_path: Path, out_path: Path, model: str, url: str, limit=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _resume_done_images(out_path)

    records = load_run(input_path)
    if limit:
        records = records[:limit]

    pending = [r for r in records if r.get("image", "") not in done]
    if not pending:
        print(f"  All {len(done)} records already extracted — nothing to do.")
        return out_path

    print(f"  Extracting elements from {len(pending)} records with {model} ...")

    errors = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(pending):
            image = rec.get("image", "")
            recovery_text = get_field_text(rec, "recovery_considerations") or ""
            user_prompt = build_extract_prompt(recovery_text)
            parsed, raw, _elapsed = call_ollama(_SYSTEM_PROMPT, user_prompt, model, url)
            result = parse_extract_result(parsed)
            output_rec = build_output_record(image, result, raw_response=raw)
            out_f.write(json.dumps(output_rec) + "\n")
            if not result["parse_ok"]:
                errors += 1
                print(f"  PARSE_FAIL [{i + 1}/{len(pending)}] {image[-50:]}")

    print(f"\n  Done. extracted={len(pending)}  skipped={len(done)}  parse_errors={errors}")
    print(f"  Output -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# vLLM backend (cluster runs -- no Ollama available)
# ---------------------------------------------------------------------------

def _run_vllm_batch(user_prompts: list, model_dir: str, tp_size: int,
                     max_model_len: int, max_tokens: int, prefill: str = None,
                     guided_json: bool = False, temperature: float = 0.1) -> list:
    """Load the model once and extract elements for every prompt in one batch
    pass. prefill=None lets the model think normally (its <think> block gets
    stripped by clean_and_parse_json downstream); pass a non-empty prefill
    (e.g. an empty think block) to force-skip reasoning instead.
    guided_json=True constrains output to _ELEMENTS_JSON_SCHEMA at the
    token level -- only safe for non-reasoning models, since it forces valid
    JSON from the very first token and would suppress a <think> block
    entirely. That same lack of working-space also makes low-temperature
    guided decoding prone to collapsing onto the "safest" completion
    (observed: 20/20 records returned {"elements": []} even for inputs
    known to have real extractable content) -- guided_json configs should
    use a higher temperature than reasoning configs to counteract this."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"  [vLLM] Loading model from {model_dir} (tp={tp_size}, max_model_len={max_model_len}) ...")
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp_size,
        dtype="auto",
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    # temperature + repetition_penalty guard against greedy decoding getting
    # stuck in a degenerate loop (observed: "tug", "tug", "tug", ... repeated
    # until max_tokens was exhausted) -- temperature=0 (used elsewhere in
    # this codebase for reproducibility) has zero randomness to break out of
    # a repetitive rut once it starts one. This is a deliberate exception to
    # the temperature=0 convention, traded for robustness; Stage 1
    # extraction is not expected to be bit-for-bit reproducible as a result.
    #
    # repetition_penalty is deliberately skipped under guided_json: a
    # runaway text loop is structurally impossible once output is
    # schema-constrained, so it has nothing to protect against there -- but
    # it can actively punish the model for reusing the same structural JSON
    # tokens (quotes, commas) that every additional array element mechanically
    # requires. Observed: qwen25_72b returned {"elements": []} deterministically
    # (byte-for-byte identical across 20 different inputs, unaffected by
    # raising temperature 0.1->0.4) with repetition_penalty on -- consistent
    # with the penalty making the empty array the "cheapest" valid completion
    # regardless of what the input actually contains.
    sampling_kwargs = dict(temperature=temperature, max_tokens=max_tokens)
    if guided_json:
        from vllm.sampling_params import GuidedDecodingParams
        sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=_ELEMENTS_JSON_SCHEMA)
    else:
        sampling_kwargs["repetition_penalty"] = 1.15
    params = SamplingParams(**sampling_kwargs)

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    raw_prompts = []
    for up in user_prompts:
        msgs = [{"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": up}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        raw_prompts.append(text + (prefill or ""))

    print(f"  [vLLM] Extracting from {len(raw_prompts)} records ...")
    outputs = llm.generate(raw_prompts, sampling_params=params)
    return [o.outputs[0].text for o in outputs]


def run_vllm(input_path: Path, out_path: Path, model_dir: str, tp_size: int = 1,
             model_key: str = "qwen25_72b", limit=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _resume_done_images(out_path)

    records = load_run(input_path)
    if limit:
        records = records[:limit]

    pending = [r for r in records if r.get("image", "") not in done]
    if not pending:
        print(f"  All {len(done)} records already extracted — nothing to do.")
        return out_path

    images = [r.get("image", "") for r in pending]
    user_prompts = [
        build_extract_prompt(get_field_text(r, "recovery_considerations") or "")
        for r in pending
    ]

    cfg = _VLLM_MODEL_CONFIG[model_key]
    raw_responses = _run_vllm_batch(
        user_prompts, model_dir, tp_size,
        max_model_len=cfg["max_model_len"], max_tokens=cfg["max_tokens"], prefill=cfg["prefill"],
        guided_json=cfg.get("guided_json", False), temperature=cfg.get("temperature", 0.1),
    )

    errors = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, (image, raw) in enumerate(zip(images, raw_responses)):
            raw = raw if isinstance(raw, str) else ""
            parsed = clean_and_parse_json(raw)
            result = parse_extract_result(parsed)
            output_rec = build_output_record(image, result, raw_response=raw)
            out_f.write(json.dumps(output_rec) + "\n")
            if not result["parse_ok"]:
                errors += 1
                print(f"  PARSE_FAIL [{i + 1}/{len(pending)}] {image[-50:]}")

    print(f"\n  Done. extracted={len(pending)}  skipped={len(done)}  parse_errors={errors}")
    print(f"  Output -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Stage 1: extract salvage elements from recovery plans (Pipeline 6)"
    )
    ap.add_argument("--run", help="Run name; resolves --input/--out defaults if not given explicitly")
    ap.add_argument("--input", type=Path, help="Full-answer inference JSONL (default: derived from --run; "
                                                "if omitted and no full-answer file exists, separated-into-parts "
                                                "shards for --run are auto-combined first)")
    ap.add_argument("--out", type=Path, help="Output checkpoint JSONL (default: derived from --run)")
    ap.add_argument("--backend", choices=["ollama", "vllm"], default="ollama",
                    help="'ollama' for local dev (needs a running Ollama server); "
                         "'vllm' for cluster runs (no Ollama there -- loads a local HF checkpoint directly)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model (--backend ollama only)")
    ap.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama chat endpoint (--backend ollama only)")
    ap.add_argument("--model-dir", help="Path to HF checkpoint dir (--backend vllm only, required)")
    ap.add_argument("--model-key", default="qwen25_72b", choices=list(_VLLM_MODEL_CONFIG),
                    help="vLLM model config key (--backend vllm only). qwen25_72b (default) is "
                         "recommended -- not a reasoning model, so guided JSON decoding applies "
                         "cleanly and structurally rules out malformed-JSON failures. deepseek_r1 "
                         "is kept as a fallback but hit repeated reasoning-related failure modes "
                         "(rule recitation, repetition loops) in practice.")
    ap.add_argument("--tp", type=int, default=1, help="Tensor-parallel size (--backend vllm only)")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N records (smoke test)")
    args = ap.parse_args()

    if args.input is None and args.run is None:
        ap.error("Either --run or --input must be given.")
    run_name = args.run or args.input.stem
    input_path = resolve_input_path(run_name, args.input, RESULTS_IN, paths.run_dir(run_name))
    out_path = args.out or paths.raw_elements_path(run_name)

    if args.backend == "vllm":
        if not args.model_dir:
            ap.error("--backend vllm requires --model-dir")
        run_vllm(input_path, out_path, args.model_dir, args.tp, args.model_key, args.limit)
    else:
        run(input_path, out_path, args.model, args.url, args.limit)


if __name__ == "__main__":
    main()
