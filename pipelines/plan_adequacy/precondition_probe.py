"""
precondition_probe.py -- text-only sibling of run_inference.py, for Phase 2/3
of the revised Direction-4 protocol (see reports/p9/precondition_position_phase1.txt
and pipelines/plan_adequacy/precondition_probe_prompts.py for how the input
JSONLs are built).

pipelines/plan_coherence/improved/inference/run_inference.py loads the
exact same model to generate the 330-plan corpus this probe is
diagnosing; the model-loading block below is copied verbatim from it so
Phase 2/3 queries the identical weights and runtime the same way. This
script lives under plan_adequacy/ instead (this is P9's own diagnostic
work, not plan_coherence's), but still needs run_inference.py's
container -- castor_qwen.sif, not P9's usual castor_judge.sif -- since
that is the only one with the Qwen3-VL stack (transformers>=4.51 +
qwen-vl-utils) installed. See submit_precondition_probe.sh for how that
cross-pipeline container dependency is wired.

The only real change from run_inference.py is the encode step: no image
content block, since these are isolated fact-check questions (Phase 2) or
plan continuations from given prefix text (Phase 3), never a photo.

Decoding is forced greedy (do_sample=False) regardless of any config file --
the whole Phase 1-3 design assumes deterministic decoding (see Phase 1's
own docstring on why: "n=1 per cell is the model's actual answer, not a
noisy draw of it" applies here too), so this script does not read
config.yaml's temperature at all.

Usage (inside castor_qwen.sif, same container run_inference.py uses):
    python precondition_probe.py \\
        --vlm-dir /data/$USER/qwen3vl-8b \\
        --input   reports/p9/phase2_candidates.jsonl \\
        --output  reports/p9/phase2_responses.jsonl \\
        --max-new-tokens 200

    python precondition_probe.py \\
        --vlm-dir /data/$USER/qwen3vl-8b \\
        --input   reports/p9/phase3_items.jsonl \\
        --output  reports/p9/phase3_responses.jsonl \\
        --max-new-tokens 400 \\
        --field-prompt continuation_prompt --field-context prefix_plan_text
"""
import argparse
import json
import time
from pathlib import Path

import torch

try:
    from transformers import Qwen3VLForConditionalGeneration as _QwenVL
    _QWEN_CLASS = "Qwen3VLForConditionalGeneration"
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _QwenVL
        _QWEN_CLASS = "Qwen2_5_VLForConditionalGeneration"
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as _QwenVL
        _QWEN_CLASS = "Qwen2VLForConditionalGeneration"

from transformers import AutoProcessor


def load_rows(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_text_prompt(row: dict, field_prompt: str, field_context: str | None) -> str:
    """Phase 2: field_prompt='question', field_context=None -- the question
    stands alone. Phase 3: field_prompt='continuation_prompt',
    field_context='prefix_plan_text' -- the prefix plan text (filled in by a
    person from a clean base plan, see precondition_probe_prompts.py) comes
    first, then the continuation instruction."""
    prompt = row[field_prompt]
    if field_context:
        context = row.get(field_context)
        if not context:
            raise ValueError(
                f"row for relationship={row.get('relationship')!r} k={row.get('k')!r} "
                f"has no {field_context!r} filled in -- fill prefix_plan_text for every "
                f"row before running Phase 3, this is not optional")
        prompt = f"{context}\n\n{prompt}"
    return prompt


def main():
    ap = argparse.ArgumentParser(description="Phase 2/3 text-only probe (Direction 4)")
    ap.add_argument("--vlm-dir", required=True, help="model weights dir, e.g. /data/$USER/qwen3vl-8b")
    ap.add_argument("--input", required=True, help="phase2_candidates.jsonl or phase3_items.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--field-prompt", default="question")
    ap.add_argument("--field-context", default=None,
                    help="Phase 3 only: set to 'prefix_plan_text'")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    rows = load_rows(Path(args.input))
    done_ids = set()
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add((r.get("relationship"), r.get("k"), r.get("image")))
                except Exception:
                    pass
        print(f"Resuming -- {len(done_ids)} already done.")

    print(f"Qwen class : {_QWEN_CLASS}")
    print(f"Loading VLM from {args.vlm_dir} ...")
    model = _QwenVL.from_pretrained(
        args.vlm_dir, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.vlm_dir)

    with open(out_path, "a", encoding="utf-8") as fout:
        for row in rows:
            row_id = (row.get("relationship"), row.get("k"), row.get("image"))
            if row_id in done_ids:
                continue

            prompt_text = build_text_prompt(row, args.field_prompt, args.field_context)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            # No process_vision_info call -- there is no image in this message.
            inputs = processor(text=[text_input], padding=True,
                               return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            elapsed = time.time() - t0

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[0][input_len:]
            answer = processor.decode(output_ids, skip_special_tokens=True).strip()

            entry = {**row, "prompt_sent": prompt_text, "response": answer,
                     "timing": round(elapsed, 2), "model_id": args.vlm_dir}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{row_id} -- {elapsed:.1f}s")

    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
