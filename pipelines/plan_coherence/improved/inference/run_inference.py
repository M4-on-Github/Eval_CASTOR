"""
run_inference.py — Qwen3-VL 8B inference for the improved CASTOR experiment.

For each condition (standard_v2 / control_v2 / ablation_v2):
  - Loads prompt template from prompts/prompt_{condition}.txt
  - Runs all images through Qwen3-VL 8B (greedy, max_new_tokens=1024)
  - Writes one JSONL per condition:
      results/answers_qwen3vl8b_baseline_{condition}_improved.jsonl

Usage (single-node, run inside castor_qwen.sif):
    python improved/inference/run_inference.py --config improved/config.yaml

Records per JSONL line:
    question_id, image, prompt, text, model_id, model_tag, method,
    condition, timing, run_name
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml
import torch
from PIL import Image

# Qwen class: try Qwen3-VL first (transformers>=4.51), fall back to older variants
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

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "qwen-vl-utils not found. Run inside castor_qwen.sif "
        "(built by QWEN/CASTOR/build_container.sh)."
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand(s: str) -> str:
    """Expand ${USER}, ${HOME}, $USER, $HOME in a path string."""
    return os.path.expandvars(s)


def load_prompt(prompts_dir: Path, condition: str) -> str:
    p = prompts_dir / f"prompt_{condition}.txt"
    return p.read_text().strip()


def gather_images(images_dir: Path) -> list[dict]:
    """Walk images_dir, return list of {image_path, gt_state, question_id}."""
    state_dirs = ["aground", "capsized", "sunken", "on_fire"]
    records = []
    for state in state_dirs:
        state_path = images_dir / state
        if not state_path.is_dir():
            continue
        _EXTS = {".jpg", ".jpeg", ".png"}
        imgs = sorted(p for p in state_path.iterdir() if p.suffix.lower() in _EXTS)
        for img in imgs:
            records.append({
                "question_id": f"{state}/{img.name}",
                "image": str(img.relative_to(images_dir)),
                "abs_path": str(img),
                "gt_state": state,
            })
    return records


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_condition(
    model,
    processor,
    images_dir: Path,
    prompts_dir: Path,
    results_dir: Path,
    condition: str,
    cfg: dict,
):
    prompt_text = load_prompt(prompts_dir, condition)
    records = gather_images(images_dir)
    out_path = results_dir / f"answers_qwen3vl8b_baseline_{condition}_improved.jsonl"

    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_id"])
                except Exception:
                    pass
        print(f"[{condition}] Resuming — {len(done_ids)} already done, "
              f"{len(records) - len(done_ids)} remaining.")

    max_new_tokens = cfg["inference"]["max_new_tokens"]
    temperature = cfg["inference"]["temperature"]
    model_tag = cfg["models"]["vlm_tag"]
    model_id = cfg["models"]["vlm_dir"]

    with open(out_path, "a") as fout:
        for rec in records:
            qid = rec["question_id"]
            if qid in done_ids:
                continue

            t0 = time.time()

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": rec["abs_path"],
                            "max_pixels": 1280 * 28 * 28,  # cap: ~1M px, prevents OOM on high-res PNGs
                        },
                        {"type": "text",  "text": prompt_text},
                    ],
                }
            ]

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with torch.no_grad():
                generated_ids = model.generate(**inputs, **gen_kwargs)

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[0][input_len:]
            answer = processor.decode(output_ids, skip_special_tokens=True).strip()

            elapsed = time.time() - t0

            entry = {
                "question_id": qid,
                "image":       rec["image"],
                "prompt":      prompt_text,
                "text":        answer,
                "model_id":    model_id,
                "model_tag":   model_tag,
                "method":      cfg["inference"]["method_tag"],
                "condition":   condition,
                "timing":      round(elapsed, 2),
                "run_name":    f"castor_improved_{condition}",
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"[{condition}] {qid} — {elapsed:.1f}s")

    print(f"[{condition}] Done → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="improved/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pipeline_dir   = Path(expand(cfg["paths"]["pipeline_dir"]))
    images_dir     = Path(expand(cfg["paths"]["images_dir"]))
    user_models_dir = Path(expand(cfg["paths"]["user_models_dir"]))
    prompts_dir    = pipeline_dir / "prompts"
    results_dir    = pipeline_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    vlm_path = user_models_dir / cfg["models"]["vlm_dir"]
    print(f"Qwen class : {_QWEN_CLASS}")
    print(f"Loading VLM from {vlm_path} ...")
    model = _QwenVL.from_pretrained(
        str(vlm_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(vlm_path))

    for condition in cfg["inference"]["conditions"]:
        print(f"\n{'='*60}\nCondition: {condition}\n{'='*60}")
        run_condition(
            model=model,
            processor=processor,
            images_dir=images_dir,
            prompts_dir=prompts_dir,
            results_dir=results_dir,
            condition=condition,
            cfg=cfg,
        )

    print("\nAll conditions complete.")


if __name__ == "__main__":
    main()
