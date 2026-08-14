"""
AutoAWQ 4-bit quantization helper.

Called by containers/quantize_job.sh (inside the Apptainer container) to
convert an FP16 HuggingFace model directory to W4A16 AWQ format.

Usage:
  python3 quantize_model.py --src /data/$USER/phi-4-reasoning-fp16 \
                             --dst /data/$USER/phi-4-reasoning-awq
"""

import argparse
import sys
from pathlib import Path


class AWQQuantization:
    """4-bit AWQ quantization, run once per judge model before P5 can use it.

    The judge panel loads three 30B+ models. At FP16 they do not fit on the
    allocated GPUs, so each is quantized to 4-bit first — that is what makes a
    three-model panel possible on one node at all.

    A non-empty destination is SKIPPED rather than overwritten. Quantization
    takes tens of minutes and is deterministic, so re-doing it wastes an
    allocation; more importantly, overwriting mid-run would corrupt weights a
    running job has memory-mapped.

    group_size 128 is AutoAWQ's default and the value the judge configs assume.
    Changing it changes the weights, so a model quantized at a different size
    is not interchangeable with one already on disk.
    """


def main():
    ap = argparse.ArgumentParser(description="AutoAWQ 4-bit quantization.")
    ap.add_argument("--src", required=True, type=Path,
                    help="Source FP16 model directory")
    ap.add_argument("--dst", required=True, type=Path,
                    help="Destination directory for AWQ-quantized weights")
    ap.add_argument("--group-size", type=int, default=128,
                    help="AWQ group size (default: 128)")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"ERROR: source directory not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    if args.dst.exists() and any(args.dst.iterdir()):
        print(f"Destination already exists and is non-empty — skipping.")
        print(f"  {args.dst}")
        return

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("ERROR: autoawq not installed. Run: pip install autoawq", file=sys.stderr)
        sys.exit(1)

    from transformers import AutoTokenizer

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": 4,
        "version": "GEMM",
    }

    print(f"Loading FP16 model from {args.src} ...")
    model = AutoAWQForCausalLM.from_pretrained(
        str(args.src),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.src),
        trust_remote_code=True,
    )

    print(f"Quantizing (W4A16, group_size={args.group_size}) ...")
    model.quantize(tokenizer, quant_config=quant_config)

    args.dst.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {args.dst} ...")
    model.save_quantized(str(args.dst))
    tokenizer.save_pretrained(str(args.dst))

    print(f"\nDone. AWQ model saved to: {args.dst}")


if __name__ == "__main__":
    main()
