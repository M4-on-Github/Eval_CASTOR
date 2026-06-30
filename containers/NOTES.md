# Cluster Run Notes

## OOM — 70B models require AWQ quantization

Qwen2.5-72B and DeepSeek-R1-Distill-Llama-70B in bfloat16 require ~140 GB.
A single RTX 6000 Ada is 48 GB; two GPUs give 96 GB — still not enough for BF16.

**Fix applied (2026-06-30):** switched to AWQ 4-bit quantized variants (~40 GB each,
fit on 1 GPU). Model dirs and HF repo IDs updated accordingly.

| Model | HF repo | Local dir | VRAM |
|-------|---------|-----------|------|
| Qwen2.5 72B | `Qwen/Qwen2.5-72B-Instruct-AWQ` | `qwen25-72b-instruct-awq` | ~40 GB |
| DeepSeek-R1 70B | `cognitivecomputations/DeepSeek-R1-Distill-Llama-70B-AWQ` | `deepseek-r1-distill-llama-70b-awq` | ~40 GB |

## GPT-OSS 120B — unrecognized architecture in vLLM 0.5.5

Error: `KeyError: 'gpt_oss'` — transformers does not recognize `model_type: gpt_oss`.
Root cause: vLLM 0.5.5 bundles a transformers version that doesn't have `gpt_oss`
in its `CONFIG_MAPPING`, and the model repo apparently lacks an `auto_map` entry.

**Fix applied (2026-06-30):** bumped container base to `vllm/vllm-openai:v0.6.3`.
Rebuild the SIF with `build_judge_container.sh --force` and resubmit.
