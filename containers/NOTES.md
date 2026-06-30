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

Error: `KeyError: 'gpt_oss'` — transformers does not recognize `model_type: gpt_oss`
and the model repo does not register it via `auto_map` for trust_remote_code.

vLLM 0.5.5 cannot load this model. Options:
1. Rebuild the container with a newer vLLM version that adds gpt_oss support.
2. Replace the anchor judge with a different model (e.g., `meta-llama/Llama-3.1-70B-Instruct`).

Until resolved, `gptoss_120b` is skipped in `build_judge_container.sh` and the
panel runs with two judges (Qwen + DeepSeek). Aggregation still works — partial
scores are handled by `compute_consensus()` (nulls excluded from mean).
