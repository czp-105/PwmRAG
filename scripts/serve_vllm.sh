#!/usr/bin/env bash
set -euo pipefail

case "${1:-victim}" in
  victim) port=8000 ;;
  attacker) port=8001 ;;
  *) echo "usage: $0 [victim|attacker] [model]" >&2; exit 2 ;;
esac
model="${2:-Qwen/Qwen3-8B}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" HF_HUB_OFFLINE=1 \
vllm serve "$model" \
  --served-model-name "$model" \
  --host 127.0.0.1 \
  --port "$port" \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --generation-config vllm
