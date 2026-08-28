#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:=/data/magnus/models/Qwen3.8-27B-FP8-20260828}"
: "${DRAFT_DIR:=/data/magnus/models/Qwen3.8-27B-DFlash2-20260828}"
: "${MAX_MODEL_LEN:=16384}"
: "${MAX_NUM_SEQS:=1}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${SPEC_TOKENS:=3}"
: "${ENABLE_DFLASH2:=0}"
: "${DTYPE:=auto}"
: "${KV_CACHE_DTYPE:=fp8}"

if [[ -z "${MAGNUS_PORT:-}" ]]; then
  echo "FATAL: MAGNUS_PORT is required" >&2
  exit 2
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "FATAL: model config not found: $MODEL_DIR/config.json" >&2
  exit 2
fi
if [[ "$ENABLE_DFLASH2" == "1" && ! -f "$DRAFT_DIR/config.json" ]]; then
  echo "FATAL: DFlash2 config not found: $DRAFT_DIR/config.json" >&2
  exit 2
fi

args=(
  vllm serve "$MODEL_DIR"
  --host 0.0.0.0
  --port "$MAGNUS_PORT"
  --served-model-name Qwen3.8-27B
  --tensor-parallel-size 1
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens 8192
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --enable-prefix-caching
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --trust-remote-code
)

if [[ "$ENABLE_DFLASH2" == "1" ]]; then
  args+=(--speculative-config "{\"method\":\"draft_model\",\"model\":\"$DRAFT_DIR\",\"num_speculative_tokens\":$SPEC_TOKENS}")
fi

echo "[qwen38] starting single-GPU low-concurrency API"
echo "[qwen38] model=$MODEL_DIR max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS"
echo "[qwen38] dflash2=$ENABLE_DFLASH2 spec_tokens=$SPEC_TOKENS kv_cache=$KV_CACHE_DTYPE dtype=$DTYPE"
exec "${args[@]}"
