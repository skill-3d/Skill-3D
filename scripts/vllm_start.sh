#!/bin/bash
set -euo pipefail

# Usage examples:
#   MODEL_PATH=/path/to/checkpoint SERVED_MODEL_NAME=Skill-3D-GRPO-4B GPU_DEVICE=0 PORT=8001 bash scripts/vllm_start.sh
#   MODEL_PATH=model_links/Skill-3D-SFT-4B SERVED_MODEL_NAME=Skill-3D-SFT-4B GPU_DEVICE=0 PORT=8001 bash scripts/vllm_start.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "${PROJECT_ROOT}"

MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Skill-3D-GRPO-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Skill-3D-GRPO-4B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
GPU_DEVICE="${GPU_DEVICE:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

# Save logs under logs/vLLM/<model_name>/.
LOG_DIR="${LOG_ROOT}/vLLM/${SERVED_MODEL_NAME}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Logging to $LOG_FILE"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "[INFO] HOST=$HOST PORT=$PORT"
echo "[INFO] GPU_DEVICE=$GPU_DEVICE TP=$TENSOR_PARALLEL_SIZE"
echo "[INFO] XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "[INFO] TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH" >&2
  echo "[HINT] Put a symlink under ${MODEL_ROOT}, or pass MODEL_PATH=/path/to/checkpoint." >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[ERROR] MODEL_PATH is not a full model checkpoint: $MODEL_PATH" >&2
  echo "[INFO] Expected config.json under MODEL_PATH." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"
