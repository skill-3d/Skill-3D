#!/usr/bin/env bash
set -euo pipefail

# Skill-3D local Qwen/vLLM inference on held-out test splits.
#
# Start vLLM first, for example:
#   MODEL_PATH=/path/to/checkpoint SERVED_MODEL_NAME=Skill-3D-SFT-4B GPU_DEVICE=0 PORT=8001 bash scripts/vllm_start.sh
#
# Then run:
#   BENCHMARK=vsi MAX_SAMPLES=20 bash scripts/run_skill3d_qwen_sft_inference.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "$PROJECT_ROOT"

USE_MODEL="${USE_MODEL:-Skill-3D-GRPO-4B}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8001/v1}"
BENCHMARK="${BENCHMARK:-vsi}"
case "$BENCHMARK" in
  vsi)
    DEFAULT_DATA_PATH="dataset/skill3d_splits/vsi_test.jsonl"
    ;;
  mmsi_pr)
    DEFAULT_DATA_PATH="dataset/skill3d_splits/mmsi_pr_test.jsonl"
    ;;
  cv3d)
    DEFAULT_DATA_PATH="dataset/skill3d_splits/cv3d_test.jsonl"
    ;;
  blink_multiview)
    DEFAULT_DATA_PATH="dataset/skill3d_splits/blink_multiview_test.jsonl"
    ;;
  all)
    DEFAULT_DATA_PATH="dataset/skill3d_splits/all_test.jsonl"
    ;;
  *)
    DEFAULT_DATA_PATH="dataset/skill3d_splits/${BENCHMARK}_test.jsonl"
    ;;
esac

DATA_PATH="${DATA_PATH:-$DEFAULT_DATA_PATH}"
IMAGE_BASE_PATH="${IMAGE_BASE_PATH:-dataset}"
TASK="${TASK:-skill3d-${BENCHMARK}-qwen-sft-test}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
MAX_SAMPLES="${MAX_SAMPLES-}"
case "${MAX_SAMPLES,,}" in
  all|full|none)
    MAX_SAMPLES=""
    ;;
esac
MAX_WORKERS="${MAX_WORKERS:-8}"
PROMPT_STYLE="${PROMPT_STYLE:-skills}"
MEMORY_MODE="${MEMORY_MODE:-hybrid}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-hybrid}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_ROOT_BASE}/skill3d_qwen_sft_inference}"
RESUME="${RESUME:-0}"
RESUME_RETRY_FAILED="${RESUME_RETRY_FAILED:-0}"

BENCHMARK_MEMORY_ROOT="${BENCHMARK_MEMORY_ROOT:-${MEMORY_ROOT}}"
export SKILL3D_SKILL_STORAGE_PATH="${SKILL3D_SKILL_STORAGE_PATH:-${SPAGENT_SKILL_STORAGE_PATH:-${BENCHMARK_MEMORY_ROOT}/learned_skills.json}}"
export SPAGENT_SKILL_STORAGE_PATH="${SPAGENT_SKILL_STORAGE_PATH:-${SKILL3D_SKILL_STORAGE_PATH}}"
export SKILL3D_HIERARCHICAL_MEMORY_DIR="${SKILL3D_HIERARCHICAL_MEMORY_DIR:-${SPAGENT_HIERARCHICAL_MEMORY_DIR:-${BENCHMARK_MEMORY_ROOT}/memory}}"
export SPAGENT_HIERARCHICAL_MEMORY_DIR="${SPAGENT_HIERARCHICAL_MEMORY_DIR:-${SKILL3D_HIERARCHICAL_MEMORY_DIR}}"
export SPAGENT_SKILL_PROMPT_MODE="${SPAGENT_SKILL_PROMPT_MODE:-compact}"
export SPAGENT_SUPPRESS_USER_FORMAT_PROMPT="${SPAGENT_SUPPRESS_USER_FORMAT_PROMPT:-1}"
export SPAGENT_ALLOW_MODEL_SKILL_CHOICE="${SPAGENT_ALLOW_MODEL_SKILL_CHOICE:-1}"
export SPAGENT_AUTO_SELECT_SKILLS="${SPAGENT_AUTO_SELECT_SKILLS:-0}"
export SPAGENT_FORCE_SKILL_TOOL_CALL="${SPAGENT_FORCE_SKILL_TOOL_CALL:-1}"
export SPAGENT_AUTO_SELECT_SKILL_TOP_K="${SPAGENT_AUTO_SELECT_SKILL_TOP_K:-1}"
export SPAGENT_ENFORCE_MUST_TRY_TOOLS="${SPAGENT_ENFORCE_MUST_TRY_TOOLS:-0}"
export SPAGENT_STOP_ON_ANSWER="${SPAGENT_STOP_ON_ANSWER:-1}"
export SPAGENT_FILTER_INVALID_SKILL3D_SFT="${SPAGENT_FILTER_INVALID_SKILL3D_SFT:-1}"

export OPENAI_API_BASE="$LLM_BASE_URL"
export OPENAI_BASE_URL="$LLM_BASE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-0}"
export SPAGENT_VLLM_SKIP_SPECIAL_TOKENS="${SPAGENT_VLLM_SKIP_SPECIAL_TOKENS:-0}"
export SPAGENT_QWEN_TEMPERATURE="${SPAGENT_QWEN_TEMPERATURE:-0.7}"
LLM_HOST="$(printf '%s\n' "$LLM_BASE_URL" | sed -E 's#^https?://([^/:]+).*#\1#')"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="127.0.0.1,localhost,$LLM_HOST"
export no_proxy="$NO_PROXY"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${USE_MODEL}-Skill3D-${BENCHMARK}-QwenSFT-${TS}"
if [[ "$RESUME" == "1" ]]; then
  if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    DATA_OUTPUT_DIR="$RESUME_RUN_DIR"
  else
    LATEST_RUN_DIR="$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "${USE_MODEL}-Skill3D-${BENCHMARK}-QwenSFT-*" 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "$LATEST_RUN_DIR" ]]; then
      DATA_OUTPUT_DIR="$LATEST_RUN_DIR"
    else
      DATA_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
    fi
  fi
  RUN_NAME="$(basename "$DATA_OUTPUT_DIR")"
else
  DATA_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
fi

LOG_DIR="logs/skill3d-qwen-sft-inference/${USE_MODEL}"
RETRIEVAL_TRACE_PATH="${DATA_OUTPUT_DIR}/retrieval_trace.jsonl"
CHECKPOINT_PATH="${DATA_OUTPUT_DIR}/inference_checkpoint.jsonl"

mkdir -p "$DATA_OUTPUT_DIR" "$LOG_DIR"
LOGFILE="${LOG_DIR}/${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Logging to: $LOGFILE"
echo "[INFO] Model: $USE_MODEL"
echo "[INFO] API base: $OPENAI_BASE_URL"
echo "[INFO] Benchmark: $BENCHMARK"
echo "[INFO] Data: $DATA_PATH"
echo "[INFO] Output dir: $DATA_OUTPUT_DIR"
echo "[INFO] Retrieval model: ${SKILL3D_RETRIEVAL_MODEL}"
echo "[INFO] Skill prompt mode: ${SPAGENT_SKILL_PROMPT_MODE}"
echo "[INFO] Suppress user format prompt: ${SPAGENT_SUPPRESS_USER_FORMAT_PROMPT}"
echo "[INFO] Allow model skill choice: ${SPAGENT_ALLOW_MODEL_SKILL_CHOICE}"
echo "[INFO] Auto skill selection: ${SPAGENT_AUTO_SELECT_SKILLS}"
echo "[INFO] Force skill+tool before answer: ${SPAGENT_FORCE_SKILL_TOOL_CALL}"
echo "[INFO] Stop on answer: ${SPAGENT_STOP_ON_ANSWER}"
echo "[INFO] Qwen temperature: ${SPAGENT_QWEN_TEMPERATURE}"
echo "[INFO] Enforce must-try tools: $SPAGENT_ENFORCE_MUST_TRY_TOOLS"

if ! curl -sS -m 8 --noproxy "*" "$OPENAI_BASE_URL/models" >/dev/null; then
  echo "[ERROR] Cannot connect to $OPENAI_BASE_URL/models"
  echo "[HINT] Start vLLM first, or override LLM_BASE_URL."
  exit 1
fi

CMD=(
  python examples/evaluation/evaluate_img_alltools.py
  --data_path "$DATA_PATH"
  --image_base_path "$IMAGE_BASE_PATH"
  --model "$USE_MODEL"
  --model_backend qwen_vllm
  --max_iterations "$MAX_ITERATIONS"
  --max_workers "$MAX_WORKERS"
  --task "$TASK"
  --prompt_style "$PROMPT_STYLE"
  --memory_mode "$MEMORY_MODE"
  --retrieval_mode "$RETRIEVAL_MODE"
  --retrieval_top_k "$RETRIEVAL_TOP_K"
  --retrieval_trace_path "$RETRIEVAL_TRACE_PATH"
  --skill_storage_path "$SPAGENT_SKILL_STORAGE_PATH"
  --checkpoint_path "$CHECKPOINT_PATH"
  --disable_skill_update
)

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi
if [[ "$RESUME_RETRY_FAILED" == "1" ]]; then
  CMD+=(--resume_retry_failed)
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max_samples "$MAX_SAMPLES")
fi
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[INFO] Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[INFO] Done."
