#!/usr/bin/env bash
set -euo pipefail

# Skill-3D GPT-5.4 skill-guided inference on held-out test splits.
#
# Usage:
#   BENCHMARK=vsi bash scripts/run_skill3d_gpt54_inference.sh
#   BENCHMARK=vsi MAX_SAMPLES=20 bash scripts/run_skill3d_gpt54_inference.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
cd "$PROJECT_ROOT"

USE_MODEL="${USE_MODEL:-Skill-3D-SFT-4B}"
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
TASK="${TASK:-skill3d-${BENCHMARK}-test}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_WORKERS="${MAX_WORKERS:-1}"
PROMPT_STYLE="${PROMPT_STYLE:-skills}"
MEMORY_MODE="${MEMORY_MODE:-hybrid}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-hybrid}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-6}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_ROOT_BASE}/skill3d_inference}"
RESUME="${RESUME:-0}"
RESUME_RETRY_FAILED="${RESUME_RETRY_FAILED:-0}"

BENCHMARK_MEMORY_ROOT="${BENCHMARK_MEMORY_ROOT:-${MEMORY_ROOT}}"
export SKILL3D_SKILL_STORAGE_PATH="${SKILL3D_SKILL_STORAGE_PATH:-${SPAGENT_SKILL_STORAGE_PATH:-${BENCHMARK_MEMORY_ROOT}/learned_skills.json}}"
export SPAGENT_SKILL_STORAGE_PATH="${SPAGENT_SKILL_STORAGE_PATH:-${SKILL3D_SKILL_STORAGE_PATH}}"
export SKILL3D_HIERARCHICAL_MEMORY_DIR="${SKILL3D_HIERARCHICAL_MEMORY_DIR:-${SPAGENT_HIERARCHICAL_MEMORY_DIR:-${BENCHMARK_MEMORY_ROOT}/memory}}"
export SPAGENT_HIERARCHICAL_MEMORY_DIR="${SPAGENT_HIERARCHICAL_MEMORY_DIR:-${SKILL3D_HIERARCHICAL_MEMORY_DIR}}"
export SPAGENT_AUTO_SELECT_SKILLS="${SPAGENT_AUTO_SELECT_SKILLS:-1}"
export SPAGENT_AUTO_SELECT_SKILL_TOP_K="${SPAGENT_AUTO_SELECT_SKILL_TOP_K:-1}"

export SPAGENT_FILTER_INVALID_SKILL3D_SFT="${SPAGENT_FILTER_INVALID_SKILL3D_SFT:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${USE_MODEL}-Skill3D-${BENCHMARK}-Inference-${TS}"
if [[ "$RESUME" == "1" ]]; then
  if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
    DATA_OUTPUT_DIR="$RESUME_RUN_DIR"
  else
    LATEST_RUN_DIR="$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name "${USE_MODEL}-Skill3D-${BENCHMARK}-Inference-*" 2>/dev/null | sort | tail -n 1 || true)"
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

LOG_DIR="logs/skill3d-inference/${USE_MODEL}"
RETRIEVAL_TRACE_PATH="${DATA_OUTPUT_DIR}/retrieval_trace.jsonl"
CHECKPOINT_PATH="${DATA_OUTPUT_DIR}/inference_checkpoint.jsonl"

mkdir -p "$DATA_OUTPUT_DIR" "$LOG_DIR"

LOGFILE="${LOG_DIR}/${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Logging to: $LOGFILE"
echo "[INFO] Model: $USE_MODEL"
echo "[INFO] Benchmark: $BENCHMARK"
echo "[INFO] Data: $DATA_PATH"
echo "[INFO] Skill storage path: $SPAGENT_SKILL_STORAGE_PATH"
echo "[INFO] Hierarchical memory dir: $SPAGENT_HIERARCHICAL_MEMORY_DIR"
echo "[INFO] Prompt style: $PROMPT_STYLE"
echo "[INFO] Memory mode: $MEMORY_MODE"
echo "[INFO] Retrieval mode: $RETRIEVAL_MODE"
echo "[INFO] Retrieval top-k: $RETRIEVAL_TOP_K"
echo "[INFO] Auto skill selection: $SPAGENT_AUTO_SELECT_SKILLS"
echo "[INFO] Auto skill selection top-k: $SPAGENT_AUTO_SELECT_SKILL_TOP_K"
echo "[INFO] Skill update: disabled for test inference"
echo "[INFO] Output dir: $DATA_OUTPUT_DIR"
echo "[INFO] Resume: $RESUME"
echo "[INFO] Checkpoint: $CHECKPOINT_PATH"

CMD=(
  python examples/evaluation/evaluate_img_alltools.py
  --data_path "$DATA_PATH"
  --image_base_path "$IMAGE_BASE_PATH"
  --model "$USE_MODEL"
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
