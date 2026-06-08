#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
source "${project_dir}/scripts/common_env.sh"
cd "${PROJECT_ROOT}"
log_dir="${script_dir}/logs"
mkdir -p "${log_dir}"

logfile="${log_dir}/$(date +%Y%m%d_%H%M%S)_sft.log"
exec > >(tee -a "${logfile}") 2>&1
echo "[INFO] Logging to ${logfile}"

SWIFT_BIN="${SWIFT_BIN:-swift}"
if ! command -v "${SWIFT_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] swift command not found. Activate the ms-swift environment first, or set SWIFT_BIN=/path/to/swift." >&2
  exit 127
fi

MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen3-VL-8B-Instruct}"
DATASET_JSONL="${DATASET_JSONL:-${DATA_ROOT}/skill3d_sft_messages.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/sft_skill3d_qwen3_vl}"
SPECIAL_TOKENS_FILE="${SPECIAL_TOKENS_FILE:-${PROJECT_ROOT}/train/special_tokens/skill3d_agentic_spatial.txt}"

# The dataset rows already contain a role=system message. Keep SYSTEM_PROMPT
# empty by default to avoid duplicated or conflicting SFT instructions.
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -s "${DATASET_JSONL}" ]]; then
  echo "[ERROR] DATASET_JSONL not found or empty: ${DATASET_JSONL}" >&2
  exit 1
fi
if [[ ! -s "${SPECIAL_TOKENS_FILE}" ]]; then
  echo "[ERROR] SPECIAL_TOKENS_FILE not found or empty: ${SPECIAL_TOKENS_FILE}" >&2
  exit 1
fi

TRAIN_TYPE="${TRAIN_TYPE:-full}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-1}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-1}"
GRAD_ACC="${GRAD_ACC:-1}"
LR="${LR:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MAX_PIXELS="${MAX_PIXELS:-131072}"
SAVE_STEPS="${SAVE_STEPS:-25}"
EVAL_STEPS="${EVAL_STEPS:-25}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
REPORT_TO="${REPORT_TO:-none}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-false}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
FREEZE_VIT="${FREEZE_VIT:-true}"
DEEPSPEED="${DEEPSPEED:-zero2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29610}"

# Keep distributed comms on local loopback, matching the current machine setup.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
deepspeed_args=()
case "${DEEPSPEED,,}" in
  ""|"0"|"false"|"none"|"off"|"no")
    echo "[INFO] DeepSpeed: disabled"
    ;;
  *)
    deepspeed_args=(--deepspeed "${DEEPSPEED}")
    ;;
esac

system_args=()
if [[ -n "${SYSTEM_PROMPT}" ]]; then
  if [[ ! -s "${SYSTEM_PROMPT}" ]]; then
    echo "[ERROR] SYSTEM_PROMPT was set but file does not exist: ${SYSTEM_PROMPT}" >&2
    exit 1
  fi
  system_args=(--system "${SYSTEM_PROMPT}")
fi

echo "[INFO] Swift bin: ${SWIFT_BIN}"
echo "[INFO] Model: ${MODEL_PATH}"
echo "[INFO] Dataset: ${DATASET_JSONL}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Train type: ${TRAIN_TYPE}"
echo "[INFO] Epochs: ${NUM_EPOCHS}"
echo "[INFO] Learning rate: ${LR}"
echo "[INFO] Save steps: ${SAVE_STEPS}"
echo "[INFO] Attention impl: ${ATTN_IMPL}"
echo "[INFO] New special tokens: ${SPECIAL_TOKENS_FILE}"
echo "[INFO] DeepSpeed setting: ${DEEPSPEED}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "[INFO] MASTER_ADDR: ${MASTER_ADDR}"
echo "[INFO] MASTER_PORT: ${MASTER_PORT}"
echo "[INFO] SYSTEM_PROMPT: ${SYSTEM_PROMPT:-<dataset role=system only>}"
echo "[INFO] XDG_CACHE_HOME: ${XDG_CACHE_HOME}"
echo "[INFO] TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"

MAX_PIXELS="${MAX_PIXELS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
"${SWIFT_BIN}" sft \
  --model "${MODEL_PATH}" \
  --train_type "${TRAIN_TYPE}" \
  --attn_impl "${ATTN_IMPL}" \
  --new_special_tokens "${SPECIAL_TOKENS_FILE}" \
  --dataset "${DATASET_JSONL}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS}" \
  --learning_rate "${LR}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --dataloader_num_workers "${NUM_WORKERS}" \
  --load_from_cache_file "${LOAD_FROM_CACHE_FILE}" \
  --freeze_vit "${FREEZE_VIT}" \
  --model_author skill3d \
  --model_name skill3d-qwen3-8b-vsi-gpt54-format-tool-skill \
  --report_to "${REPORT_TO}" \
  "${system_args[@]}" \
  "${deepspeed_args[@]}"
