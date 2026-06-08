#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
source "${project_dir}/scripts/common_env.sh"
cd "${PROJECT_ROOT}"
log_dir="${script_dir}/logs"
mkdir -p "${log_dir}"

logfile="${log_dir}/$(date +%Y%m%d_%H%M%S)_grpo_skill3d_agentic.log"
exec > >(tee -a "$logfile") 2>&1
echo "[INFO] Logging to $logfile"

SWIFT_BIN="${SWIFT_BIN:-swift}"
if ! command -v "${SWIFT_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] swift command not found. Activate the ms-swift environment first, or set SWIFT_BIN=/path/to/swift." >&2
  exit 127
fi

DEFAULT_SFT_OUTPUT_DIR="${DEFAULT_SFT_OUTPUT_DIR:-${MODEL_ROOT}/Skill-3D-SFT-4B}"
DEFAULT_MODEL_GLOB="${DEFAULT_MODEL_GLOB:-${DEFAULT_SFT_OUTPUT_DIR}/v*/checkpoint-*}"
MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(ls -td ${DEFAULT_MODEL_GLOB} 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="${DEFAULT_SFT_OUTPUT_DIR}"
fi
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/Skill-3D-GRPO-mixed1000.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/grpo_skill3d_sft4b}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-${PROJECT_ROOT}/train/system_prompt/system_prompt_skill3d_agentic.txt}"

export ROOT_IMAGE_DIR="${ROOT_IMAGE_DIR:-${DATA_ROOT}}"
export SPAGENT_TRAIN_NUM_FRAMES="${SPAGENT_TRAIN_NUM_FRAMES:-7}"
export SPAGENT_RL_MEMORY_MODE="${SPAGENT_RL_MEMORY_MODE:-hybrid}"
export SPAGENT_RL_RETRIEVAL_MODE="${SPAGENT_RL_RETRIEVAL_MODE:-hybrid}"
export SPAGENT_RL_REQUIRE_SKILL_CHOICE="${SPAGENT_RL_REQUIRE_SKILL_CHOICE:-1}"
export SPAGENT_RL_UPDATE_SKILL_MEMORY="${SPAGENT_RL_UPDATE_SKILL_MEMORY:-0}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29631}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

export SPAGENT_MAX_TURNS="${SPAGENT_MAX_TURNS:-3}"
export SPAGENT_MIN_TOOL_CALLS_BEFORE_ANSWER="${SPAGENT_MIN_TOOL_CALLS_BEFORE_ANSWER:-1}"
export SPAGENT_MIN_PI3_CALLS_BEFORE_ANSWER="${SPAGENT_MIN_PI3_CALLS_BEFORE_ANSWER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MAX_PIXELS="${MAX_PIXELS:-262144}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
ACC_WEIGHT="${ACC_WEIGHT:-0.6}"
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.2}"
TOOL_USE_WEIGHT="${TOOL_USE_WEIGHT:-0.2}"
LR="${LR:-1e-6}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-1}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEEPSPEED="${DEEPSPEED:-none}"
REPORT_TO="${REPORT_TO:-none}"
RL_BETA="${BETA:-0.05}"
TRAIN_TYPE="${TRAIN_TYPE:-full}"
export SPAGENT_RL_TOOL_BUDGET="${SPAGENT_RL_TOOL_BUDGET:-${SPAGENT_MAX_TURNS}}"

deepspeed_args=()
case "${DEEPSPEED,,}" in
  ""|"0"|"false"|"none"|"off"|"no")
    echo "[INFO] DeepSpeed: disabled"
    ;;
  *)
    deepspeed_args=(--deepspeed "${DEEPSPEED}")
    ;;
esac

tuner_flag="--train_type"
if "${SWIFT_BIN}" rlhf -h 2>&1 | grep -q -- '--tuner_type'; then
  tuner_flag="--tuner_type"
fi

echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] dataset=${DATASET_PATH}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] reward=external_r1v_acc:${ACC_WEIGHT} external_agentic_skill3d_format:${FORMAT_WEIGHT} external_skill3d_tool_use:${TOOL_USE_WEIGHT}"
echo "[INFO] max_turns=${SPAGENT_MAX_TURNS} min_tool_calls=${SPAGENT_MIN_TOOL_CALLS_BEFORE_ANSWER}"
echo "[INFO] grpo=num_generations:${NUM_GENERATIONS} generation_batch_size:${GENERATION_BATCH_SIZE} grad_acc:${GRAD_ACC} beta:${RL_BETA} temperature:${TEMPERATURE:-0.6}"
echo "[INFO] train_type=${TRAIN_TYPE} lr=${LR} per_device_train_bs=${PER_DEVICE_TRAIN_BS}"
echo "[INFO] tool_budget=${SPAGENT_RL_TOOL_BUDGET}"
echo "[INFO] retrieval_model=${SKILL3D_RETRIEVAL_MODEL}"
echo "[INFO] rl_memory_mode=${SPAGENT_RL_MEMORY_MODE} rl_retrieval_mode=${SPAGENT_RL_RETRIEVAL_MODE} top_k=${SPAGENT_RL_RETRIEVAL_TOP_K}"
echo "[INFO] require_skill_choice=${SPAGENT_RL_REQUIRE_SKILL_CHOICE}"
echo "[INFO] update_skill_memory=${SPAGENT_RL_UPDATE_SKILL_MEMORY}"
echo "[INFO] stop_on_answer=${SPAGENT_STOP_ON_ANSWER}"

train_type_args=(${tuner_flag} "${TRAIN_TYPE}")
case "${TRAIN_TYPE,,}" in
  lora)
    train_type_args+=(
      --lora_rank "${LORA_RANK:-8}"
      --lora_alpha "${LORA_ALPHA:-32}"
      --target_modules "${TARGET_MODULES:-all-linear}"
    )
    ;;
  full)
    ;;
  *)
    echo "[ERROR] Unsupported TRAIN_TYPE for this script: ${TRAIN_TYPE}. Use lora or full." >&2
    exit 1
    ;;
esac

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  echo "[INFO] Put the SFT checkpoint under ${MODEL_ROOT}, or override MODEL_PATH=/path/to/model_or_checkpoint." >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" && ! -f "${MODEL_PATH}/adapter_config.json" ]]; then
  echo "[ERROR] MODEL_PATH is not a model/checkpoint directory: ${MODEL_PATH}" >&2
  echo "[INFO] Expected config.json or adapter_config.json. Override MODEL_PATH to a specific checkpoint if needed." >&2
  exit 1
fi
if [[ ! -s "${DATASET_PATH}" ]]; then
  echo "[ERROR] DATASET_PATH not found or empty: ${DATASET_PATH}" >&2
  exit 1
fi

MAX_PIXELS="${MAX_PIXELS}" \
MASTER_ADDR="${MASTER_ADDR}" \
MASTER_PORT="${MASTER_PORT}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
"${SWIFT_BIN}" rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  --external_plugins "${project_dir}/plugin/plugin_all_angles.py" \
  --multi_turn_scheduler spagent_tool_call_scheduler \
  --max_turns "${SPAGENT_MAX_TURNS}" \
  --reward_funcs external_r1v_acc external_agentic_skill3d_format external_skill3d_tool_use \
  --reward_weights "${ACC_WEIGHT}" "${FORMAT_WEIGHT}" "${TOOL_USE_WEIGHT}" \
  "${train_type_args[@]}" \
  --torch_dtype bfloat16 \
  --dataset "${DATASET_PATH}" \
  --load_from_cache_file true \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --max_length "${MAX_LENGTH}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS}" \
  --learning_rate "${LR}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --save_strategy steps \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS:-400}" \
  --save_steps "${SAVE_STEPS:-70}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --logging_steps 1 \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.05 \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE:-0.6}" \
  --repetition_penalty 1.1 \
  --system "${SYSTEM_PROMPT}" \
  --log_completions true \
  --report_to "${REPORT_TO}" \
  --num_iterations 1 \
  --dataloader_num_workers "${NUM_WORKERS}" \
  --beta "${RL_BETA}" \
  --max_grad_norm 0.5 \
  --truncation_strategy left \
  "${deepspeed_args[@]}"
