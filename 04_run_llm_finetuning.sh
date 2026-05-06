#!/usr/bin/env bash
# =============================================================================
# 04_run_llm_finetuning.sh
# Run the MLPerf LLM Fine-Tuning benchmark (Llama 2 70B on SCROLLS GovReport).
# Uses LoRA fine-tuning with DeepSpeed ZeRO-3.
#
# Usage: ./04_run_llm_finetuning.sh <num_gpus> [num_runs]
#
# Examples:
#   ./04_run_llm_finetuning.sh 8        # 8 GPUs, 10 runs
#   ./04_run_llm_finetuning.sh 2 10     # 2 GPUs, 10 runs
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

NUM_GPUS="${1:?Usage: $0 <num_gpus> [num_runs]}"
NUM_RUNS="${2:-${NUM_RUNS}}"
RUN_TAG="llm_finetuning_${NUM_GPUS}xH200_$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="${LLM_LOG_DIR}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${OUT_DIR}/driver.log"; }
die()  { log "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
[[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || die "num_gpus must be an integer"
[[ "${NUM_GPUS}" -le 8 ]]       || die "This node has 8 GPUs max"
# Llama 2 70B with ZeRO-3 needs ≥2 GPUs (frozen params ~138 GB > 1x H200 140 GB)
[[ "${NUM_GPUS}" -ge 2 ]] || { log "SKIP: LLM fine-tuning requires ≥2 GPUs (have ${NUM_GPUS}); skipping."; exit 0; }

check_gpus_idle "${NUM_GPUS}" || die "Aborting: GPUs not idle. Free all GPU processes before benchmarking."

# ---------------------------------------------------------------------------
# Hyperparameters (from MLCommons llama2_70b_lora reference)
# ---------------------------------------------------------------------------
LLM_SRC="${MLPERF_TRAINING_DIR}/training/${LLM_SRC_SUBDIR}"
GPU_IDS=$(seq -s, 0 $(( NUM_GPUS - 1 )))

NUMA_ARGS=$(get_numa_args "${GPU_IDS}")

log "Starting LLM Fine-Tuning benchmark: ${NUM_GPUS} GPU(s), ${NUM_RUNS} runs"
log "GPU IDs      : ${GPU_IDS}"
log "NUMA affinity: ${NUMA_ARGS}"
log "Model src    : ${LLM_SRC}"
log "Logs         : ${OUT_DIR}"

# ---------------------------------------------------------------------------
# Write per-GPU-count accelerate config (overrides num_processes in default)
# ---------------------------------------------------------------------------
ACCEL_CONFIG="${OUT_DIR}/accelerate_config_${NUM_GPUS}gpu.yaml"
cat > "${ACCEL_CONFIG}" <<ACCELEOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_clipping: 0.3
  gradient_accumulation_steps: 1
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_GPUS}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
ACCELEOF

log "Accelerate config written to ${ACCEL_CONFIG}"

# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------
TIMES_FILE="${OUT_DIR}/wall_times.txt"
echo "# run_num, status, wall_time_min" > "${TIMES_FILE}"

for run in $(seq 1 "${NUM_RUNS}"); do
    RUN_LOG="${OUT_DIR}/run_${run}.log"
    log "--- Run ${run}/${NUM_RUNS} ---"

    START_EPOCH=$(date +%s%N)

    # Use a unique seed per run for stochasticity (MLPerf requirement)
    RUN_SEED=$(( RANDOM * RANDOM % 65536 ))

    ${DOCKER_CMD} run --rm \
        --gpus "\"device=${GPU_IDS}\"" \
        --net=host \
        --uts=host \
        --shm-size="${DOCKER_SHM_SIZE}" \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        ${NUMA_ARGS} \
        -v "${LLAMA2_MODEL_DIR}:/workspace/ft-llm/models/llama-v2-fused-qkv:ro" \
        -v "${GOVREPORT_DATA_DIR}:/workspace/ft-llm/dataset:ro" \
        -v "${LLM_RESULT_DIR}:/workspace/ft-llm/results" \
        -v "${ACCEL_CONFIG}:/workspace/ft-llm/configs/run_config.yaml:ro" \
        -v "${OUT_DIR}:/run_logs" \
        -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
        -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}" \
        -e NCCL_DEBUG="${NCCL_DEBUG}" \
        -e NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE}" \
        -e NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL}" \
        -e CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS}" \
        -e TOKENIZERS_PARALLELISM=false \
        "${LLM_IMAGE}" \
        bash -c "
            set -e
            cd /workspace/ft-llm

            accelerate launch --config_file configs/run_config.yaml \
                scripts/train.py \
                    --dataset_path /workspace/ft-llm/dataset \
                    --model_path ./models/llama-v2-fused-qkv \
                    --max_seq_len 8192 \
                    --bf16 True \
                    --logging_steps 24 \
                    --eval_steps 48 \
                    --output_dir ./results/run_${run} \
                    --per_device_train_batch_size 1 \
                    --gradient_accumulation_steps 1 \
                    --lr_scheduler_type cosine \
                    --learning_rate 4e-4 \
                    --weight_decay 0.0001 \
                    --warmup_ratio 0 \
                    --max_grad_norm 0.3 \
                    --use_gradient_checkpointing True \
                    --target_eval_loss ${LLM_TARGET_LOSS} \
                    --use_peft_lora True \
                    --lora_r 16 \
                    --lora_alpha 32 \
                    --lora_dropout 0.1 \
                    --lora_target_modules "qkv_proj,o_proj" \
                    --max_steps 1024 \
                    --use_flash_attn \
                    --seed ${RUN_SEED} \
                    2>&1 | tee /run_logs/run_${run}_docker.log
        " 2>&1 | tee "${RUN_LOG}"

    END_EPOCH=$(date +%s%N)
    WALL_TIME_SEC=$(( (END_EPOCH - START_EPOCH) / 1000000000 ))
    WALL_TIME_MIN=$(echo "${WALL_TIME_SEC}" | awk '{printf "%.4f", $1/60}')

    STATUS="unknown"
    if grep -q '"key": "run_stop"' "${RUN_LOG}" 2>/dev/null; then
        if grep '"run_stop"' "${RUN_LOG}" | grep -q '"status": "success"'; then
            STATUS="success"
        else
            STATUS="aborted"
        fi
    elif [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        STATUS="failed"
    fi

    log "Run ${run} finished | status=${STATUS} | wall_time=${WALL_TIME_MIN} min"
    echo "${run},${STATUS},${WALL_TIME_MIN}" >> "${TIMES_FILE}"
done

log "All ${NUM_RUNS} runs complete."
log "Results written to ${TIMES_FILE}"
log "Run 'python3 ${SCRIPT_DIR}/process_results.py ${OUT_DIR}' to compute final stats."
