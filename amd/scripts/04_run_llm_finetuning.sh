#!/bin/bash
# AMD MI350X — Llama 2 70B LoRA Fine-Tuning
# Usage: bash 04_run_llm_finetuning.sh <num_gpus> [num_runs]
#   num_gpus: 8 (single node) or 16 (2-node)
#   num_runs: default 1 (set to 10 for full MLPerf submission)
#
# Requires HF_TOKEN env var for Llama 2 model download.
# Run from Node 1 (amd1).

set -euo pipefail
source "$(dirname "$0")/config.env"

NUM_GPUS=${1:-8}
NUM_RUNS=${2:-1}

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable not set."
    echo "Usage: HF_TOKEN=hf_xxxx bash $0 ${NUM_GPUS}"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="llm_finetuning_${NUM_GPUS}xMI350X_${TIMESTAMP}"
RUN_LOG_DIR="${LOG_DIR}/llm_finetuning/${RUN_TAG}"
mkdir -p "${RUN_LOG_DIR}"

echo "=== Llama 2 70B Fine-Tuning — AMD MI350X ==="
echo "GPUs: ${NUM_GPUS} | Runs: ${NUM_RUNS} | Tag: ${RUN_TAG}"
echo "Start: $(date)"

# Determine node configuration
if [ "${NUM_GPUS}" -le 8 ]; then
    NUM_NODES=1
    MASTER_ADDR="localhost"
else
    NUM_NODES=2
    MASTER_ADDR="${NODE1_FABRIC}"
fi

run_once() {
    local RUN_ID=$1
    local SEED=$((RANDOM * RANDOM % 65535))
    local LOG_FILE="${RUN_LOG_DIR}/run_${RUN_ID}.log"

    echo "[Run ${RUN_ID}/${NUM_RUNS}] seed=${SEED} → ${LOG_FILE}"

    DOCKER_ARGS=(
        --rm
        --network host
        --ipc host
        --privileged
        --device /dev/kfd
        --device /dev/dri
        --group-add video
        --group-add render
        --ulimit nofile=65536:65536
        --ulimit memlock=-1:-1
        -v "${GOVREPORT_DIR}:/data/govreport:ro"
        -v "${LLAMA2_MODEL_DIR}:/data/model:ro"
        -v "${RUN_LOG_DIR}:/results"
        -v /tmp/miopen_cache:/tmp/miopen_cache
        -e NCCL_DEBUG="${NCCL_DEBUG}"
        -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
        -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"
        -e MASTER_ADDR="${MASTER_ADDR}"
        -e MASTER_PORT=29501
        -e WORLD_SIZE="${NUM_GPUS}"
        -e NPROC_PER_NODE="${GPUS_PER_NODE}"
        -e NODE_RANK=0
        -e HF_TOKEN="${HF_TOKEN}"
        -e SEED="${SEED}"
        -e LR="${FT_LR}"
        -e TARGET_EVAL_LOSS="${FT_TARGET_LOSS}"
        -e LORA_RANK="${FT_LORA_RANK}"
        -e LORA_ALPHA="${FT_LORA_ALPHA}"
        -e MAX_STEPS="${FT_MAX_STEPS}"
        -e MIOPEN_USER_DB_PATH=/tmp/miopen_cache
    )

    if [ "${NUM_NODES}" -gt 1 ]; then
        ssh root@${NODE2_FABRIC} "
            docker run ${DOCKER_ARGS[*]} \
                -e NODE_RANK=1 \
                ${FINETUNE_IMAGE} \
                bash run_finetuning.sh
        " > "${RUN_LOG_DIR}/run_${RUN_ID}_node2.log" 2>&1 &
        NODE2_PID=$!
    fi

    START_TIME=$(date +%s)
    docker run "${DOCKER_ARGS[@]}" \
        -e NODE_RANK=0 \
        "${FINETUNE_IMAGE}" \
        bash run_finetuning.sh 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    WALLTIME=$((END_TIME - START_TIME))

    if [ "${NUM_NODES}" -gt 1 ]; then
        wait ${NODE2_PID} 2>/dev/null || true
    fi

    CONVERGED="NO"
    FINAL_LOSS="N/A"
    if grep -q "run_stop.*success" "${LOG_FILE}" 2>/dev/null; then
        CONVERGED="YES"
        FINAL_LOSS=$(grep -oP 'eval_loss["\s:]+\K[0-9.]+' "${LOG_FILE}" | tail -1)
    fi

    echo "[Run ${RUN_ID}] walltime=${WALLTIME}s converged=${CONVERGED} loss=${FINAL_LOSS} exit=${EXIT_CODE}"
    echo "${RUN_ID},${WALLTIME},${CONVERGED},${FINAL_LOSS},${EXIT_CODE}" >> "${RUN_LOG_DIR}/results.csv"
}

echo "run_id,walltime_sec,converged,final_loss,exit_code" > "${RUN_LOG_DIR}/results.csv"
for i in $(seq 1 "${NUM_RUNS}"); do
    run_once "${i}"
done

echo ""
echo "=== All runs complete ==="
echo "Results in: ${RUN_LOG_DIR}/results.csv"
cat "${RUN_LOG_DIR}/results.csv"
