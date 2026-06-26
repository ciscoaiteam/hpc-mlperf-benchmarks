#!/bin/bash
# AMD MI350X — Llama 3.1 8B Pretraining
# Usage: bash 03_run_llama31_pretraining.sh <num_gpus> [num_runs]
#   num_gpus: 8 (single node) or 16 (2-node)
#   num_runs: default 1 (set to 10 for full MLPerf submission)
#
# Run from Node 1 (amd1). For 16-GPU runs, Node 2 must be reachable via fabric.

set -euo pipefail
source "$(dirname "$0")/config.env"

NUM_GPUS=${1:-8}
NUM_RUNS=${2:-1}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TAG="llama31_pretraining_${NUM_GPUS}xMI350X_${TIMESTAMP}"
RUN_LOG_DIR="${LOG_DIR}/llama31_pretraining/${RUN_TAG}"
mkdir -p "${RUN_LOG_DIR}"

echo "=== Llama 3.1 8B Pretraining — AMD MI350X ==="
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

# MPI hostfile for multi-node
HOSTFILE="/tmp/hostfile_llama31"
if [ "${NUM_NODES}" -gt 1 ]; then
    echo "${NODE1_FABRIC}:${GPUS_PER_NODE}" > "${HOSTFILE}"
    echo "${NODE2_FABRIC}:${GPUS_PER_NODE}" >> "${HOSTFILE}"
fi

# Docker run function
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
        -v "${C4_PREPROCESSED_DIR}:/data/preprocessed:ro"
        -v "${LLAMA31_TOKENIZER_DIR}:/data/tokenizer:ro"
        -v "${RUN_LOG_DIR}:/results"
        -v /tmp/miopen_cache:/tmp/miopen_cache
        -e NCCL_DEBUG="${NCCL_DEBUG}"
        -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
        -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE}"
        -e MASTER_ADDR="${MASTER_ADDR}"
        -e MASTER_PORT=29500
        -e WORLD_SIZE="${NUM_GPUS}"
        -e NPROC_PER_NODE="${GPUS_PER_NODE}"
        -e NODE_RANK=0
        -e GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
        -e MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE}"
        -e TARGET_LOG_PPL="${TARGET_LOG_PPL}"
        -e SEED="${SEED}"
        -e MIOPEN_USER_DB_PATH=/tmp/miopen_cache
    )

    if [ "${NUM_NODES}" -gt 1 ]; then
        # Launch on Node 2 in background
        ssh root@${NODE2_FABRIC} "
            docker run ${DOCKER_ARGS[*]} \
                -e NODE_RANK=1 \
                ${PRETRAIN_IMAGE} \
                bash run_llama31.sh
        " > "${RUN_LOG_DIR}/run_${RUN_ID}_node2.log" 2>&1 &
        NODE2_PID=$!
    fi

    # Launch on Node 1 (foreground — captures exit code)
    START_TIME=$(date +%s)
    docker run "${DOCKER_ARGS[@]}" \
        -e NODE_RANK=0 \
        "${PRETRAIN_IMAGE}" \
        bash run_llama31.sh 2>&1 | tee "${LOG_FILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    WALLTIME=$((END_TIME - START_TIME))

    if [ "${NUM_NODES}" -gt 1 ]; then
        wait ${NODE2_PID} 2>/dev/null || true
    fi

    # Parse result
    CONVERGED="NO"
    FINAL_PPL="N/A"
    if grep -q "run_stop.*success" "${LOG_FILE}" 2>/dev/null; then
        CONVERGED="YES"
        FINAL_PPL=$(grep -oP 'val_loss["\s:]+\K[0-9.]+' "${LOG_FILE}" | tail -1)
    fi

    echo "[Run ${RUN_ID}] walltime=${WALLTIME}s converged=${CONVERGED} ppl=${FINAL_PPL} exit=${EXIT_CODE}"
    echo "${RUN_ID},${WALLTIME},${CONVERGED},${FINAL_PPL},${EXIT_CODE}" >> "${RUN_LOG_DIR}/results.csv"
}

# Main loop
echo "run_id,walltime_sec,converged,final_ppl,exit_code" > "${RUN_LOG_DIR}/results.csv"
for i in $(seq 1 "${NUM_RUNS}"); do
    run_once "${i}"
done

echo ""
echo "=== All runs complete ==="
echo "Results in: ${RUN_LOG_DIR}/results.csv"
cat "${RUN_LOG_DIR}/results.csv"
