#!/usr/bin/env bash
# =============================================================================
# 05_run_llama31_multinode.sh
# 2-node (16× H200) Llama 3.1 8B pretraining — direct torchrun multi-node
#
# Node 2  ${NODE2_IP}  (IB: ${MASTER_IB_ADDR})  → master / rank-0 node
# Node 1  ${NODE1_IP}  (IB: worker)              → worker / rank-1 node
#
# Strategy: bypass NeMo-Run LocalExecutor (single-node only in v0.4.0).
# Both nodes run identical torchrun commands pointing at the same c10d
# rendezvous endpoint on Node 2's IB interface.
#
# Usage:
#   ./05_run_llama31_multinode.sh [SEED]
#   SEED defaults to a random value; pass a fixed seed to reuse npy_index.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# ─── Node identifiers ────────────────────────────────────────────────────────
MASTER_SSH="${NODE2_IP:?Set NODE2_IP}"       # Node 2 (master) mgmt IP
WORKER_SSH="${NODE1_IP:?Set NODE1_IP}"       # Node 1 (worker) mgmt IP
MASTER_IB_ADDR="${MASTER_IB_ADDR:?Set MASTER_IB_ADDR}"  # Node 2 GPU fabric IP
RDZV_PORT="29500"
NUM_NODES=2
GPUS_PER_NODE=8

RUN_SEED="${1:-$(shuf -i 1-32767 -n 1)}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ─── Output / log paths (must exist on BOTH nodes) ───────────────────────────
RUN_OUT="${LLAMA31_RESULT_DIR}/run_16gpu"
LOG_DIR="${LLAMA31_LOG_DIR}/llama31_pretraining_16xH200_${TIMESTAMP}"

echo "============================================================"
echo "  16× H200 Llama 3.1 8B Pretraining (2-node torchrun)"
echo "  Master : ${MASTER_SSH}  (IB ${MASTER_IB_ADDR})"
echo "  Worker : ${WORKER_SSH}"
echo "  Seed   : ${RUN_SEED}"
echo "  Output : ${RUN_OUT}"
echo "  Logs   : ${LOG_DIR}"
echo "============================================================"

echo "[setup] Creating output + log dirs on both nodes..."
ssh "${MASTER_SSH}" "mkdir -p '${RUN_OUT}' '${LOG_DIR}'"
ssh "${WORKER_SSH}" "mkdir -p '${RUN_OUT}' '${LOG_DIR}'"

echo "[setup] Syncing multinode_wrapper.py to both nodes..."
rsync -q "${SCRIPT_DIR}/multinode_wrapper.py" "${MASTER_SSH}:/root/mlperf-h200/multinode_wrapper.py"
rsync -q "${SCRIPT_DIR}/multinode_wrapper.py" "${WORKER_SSH}:/root/mlperf-h200/multinode_wrapper.py"

echo "[setup] Syncing npy_index from master to worker (new files only)..."
ssh "${MASTER_SSH}" "rsync -a --ignore-existing ${LLAMA31_NPY_INDEX_DIR}/ ${WORKER_SSH}:${LLAMA31_NPY_INDEX_DIR}/"

# ─── Generate per-node launch scripts (explicit node_rank, no c10d rendezvous) ──
# Node 2 = rank 0 (master, owns the TCPStore); Node 1 = rank 1 (worker)
for NODE_RANK in 0 1; do
cat > /tmp/pretrain16_node${NODE_RANK}.sh << HEREDOC
#!/usr/bin/env bash
set -euo pipefail
docker run --rm \
  --gpus all \
  --net=host --uts=host --ipc=host \
  --shm-size=128g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
  --cap-add IPC_LOCK \
  --device /dev/infiniband/ \
  -v ${C4_PREPROCESSED_DIR}/llama3_1_8b_preprocessed_c4_dataset:/preproc_data:ro \
  -v ${LLAMA31_TOKENIZER_DIR}/llama3_1_8b_tokenizer:/tokenizer:ro \
  -v ${LLAMA31_NPY_INDEX_DIR}:/npy_index \
  -v ${RUN_OUT}:/mlperf-outputs \
  -v /root/mlperf-h200/multinode_wrapper.py:/workspace/code/multinode_wrapper.py:ro \
  -e PREPROCESSED_PATH=/preproc_data \
  -e GBS=${LLAMA31_GBS} \
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=${NCCL_IB_HCA} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
  -e NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT} \
  -e NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT} \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_FILE=/mlperf-outputs/nccl_debug_%h_%p.log \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL} \
  -e NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \
  -e GLOO_SOCKET_IFNAME=ens201np0 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
  -e NVTE_DP_AMAX_REDUCE_INTERVAL=0 \
  -e NVTE_ASYNC_AMAX_REDUCTION=1 \
  -e NVTE_BWD_LAYERNORM_SM_MARGIN=16 \
  -e NVTE_FWD_LAYERNORM_SM_MARGIN=16 \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONUNBUFFERED=1 \
  ${LLAMA31_IMAGE} \
  torchrun \
    --node_rank=${NODE_RANK} \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${MASTER_IB_ADDR} \
    --master_port=${RDZV_PORT} \
    /workspace/code/multinode_wrapper.py \
    --nodes ${NUM_NODES} \
    --gpus_per_node ${GPUS_PER_NODE} \
    --gbs ${LLAMA31_GBS} \
    --mbs ${LLAMA31_MBS} \
    --max_lr ${LLAMA31_MAX_LR} \
    --warmup_steps ${LLAMA31_WARMUP_STEPS} \
    --eval_every ${LLAMA31_EVAL_EVERY} \
    --max_steps 1200000 \
    --seed ${RUN_SEED} \
    --tokenizer_path /tokenizer \
    --target_log_ppl ${LLAMA31_TARGET_LOG_PPL}
HEREDOC
done

chmod +x /tmp/pretrain16_node0.sh /tmp/pretrain16_node1.sh
rsync -q /tmp/pretrain16_node0.sh "${MASTER_SSH}:/root/pretrain16_launch.sh"
rsync -q /tmp/pretrain16_node1.sh "${WORKER_SSH}:/root/pretrain16_launch.sh"
ssh "${MASTER_SSH}" "chmod +x /root/pretrain16_launch.sh"
ssh "${WORKER_SSH}" "chmod +x /root/pretrain16_launch.sh"

# ─── Launch ──────────────────────────────────────────────────────────────────
# Worker starts first so it is listening on the rendezvous before master joins.
echo "[launch] Starting worker (Node 1)..."
ssh "${WORKER_SSH}" \
  "tmux new-session -d -s pretrain16 \
   'bash /root/pretrain16_launch.sh 2>&1 | tee ${LOG_DIR}/node1_docker.log \
    && echo \"[node1] DONE\" >> ${LOG_DIR}/node1_docker.log \
    || echo \"[node1] FAILED\" >> ${LOG_DIR}/node1_docker.log'"

sleep 5    # give worker time to attach to rendezvous

echo "[launch] Starting master (Node 2)..."
ssh "${MASTER_SSH}" \
  "tmux new-session -d -s pretrain16 \
   'bash /root/pretrain16_launch.sh 2>&1 | tee ${LOG_DIR}/node2_docker.log \
    && echo \"[node2] DONE\" >> ${LOG_DIR}/node2_docker.log \
    || echo \"[node2] FAILED\" >> ${LOG_DIR}/node2_docker.log'"

echo ""
echo "============================================================"
echo "  Both nodes launched.  Monitor with:"
echo "    ssh ${MASTER_SSH} 'tmux attach -t pretrain16'"
echo "    ssh ${WORKER_SSH} 'tmux attach -t pretrain16'"
echo "    tail -f ${LOG_DIR}/node2_docker.log"
echo "  Results will appear in: ${RUN_OUT}/"
echo "============================================================"
