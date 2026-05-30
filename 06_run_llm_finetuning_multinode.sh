#!/usr/bin/env bash
# =============================================================================
# 06_run_llm_finetuning_multinode.sh
# 2-node (16× H200) Llama 2 70B LoRA fine-tuning on SCROLLS GovReport
# Uses DeepSpeed ZeRO-3 via HuggingFace accelerate.
#
# Node 2  <NODE2_MGMT_IP>  → master / machine_rank=0
# Node 1  <NODE1_MGMT_IP>  → worker / machine_rank=1
#
# Usage:
#   ./06_run_llm_finetuning_multinode.sh [num_runs]
#   num_runs defaults to 1
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# ─── Node identifiers ────────────────────────────────────────────────────────
MASTER_SSH="<NODE2_MGMT_IP>"   # Node 2 (mlperf2) mgmt IP
WORKER_SSH="<NODE1_MGMT_IP>"   # Node 1 (mlperf1) mgmt IP
MASTER_IB_ADDR="<NODE2_GPU_FABRIC_IP>"   # Node 2 GPU fabric IP (ens201np0, mlx5_4)
ACCEL_PORT="29600"           # avoid clash with torchrun 29500
NUM_NODES=2
GPUS_PER_NODE=8
TOTAL_GPUS=$(( NUM_NODES * GPUS_PER_NODE ))

NUM_RUNS="${1:-1}"

# Override container image to use rebuilt NCCL 2.25.1 + IBext v8 plugin
LLM_IMAGE="mlperf-llm-finetuning:h200-nccl225"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ─── Output / log paths (must exist on BOTH nodes) ───────────────────────────
RUN_OUT="${LLM_RESULT_DIR}/run_16gpu"
LOG_DIR="${LLM_LOG_DIR}/llm_finetuning_16xH200_${TIMESTAMP}"

echo "============================================================"
echo "  16× H200 Llama 2 70B LoRA Fine-Tuning (2-node DeepSpeed)"
echo "  Master : ${MASTER_SSH}  (IB ${MASTER_IB_ADDR})"
echo "  Worker : ${WORKER_SSH}"
echo "  Runs   : ${NUM_RUNS}"
echo "  Output : ${RUN_OUT}"
echo "  Logs   : ${LOG_DIR}"
echo "============================================================"

echo "[setup] Creating output + log dirs on both nodes..."
ssh "${MASTER_SSH}" "mkdir -p '${RUN_OUT}' '${LOG_DIR}'"
ssh "${WORKER_SSH}" "mkdir -p '${RUN_OUT}' '${LOG_DIR}'"

# ─── Generate DeepSpeed ZeRO-3 config JSON ────────────────────────────────
cat > /tmp/ft16_ds_config.json << 'DSEOF'
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto"
  },
  "gradient_clipping": 0.3,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,
  "wall_clock_breakdown": false
}
DSEOF

rsync -q /tmp/ft16_ds_config.json "${MASTER_SSH}:/root/ft16_ds_config.json"
rsync -q /tmp/ft16_ds_config.json "${WORKER_SSH}:/root/ft16_ds_config.json"

echo "[setup] Syncing ft_multinode_wrapper.py to both nodes..."
rsync -q "${SCRIPT_DIR}/ft_multinode_wrapper.py" "${MASTER_SSH}:/root/mlperf-h200/ft_multinode_wrapper.py"
rsync -q "${SCRIPT_DIR}/ft_multinode_wrapper.py" "${WORKER_SSH}:/root/mlperf-h200/ft_multinode_wrapper.py"

# ─── Generate per-node launch scripts ─────────────────────────────────────
for MACHINE_RANK in 0 1; do
cat > /tmp/ft16_node${MACHINE_RANK}.sh << 'HEREDOC'
#!/usr/bin/env bash
set -euo pipefail
HEREDOC

cat >> /tmp/ft16_node${MACHINE_RANK}.sh << HEREDOC
RUN_SEED=\${1:-42}
docker run --rm \\
  -e RUN_SEED=\${RUN_SEED} \\
  --gpus all \\
  --net=host --uts=host --ipc=host \\
  --shm-size=128g \\
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \\
  --cap-add IPC_LOCK \\
  --device /dev/infiniband/ \\
  -v ${LLAMA2_MODEL_DIR}:/workspace/ft-llm/models/llama-v2-fused-qkv:ro \\
  -v ${GOVREPORT_DATA_DIR}:/workspace/ft-llm/dataset:ro \\
  -v ${RUN_OUT}:/workspace/ft-llm/results \\
  -v /root/ft16_ds_config.json:/workspace/ft-llm/configs/ds_config.json:ro \\
  -v /root/mlperf-h200/ft_multinode_wrapper.py:/workspace/ft-llm/ft_multinode_wrapper.py:ro \\
  -v ${LOG_DIR}:/run_logs \\
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \\
  -e TORCH_NCCL_AVOID_RECORD_STREAMS=1 \\
  -e NCCL_IB_HCA=${NCCL_IB_HCA} \\
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \\
  -e NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT} \\
  -e NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT} \\
  -e NCCL_DEBUG=INFO \\
  -e NCCL_DEBUG_FILE=/run_logs/nccl_debug_ft_%h_%p.log \\
  -e NCCL_NVLS_ENABLE=0 \\
  -e NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL} \\
  -e NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL} \\
  -e GLOO_SOCKET_IFNAME=ens201np0 \\
  -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS} \\
  -e TOKENIZERS_PARALLELISM=false \\
  -e PYTHONUNBUFFERED=1 \\
  ${LLM_IMAGE} \\
  torchrun \\
    --node_rank=${MACHINE_RANK} \\
    --nnodes=${NUM_NODES} \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --master_addr=${MASTER_IB_ADDR} \\
    --master_port=${ACCEL_PORT} \\
    /workspace/ft-llm/ft_multinode_wrapper.py \\
        --deepspeed /workspace/ft-llm/configs/ds_config.json \\
        --dataset_path /workspace/ft-llm/dataset \\
        --model_path ./models/llama-v2-fused-qkv \\
        --max_seq_len 8192 \\
        --bf16 True \\
        --logging_steps 24 \\
        --eval_steps 48 \\
        --output_dir ./results/run_16gpu_\${RUN_SEED} \\
        --per_device_train_batch_size 1 \\
        --gradient_accumulation_steps 1 \\
        --lr_scheduler_type cosine \\
        --learning_rate 4e-4 \\
        --weight_decay 0.0001 \\
        --warmup_ratio 0 \\
        --max_grad_norm 0.3 \\
        --use_gradient_checkpointing True \\
        --target_eval_loss ${LLM_TARGET_LOSS} \\
        --use_peft_lora True \\
        --lora_r 16 \\
        --lora_alpha 32 \\
        --lora_dropout 0.1 \\
        --lora_target_modules 'qkv_proj,o_proj' \\
        --max_steps 1024 \\
        --use_flash_attn \\
        --seed \${RUN_SEED}
HEREDOC
done

chmod +x /tmp/ft16_node0.sh /tmp/ft16_node1.sh
rsync -q /tmp/ft16_node0.sh "${MASTER_SSH}:/root/ft16_launch.sh"
rsync -q /tmp/ft16_node1.sh "${WORKER_SSH}:/root/ft16_launch.sh"
ssh "${MASTER_SSH}" "chmod +x /root/ft16_launch.sh"
ssh "${WORKER_SSH}" "chmod +x /root/ft16_launch.sh"

# ─── Run loop ──────────────────────────────────────────────────────────────
for run in $(seq 1 "${NUM_RUNS}"); do
    RUN_SEED=$(( RANDOM * RANDOM % 65536 ))
    echo ""
    echo "[run ${run}/${NUM_RUNS}] seed=${RUN_SEED}"

    # Worker first
    echo "[launch] Starting worker (Node 1)..."
    ssh "${WORKER_SSH}" \
      "tmux kill-session -t ft16 2>/dev/null; true; \
       tmux new-session -d -s ft16 \
       'bash /root/ft16_launch.sh ${RUN_SEED} 2>&1 | tee ${LOG_DIR}/node1_run${run}.log \
        && echo \"[node1] DONE\" >> ${LOG_DIR}/node1_run${run}.log \
        || echo \"[node1] FAILED\" >> ${LOG_DIR}/node1_run${run}.log'"

    sleep 5

    echo "[launch] Starting master (Node 2)..."
    ssh "${MASTER_SSH}" \
      "tmux kill-session -t ft16 2>/dev/null; true; \
       tmux new-session -d -s ft16 \
       'bash /root/ft16_launch.sh ${RUN_SEED} 2>&1 | tee ${LOG_DIR}/node2_run${run}.log \
        && echo \"[node2] DONE\" >> ${LOG_DIR}/node2_run${run}.log \
        || echo \"[node2] FAILED\" >> ${LOG_DIR}/node2_run${run}.log'"

    echo "  Monitor: ssh ${MASTER_SSH} 'tmux attach -t ft16'"
    echo "           tail -f ${LOG_DIR}/node2_run${run}.log"

    # Wait for master to finish (poll every 30s)
    echo "[wait] Waiting for run ${run} to complete..."
    while true; do
        if ssh "${MASTER_SSH}" "grep -q 'DONE\|FAILED' ${LOG_DIR}/node2_run${run}.log 2>/dev/null"; then
            break
        fi
        sleep 30
    done

    # Check result
    if ssh "${MASTER_SSH}" "grep -q 'run_stop.*success' ${LOG_DIR}/node2_run${run}.log 2>/dev/null"; then
        echo "[run ${run}] ✓ SUCCESS"
    else
        echo "[run ${run}] ✗ FAILED or did not converge"
    fi
done

echo ""
echo "============================================================"
echo "  All ${NUM_RUNS} runs complete."
echo "  Logs: ${LOG_DIR}"
echo "  Results: ${RUN_OUT}"
echo "============================================================"
