#!/usr/bin/env bash
# =============================================================================
# 03_run_retinanet_multinode.sh
# Launch a 16-GPU (2-node) RetinaNet benchmark run.
# Run this on mlperf1 only — it SSHs into mlperf2 automatically.
#
# Usage:
#   ./03_run_retinanet_multinode.sh [num_runs]
#
# Example:
#   ./03_run_retinanet_multinode.sh 10
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

NUM_RUNS="${1:-${NUM_RUNS}}"
TOTAL_GPUS=16
NODE1_SSH="10.195.0.10"      # mlperf1 mgmt IP (for SSH)
NODE2_SSH="10.195.0.20"      # mlperf2 mgmt IP (for SSH)
NODE1_ADDR="<NODE1_GPU_FABRIC_IP>"  # mlperf1 GPU fabric IP (NCCL/torchrun rendezvous)
NODE2_ADDR="<NODE2_GPU_FABRIC_IP>"  # mlperf2 GPU fabric IP
REMOTE_SCRIPT="/opt/mlperf/03_run_retinanet.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

log "=== 16-GPU Multi-Node RetinaNet Benchmark ==="
log "mlperf1: ${NODE1_ADDR} (rank 0 / master)"
log "mlperf2: ${NODE2_ADDR} (rank 1)"
log "Runs   : ${NUM_RUNS}"

# Verify mlperf2 is reachable and script is present
ssh -o ConnectTimeout=10 root@${NODE2_SSH} "test -f ${REMOTE_SCRIPT}" \
    || die "Cannot reach mlperf2 or script not deployed at ${REMOTE_SCRIPT}"

# Sync any local script changes to mlperf2 before starting
rsync -q "${SCRIPT_DIR}/config.env" "${SCRIPT_DIR}/03_run_retinanet.sh" \
    root@${NODE2_SSH}:/opt/mlperf/
log "Scripts synced to mlperf2"

# Launch worker on mlperf2 in background (non-blocking SSH)
log "Launching rank 1 on mlperf2..."
ssh root@${NODE2_SSH} \
    "nohup bash ${REMOTE_SCRIPT} ${TOTAL_GPUS} ${NUM_RUNS} 1 ${NODE1_ADDR} \
     > /opt/mlperf/logs/retinanet_multinode_rank1.log 2>&1 &
     echo \$!" &
SSH_PID=$!

# Small delay to let mlperf2 start its rendezvous listener
sleep 5

# Launch master on this node (blocking — waits for run completion)
log "Launching rank 0 on mlperf1..."
bash "${SCRIPT_DIR}/03_run_retinanet.sh" \
    "${TOTAL_GPUS}" "${NUM_RUNS}" 0 "${NODE1_ADDR}"

# Wait for SSH background job to finish
wait ${SSH_PID} || true

log "=== Multi-node benchmark complete. ==="
log "mlperf1 logs: ${LOG_DIR}/retinanet/"
log "mlperf2 logs: ssh root@${NODE2_ADDR} ls /opt/mlperf/logs/retinanet/"
