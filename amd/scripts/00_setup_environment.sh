#!/bin/bash
# AMD MI350X — Environment Setup
# Run on Node 1 (amd1). Sets up both nodes.

set -euo pipefail

NODE1_IP="192.168.200.11"
NODE2_IP="192.168.200.21"
MLPERF_DIR="/opt/mlperf"

echo "=== AMD MI350X MLPerf Setup ==="
echo "Node: $(hostname), $(date)"
echo ""

# Create data directories
mkdir -p ${MLPERF_DIR}/{llama31-c4-preprocessed,llama31-tokenizer,govreport,models/llama2-70b,logs/llama31_pretraining,logs/llm_finetuning,results}
echo "[+] Created ${MLPERF_DIR} directory tree"

# Verify ROCm
echo ""
echo "=== ROCm Version ==="
cat /opt/rocm/.info/version
rocm-smi --showproductname 2>/dev/null | grep -E 'GPU\[|MI350' | head -9

# Verify RCCL
echo ""
echo "=== RCCL Version ==="
dpkg -l rccl 2>/dev/null | grep ^ii | awk '{print $2, $3}'

# Verify Docker
echo ""
echo "=== Docker Version ==="
docker --version

# Pull MLPerf containers (both tasks)
echo ""
echo "=== Pulling AMD MLPerf containers ==="
echo "Pulling Llama 3.1 8B pretraining container (~27 GB)..."
docker pull rocm/amd-mlperf:llama31_8b_training_6.0

echo "Pulling Llama 2 70B fine-tuning container (~27 GB)..."
docker pull rocm/amd-mlperf:llama2_70b_training_6.0

# Set kernel parameters
echo ""
echo "=== Kernel parameters ==="
sysctl -w vm.max_map_count=1048576
echo "vm.max_map_count=1048576" >> /etc/sysctl.conf 2>/dev/null || true

# Replicate setup to Node 2
echo ""
echo "=== Replicating setup to Node 2 (${NODE2_IP}) ==="
ssh -o StrictHostKeyChecking=no root@${NODE2_IP} "
    mkdir -p ${MLPERF_DIR}/{llama31-c4-preprocessed,llama31-tokenizer,govreport,models/llama2-70b,logs/llama31_pretraining,logs/llm_finetuning,results}
    sysctl -w vm.max_map_count=1048576
    echo 'vm.max_map_count=1048576' >> /etc/sysctl.conf 2>/dev/null || true
    echo '[Node2] Directory tree created'
"

echo ""
echo "=== Setup complete ==="
echo "Next step: bash 01_download_llama31_data.sh"
