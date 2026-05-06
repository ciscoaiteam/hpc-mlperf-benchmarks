#!/usr/bin/env bash
# =============================================================================
# 00_setup_environment.sh
# One-time setup: verify/install Docker, nvidia-container-toolkit,
# clone MLCommons training repo, build benchmark Docker images, create dirs.
# Run as root on the H200 server.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# 1. Create directory layout
# ---------------------------------------------------------------------------
log "Creating directory structure under ${SCRATCH_DIR}..."
mkdir -p "${C4_PREPROCESSED_DIR}" "${LLAMA31_TOKENIZER_DIR}" "${MODELS_DIR}/llama2-70b" \
         "${GOVREPORT_DATA_DIR}" "${LLAMA31_RESULT_DIR}" "${LLM_RESULT_DIR}" \
         "${LLAMA31_LOG_DIR}" "${LLM_LOG_DIR}" "${MLPERF_TRAINING_DIR}"

# ---------------------------------------------------------------------------
# 2. Verify NVIDIA driver & GPU visibility
# ---------------------------------------------------------------------------
log "Verifying GPU access..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
log "Detected ${GPU_COUNT} GPU(s)."
if [[ ${GPU_COUNT} -lt 8 ]]; then
    log "WARNING: Expected 8 GPUs, found ${GPU_COUNT}. Check driver installation."
fi

# ---------------------------------------------------------------------------
# 3. Docker
# ---------------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
    log "Docker not found. Installing..."
    apt-get update -qq
    apt-get install -y ca-certificates curl gnupg lsb-release
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update -qq
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    systemctl enable --now docker
else
    log "Docker already installed: $(docker --version)"
fi

# ---------------------------------------------------------------------------
# 4. NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
if ! dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    log "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    DIST=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
    curl -sL "https://nvidia.github.io/libnvidia-container/${DIST}/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
else
    log "NVIDIA Container Toolkit already installed."
fi

# Test GPU Docker access
log "Testing Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L

# ---------------------------------------------------------------------------
# 5. Clone MLCommons training repository
# ---------------------------------------------------------------------------
if [[ ! -d "${MLPERF_TRAINING_DIR}/training" ]]; then
    log "Cloning MLCommons training repo..."
    git clone --depth 1 https://github.com/mlcommons/training.git \
        "${MLPERF_TRAINING_DIR}/training"
else
    log "MLCommons training repo already present. Pulling latest..."
    git -C "${MLPERF_TRAINING_DIR}/training" pull --ff-only
fi

# ---------------------------------------------------------------------------
# 6. Build Llama 3.1 8B Pre-Training Docker image (NeMo)
# ---------------------------------------------------------------------------
LLAMA31_SRC="${MLPERF_TRAINING_DIR}/training/${LLAMA31_SRC_SUBDIR}"
if [[ -d "${LLAMA31_SRC}" ]]; then
    log "Building Llama 3.1 8B pre-training Docker image (${LLAMA31_IMAGE})..."
    ${DOCKER_CMD} build -f "${LLAMA31_SRC}/Dockerfile.h200" -t "${LLAMA31_IMAGE}" "${LLAMA31_SRC}"
else
    log "ERROR: Llama 3.1 8B source not found at ${LLAMA31_SRC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 7. Build LLM Fine-Tuning Docker image
# ---------------------------------------------------------------------------
LLM_SRC="${MLPERF_TRAINING_DIR}/training/${LLM_SRC_SUBDIR}"
if [[ -d "${LLM_SRC}" ]]; then
    log "Building LLM fine-tuning Docker image (${LLM_IMAGE})..."
    ${DOCKER_CMD} build -t "${LLM_IMAGE}" "${LLM_SRC}"
else
    log "ERROR: LLM fine-tuning source not found at ${LLM_SRC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 8. Install Python result-processing deps (on host)
# ---------------------------------------------------------------------------
log "Installing Python dependencies for result processing..."
pip3 install --quiet numpy pandas tabulate 2>/dev/null || \
    python3 -m pip install --quiet numpy pandas tabulate

log ""
log "=========================================="
log "  Setup complete."
log "  Scratch dir : ${SCRATCH_DIR}"
log "  Training src: ${MLPERF_TRAINING_DIR}/training"
log "=========================================="
log "Next steps:"
log "  1. Run ./01_download_llama31_data.sh"
log "  2. Run ./02_prepare_llm_assets.sh"
log "  3. Run ./05_run_all_benchmarks.sh  (or individual run scripts)"
