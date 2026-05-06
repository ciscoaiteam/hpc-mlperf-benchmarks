#!/usr/bin/env bash
# =============================================================================
# 01_download_retinanet_data.sh
# Download and preprocess the OpenImages v6 dataset and ResNet-50 checkpoint
# required for the MLPerf RetinaNet (Object Detection Lightweight) benchmark.
#
# Approximate download sizes:
#   OpenImages train subset (~1.7M images): ~500 GB
#   OpenImages validation subset (~41K images): ~12 GB
#   ResNet-50 pretrained checkpoint: ~100 MB
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

RETINANET_SRC="${MLPERF_TRAINING_DIR}/training/${RETINANET_SRC_SUBDIR}"
RETINANET_SCRIPTS="${RETINANET_SRC}/scripts"

# ---------------------------------------------------------------------------
# 1. Download OpenImages dataset via MLCommons script
# ---------------------------------------------------------------------------
log "Downloading OpenImages v6 (MLPerf subset) to ${OPENIMAGES_DIR}..."
log "This will take a long time. Run in a tmux session."

if [[ ! -f "${OPENIMAGES_DIR}/.download_complete" ]]; then
    mkdir -p "${OPENIMAGES_DIR}"

    MLPERF_DL_SCRIPT="${RETINANET_SCRIPTS}/download_openimages_mlperf.sh"
    if [[ -f "${MLPERF_DL_SCRIPT}" ]]; then
        log "Using MLCommons download script: ${MLPERF_DL_SCRIPT}"
        # The script expects to be run from the retinanet src dir
        # and writes data to the current directory or DATASET_DIR
        pushd "${RETINANET_SRC}" > /dev/null
        DATASET_DIR="${OPENIMAGES_DIR}" bash "${MLPERF_DL_SCRIPT}"
        popd > /dev/null
    else
        log "ERROR: MLCommons download script not found at ${MLPERF_DL_SCRIPT}"
        log "Available scripts: $(ls ${RETINANET_SCRIPTS}/)"
        exit 1
    fi

    touch "${OPENIMAGES_DIR}/.download_complete"
    log "OpenImages download complete."
else
    log "OpenImages already downloaded (found .download_complete marker)."
fi

# ---------------------------------------------------------------------------
# 2. Download RetinaNet backbone checkpoint
# ---------------------------------------------------------------------------
CKPT_FILE="${RETINANET_CHECKPOINT}"
if [[ ! -f "${CKPT_FILE}" ]]; then
    log "Downloading RetinaNet backbone checkpoint..."
    mkdir -p "$(dirname "${CKPT_FILE}")"
    BACKBONE_SCRIPT="${RETINANET_SCRIPTS}/download_backbone.sh"
    if [[ -f "${BACKBONE_SCRIPT}" ]]; then
        CHECKPOINT_DIR="$(dirname "${CKPT_FILE}")" bash "${BACKBONE_SCRIPT}"
    else
        # Fallback: download directly from PyTorch hub
        CKPT_URL="https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
        curl -C - -L -o "${CKPT_FILE}" "${CKPT_URL}"
    fi
    log "Backbone checkpoint saved to $(dirname "${CKPT_FILE}")"
else
    log "Backbone checkpoint already present."
fi

# ---------------------------------------------------------------------------
# 3. Verify dataset structure
# ---------------------------------------------------------------------------
log "Verifying dataset structure..."
for d in "${OPENIMAGES_DIR}/train" "${OPENIMAGES_DIR}/validation" \
          "${OPENIMAGES_DIR}/annotations"; do
    if [[ -d "${d}" ]]; then
        COUNT=$(find "${d}" -type f | wc -l)
        log "  ${d}: ${COUNT} files"
    else
        log "  WARNING: ${d} not found"
    fi
done

log ""
log "Data preparation complete."
log "  OpenImages dir : ${OPENIMAGES_DIR}"
log "  ResNet-50 ckpt : ${CKPT_FILE}"
