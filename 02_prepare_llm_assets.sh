#!/usr/bin/env bash
# =============================================================================
# 02_prepare_llm_assets.sh
# Download Llama 2 70B fused-QKV weights and pre-tokenized SCROLLS GovReport
# dataset for the MLPerf LLM Fine-Tuning benchmark via MLCommons R2 downloader.
#
# PREREQUISITES:
#   MLCommons membership — browser auth via Cloudflare Access is required.
#   The script will print a URL; open it in any browser to authenticate.
#   Auth is cached in ~/.cloudflared/ for subsequent runs.
#
# Approximate sizes:
#   Llama 2 70B fused-QKV weights: ~130 GB
#   SCROLLS GovReport (pre-tokenized 8k): ~1.5 GB
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

MLC_DOWNLOADER_URL="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"
MODEL_URI="https://llama2.mlcommons-storage.org/metadata/llama-2-70b-fused-qkv-mlperf.uri"
SCROLLS_URI="https://llama2.mlcommons-storage.org/metadata/scrolls-gov-report-8k.uri"

# ---------------------------------------------------------------------------
# 1. Ensure wget is available (downloader dependency)
# ---------------------------------------------------------------------------
if ! command -v wget &>/dev/null; then
    log "Installing wget..."
    apt-get install -y --no-install-recommends wget
fi

# ---------------------------------------------------------------------------
# 2. Download Llama 2 70B fused-QKV weights via MLCommons R2 downloader
# ---------------------------------------------------------------------------
if [[ ! -f "${LLAMA2_MODEL_DIR}/config.json" ]]; then
    log "Downloading Llama 2 70B fused-QKV weights to ${LLAMA2_MODEL_DIR} (~130 GB)..."
    log ">>> Cloudflare Access auth required — open the printed URL in your browser <<<"
    log "This will take a long time. Run in tmux."
    mkdir -p "${LLAMA2_MODEL_DIR}"
    bash <(curl -s "${MLC_DOWNLOADER_URL}") \
        -d "${LLAMA2_MODEL_DIR}" \
        "${MODEL_URI}"
    log "Llama 2 70B download complete."
else
    log "Llama 2 70B weights already present at ${LLAMA2_MODEL_DIR}."
fi

# ---------------------------------------------------------------------------
# 3. Download pre-tokenized SCROLLS GovReport 8k dataset
# ---------------------------------------------------------------------------
if [[ ! -f "${GOVREPORT_DATA_DIR}/.download_complete" ]]; then
    log "Downloading SCROLLS GovReport 8k dataset to ${GOVREPORT_DATA_DIR} (~1.5 GB)..."
    mkdir -p "${GOVREPORT_DATA_DIR}"
    bash <(curl -s "${MLC_DOWNLOADER_URL}") \
        -d "${GOVREPORT_DATA_DIR}" \
        "${SCROLLS_URI}"
    touch "${GOVREPORT_DATA_DIR}/.download_complete"
    log "GovReport dataset download complete."
else
    log "GovReport dataset already present."
fi

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
log "Verifying assets..."
log "  Llama 2 70B : $(du -sh "${LLAMA2_MODEL_DIR}" 2>/dev/null | cut -f1) at ${LLAMA2_MODEL_DIR}"
log "  GovReport   : $(du -sh "${GOVREPORT_DATA_DIR}" 2>/dev/null | cut -f1) at ${GOVREPORT_DATA_DIR}"

log ""
log "LLM asset preparation complete."
