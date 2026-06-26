#!/usr/bin/env bash
# =============================================================================
# 01_download_llama31_data.sh
# Download the pre-tokenized C4/en dataset and Llama 3.1 8B tokenizer
# required for the MLPerf Small LLM Pre-Training benchmark.
#
# Both assets are served via the MLCommons R2 Downloader.
# No HuggingFace token is required — the weights are not needed since
# the benchmark trains from random initialization to target perplexity.
#
# Approximate download sizes:
#   Pre-tokenized C4/en (megatron .bin/.idx): ~350 GB
#   Llama 3.1 8B tokenizer:                   ~  1 MB
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

R2_DOWNLOADER_URL="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"
MLCOMMONS_STORAGE="https://training.mlcommons-storage.org"

# ---------------------------------------------------------------------------
# 1. Download pre-tokenized C4/en dataset
# ---------------------------------------------------------------------------
if [[ ! -f "${C4_PREPROCESSED_DIR}/.download_complete" ]]; then
    log "Downloading pre-tokenized C4/en dataset to ${C4_PREPROCESSED_DIR}..."
    log "Expected size: ~350 GB. Run in a tmux session."
    mkdir -p "${C4_PREPROCESSED_DIR}"
    pushd "${C4_PREPROCESSED_DIR}" > /dev/null
    bash <(curl -s "${R2_DOWNLOADER_URL}") \
        -d llama3_1_8b_preprocessed_c4_dataset \
        "${MLCOMMONS_STORAGE}/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri"
    popd > /dev/null
    touch "${C4_PREPROCESSED_DIR}/.download_complete"
    log "C4 dataset download complete."
else
    log "C4 preprocessed dataset already present (found .download_complete marker)."
fi

# ---------------------------------------------------------------------------
# 2. Download Llama 3.1 8B tokenizer
# ---------------------------------------------------------------------------
if [[ ! -f "${LLAMA31_TOKENIZER_DIR}/.download_complete" ]]; then
    log "Downloading Llama 3.1 8B tokenizer to ${LLAMA31_TOKENIZER_DIR}..."
    mkdir -p "${LLAMA31_TOKENIZER_DIR}"
    pushd "${LLAMA31_TOKENIZER_DIR}" > /dev/null
    bash <(curl -s "${R2_DOWNLOADER_URL}") \
        -d llama3_1_8b_tokenizer \
        "${MLCOMMONS_STORAGE}/metadata/llama-3-1-8b-tokenizer.uri"
    popd > /dev/null
    touch "${LLAMA31_TOKENIZER_DIR}/.download_complete"
    log "Tokenizer download complete."
else
    log "Tokenizer already present (found .download_complete marker)."
fi

# ---------------------------------------------------------------------------
# 3. Verify asset structure
# ---------------------------------------------------------------------------
log "Verifying asset structure..."
for d in "${C4_PREPROCESSED_DIR}" "${LLAMA31_TOKENIZER_DIR}"; do
    if [[ -d "${d}" ]]; then
        COUNT=$(find "${d}" -type f | wc -l)
        log "  ${d}: ${COUNT} files"
    else
        log "  WARNING: ${d} not found"
    fi
done

log ""
log "Data preparation complete."
log "  C4 preprocessed : ${C4_PREPROCESSED_DIR}"
log "  Tokenizer        : ${LLAMA31_TOKENIZER_DIR}"
