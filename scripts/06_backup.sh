#!/usr/bin/env bash
# =============================================================================
# 06_backup.sh
# Create a compressed backup of MLPerf scripts, results, logs, and optionally
# all model/dataset files and Docker images for fully self-contained replay.
#
# Usage:
#   bash 06_backup.sh                         # scripts + results + logs only
#   bash 06_backup.sh --with-images           # + Docker images (~120 GB)
#   bash 06_backup.sh --with-data             # + all models/datasets + Docker images
#   bash 06_backup.sh --with-data --dest /mnt/nas
#
# Compression:
#   Uses zstd -T0 (all cores, level 1) for speed — model weights compress <5%.
#   Falls back to pigz (parallel gzip) if zstd is unavailable.
#   Archive extension: .tar.zst (zstd) or .tar.gz (pigz/gzip fallback).
#
# Disk requirements for --with-data:
#   ~380 GB source data + ~120 GB Docker image staging + final archive (~450 GB)
#   Ensure ~600 GB free before running.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
WITH_IMAGES=false
WITH_DATA=false
DEST_DIR="${SCRATCH_DIR}/backups"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-images) WITH_IMAGES=true ;;
        --with-data)   WITH_DATA=true; WITH_IMAGES=true ;;
        --dest)        DEST_DIR="$2"; shift ;;
        --dest=*)      DEST_DIR="${1#--dest=}" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "${DEST_DIR}"

# ---------------------------------------------------------------------------
# Pick compressor: prefer zstd (fast parallel), fall back to pigz, then gzip
# ---------------------------------------------------------------------------
if command -v pigz &>/dev/null; then
    COMPRESS_CMD="pigz -1"
    ARCHIVE_EXT="tar.gz"
    DECOMPRESS_HINT="tar -xzf mlperf_backup_*.tar.gz -C /opt/mlperf-restore/"
elif command -v zstd &>/dev/null; then
    COMPRESS_CMD="zstd -1"
    ARCHIVE_EXT="tar.zst"
    DECOMPRESS_HINT="zstd -d mlperf_backup_*.tar.zst | tar -x -C /opt/mlperf-restore/"
else
    COMPRESS_CMD="gzip -1"
    ARCHIVE_EXT="tar.gz"
    DECOMPRESS_HINT="tar -xzf mlperf_backup_*.tar.gz -C /opt/mlperf-restore/"
fi

DATE=$(date +%Y%m%d_%H%M%S)
ARCHIVE="${DEST_DIR}/mlperf_backup_${DATE}.${ARCHIVE_EXT}"

STAGING=$(mktemp -d /tmp/mlperf_backup_XXXXXX)
trap 'log "Cleaning up staging..."; rm -rf "${STAGING}"' EXIT

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Size estimate up front
# ---------------------------------------------------------------------------
log "====================================================="
log "  MLPerf Backup"
log "  Date      : $(date)"
log "  Archive   : ${ARCHIVE}"
log "  Compressor: ${COMPRESS_CMD%% *}"
log "  With data : ${WITH_DATA}"
log "  With imgs : ${WITH_IMAGES}"
log "====================================================="
log ""
log "Estimating source sizes..."
TOTAL_BYTES=0
add_size() {
    local path="$1" label="$2"
    if [[ -e "${path}" ]]; then
        local s
        s=$(du -sb "${path}" 2>/dev/null | awk '{print $1}')
        local h
        h=$(du -sh "${path}" 2>/dev/null | awk '{print $1}')
        log "  ${h}   ${label}"
        TOTAL_BYTES=$(( TOTAL_BYTES + s ))
    fi
}

add_size "${SCRIPT_DIR}"           "scripts + config"
add_size "${RESULTS_DIR}"          "results"
add_size "${LOG_DIR}"              "logs"
if ${WITH_DATA}; then
    add_size "${LLAMA2_MODEL_DIR%/*}"         "Llama 2 70B weights"
    add_size "${C4_PREPROCESSED_DIR}"         "C4 preprocessed dataset"
    add_size "${LLAMA31_TOKENIZER_DIR}"       "Llama 3.1 tokenizer"
    add_size "${GOVREPORT_DATA_DIR}"          "GovReport dataset"
    add_size "${SCRATCH_DIR}/openimages"      "OpenImages"
    add_size "${SCRATCH_DIR}/models"          "RetinaNet models"
fi
if ${WITH_IMAGES}; then
    log "  ~120 GB  Docker images (staged before archiving)"
fi

TOTAL_GB=$(echo "scale=1; ${TOTAL_BYTES}/1073741824" | bc)
log ""
log "  Estimated source total : ~${TOTAL_GB} GB"
log "  Note: model weights compress <5%; archive will be similar size."
log ""

# ---------------------------------------------------------------------------
# 1. Scripts + config (small, stage to temp)
# ---------------------------------------------------------------------------
log "Collecting scripts and config..."
SCRIPTS_STAGE="${STAGING}/mlperf-scripts"
mkdir -p "${SCRIPTS_STAGE}"
cp "${SCRIPT_DIR}"/*.sh "${SCRIPT_DIR}/config.env" "${SCRIPTS_STAGE}/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Results + logs (small, stage to temp)
# ---------------------------------------------------------------------------
log "Collecting results and logs..."
mkdir -p "${STAGING}/results" "${STAGING}/logs"
[[ -d "${RESULTS_DIR}" ]] && cp -r "${RESULTS_DIR}/." "${STAGING}/results/"
[[ -d "${LOG_DIR}" ]]     && cp -r "${LOG_DIR}/."     "${STAGING}/logs/"

# ---------------------------------------------------------------------------
# 3. Manifest
# ---------------------------------------------------------------------------
log "Writing manifest..."
MANIFEST="${STAGING}/MANIFEST.txt"
{
    echo "MLPerf H200 Backup"
    echo "Generated   : $(date)"
    echo "Host        : $(hostname)"
    echo "Kernel      : $(uname -r)"
    echo "With data   : ${WITH_DATA}"
    echo "With images : ${WITH_IMAGES}"
    echo ""
    echo "=== Docker images present at backup time ==="
    docker images --format "{{.Repository}}:{{.Tag}}  {{.Size}}  {{.ID}}" 2>/dev/null || true
    echo ""
    echo "=== To replay on a new server ==="
    echo "  1. Extract archive:"
    echo "       mkdir -p /opt/mlperf-restore && ${DECOMPRESS_HINT}"
    if ${WITH_IMAGES}; then
        echo "  2. Load Docker images:"
        echo "       docker load < /opt/mlperf-restore/docker-images/mlperf-llama31-pretraining_h200.tar"
        echo "       docker load < /opt/mlperf-restore/docker-images/mlperf-llm-finetuning_h200.tar"
        echo "       docker load < /opt/mlperf-restore/docker-images/mlperf-retinanet_h200.tar"
    else
        echo "  2. Build Docker images: bash mlperf-scripts/00_setup_environment.sh"
    fi
    if ${WITH_DATA}; then
        echo "  3. Data already included — update paths in config.env to match new server."
    else
        echo "  3. Download data: bash mlperf-scripts/01_download_llama31_data.sh"
        echo "                    bash mlperf-scripts/02_prepare_llm_assets.sh"
    fi
    echo "  4. Run: bash mlperf-scripts/05_run_all_benchmarks.sh llama31 8 --runs 1"
    echo ""
    echo "=== Results summary ==="
    cat "${RESULTS_DIR}"/mlperf_summary_*.txt 2>/dev/null | tail -60 || echo "  (no summary found)"
} > "${MANIFEST}"

# ---------------------------------------------------------------------------
# 4. Docker images — save to staging dir (uncompressed; zstd handles it)
# ---------------------------------------------------------------------------
if ${WITH_IMAGES}; then
    log "Saving Docker images to staging (this takes 10–20 min per image)..."
    DOCKER_STAGE="${STAGING}/docker-images"
    mkdir -p "${DOCKER_STAGE}"

    save_image() {
        local IMAGE="$1"
        local FNAME="${DOCKER_STAGE}/$(echo "${IMAGE}" | tr ':/' '_').tar"
        if docker image inspect "${IMAGE}" &>/dev/null; then
            log "  Saving ${IMAGE} → $(basename "${FNAME}")"
            docker save "${IMAGE}" > "${FNAME}"
            log "  Done: $(du -sh "${FNAME}" | cut -f1)"
        else
            log "  SKIP: ${IMAGE} not found locally"
        fi
    }

    save_image "${LLAMA31_IMAGE}"
    save_image "${LLM_IMAGE}"
    save_image "mlperf-retinanet:h200"
fi

# ---------------------------------------------------------------------------
# 5. Build tar include list (staging + optional large data dirs)
# ---------------------------------------------------------------------------
log "Building archive..."
TAR_SOURCES=("${STAGING}")
if ${WITH_DATA}; then
    [[ -d "${LLAMA2_MODEL_DIR%/*}" ]]    && TAR_SOURCES+=("${LLAMA2_MODEL_DIR%/*}")
    [[ -d "${C4_PREPROCESSED_DIR}" ]]    && TAR_SOURCES+=("${C4_PREPROCESSED_DIR}")
    [[ -d "${LLAMA31_TOKENIZER_DIR}" ]]  && TAR_SOURCES+=("${LLAMA31_TOKENIZER_DIR}")
    [[ -d "${GOVREPORT_DATA_DIR}" ]]     && TAR_SOURCES+=("${GOVREPORT_DATA_DIR}")
    [[ -d "${SCRATCH_DIR}/openimages" ]] && TAR_SOURCES+=("${SCRATCH_DIR}/openimages")
    [[ -d "${SCRATCH_DIR}/models" ]]     && TAR_SOURCES+=("${SCRATCH_DIR}/models")
fi

log "  Sources: ${TAR_SOURCES[*]}"
log "  Compressing with: ${COMPRESS_CMD%% *} (this will take a while for large data)..."
log "  Started: $(date)"

tar -cf - "${TAR_SOURCES[@]}" | ${COMPRESS_CMD} > "${ARCHIVE}"

ARCHIVE_SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
log ""
log "====================================================="
log "  Backup complete!"
log "  Finished : $(date)"
log "  Archive  : ${ARCHIVE}"
log "  Size     : ${ARCHIVE_SIZE}"
log "====================================================="
log ""
log "  To copy to another server (over high-speed link):"
log "    rsync -avP --progress ${ARCHIVE} user@new-server:/destination/"
log "  To extract:"
log "    ${DECOMPRESS_HINT}"
