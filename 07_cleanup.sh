#!/usr/bin/env bash
# =============================================================================
# 07_cleanup.sh
# Remove scratch artifacts from /opt/mlperf that are no longer needed.
# Safe to run between benchmark sessions. Does NOT touch datasets, models,
# Docker images, or official result/log directories.
#
# Usage:
#   bash 07_cleanup.sh            # dry-run (shows what would be removed)
#   bash 07_cleanup.sh --execute  # actually deletes
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

DRY_RUN=true
[[ "${1:-}" == "--execute" ]] && DRY_RUN=false

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
remove() {
    local target="$1"
    local desc="${2:-}"
    if [[ ! -e "${target}" ]]; then return; fi
    local size
    size=$(du -sh "${target}" 2>/dev/null | cut -f1)
    if ${DRY_RUN}; then
        echo "  [DRY-RUN] would remove: ${target}  (${size})  ${desc}"
    else
        echo "  removing: ${target}  (${size})"
        rm -rf "${target}"
    fi
}

log "====================================================="
log "  MLPerf Cleanup — Scratch /opt/mlperf"
${DRY_RUN} && log "  Mode: DRY-RUN (pass --execute to actually delete)"
log "====================================================="

# ---------------------------------------------------------------------------
# 1. Large one-off setup/download logs (no replay value)
# ---------------------------------------------------------------------------
log ""
log "--- Setup / Download Logs ---"
for f in \
    "${SCRATCH_DIR}/setup_mlperf1.log" \
    "${SCRATCH_DIR}/setup_mlperf2.log" \
    "${SCRATCH_DIR}/download_openimages.log" \
    "${SCRATCH_DIR}/coco_export.log" \
    "${SCRATCH_DIR}/llm_download.log" \
    "${SCRATCH_DIR}/cf_auth.log" \
    "${SCRATCH_DIR}/rsync_train_images.log" \
    "${SCRATCH_DIR}/rsync_val_images.log" \
    "${SCRATCH_DIR}/rsync_govreport.log" \
    "${SCRATCH_DIR}/rsync_llama2.log" \
    "${SCRATCH_DIR}/rsync_models.log" \
    "${SCRATCH_DIR}/rsync_openimages.log"; do
    remove "${f}"
done

# ---------------------------------------------------------------------------
# 2. Image scanning artifacts (bad_images_*.txt, scan*.log)
# ---------------------------------------------------------------------------
log ""
log "--- Image Scan Artifacts ---"
for f in "${SCRATCH_DIR}"/bad_images_*.txt \
          "${SCRATCH_DIR}"/scan*.log \
          "${SCRATCH_DIR}"/scan_rank*.log; do
    [[ -e "${f}" ]] && remove "${f}"
done

# ---------------------------------------------------------------------------
# 3. RetinaNet logs (large — 979 MB — keep results, remove raw training logs)
# ---------------------------------------------------------------------------
log ""
log "--- RetinaNet Training Logs (raw Docker output) ---"
for f in "${LOG_DIR}"/retinanet/*/run_*_docker.log; do
    [[ -e "${f}" ]] && remove "${f}" "(raw Docker output; wall_times.txt kept)"
done

# ---------------------------------------------------------------------------
# 4. Llama 3.1 NPY index cache (auto-rebuilt at run time)
# ---------------------------------------------------------------------------
log ""
log "--- NPY Index Cache (auto-rebuilt) ---"
remove "${LLAMA31_NPY_INDEX_DIR}" "(rebuilt automatically at run start)"

# ---------------------------------------------------------------------------
# 5. Any partial/failed run Docker logs (size 0 or runs that never started)
# ---------------------------------------------------------------------------
log ""
log "--- Empty or failed run log dirs ---"
for dir in "${LOG_DIR}"/llm_finetuning/*/  \
            "${LOG_DIR}"/llama31_pretraining/*/; do
    [[ ! -d "${dir}" ]] && continue
    # A run dir with no wall_times data and only 0-byte logs is a failed start
    local_wt="${dir}/wall_times.txt"
    if [[ -f "${local_wt}" ]] && grep -q "^[0-9]" "${local_wt}" 2>/dev/null; then
        continue  # has at least one completed run — keep
    fi
    docker_logs=$(find "${dir}" -name "run_*_docker.log" -size +1k 2>/dev/null | wc -l)
    if [[ "${docker_logs}" -eq 0 ]]; then
        remove "${dir}" "(no successful runs, no substantive logs)"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
if ${DRY_RUN}; then
    log "Dry-run complete. Re-run with --execute to apply."
else
    log "Cleanup complete."
fi
log ""
log "Disk usage after cleanup:"
df -h "${SCRATCH_DIR}" | tail -1
