#!/usr/bin/env bash
# =============================================================================
# 05_run_all_benchmarks.sh
# Orchestrate all MLPerf Training runs across all GPU configurations.
# Runs both benchmarks (Llama 3.1 8B Pre-Training + LLM Fine-Tuning) for 1, 2, 4, 8 GPUs,
# 10 runs each.  Results are aggregated at the end.
#
# Usage:
#   ./05_run_all_benchmarks.sh                         # all benchmarks, all GPU configs
#   ./05_run_all_benchmarks.sh llama31                 # Llama 3.1 8B pre-training only
#   ./05_run_all_benchmarks.sh llm                     # LLM fine-tuning only
#   ./05_run_all_benchmarks.sh llama31 4 8             # Llama 3.1 8B, 4 and 8 GPUs only
#   ./05_run_all_benchmarks.sh llama31 8 --runs 1      # single run (quick test)
#   ./05_run_all_benchmarks.sh llama31 8 --runs 10     # full 10-run suite
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

# ---------------------------------------------------------------------------
# Strip --runs N from args before positional parsing
# ---------------------------------------------------------------------------
REQUESTED_RUNS=""
FILTERED_ARGS=()
SKIP_NEXT=false
for arg in "$@"; do
    if ${SKIP_NEXT}; then
        REQUESTED_RUNS="${arg}"
        SKIP_NEXT=false
    elif [[ "${arg}" == "--runs="* ]]; then
        REQUESTED_RUNS="${arg#--runs=}"
    elif [[ "${arg}" == "--runs" ]]; then
        SKIP_NEXT=true
    else
        FILTERED_ARGS+=("${arg}")
    fi
done
set -- "${FILTERED_ARGS[@]+"${FILTERED_ARGS[@]}"}"

# Override NUM_RUNS if --runs was given
[[ -n "${REQUESTED_RUNS}" ]] && NUM_RUNS="${REQUESTED_RUNS}"

BENCHMARK_FILTER="${1:-all}"  # all | llama31 | llm
shift 2>/dev/null || true
CUSTOM_GPUS=("$@")            # optional override GPU list

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Fail fast if any GPU is occupied before we start the full suite
check_gpus_idle || { log "ERROR: GPUs not idle. Aborting full benchmark suite."; exit 1; }

# Use command-line GPU override if provided, else default from config.env
if [[ ${#CUSTOM_GPUS[@]} -gt 0 ]]; then
    CONFIGS=("${CUSTOM_GPUS[@]}")
else
    CONFIGS=("${GPU_CONFIGS[@]}")
fi

MASTER_SUMMARY="${SCRATCH_DIR}/results/mlperf_summary_$(date '+%Y%m%d_%H%M%S').txt"
mkdir -p "$(dirname "${MASTER_SUMMARY}")"

echo "MLPerf Training Benchmark Summary" > "${MASTER_SUMMARY}"
echo "Run date: $(date)"               >> "${MASTER_SUMMARY}"
echo "Hardware: H200 x ${#CONFIGS[@]} config(s): ${CONFIGS[*]}" >> "${MASTER_SUMMARY}"
echo "" >> "${MASTER_SUMMARY}"

FAILED_RUNS=()

run_benchmark() {
    local BENCH="$1"
    local NGPU="$2"
    local SCRIPT

    if [[ "${BENCH}" == "llama31" ]]; then
        SCRIPT="${SCRIPT_DIR}/03_run_llama31_pretraining.sh"
    else
        SCRIPT="${SCRIPT_DIR}/04_run_llm_finetuning.sh"
    fi

    log "======================================================"
    log "  Benchmark : ${BENCH}"
    log "  GPUs      : ${NGPU}"
    log "  Runs      : ${NUM_RUNS}"
    log "======================================================"

    if bash "${SCRIPT}" "${NGPU}" "${NUM_RUNS}"; then
        log "  DONE: ${BENCH} @ ${NGPU} GPU(s)"
    else
        log "  FAILED: ${BENCH} @ ${NGPU} GPU(s)"
        FAILED_RUNS+=("${BENCH}_${NGPU}xH200")
    fi
}

# ---------------------------------------------------------------------------
# Execute all requested combinations
# ---------------------------------------------------------------------------
for NGPU in "${CONFIGS[@]}"; do
    if [[ "${BENCHMARK_FILTER}" == "all" || "${BENCHMARK_FILTER}" == "llama31" ]]; then
        run_benchmark "llama31" "${NGPU}"
    fi
    if [[ "${BENCHMARK_FILTER}" == "all" || "${BENCHMARK_FILTER}" == "llm" ]]; then
        run_benchmark "llm" "${NGPU}"
    fi
done

# ---------------------------------------------------------------------------
# Aggregate results from all log directories
# ---------------------------------------------------------------------------
log "Aggregating results..."
python3 "${SCRIPT_DIR}/process_results.py" \
    --log-root "${SCRATCH_DIR}/logs" \
    --output "${MASTER_SUMMARY}" \
    --append

log ""
log "===== ALL BENCHMARKS COMPLETE ====="
log "Master summary: ${MASTER_SUMMARY}"
cat "${MASTER_SUMMARY}"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    log "WARNING: The following run groups had failures:"
    for f in "${FAILED_RUNS[@]}"; do
        log "  - ${f}"
    done
    exit 1
fi
