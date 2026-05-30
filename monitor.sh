#!/bin/bash
# MLPerf Training Monitor — LLM Fine-Tuning + Llama 3.1 8B Pretraining
# Usage: bash monitor.sh [interval_seconds]   (default: 30s)
#
# Tracks: GPU util/mem/power, benchmark progress, data download,
#         Docker image build, and BMC system power on both nodes.

N1="root@${NODE1_IP:?Set NODE1_IP}"
N2="root@${NODE2_IP:?Set NODE2_IP}"
BMC1="${BMC1_IP:?Set BMC1_IP}"
BMC2="${BMC2_IP:?Set BMC2_IP}"
BMC_CREDS="${BMC_CREDS:?Set BMC_CREDS (user:pass)}"
INTERVAL="${1:-30}"

SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
divider() { echo "────────────────────────────────────────────────────────"; }

bmc_power() {
    local bmc=$1 sys_w gpu_w=0 w sensor
    sys_w=$(curl -sk --connect-timeout 3 -u "$BMC_CREDS" \
        "https://$bmc/redfish/v1/Chassis/PlatformSensors/Sensors/power_PWR_System" \
        | jq -r '.Reading // "?"' 2>/dev/null)
    for sensor in power_PWR_GB_GPU1 power_PWR_GB_GPU2 power_PWR_GB_GPU3 power_PWR_GB_GPU4 \
                  power_PWR_GB_GPU5 power_PWR_GB_GPU6 power_PWR_GB_GPU7 power_PWR_GB_GPU8; do
        w=$(curl -sk --connect-timeout 2 -u "$BMC_CREDS" \
            "https://$bmc/redfish/v1/Chassis/PlatformSensors/Sensors/$sensor" \
            | jq -r '.Reading // 0' 2>/dev/null)
        gpu_w=$(( gpu_w + ${w:-0} ))
    done
    echo "sys=${sys_w:-?}W  gpu_total=${gpu_w}W"
}

gpu_table() {
    local host=$1
    ssh $SSH_OPTS -T "$host" \
        'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
         --format=csv,noheader,nounits 2>/dev/null' 2>/dev/null \
    | while IFS=',' read -r idx util mem_used mem_tot temp pwr; do
        printf "  GPU%-2s  util:%-4s%%  mem:%6s/%-6sMiB  %s°C  %sW\n" \
            "${idx// /}" "${util// /}" "${mem_used// /}" "${mem_tot// /}" \
            "${temp// /}" "${pwr// /}"
      done
}

bench_summary() {
    local host=$1 log_dir=$2 bench_label=$3 progress_grep=$4
    local latest_run wall_times progress
    latest_run=$(ssh $SSH_OPTS -T "$host" \
        "ls -td ${log_dir}/* 2>/dev/null | head -1" 2>/dev/null)
    if [[ -z "$latest_run" ]]; then
        echo "  Not started yet"; return
    fi
    echo "  Run dir : $(basename "$latest_run")"
    wall_times=$(ssh $SSH_OPTS -T "$host" \
        "grep -v '^#' '${latest_run}/wall_times.txt' 2>/dev/null" 2>/dev/null)
    if [[ -n "$wall_times" ]]; then
        echo "$wall_times" | while IFS=',' read -r run status mins; do
            printf "    run %-3s  %-8s  %s min\n" "$run" "$status" "$mins"
        done
    fi
    local latest_docker
    latest_docker=$(ssh $SSH_OPTS -T "$host" \
        "ls '${latest_run}'/run_*_docker.log 2>/dev/null | tail -1" 2>/dev/null)
    if [[ -n "$latest_docker" ]]; then
        progress=$(ssh $SSH_OPTS -T "$host" \
            "grep -E '${progress_grep}' '${latest_docker}' 2>/dev/null | tail -3" 2>/dev/null)
        [[ -n "$progress" ]] && echo "  Latest  :" && echo "$progress" | sed 's/^/    /'
    fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while true; do
    clear
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    echo "╔══════════════════════════════════════════════════════╗"
    echo "  MLPerf Training Monitor   [$TS]"
    echo "╚══════════════════════════════════════════════════════╝"

    # --- Job status ---
    echo ""; divider; echo "  JOB STATUS  (Node1=${NODE1_IP} | Node2=${NODE2_IP})"
    divider
    echo "  Node 1:"
    ssh $SSH_OPTS -T "$N1" 'tmux ls 2>/dev/null || echo "  no sessions"' 2>/dev/null | sed 's/^/    /'
    echo "  Node 2:"
    ssh $SSH_OPTS -T "$N2" 'tmux ls 2>/dev/null || echo "  no sessions"' 2>/dev/null | sed 's/^/    /'

    # --- Node 1 GPUs ---
    echo ""; divider; echo "  NODE 1  (${NODE1_IP})"
    divider
    gpu_table "$N1"

    # --- Node 2 GPUs ---
    echo ""; divider; echo "  NODE 2  (${NODE2_IP})"
    divider
    gpu_table "$N2"

    # --- LLM Fine-tuning ---
    echo ""; divider; echo "  LLM FINE-TUNING  (Llama 2 70B, target loss ≤ 0.925)"
    divider
    bench_summary "$N1" "/opt/mlperf/logs/llm_finetuning" "llm" \
        "step|loss|eval_loss|run_stop|run_start"

    # --- Llama 3.1 8B Pretraining (Node 2) ---
    echo ""; divider; echo "  LLAMA 3.1 8B PRETRAINING  (Node 2 — target log-ppl ≤ 3.3)"
    divider
    bench_summary "$N2" "/opt/mlperf/logs/llama31_pretraining" "llama31" \
        "train_loss|eval_loss|step_time|run_stop|run_start|global_step"
    # Pull live NeMo training line from rank-0 torchelastic log inside container
    NEMO_STEP=$(ssh $SSH_OPTS -T "$N2" \
        'CID=$(docker ps -q 2>/dev/null | head -1); \
         [[ -n "$CID" ]] && docker exec "$CID" \
           sh -c "find /root/.nemo_run/experiments -path \"*/torchelastic/*/attempt_0/0/stdout.log\" 2>/dev/null \
                  | grep -v build_data | head -1 \
                  | xargs grep -E \"iteration|train_loss\" 2>/dev/null | tail -1"' \
        2>/dev/null)
    [[ -n "$NEMO_STEP" ]] && echo "  NeMo  : $NEMO_STEP"
    # Latest eval from MLPerf log
    LATEST_EVAL=$(ssh $SSH_OPTS -T "$N2" \
        'grep "eval_accuracy" $(ls -t /opt/mlperf/results/llama31_pretraining/run_*/mlperf_llama31_8b.log 2>/dev/null | head -1) 2>/dev/null | tail -1 | grep -oP ".eval_accuracy.*" | head -c 80' \
        2>/dev/null)
    [[ -n "$LATEST_EVAL" ]] && echo "  Eval  : $LATEST_EVAL"

    # --- Data download ---
    DL_DONE=$(ssh $SSH_OPTS -T "$N1" \
        "test -f /opt/mlperf/llama31-c4-preprocessed/.download_complete && echo yes" 2>/dev/null)
    if [[ "$DL_DONE" != "yes" ]]; then
        echo ""; divider; echo "  DATA DOWNLOAD  (C4 preprocessed + tokenizer)"
        divider
        ssh $SSH_OPTS -T "$N1" \
            "tail -4 /opt/mlperf/logs/llama31_data_download.log 2>/dev/null | grep -v '^$'" \
            2>/dev/null | sed 's/^/  /'
    fi

    # --- Docker image build ---
    IMG_READY=$(ssh $SSH_OPTS -T "$N1" \
        "docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep 'mlperf-llama31-pretraining:h200'" \
        2>/dev/null)
    if [[ -z "$IMG_READY" ]]; then
        echo ""; divider; echo "  DOCKER IMAGE BUILD  (mlperf-llama31-pretraining:h200)"
        divider
        BSTEP=$(ssh $SSH_OPTS -T "$N1" \
            "grep '^Step' /opt/mlperf/logs/llama31_image_build.log 2>/dev/null | tail -1" 2>/dev/null)
        echo "  ${BSTEP:-building...}"
    else
        echo ""; echo "  ✓ Docker image ready: $IMG_READY"
    fi

    # --- BMC power ---
    echo ""; divider; echo "  POWER  (BMC Redfish)"
    divider
    printf "  Node 1: "; bmc_power "$BMC1"
    printf "  Node 2: "; bmc_power "$BMC2"

    echo ""
    echo "  [Refreshes every ${INTERVAL}s — Ctrl+C to exit]"
    sleep "$INTERVAL"
done
