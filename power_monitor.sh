#!/bin/bash
# Power + training monitor — runs locally, polls BMCs and mlperf1 via SSH
BMC1="<BMC1_IP>"
BMC2="<BMC2_IP>"
CREDS="root:<BMC_PASSWORD>"
URL="/redfish/v1/Chassis/PlatformSensors/Sensors/power_PWR_System"
LOG="/Users/mikcochr/gitprojects/mlperf-h200/power_log.csv"
INTERVAL=10

echo "timestamp,n1_watts,n2_watts,total_watts" > "$LOG"

min_n1=99999; max_n1=0; sum_n1=0
min_n2=99999; max_n2=0; sum_n2=0
min_tot=99999; max_tot=0; sum_tot=0
count=0

echo "[$(date +%H:%M:%S)] Power logger started — sampling every ${INTERVAL}s"
echo "[$(date +%H:%M:%S)] Log: $LOG"
echo ""
printf "%-10s  %7s %7s %8s  |  %8s %8s %8s  n\n" \
    "Time" "N1(W)" "N2(W)" "Total(W)" "Min(W)" "Avg(W)" "Max(W)"
printf '%s\n' "--------------------------------------------------------------------------"

while true; do
    n1=$(curl -sk --connect-timeout 5 -u "$CREDS" "https://$BMC1$URL" | jq -r '.Reading // 0' 2>/dev/null)
    n2=$(curl -sk --connect-timeout 5 -u "$CREDS" "https://$BMC2$URL" | jq -r '.Reading // 0' 2>/dev/null)
    n1=${n1:-0}; n2=${n2:-0}
    tot=$((n1 + n2))
    ts=$(date +"%Y-%m-%dT%H:%M:%S")
    echo "$ts,$n1,$n2,$tot" >> "$LOG"
    count=$((count + 1))

    ((n1 < min_n1)) && min_n1=$n1
    ((n1 > max_n1)) && max_n1=$n1
    sum_n1=$((sum_n1 + n1))

    ((n2 < min_n2)) && min_n2=$n2
    ((n2 > max_n2)) && max_n2=$n2
    sum_n2=$((sum_n2 + n2))

    ((tot < min_tot)) && min_tot=$tot
    ((tot > max_tot)) && max_tot=$tot
    sum_tot=$((sum_tot + tot))

    avg_n1=$((sum_n1  / count))
    avg_n2=$((sum_n2  / count))
    avg_tot=$((sum_tot / count))

    printf "%-10s  %7d %7d %8d  |  %8d %8d %8d  %d\n" \
        "$(date +%H:%M:%S)" "$n1" "$n2" "$tot" "$min_tot" "$avg_tot" "$max_tot" "$count"

    # Every 10 samples also print training progress
    if (( count % 10 == 0 )); then
        prog=$(ssh -T mlperf1 'RUN_DIR=$(ls -td /opt/mlperf/logs/retinanet/retinanet_16xH200_* 2>/dev/null | head -1) && strings "$RUN_DIR/run_1.log" 2>/dev/null | grep "Epoch:" | tail -1' 2>/dev/null)
        err=$(ssh -T mlperf1 'RUN_DIR=$(ls -td /opt/mlperf/logs/retinanet/retinanet_16xH200_* 2>/dev/null | head -1) && strings "$RUN_DIR/run_1.log" 2>/dev/null | grep -E "killed by signal|run_stop|ChildFailed" | tail -1' 2>/dev/null)
        printf '%s\n' "--------------------------------------------------------------------------"
        echo "  TRAINING: ${prog:-no progress yet}"
        [[ -n "$err" ]] && echo "  *** $err"
        printf '%s\n' "--------------------------------------------------------------------------"
    fi

    sleep "$INTERVAL"
done
