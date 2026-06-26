#!/bin/bash
# AMD MI350X — Download Llama 3.1 8B pretraining data
# Run in tmux: tmux new -s llama31_data

set -euo pipefail

MLPERF_DIR="/opt/mlperf"
C4_DIR="${MLPERF_DIR}/llama31-c4-preprocessed"
TOKENIZER_DIR="${MLPERF_DIR}/llama31-tokenizer"

echo "=== Downloading Llama 3.1 8B pretraining data ==="
echo "Target: ${C4_DIR} (~350 GB)"
echo "Start: $(date)"

# Download pre-tokenized C4/en dataset via MLCommons rclone
docker run --rm \
    -v ${C4_DIR}:/data/preprocessed \
    -v ${TOKENIZER_DIR}:/data/tokenizer \
    rocm/amd-mlperf:llama31_8b_training_6.0 \
    bash -c "
        cd /workspace/llm_pretraining || cd /workspace
        # Try MLCommons downloader
        if [ -f scripts/download_dataset.sh ]; then
            bash scripts/download_dataset.sh /data/preprocessed /data/tokenizer
        elif [ -f download.sh ]; then
            bash download.sh
        else
            echo 'Checking for rclone download script...'
            find / -name 'download*.sh' 2>/dev/null | head -5
        fi
    "

echo "Download complete: $(date)"
echo "Dataset size: $(du -sh ${C4_DIR} 2>/dev/null | cut -f1)"
