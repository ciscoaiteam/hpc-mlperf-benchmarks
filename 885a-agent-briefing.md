# MLPerf 885A Agent Briefing — Next Benchmark Cycle

## Servers

- **885A Node 1 (H200-1):** `root@<NODE1_MGMT_IP>` — SSH key already installed
- **885A Node 2 (H200-2):** `root@<NODE2_MGMT_IP>` — SSH key already installed

Both are Cisco UCS C885A M8, **8× H200 SXM5 per node**, HGX architecture with full NVLink mesh
(NV18 connections between all 8 GPUs). AMD EPYC 9575F × 2 sockets, 256 CPUs, 8 NUMA nodes
(one per GPU).

---

## Backup

The 312GB backup archive is at `/root/mlperf_backup_20260506_174928.tar.gz` on **885A-2 only**
(<NODE2_MGMT_IP>). `/opt/mlperf/` is empty on both nodes — they were reprovisioned. The backup
must be extracted on both nodes.

**Archive internal path prefix:** `tmp/mlperf_backup_xhg6bB/` (2 path components deep)

### Step 1 — Extract on 885A-2

```bash
mkdir -p /opt/mlperf
tmux new-session -d -s extract \
  "tar -xzf /root/mlperf_backup_20260506_174928.tar.gz --strip-components=2 -C /opt/mlperf/ \
   2>&1 | tee /root/extract.log; echo EXIT:\$?"
```

### Step 2 — Transfer backup to 885A-1, then extract there

```bash
# On 885A-1 — pull archive from 885A-2 (need SSH key from 885A-1 → 885A-2 first):
# Bootstrap key from your laptop:
PUB=$(ssh root@<NODE1_MGMT_IP> 'cat ~/.ssh/id_ed25519.pub 2>/dev/null || \
  (ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q && cat ~/.ssh/id_ed25519.pub)')
ssh root@<NODE2_MGMT_IP> "echo '$PUB' >> ~/.ssh/authorized_keys"

# Then on 885A-1, pull and extract:
ssh root@<NODE1_MGMT_IP> 'tmux new-session -d -s xfer \
  "rsync -avz --progress -e \"ssh -o StrictHostKeyChecking=no\" \
   root@<NODE2_MGMT_IP>:/root/mlperf_backup_20260506_174928.tar.gz /root/ \
   2>&1 | tee /root/xfer.log; echo DONE:\$?"'

# After transfer completes, extract on 885A-1:
ssh root@<NODE1_MGMT_IP> 'mkdir -p /opt/mlperf && \
  tmux new-session -d -s extract \
  "tar -xzf /root/mlperf_backup_20260506_174928.tar.gz --strip-components=2 \
   -C /opt/mlperf/ 2>&1 | tee /root/extract.log; echo EXIT:\$?"'
```

### What's in the backup (from MANIFEST.txt)

```
/opt/mlperf/  (after extraction)
├── docker-images/
│   ├── mlperf-llama31-pretraining_h200.tar   (44.8 GB)
│   ├── mlperf-llm-finetuning_h200.tar        (34.3 GB)
│   └── mlperf-retinanet_h200.tar             (41.4 GB — DO NOT load, not needed)
├── llama31-c4-preprocessed/                  (C4 dataset, preprocessed for NeMo)
├── llama31-tokenizer/                        (Llama 3.1 tokenizer)
├── llama2-70b/
│   └── Llama2-70b-fused-qkv-mlperf/         (model weights for fine-tuning)
├── govreport/                                (SCROLLS GovReport fine-tuning dataset)
├── mlperf-scripts/                           (original benchmark scripts — superseded by below)
├── results/                                  (previous cycle results — keep for reference)
└── MANIFEST.txt
```

### Step 3 — Load Docker images (on both nodes, after extraction)

```bash
docker load < /opt/mlperf/docker-images/mlperf-llama31-pretraining_h200.tar
docker load < /opt/mlperf/docker-images/mlperf-llm-finetuning_h200.tar
# DO NOT load retinanet
```

---

## Benchmark Scripts

Sync the latest scripts from the laptop to both nodes:

```bash
rsync -avz /Users/mikcochr/gitprojects/mlperf-h200/*.sh \
           /Users/mikcochr/gitprojects/mlperf-h200/config.env \
           root@<NODE1_MGMT_IP>:/root/mlperf-h200/

rsync -avz /Users/mikcochr/gitprojects/mlperf-h200/*.sh \
           /Users/mikcochr/gitprojects/mlperf-h200/config.env \
           root@<NODE2_MGMT_IP>:/root/mlperf-h200/
```

### IMPORTANT — config.env must be patched on 885A nodes

The repo `config.env` is currently set for the 845A MGX platform. Patch it back to 885A HGX
settings on each node after syncing:

```bash
for HOST in <NODE1_MGMT_IP> <NODE2_MGMT_IP>; do
  ssh root@$HOST 'sed -i \
    -e "s/^GPU_CONFIGS=.*/GPU_CONFIGS=(4 8)/" \
    -e "s/^export NCCL_SOCKET_IFNAME=.*/export NCCL_SOCKET_IFNAME=ens211f0np0/" \
    -e "s/^export NCCL_IB_HCA=.*/export NCCL_IB_HCA=mlx5_2/" \
    -e "s/^export NCCL_P2P_LEVEL=.*/export NCCL_P2P_LEVEL=NVL/" \
    -e "s/^MLPERF_MASTER_ADDR=.*/MLPERF_MASTER_ADDR=\"<NODE1_MGMT_IP>\"/" \
    -e "s/^ALL_CPUS=.*/ALL_CPUS=\"0-127,128-255\"/" \
    /root/mlperf-h200/config.env'
done
```

Also restore the full 8-GPU `GPU_CPUS` array and the `>= 8` threshold in `get_numa_args` in
`config.env`. The correct 885A values are:

```bash
ALL_CPUS="0-127,128-255"

declare -A GPU_CPUS=(
    [0]="16-31,144-159"   [1]="32-47,160-175"
    [2]="48-63,176-191"   [3]="0-15,128-143"
    [4]="80-95,208-223"   [5]="96-111,224-239"
    [6]="112-127,240-255" [7]="64-79,192-207"
)

# In get_numa_args — threshold should be >= 8 (not >= 4):
if [[ "${NUM_GPUS}" -ge 8 ]]; then
```

---

## What to Run

**Key change this cycle: GBS=512 (up from 32 last cycle) — already set in config.env**
**MAX_LR=8e-3 (linear scaled from 5e-4 × 512/32) — already set in config.env**

| Node | Benchmark | GPUs | Script | Expected TTT |
|------|-----------|------|--------|--------------|
| H200-2 (<NODE2_MGMT_IP>) | Llama 3.1 8B pretraining | 8× | `03_run_llama31_pretraining.sh 8 1` | ~25–40 min |
| H200-1 (<NODE1_MGMT_IP>) | Llama 3.1 8B pretraining | 4× | `03_run_llama31_pretraining.sh 4 1` | ~50–80 min |
| H200-2 (<NODE2_MGMT_IP>) | LLaMA2 70B LoRA fine-tuning | 8× | `04_run_llm_finetuning.sh 8 1` | ~60–70 min |

Run from `/root/mlperf-h200/` on each respective node. Use `tmux` so runs survive disconnection.

### Convergence targets

- Llama 3.1 pretraining: `val_loss ≤ 3.300` and `run_stop status=success` in MLPerf log
- LLaMA2 fine-tuning: `val_loss ≤ 0.9246` and `run_stop status=success`
- **Exclude RetinaNet from all runs and reporting**

---

## Pull Results When Done

```bash
# Run from laptop:
rsync -avz root@<NODE1_MGMT_IP>:/opt/mlperf/results/ \
      /Users/mikcochr/gitprojects/mlperf-h200/results/node1_885a/

rsync -avz root@<NODE2_MGMT_IP>:/opt/mlperf/results/ \
      /Users/mikcochr/gitprojects/mlperf-h200/results/node2_885a/

cd /Users/mikcochr/gitprojects/mlperf-h200 && git add results/ && git commit -m "885A results: GBS=512 benchmark cycle"
```

---

## Previous Cycle Reference Results (GBS=32, for comparison)

| Benchmark | GPUs | TTT | Final val_loss | Status |
|-----------|------|-----|----------------|--------|
| Llama 3.1 pretraining | 8× H200 SXM5 | 269.8 min | 3.291 | ✅ Success |
| Llama 3.1 pretraining | 4× H200 SXM5 | 525.0 min | 3.291 | ✅ Success |
| LLaMA2 70B fine-tuning | 8× H200 SXM5 | 64.3 min | ≤ 0.9246 | ✅ Success |

MLPerf v5.1 official target (same hardware, GBS=512): **23.6 min** for 8× H200 pretraining.
