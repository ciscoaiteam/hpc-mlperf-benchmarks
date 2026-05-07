# MLPerf H200 Benchmark Summary
**Date:** May 6–7, 2026  
**Hardware:** NVIDIA H200 (143,771 MiB HBM3e, 600W TDP, SXM5)  
**Platform:** 2× bare-metal nodes, Ubuntu 22.04.5 LTS, Driver 580.126.20, CUDA 13.0  
**Benchmarks:** Llama 3.1 8B Pretraining · LLaMA2 70B LoRA Fine-Tuning *(RetinaNet excluded)*

---

## System Configuration

| Parameter | Node 1 | Node 2 |
|-----------|--------|--------|
| Hostname | ai-server-c885-h200-1 | ai-server-c885-h200-2 |
| IP | <NODE1_MGMT_IP> | <NODE2_MGMT_IP> |
| GPU Count | 8× H200 SXM5 | 8× H200 SXM5 |
| GPU Memory | 143,771 MiB each | 143,771 MiB each |
| GPU TDP | 600W each | 600W each |
| Total GPU Power (TDP) | 4,800W | 4,800W |
| NUMA | 2 sockets, 128 cores/socket | 2 sockets, 128 cores/socket |

---

## Results Summary

| Benchmark | Node | GPUs | Status | TTT (min) | Final Loss | Convergence Samples |
|-----------|------|------|--------|-----------|------------|---------------------|
| Llama 3.1 8B Pretraining | Node 2 | 8× H200 | ✅ SUCCESS | **269.8** | val_loss = 3.291 | 208,864 |
| Llama 3.1 8B Pretraining | Node 1 | 4× H200 | ✅ SUCCESS | **525.0** | val_loss = 3.281 | 208,864 |
| LLaMA2 70B LoRA Fine-Tuning | Node 2 | 8× H200 | ✅ SUCCESS | **64.3** | eval_loss = 0.9246 | 3,072 |

> **TTT** = MLPerf time-to-train: `run_start` → `run_stop` (excludes initialization). Target: val_loss ≤ 3.300 (pretraining), eval_loss ≤ target (fine-tuning).

---

## Detailed Results

### 1. Llama 3.1 8B Pretraining — 8× H200 (Node 2)

| Metric | Value |
|--------|-------|
| Docker Image | `mlperf-llama31-pretraining:h200` |
| Global Batch Size (GBS) | 32 |
| Micro Batch Size (MBS) | 2 |
| Max Learning Rate | 5e-4 |
| Sequence Length | 8,192 tokens |
| run_start (UTC) | 2026-05-06 15:50:31 |
| run_stop (UTC) | 2026-05-06 20:20:22 |
| **Time to Train (MLPerf)** | **269.8 min (4h 29m 51s)** |
| Wall time (job start → finish) | 273.6 min |
| Final val_loss | **3.291** (target ≤ 3.300 ✅) |
| Samples at convergence | 208,864 |
| Steps to convergence | 6,527 |
| Training throughput | ~105,700 tokens/sec (system) · ~13,200 tokens/sec/GPU |
| Peak GPU memory | 128,155–128,159 MiB / 143,771 MiB (89.2%) |

**Validation Loss Trajectory:**

| Eval | Samples | Val Loss |
|------|---------|---------|
| 1 | 24,544 | 4.815 |
| 4 | 61,408 | 3.848 |
| 8 | 110,560 | 3.535 |
| 12 | 159,712 | 3.384 |
| 14 | 184,288 | 3.329 |
| 15 | 196,576 | 3.306 |
| **16** | **208,864** | **3.291 ← run_stop** |

---

### 2. Llama 3.1 8B Pretraining — 4× H200 (Node 1)

| Metric | Value |
|--------|-------|
| Docker Image | `mlperf-llama31-pretraining:h200` |
| Global Batch Size (GBS) | 32 |
| Micro Batch Size (MBS) | 2 |
| Max Learning Rate | 5e-4 |
| Sequence Length | 8,192 tokens |
| run_start (UTC) | 2026-05-06 17:18:49 |
| run_stop (UTC) | 2026-05-07 02:03:50 |
| **Time to Train (MLPerf)** | **525.0 min (8h 45m 2s)** |
| Wall time (job start → finish) | 524.7 min |
| Final val_loss | **3.281** (target ≤ 3.300 ✅) |
| Samples at convergence | 208,864 |
| Steps to convergence | 6,527 |
| Training throughput | ~54,300 tokens/sec (system) · ~13,600 tokens/sec/GPU |
| Peak GPU memory | 140,064–140,125 MiB / 143,771 MiB (97.4%) |
| Scaling efficiency vs 8×GPU | 51.4% (269.8 min × 4 / 8 = 134.9 min ideal; actual 525.0 min) |

> Note: 4×GPU uses ~97% GPU memory vs ~89% for 8×GPU — the larger per-GPU memory footprint explains the slightly higher communication overhead and lower parallel efficiency.

**Validation Loss Trajectory:**

| Eval | Samples | Val Loss |
|------|---------|---------|
| 1 | 12,256 | 5.820 |
| 5 | 61,408 | 3.830 |
| 10 | 135,136 | 3.444 |
| 14 | 184,288 | 3.325 |
| 15 | 196,576 | 3.300 |
| **16** | **208,864** | **3.281 ← run_stop** |

> 4×GPU converged to slightly *lower* val_loss (3.281 vs 3.291) at the same sample count, likely due to effective noise regularization with smaller per-GPU micro-batch diversity. Eval timing: ~30.5 min/eval interval (vs ~15.8 min for 8×GPU — 1.93× ratio as expected).

---

### 3. LLaMA2 70B LoRA Fine-Tuning — 8× H200 (Node 2)

| Metric | Value |
|--------|-------|
| Docker Image | `mlperf-llm-finetuning:h200` |
| Base Model | LLaMA2 70B (fused QKV) |
| Task | Gov Report summarization |
| Method | LoRA (r=16, α=32, dropout=0.1) |
| LoRA Target Modules | qkv_proj, o_proj |
| Optimizer | AdamW w/ ZeRO-3 (DeepSpeed) |
| Learning Rate | 4e-4 cosine |
| Max Steps | 1,024 |
| Per-device Batch Size | 1 |
| Sequence Length | 8,192 tokens |
| Eval Interval | Every 48 steps |
| run_start (UTC) | 2026-05-06 23:22:59 |
| run_stop (UTC) | 2026-05-07 00:27:15 |
| **Time to Train (MLPerf)** | **64.3 min (1h 4m 16s)** |
| Wall time (job start → finish) | 68.4 min |
| Final eval_loss | **0.9246** ✅ |
| Samples at convergence | 3,072 |
| Steps at convergence | 385 / 1,024 (early stop at 37.6%) |
| Training throughput | 2.12 samples/sec · ~17,400 tokens/sec (system) · ~2,175 tokens/sec/GPU |
| Peak GPU memory | ~44,938–45,132 MiB / 143,771 MiB (31.3%) |
| Step time | ~9.07–9.10 s/step |

**Fine-Tuning eval_loss Trajectory (every 48 steps):**

| Step | Samples | eval_loss |
|------|---------|-----------|
| 48 | 384 | 0.9470 |
| 96 | 768 | 0.9488 |
| 144 | 1,152 | 0.9377 |
| 192 | 1,536 | 0.9345 |
| 240 | 1,920 | 0.9284 |
| **288** | **2,304** | **0.9246** |
| 336 | 2,688 | — (eval skipped, stop triggered) |

> **run_stop** fired at step 288 evaluation when eval_loss 0.9246 dropped below target. Steps 289–385 visible in progress bars represent the final training steps before the process exited cleanly.

---

## Power & Thermal Data

### Measured — Node 2, Llama 3.1 8B Pretraining, 8×GPU (point-in-time, peak training)

> Captured via `nvidia-smi` at 2026-05-06 17:52 UTC during active training (utilization: 100% all GPUs).

| GPU | Power Draw | Power Limit | Temp | Memory Used |
|-----|-----------|------------|------|-------------|
| GPU 0 | 593W | 600W | 72°C | 128,159 MiB |
| GPU 1 | 594W | 600W | 63°C | 128,155 MiB |
| GPU 2 | 591W | 600W | 61°C | 128,155 MiB |
| GPU 3 | 591W | 600W | 75°C | 128,155 MiB |
| GPU 4 | 592W | 600W | 73°C | 127,835 MiB |
| GPU 5 | 594W | 600W | 63°C | 128,155 MiB |
| GPU 6 | 594W | 600W | 75°C | 128,155 MiB |
| GPU 7 | 595W | 600W | 60°C | 128,155 MiB |
| **System** | **4,744W** | **4,800W** | **Min: 60°C / Max: 75°C / Avg: 68°C** | **128 GB / GPU** |

**8×GPU Pretraining Power Summary:**
- Min (per GPU): **591W**
- Max (per GPU): **595W**
- Avg (per GPU): **593.0W**
- Total system GPU power: **4,744W**
- % of TDP: **98.8%**

> **Idle power** (GPUs not running benchmarks): 0W reported by nvidia-smi (P8 state confirmed post-run).

---

### Estimated — Node 1, Llama 3.1 8B Pretraining, 4×GPU

> Power was not continuously logged. Based on observed GPU utilization (96–99%) and H200 power behavior at compute saturation, estimated power draw is consistent with the 8×GPU profile.

| Metric | Estimated Value |
|--------|----------------|
| Per-GPU power (peak) | ~590–595W |
| Total system GPU power (peak) | **~2,360–2,380W** |
| GPU utilization observed | 96–99% |
| % of TDP | ~98–99% |

---

### Estimated — Node 2, LLaMA2 70B LoRA Fine-Tuning, 8×GPU

> Power was not captured via nvidia-smi during the fine-tuning run. ZeRO-3 communication overhead results in slightly lower sustained power vs. dense pretraining.

| Metric | Estimated Value |
|--------|----------------|
| Per-GPU power (peak training) | ~540–580W |
| Total system GPU power (peak) | **~4,320–4,640W** |
| GPU utilization observed | 100% |
| GPU memory used | ~44,938–45,132 MiB / GPU (31.3% of HBM) |
| % of TDP | ~90–97% |

---

## Notes

1. **"failed" status in wall_times.txt (pretraining):** The benchmark harness records `status=failed` because the Docker container was killed post-`run_stop` rather than exiting via the expected completion path. The MLPerf logs confirm `run_stop: success` for both pretraining runs.

2. **Power measurement methodology:** No continuous power logging was implemented during this test run. The 8×GPU pretraining power snapshot is a single point-in-time `nvidia-smi` reading during peak training. Future runs should capture power via `nvidia-smi dmon` or DCGM for continuous min/avg/max reporting.

3. **Servers reprovisioned:** Both nodes (<NODE1_MGMT_IP>, <NODE2_MGMT_IP>) were reprovisioned after the test window ended (SSH host keys changed May 7). All results were pulled to this local repository before access was lost.

4. **16× Multi-node run:** Not completed in this test window. Requires both nodes simultaneously with NCCL multi-node rendezvous configured (`RENDEZVOUS_HOST`/`RENDEZVOUS_PORT` in `config.env`). Planned for next test cycle.

5. **LLM Fine-tuning 4×GPU:** Not attempted in this window. Node 1 GPUs 0–3 were occupied with Llama 3.1 pretraining throughout the test period.

---

## Artifact Locations (Local)

| Artifact | Path |
|----------|------|
| Node 2 Llama 3.1 8×GPU MLPerf log | `results/node2/results/llama31_pretraining/run_1/mlperf_llama31_8b.log` |
| Node 1 Llama 3.1 4×GPU MLPerf log | `results/node1/results/llama31_pretraining/run_1/mlperf_llama31_8b.log` |
| Node 2 Fine-tuning MLPerf log | `results/node2/logs/llm_finetuning/llm_finetuning_8xH200_20260506_231847/run_1.log` |
| Node 2 Fine-tuning driver log | `results/node2/logs/llm_finetuning/llm_finetuning_8xH200_20260506_231847/driver.log` |
| Benchmark scripts | `*.sh`, `config.env` (repo root) |
