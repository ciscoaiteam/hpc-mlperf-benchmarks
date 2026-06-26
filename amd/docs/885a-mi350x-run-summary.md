# Cisco UCS C885A — AMD Instinct MI350X Benchmark Results

**Hardware:** 2× Cisco UCS C885A | 8× AMD Instinct MI350X per node (16 total)  
**Software:** ROCm 7.1.0 | RCCL 2.27.7 | Docker 29.1.3 | Ubuntu 22.04.5  
**Date:** June 2026

---

## System Overview

| Component | Specification |
|-----------|---------------|
| GPU | AMD Instinct MI350X (gfx950) × 8 per node |
| VRAM | 288 GB HBM3e per GPU (2.3 TB per node) |
| GPU Interconnect | XGMI full mesh (all-to-all, 1 hop, weight 15) |
| CPU | 2× AMD EPYC 9575F 64-Core (128 CUs/socket) |
| TDP | 1000W per GPU |
| OS | Ubuntu 22.04.5 LTS, kernel 5.15.0-181-generic |
| ROCm | 7.1.0 |
| RCCL | 2.27.7 |
| GPU arch | gfx950 |

### Network

| Interface | Type | Speed | Subnet | Notes |
|-----------|------|-------|--------|-------|
| ens99f0 | Mgmt | 1G | 198.18.130.x | Management |
| enp86s0–enp159s0 | AMD Pensando ionic | 400G × 8 | 192.168.200.x | GPU fabric (TCP) |
| ens214f0np0/f1np1 | Mellanox CX-7 | 100G × 2 | 10.195.0.x | OOB |

**Node 1 fabric IPs:** 192.168.200.11–18  
**Node 2 fabric IPs:** 192.168.200.21–28

---

## RCCL Baseline — Network Verification

| Config | Transport | Message Size | Bus BW | Errors |
|--------|-----------|-------------|--------|--------|
| 8-GPU single node | XGMI | 8 GB | ~229 GB/s | 0 |
| 16-GPU 2-node | TCP/Socket (400G) | 8 GB | ~394 GB/s avg | 0 |

> RDMA via AMD Pensando ionic NICs not yet configured (ionic_rdma module not available). Benchmarks run with TCP socket transport over 400G fabric.

---

## Benchmark 1: MLPerf LLM Pre-Training (Llama 3.1 8B)

**Target:** Validation log perplexity ≤ 3.3  
**Container:** `rocm/amd-mlperf:llama31_8b_training_6.0`  
**Dataset:** Pre-tokenized C4/en/3.0.1 (~350 GB)

### Results

| GPUs | Nodes | Transport | Time-to-Train | Steps | Final val PPL | Status |
|------|-------|-----------|--------------|-------|---------------|--------|
| 8 | 1 | XGMI | TBD | TBD | TBD | TBD |
| 16 | 2 | TCP/400G | TBD | TBD | TBD | TBD |

---

## Benchmark 2: MLPerf LLM Fine-Tuning (Llama 2 70B LoRA)

**Target:** Eval cross-entropy loss ≤ 0.925  
**Container:** `rocm/amd-mlperf:llama2_70b_training_6.0`  
**Dataset:** SCROLLS GovReport

### Results

| GPUs | Nodes | Transport | Time-to-Train | Steps | Final eval loss | Status |
|------|-------|-----------|--------------|-------|-----------------|--------|
| 8 | 1 | XGMI | TBD | TBD | TBD | TBD |
| 16 | 2 | TCP/400G | TBD | TBD | TBD | TBD |

---

## Benchmark 3: FUN3D 14.2 ABTP HVW-GPU CFD

**Target:** 3000 steps, accuracy check PASS (CL ≈ 0.334, CD ≈ 0.0927)  
**Grid:** HVW 7.75M node, HLLE++ flux, SA-neg turbulence  
**FLUDA binary:** amd-x86_64-mi250x (compatibility test on MI350X)

### Results

| GPUs | Nodes | Transport | sec/step | Solver | Walltime | CL | CD | Status |
|------|-------|-----------|----------|--------|----------|----|----|--------|
| 1 | 1 | N/A | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | 1 | XGMI | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | 1 | XGMI | TBD | TBD | TBD | TBD | TBD | TBD |
| 16 | 2 | TCP/400G | TBD | TBD | TBD | TBD | TBD | TBD |

---

## Software Stack

### ROCm / RCCL
- ROCm 7.1.0 (`/opt/rocm`)
- RCCL 2.27.7 (system package)
- Docker 29.1.3

### MLPerf Containers (AMD)
- Pretraining: `rocm/amd-mlperf:llama31_8b_training_6.0`
- Fine-tuning: `rocm/amd-mlperf:llama2_70b_training_6.0`

### FUN3D
- Version: 14.2-ffaff71
- FLUDA: `fun3d_intg-fluda-binaries-14.2-ffaff71-amd-x86_64-mi250x.tar.gz`
- Build: `--with-rocm=/opt/rocm`
- MPI: MPICH 4.0

---

## MPI Configuration

### Single-node (8 GPU)
```bash
docker run --device /dev/kfd --device /dev/dri --group-add video --group-add render \
    --network host --ipc host --privileged ...
```

### Multi-node (16 GPU, 2-node over 400G fabric)
```bash
# Hostfile (192.168.200.x fabric IPs)
192.168.200.11:8
192.168.200.21:8

# RCCL env
NCCL_SOCKET_IFNAME=enp86s0
NCCL_IB_DISABLE=1
MASTER_ADDR=192.168.200.11
```

---

## Key Differences vs NVIDIA H200 Setup

| Aspect | NVIDIA H200 (C885A) | AMD MI350X (C885A) |
|--------|--------------------|--------------------|
| GPU | H200 SXM5 141 GB | MI350X 288 GB |
| GPU interconnect | NVSwitch (full mesh) | XGMI (full mesh) |
| Collective lib | NCCL 2.25.1 | RCCL 2.27.7 |
| Framework | CUDA 12.8 / NeMo | ROCm 7.1.0 / NeMo-ROCm |
| Inter-node RDMA | RoCEv2 (Mellanox CX-7) | TCP/400G (ionic, RDMA TBD) |
| VRAM | 141 GB / GPU | 288 GB / GPU |
