# FUN3D ABTP HVW-GPU Benchmark — Cisco UCS C845A (H200 NVL)

## Results — ABTP HVW-GPU (Official NASA Benchmark) — 7.75M Nodes, 3000 Steps

| GPUs | Node | Transport | sec/step | Solver Loop | Total Walltime | Lift (CL) | Drag (CD) | Accuracy |
|------|------|-----------|----------|-------------|----------------|-----------|-----------|----------|
| 1 | C845A-1 | N/A | 1.32 s | 3,958 s | 4,323 s (72 min) | 0.3340 | 0.0927 | ✅ PASSES |
| 4 | C845A-2 | PCIe | 1.32 s | 3,959 s | 4,352 s (73 min) | 0.3340 | 0.0927 | ✅ PASSES |

**ABTP accuracy reference:** CL=0.3345468168 (±1%), CD=0.09916422295 (±10%)

**Observations:**
- Same per-step performance at 1 and 4 GPUs — grid is too small for multi-GPU scaling
- ~6% slower than C885A SXM5 (1.32 vs 1.24 s/step) due to HBM bandwidth differences
- Accuracy identical to C885A results — all within ABTP tolerance

---

## System Overview

| Component | Details |
|-----------|---------|
| **Platform** | 2× Cisco UCS C845A |
| **GPUs** | 4× NVIDIA H200 NVL (141 GB HBM3e) per node, 8 total |
| **GPU Memory** | 143,771 MiB (141 GB) per GPU |
| **GPU Interconnect (Intra-node)** | PCIe Gen5 x16 (GPU0↔GPU1 PIX, GPU2↔GPU3 PIX, cross-pair NODE) |
| **GPU Interconnect (Inter-node)** | Management network (TCP) |
| **CPU** | 2× AMD EPYC 9575F 64-Core (256 logical CPUs per node) |
| **Memory** | 1.0 TiB DDR5 per node |
| **Storage** | 7.8 TB LVM (`/dev/mapper/ubuntu--vg-ubuntu--lv`) |
| **OS** | Ubuntu 22.04.5 LTS (kernel 5.15.0-179-generic) |
| **NVIDIA Driver** | 580.159.03 |
| **NICs** | BlueField-3 integrated ConnectX-7 (mlx5_0–mlx5_5) |
| **C845-1 hostname** | `ai-server-c845-amd-1` |
| **C845-2 hostname** | `ai-server-c845-amd-2` |

### GPU Topology

```
        GPU0    GPU1    GPU2    GPU3
GPU0     X      PIX     NODE    NODE
GPU1    PIX      X      NODE    NODE
GPU2    NODE    NODE     X      PIX
GPU3    NODE    NODE    PIX      X
```

- **PIX** = single PCIe bridge (GPU0↔1, GPU2↔3 — NVL pairs)
- **NODE** = PCIe within NUMA node (cross-pair)
- No NVSwitch — inter-GPU communication is PCIe-only
- All GPUs on NUMA node 0 (CPU affinity: cores 0–63, 128–191)

---

## Software Stack

| Component | Version | Path |
|-----------|---------|------|
| **FUN3D** | 14.2-ffaff71 | `/opt/fun3d/install/bin/nodet` |
| **FLUDA** | H100 precompiled binary | `/opt/fun3d/nvidia/x86_64/h100/lib/libfluda.a` |
| **CUDA Toolkit** | 12.8.2 | `/usr/local/cuda-12.8/` |
| **OpenMPI** | 4.1.9a1 | `/usr/mpi/gcc/openmpi-4.1.9a1/` |
| **GCC** | 11.4.0 | System |
| **gfortran** | 11.4.0 | System |
| **FUN3D Source** | `fun3d_intg-14.2-ffaff71.tar.gz` | `/opt/fun3d/fun3d_intg-14.2-ffaff71/` |
| **FLUDA Binaries** | `fun3d_intg-fluda-binaries-14.2-ffaff71-nvidia-x86_64-h100.tar.gz` | `/opt/fun3d/nvidia/x86_64/h100/` |

---

## Directory Structure

```
/opt/fun3d/
├── fun3d_intg-14.2-ffaff71/          # FUN3D source tree
├── nvidia/x86_64/h100/               # FLUDA precompiled binaries
│   ├── lib/libfluda.a
│   └── include/
├── build/                            # Build directory
├── install/                          # Installed binaries
│   └── bin/nodet                     # Main FUN3D solver binary (324 MB)
└── benchmarks/
    └── hvw-gpu/                          # ABTP HVW-GPU benchmark
        ├── input/                        # Shared: hvw.b8.ugrid, hvw.mapbc, tdata, fun3d.nml
        ├── ref/abtp_fun3d_hvw_acccheck.pl
        ├── run_1gpu/                     # 1-GPU results (C845A-1)
        └── run_4gpu/                     # 4-GPU results (C845A-2)
```

---

## Date

ABTP HVW-GPU benchmarks run on **June 4, 2026**.
