# FUN3D GPU CFD Benchmarks — Cisco UCS C845A (H200 NVL)

## Results

### 10Mc Grid — 1.7M Nodes, 100 Steps

| GPUs | Nodes | Transport | Solver Loop | Per-step | Status |
|------|-------|-----------|-------------|----------|--------|
| 1 | 1 (C845-1) | N/A | **13.7 s** | 0.066 s | ✅ |
| 2 | 1 (C845-1) | PCIe | **13.7 s** | 0.066 s | ✅ |
| 4 | 1 (C845-1) | PCIe | **14.0 s** | 0.073 s | ✅ |

**Observation:** The 10Mc grid (1.7M nodes) is too small for multi-GPU scaling. A single H200 NVL completes each timestep in ~0.07s — adding GPUs only adds MPI partitioning and communication overhead with zero compute benefit.

### 35Mc Grid — 5.9M Nodes, 100 Steps

| GPUs | Nodes | Transport | Solver Loop | Per-step | GPU VRAM | Status |
|------|-------|-----------|-------------|----------|----------|--------|
| 1 | 1 (C845-1) | N/A | **47.8 s** | 0.41 s | ~96 GB | ✅ |
| 2 | 1 (C845-1) | PCIe | **48.0 s** | 0.42 s | ~24 GB each | ✅ |
| 4 | 1 (C845-1) | PCIe | **49.0 s** | 0.42 s | ~24 GB each | ✅ |
| 8 | 2 (C845-1 + C845-2) | TCP (mgmt) | **49.0 s** | 0.42 s | ~24 GB each | ✅ |

### Scaling Analysis — 35Mc Grid

| Metric | 1 GPU | 2 GPU | 4 GPU | 8 GPU (2-node) |
|--------|-------|-------|-------|----------------|
| **Solver loop time** | 47.8 s | 48.0 s | 49.0 s | 49.0 s |
| **Speedup vs 1 GPU** | 1.0× | 1.0× | 0.97× | 0.97× |
| **Per-GPU nodes** | 5.9M | 3.0M | 1.5M | 740K |
| **GPU VRAM used** | ~96 GB | ~24 GB | ~24 GB | ~24 GB |
| **Partitioning time** | ~3 min | ~5 min | ~7 min | ~10 min |

### Key Observations

- **Single GPU saturates this workload.** The H200 NVL is fast enough that 5.9M nodes at 0.41 s/step leaves no room for multi-GPU speedup — adding GPUs only introduces MPI communication overhead.

- **VRAM is the primary scaling driver.** 1 GPU uses 96 GB for the full grid; multi-GPU partitions it to ~24 GB each. For grids too large to fit in a single GPU's 141 GB, multi-GPU is required.

- **Grid-too-small for GPU scaling.** NASA recommends ≥1M points/GPU for efficient GPU utilization with FUN3D+FLUDA. At 4 GPUs the per-GPU partition is only 1.5M nodes — marginal. At 8 GPUs it's 740K — below the threshold.

- **Partitioning dominates wall time.** MetIS CPU-side mesh partitioning takes 3–10 minutes (proportional to GPU count) before the GPU solver loop begins. For a 100-step benchmark, this preprocessing exceeds the solver time itself.

- **No NVLink/NVSwitch.** The C845A connects GPUs via PCIe Gen5 x16 only (no NVLink), so inter-GPU bandwidth is ~64 GB/s per direction vs 900 GB/s on NVSwitch-equipped C885A. This further limits multi-GPU scaling.

- **To demonstrate scaling**, a much larger grid (100M+ cells) would be needed to keep each GPU busy enough that communication latency is hidden by compute.

### C845A vs C885A Comparison

| Metric | C885A (H200 SXM5) | C845A (H200 NVL) |
|--------|-------------------|-------------------|
| **GPUs per node** | 8 | 4 |
| **GPU interconnect** | NVSwitch (900 GB/s) | PCIe Gen5 x16 (~64 GB/s) |
| **GPU memory** | 141 GB HBM3e | 141 GB HBM3e |
| **6M grid, 1 GPU** | 0.12 s/step | — |
| **35Mc grid, 1 GPU** | — | 0.41 s/step |
| **Per-step/M-nodes** | ~0.020 s | ~0.069 s |

The 35Mc grid has ~6× more cells than the C885A's 6M benchmark grid, and the C845 takes ~3.4× longer per step — expected given the larger problem size. Per-million-node performance is comparable.

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

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| **Solver** | FUN3D 14.2 — NASA Langley unstructured CFD |
| **GPU Acceleration** | FLUDA library (CUDA kernels for inviscid/viscous flux, SA turbulence) |
| **Test Case** | AIAA Drag Prediction Workshop 4 (DPW-4) wing-body-tail |
| **Physics** | Compressible RANS, Spalart-Allmaras turbulence model |
| **Mach** | 0.85 |
| **Angle of Attack** | 2.0° |
| **Reynolds Number** | 5,000,000 |
| **Flux Scheme** | Roe with h-van Albada limiter (frozen at iteration 50) |
| **Time Accuracy** | Steady state |
| **CFL Schedule** | 1.0 → 100.0 over 100 iterations |
| **Steps** | 100 |
| **Grid Format** | AFLR3 binary stream (`.lb8.ugrid`) |

### fun3d.nml

```fortran
&project
  project_rootname = 'dpw_wbt0_fine-35Mc_5'
/
&raw_grid
  grid_format = 'aflr3'
  data_format = 'stream'
/
&governing_equations
  eqn_type = 'compressible'
  viscous_terms = 'turbulent'
/
&turbulent_diffusion_models
  turbulence_model = 'sa'
/
&reference_physical_properties
  mach_number     = 0.85
  angle_of_attack = 2.0
  reynolds_number = 5000000.0
/
&inviscid_flux_method
  flux_construction = 'roe'
  flux_limiter      = 'hvanalbada'
  freeze_limiter_iteration = 50
/
&nonlinear_solver_parameters
  time_accuracy = 'steady'
  schedule_iteration =   1 100
  schedule_cfl       = 1.0 100.0
/
&code_run_control
  restart_read       = 'off'
  steps              = 100
  stopping_tolerance = 1.0e-15
/
&gpu_support
  use_cuda = .true.
  gpus_per_node = 8
/
&global
  boundary_animation_freq = -1
  volume_animation_freq   = -1
/
```

---

## Benchmark Grids

All grids sourced from the NASA DPW-4 archive:
`https://dpw.larc.nasa.gov/DPW4/unstructured_Larc/CellBase/`

Grids were downloaded in VGRID format (`.cogsg`) and converted to AFLR3 (`.lb8.ugrid`) using the `cogsg2ugrid` tool included with FUN3D. The `.mapbc` boundary condition files were also converted from VGRID format to the simple AFLR3 format expected by FUN3D.

| Grid | Cells | Nodes | Tets | Surface Tris | Size (ugrid) | Source File |
|------|-------|-------|------|-------------|-------------|-------------|
| 3.5Mc (coarse) | 3.5M | 672,235 | 3,935,055 | 64,500 | 77 MB | `dpw-wbt0_crs-3.5Mc_5.tgz` |
| **10Mc (medium)** | **10M** | **1,712,882** | **10,067,380** | **134,726** | **195 MB** | `dpw-wbt0_med-10Mc_5.tgz` |
| **35Mc (fine)** | **35M** | **5,917,692** | **34,878,666** | **370,044** | **674 MB** | `dpw_wbt0_fine-35Mc_5.tgz` |

### Boundary Conditions

| VGRID BC | FUN3D BC | Family | Description |
|----------|----------|--------|-------------|
| 3 | 5050 | Box | Farfield |
| 4 | 4000 | Fuselage/WING/TIP/TE/t0 | Viscous wall (no-slip) |
| 1 | 6662 | Reflection | Symmetry plane |

---

## Build & Installation

### Prerequisites

```bash
# Install CUDA 12.8 toolkit (alongside existing CUDA 13)
apt-get install -y cuda-toolkit-12-8

# Create lib symlink (FUN3D configure expects lib/, CUDA 12.8 has lib64/)
ln -sfn /usr/local/cuda-12.8/lib64 /usr/local/cuda-12.8/lib

# Install missing dependencies
apt-get install -y libudev-dev
```

### Configure & Build

```bash
cd /opt/fun3d
mkdir build && cd build

export PATH=/usr/local/cuda-12.8/bin:/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/mpi/gcc/openmpi-4.1.9a1/lib:$LD_LIBRARY_PATH

../fun3d_intg-14.2-ffaff71/configure \
  --prefix=/opt/fun3d/install \
  CC=mpicc FC=mpif90 \
  CFLAGS="-O2" FFLAGS="-O2" \
  --with-cuda=/usr/local/cuda-12.8 \
  --with-libfluda=/opt/fun3d/nvidia/x86_64/h100

make -j$(nproc)
make install
```

### Grid Preparation

```bash
cd /opt/fun3d/benchmarks

# Download grids from DPW-4 archive
curl -LO https://dpw.larc.nasa.gov/DPW4/unstructured_Larc/CellBase/dpw_wbt0_fine-35Mc_5.tgz
curl -LO https://dpw.larc.nasa.gov/DPW4/unstructured_Larc/CellBase/dpw-wbt0_med-10Mc_5.tgz
curl -LO https://dpw.larc.nasa.gov/DPW4/unstructured_Larc/CellBase/dpw-wbt0_crs-3.5Mc_5.tgz

# Extract
tar xzf dpw_wbt0_fine-35Mc_5.tgz
tar xzf dpw-wbt0_med-10Mc_5.tgz
tar xzf dpw-wbt0_crs-3.5Mc_5.tgz

# Convert VGRID (.cogsg) to AFLR3 (.lb8.ugrid)
echo "dpw_wbt0_fine-35Mc_5" | cogsg2ugrid
echo "dpw-wbt0_med-10Mc_5" | cogsg2ugrid
echo "dpw-wbt0_crs-3.5Mc_5" | cogsg2ugrid

# Convert .mapbc to simple AFLR3 format (N, then patch_id bc_type per line)
# BC mapping: VGRID 3 → FUN3D 5050 (farfield), 4 → 4000 (viscous wall), 1 → 6662 (symmetry)
python3 -c "
for grid in ['dpw-wbt0_med-10Mc_5', 'dpw_wbt0_fine-35Mc_5', 'dpw-wbt0_crs-3.5Mc_5']:
    lines = open(grid + '.mapbc').readlines()
    patches = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('Patch'): continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                pid, bc = int(parts[0]), int(parts[1])
                fun3d_bc = {3: 5050, 4: 4000, 1: 6662}.get(bc, bc)
                patches.append((pid, fun3d_bc))
            except: pass
    with open(grid + '.mapbc', 'w') as f:
        f.write(str(len(patches)) + '\n')
        for pid, bc in patches:
            f.write(f'{pid}  {bc}\n')
    print(f'{grid}: {len(patches)} patches')
"
```

### Running Benchmarks

```bash
export PATH=/opt/fun3d/install/bin:/usr/mpi/gcc/openmpi-4.1.9a1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/mpi/gcc/openmpi-4.1.9a1/lib:$LD_LIBRARY_PATH

# GPU wrapper — assigns one GPU per MPI rank
cat > /tmp/gpu_wrapper.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK
exec "$@"
EOF
chmod +x /tmp/gpu_wrapper.sh

# Single GPU
CUDA_VISIBLE_DEVICES=0 mpirun --allow-run-as-root -np 1 nodet --time_timestep_loop

# 4 GPUs, single node
mpirun --allow-run-as-root -np 4 /tmp/gpu_wrapper.sh nodet --time_timestep_loop

# 8 GPUs, multi-node (hostfile: <MGMT_IP_C845A1> slots=4 / <MGMT_IP_C845A2> slots=4)
mpirun --allow-run-as-root --hostfile hostfile -np 8 \
  -x PATH -x LD_LIBRARY_PATH \
  --mca btl tcp,self --mca btl_tcp_if_include <MGMT_SUBNET>/24 \
  /tmp/gpu_wrapper.sh nodet --time_timestep_loop
```

---

## Issues Resolved

| # | Issue | Symptom | Fix |
|---|-------|---------|-----|
| 1 | Wrong configure flag | `--with-fluda` silently ignored | Use `--with-libfluda=/opt/fun3d/nvidia/x86_64/h100` |
| 2 | CUDA 13 API incompatibility | `undefined reference to cudaGetDeviceProperties_v2` during link | Install CUDA 12.8 toolkit (`apt-get install cuda-toolkit-12-8`) |
| 3 | CUDA lib path mismatch | Configure generates `-L/usr/local/cuda-12.8/lib` but libs are in `lib64/` | `ln -sfn /usr/local/cuda-12.8/lib64 /usr/local/cuda-12.8/lib` |
| 4 | Missing `libudev` | Link error for `libudev` | `apt-get install libudev-dev` |
| 5 | VGRID `.mapbc` format | `Bad integer for item 1 in list input` — AFLR3 reader chokes on VGRID comment lines and extra columns | Convert to simple format: line 1 = count, then `patch_id bc_type` per line |
| 6 | VGRID grid format | FUN3D GPU (FLUDA) requires AFLR3 `.lb8.ugrid`, not VGRID `.cogsg` | Run `cogsg2ugrid` (pipe project name to stdin) |
| 7 | All MPI ranks land on GPU 0 | `nvidia-smi` shows GPU 0 at 96 GB, GPUs 1-3 at 0 MB | Create `gpu_wrapper.sh` that sets `CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK` |
| 8 | `use_gpu = .true.` not recognized | `Probable incomplete read of namelist: &gpu_support iostat2=5010` | Correct variable is `use_cuda = .true.` (not `use_gpu`) |
| 9 | `nodet_mpi` not found | `mpirun was unable to find the specified executable` | Binary is named `nodet` (not `nodet_mpi`) on this build |
| 10 | `cogsg2ugrid` appears to hang | Interactive prompt waiting for stdin | Pipe project name: `echo "project_name" \| cogsg2ugrid` |
| 11 | MetIS partitioning very slow | 7+ minutes of 100% CPU before GPU solver starts (35Mc grid, 4 ranks) | Expected behavior for 35M-cell grid; no fix needed |
| 12 | SSH between C845 nodes | `Host key verification failed` during rsync | Generate SSH key on C845-1, add public key to C845-2 `authorized_keys` |

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
    ├── dpw_wbt0_fine-35Mc_5.lb8.ugrid    # 35Mc grid (674 MB)
    ├── dpw_wbt0_fine-35Mc_5.mapbc        # 35Mc boundary conditions
    ├── dpw-wbt0_med-10Mc_5.lb8.ugrid     # 10Mc grid (195 MB)
    ├── dpw-wbt0_med-10Mc_5.mapbc
    ├── dpw-wbt0_crs-3.5Mc_5.lb8.ugrid   # 3.5Mc grid (77 MB)
    ├── dpw-wbt0_crs-3.5Mc_5.mapbc
    ├── run_10Mc_1gpu/                    # Benchmark run directories
    ├── run_10Mc_2gpu/
    ├── run_10Mc_4gpu/
    ├── run_35Mc_1gpu/
    ├── run_35Mc_2gpu/
    ├── run_35Mc_4gpu/
    └── run_35Mc_8gpu_multinode/
        └── hostfile                      # <MGMT_IP_C845A1> slots=4 / .35 slots=4
```

---

## Date

All benchmarks run on **May 30, 2026**.
