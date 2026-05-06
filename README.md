# MLPerf Training Benchmarks — H200 Node

Closed-division scripts for two selected tasks:

| Task | Model | Dataset | Target |
|------|-------|---------|--------|
| LLM Pre-Training (small) | Llama 3.1 8B | C4/en/3.0.1 | log perplexity ≤ 3.3 |
| LLM Fine-Tuning | Llama 2 70B + LoRA | SCROLLS GovReport | 0.925 cross-entropy loss |

GPU configurations tested: **1, 2, 4, 8 × H200**  
Runs per configuration: **10** (fastest + slowest dropped; 8 averaged per MLPerf rules)

---

## Prerequisites

- 8× NVIDIA H200 node (driver 590+, CUDA 13.x)
- Root access (for Docker & toolkit installation)
- Docker (installed by setup script if missing)
- Hugging Face account with **Meta Llama 2 license accepted** (for the fine-tuning benchmark)
  → https://ai.meta.com/llama/
- ~500 GB free disk space on `/data/mlperf` (~350 GB C4 dataset + ~132 GB Llama 2 weights)

---

## Quickstart

```bash
# 1. One-time environment setup (installs Docker, nvidia-ctk, clones MLCommons repo)
bash 00_setup_environment.sh

# 2. Download pre-tokenized C4/en dataset + tokenizer (~350 GB — run in tmux)
tmux new -s llama31_data
bash 01_download_llama31_data.sh

# 3. Download Llama 2 70B + GovReport (~132 GB — run in tmux, needs HF_TOKEN)
tmux new -s llm_data
HF_TOKEN=hf_xxxx bash 02_prepare_llm_assets.sh

# 4a. Run everything (all benchmarks, all GPU configs)
bash 05_run_all_benchmarks.sh

# 4b. Or run individually
bash 03_run_llama31_pretraining.sh 8   # Llama 3.1 8B pretraining, 8 GPUs, 10 runs
bash 03_run_llama31_pretraining.sh 4   # Llama 3.1 8B pretraining, 4 GPUs, 10 runs
bash 04_run_llm_finetuning.sh 8        # LLM FT,   8 GPUs, 10 runs

# 5. Process results
python3 process_results.py --log-root /data/mlperf/logs
```

---

## Script Reference

| Script | Purpose |
|--------|---------|
| `config.env` | Shared paths, GPU configs, NCCL tuning |
| `00_setup_environment.sh` | Docker, nvidia-ctk, repo clone, image builds |
| `01_download_llama31_data.sh` | Pre-tokenized C4/en dataset + Llama 3.1 8B tokenizer |
| `02_prepare_llm_assets.sh` | Llama 2 70B weights + SCROLLS tokenization |
| `03_run_llama31_pretraining.sh <gpus> [runs]` | Llama 3.1 8B pretraining run loop with MLLOG capture |
| `04_run_llm_finetuning.sh <gpus> [runs]` | LLM LoRA+DeepSpeed run loop |
| `05_run_all_benchmarks.sh [bench] [gpus...]` | Full orchestration |
| `process_results.py` | Parse logs, drop hi/lo, report trimmed mean |

---

## Directory Layout (on server)

```
/opt/mlperf/
├── llama31-c4-preprocessed/ ← Pre-tokenized C4/en dataset (megatron .bin/.idx)
├── llama31-tokenizer/       ← Llama 3.1 8B tokenizer
├── govreport/               ← LLM fine-tuning dataset (SCROLLS GovReport)
├── models/
│   └── llama2-70b/          ← Llama 2 70B weights (fine-tuning benchmark)
├── results/
│   ├── llama31_pretraining/
│   └── llm_finetuning/
└── logs/
    ├── llama31_pretraining/
    │   └── llama31_pretraining_8xH200_YYYYMMDD_HHMMSS/
    │       ├── run_1.log    ← full Docker output
    │       ├── run_1_docker.log
    │       └── wall_times.txt
    └── llm_finetuning/
        └── llm_finetuning_8xH200_YYYYMMDD_HHMMSS/
```

---

## MLPerf Rules Summary

- Each task run 10× per GPU configuration (1/2/4/8 GPUs)
- Drop fastest and slowest run; average remaining 8
- If a run fails to converge it may be discarded (minimum 5 successful required)
- No maximum wall-clock time; system must reach target accuracy
- Full rules: https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc

---

## Appendix: Llama 3.1 8B Pre-Training — Setup Notes

### Data

The benchmark uses the **pre-tokenized C4/en/3.0.1** dataset served by the
MLCommons R2 Downloader (~350 GB, megatron `.bin`/`.idx` format). No HuggingFace
token is needed — the benchmark trains **from random initialisation**, so model
weights are not required.

```bash
bash 01_download_llama31_data.sh   # downloads both dataset and tokenizer
```

### Framework

The reference implementation (`small_llm_pretraining/nemo`) uses **NVIDIA NeMo** with
the Megatron-LM backend. The Docker image is built from the Dockerfile in that directory
and launched via `run_llama31.sh` inside the container.

Key environment variables consumed by `run_llama31.sh`:

| Variable | Set by run script |
|---|---|
| `PREPROCESSED_PATH` | `/data/preprocessed` (host `C4_PREPROCESSED_DIR` mounted) |
| `TOKENIZER_PATH` | `/data/tokenizer` (host `LLAMA31_TOKENIZER_DIR` mounted) |
| `NPROC_PER_NODE` | `<num_gpus>` |
| `GLOBAL_BATCH_SIZE` | 2048 sequences |
| `MICRO_BATCH_SIZE` | 1 per GPU |
| `TENSOR_MODEL_PARALLEL_SIZE` | 1 (TP=1 fits on H200 141 GB) |
| `TARGET_LOG_PPL` | 3.3 |
| `SEED` | per-run random value |

### Quality target

Training runs until **validation log perplexity ≤ 3.3** (exp(3.3) ≈ 27.1 perplexity).
With 8× H200 and a global batch of 2048 × 8192 tokens, convergence typically requires
~4 000–6 000 steps (~3–5 hours per run).

### NCCL with RoCEv2 (current node config)

This node uses RoCEv2 over `mlx5_2`. The active `config.env` settings are:

```bash
export NCCL_SOCKET_IFNAME=ens211f0np0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_2
export NCCL_NVLS_ENABLE=0   # no inter-node NVSwitch
export NCCL_P2P_LEVEL=NVL
```
