# HPC & MLPerf Benchmarks — Cisco UCS C885A

Benchmark results and scripts for MLPerf LLM training and FUN3D ABTP HVW-GPU CFD across
NVIDIA H200 and AMD MI350X hardware platforms.

---

## Platforms

### NVIDIA H200 — [`nvidia/`](nvidia/)

2× Cisco UCS C885A HGX H200 (16× H200 SXM5) + 2× C845A (8× H200 NVL)

| Results | |
|---|---|
| [C885A Run Summary](nvidia/docs/885a-run-summary.md) | MLPerf Training + FUN3D ABTP HVW-GPU on 2× C885A (16× H200 SXM5) |
| [C845A Run Summary](nvidia/docs/845a-run-summary.md) | FUN3D ABTP HVW-GPU on 2× C845A (8× H200 NVL) |

| Task | Model | Target |
|------|-------|--------|
| LLM Pre-Training | Llama 3.1 8B | log perplexity ≤ 3.3 |
| LLM Fine-Tuning | Llama 2 70B + LoRA | cross-entropy loss ≤ 0.925 |
| CFD GPU | FUN3D 14.2 ABTP HVW-GPU | 3000 steps, accuracy check pass |

Scripts: [`nvidia/scripts/`](nvidia/scripts/) — see [NVIDIA Quickstart](#nvidia-quickstart) below.

---

### AMD MI350X — [`amd/`](amd/)

2× Cisco UCS C885A with AMD Instinct MI350X 8-GPU sleds (16× MI350X total)

| Results | |
|---|---|
| [AMD C885A Run Summary](amd/docs/885a-mi350x-run-summary.md) | MLPerf Training + FUN3D ABTP HVW-GPU on 2× C885A (16× MI350X) |

| Task | Model | Target |
|------|-------|--------|
| LLM Pre-Training | Llama 3.1 8B | log perplexity ≤ 3.3 |
| LLM Fine-Tuning | Llama 2 70B + LoRA | cross-entropy loss ≤ 0.925 |
| CFD GPU | FUN3D 14.2 ABTP HVW-GPU | 3000 steps, accuracy check pass |

Scripts: [`amd/scripts/`](amd/scripts/)

---

## NVIDIA Quickstart

```bash
# 1. One-time environment setup (installs Docker, nvidia-ctk, clones MLCommons repo)
bash nvidia/scripts/00_setup_environment.sh

# 2. Download pre-tokenized C4/en dataset + tokenizer (~350 GB — run in tmux)
bash nvidia/scripts/01_download_llama31_data.sh

# 3. Download Llama 2 70B + GovReport (~132 GB — needs HF_TOKEN)
HF_TOKEN=hf_xxxx bash nvidia/scripts/02_prepare_llm_assets.sh

# 4. Run benchmarks
bash nvidia/scripts/05_run_all_benchmarks.sh

# 5. Process results
python3 nvidia/scripts/process_results.py --log-root /data/mlperf/logs
```

---

## MLPerf Rules Summary

- Each task run 10× per GPU configuration
- Drop fastest and slowest run; average remaining 8
- Minimum 5 successful runs required per configuration
- No maximum wall-clock time; system must reach target accuracy
- Full rules: https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc
