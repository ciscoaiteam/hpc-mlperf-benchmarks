#!/usr/bin/env python3
"""
multinode_wrapper.py

Direct NeMo Llama 3.1 8B training entry point for multi-node torchrun execution.
Bypasses NeMo-Run LocalExecutor (which is single-node only in v0.4.0).

Launch on BOTH nodes simultaneously:
  torchrun --nnodes=2 --nproc_per_node=8 \
           --rdzv_backend=c10d --rdzv_endpoint=<NODE2_RDMA_IP>:29500 \
           --rdzv_id=mlperf_llama31_16gpu_SEED \
           /workspace/code/multinode_wrapper.py [args]

NeMo recipe config is identical to single-node runs; only num_nodes/num_gpus differ.
Both nodes use local copies of the dataset at the same container mount paths.
"""
import os
import sys
import math
import argparse
from typing import Optional

sys.path.insert(0, '/workspace/code')

import torch
import nemo_run as run
import fiddle as fdl

from nemo.collections import llm
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo import lightning as nl
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.lightning.run import plugins

from callbacks import PreemptiveStop, MLPerfCallback, MetricsLogger
from mlperf_logging.mllog import constants


def get_data(
    gbs: int,
    mbs: int,
    seq_length: int = 8192,
    tokenizer_path: str = "/tokenizer",
    seed: int = 1234,
    use_full_dataset: bool = False,
) -> run.Config:
    tokenizer = run.Config(AutoTokenizer, pretrained_model_name=tokenizer_path)
    dataset_path = os.environ["PREPROCESSED_PATH"]

    if use_full_dataset:
        train_datasets = sum(
            [["12.5", f"{dataset_path}/c4-train.en_{idx}_text_document"] for idx in range(8)],
            [],
        )
    else:
        train_datasets = sum(
            [["10", f"{dataset_path}/c4-train.en_{idx}_text_document"] for idx in [6]],
            [],
        )

    data_paths = {
        "train": train_datasets,
        "validation": [f"{dataset_path}/c4-validation-91205-samples.en_text_document"],
        "test": [f"{dataset_path}/c4-validation-91205-samples.en_text_document"],
    }

    return run.Config(
        llm.PreTrainingDataModule,
        tokenizer=tokenizer,
        paths=data_paths,
        num_workers=128,
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        index_mapping_dir="/npy_index",
        seed=seed,
    )


def get_pretrain(
    nnodes: int,
    ngpus_per_node: int,
    max_steps: int,
    warmup_steps: int,
    data_module: run.Config,
    max_lr: float = 1e-4,
    eval_every: Optional[int] = None,
    eval_batches: Optional[int] = None,
) -> run.Partial:

    pretrain = llm.llama3_8b.pretrain_recipe(
        dir="/mlperf-outputs",
        name="8b",
        num_nodes=nnodes,
        num_gpus_per_node=ngpus_per_node,
    )

    llama31_config = run.Config(llm.gpt.model.llama.Llama31Config8B)
    llama31_config.seq_length = 8192
    pretrain.model.config = llama31_config

    pretrain.trainer.strategy.tensor_model_parallel_size = 1
    pretrain.trainer.strategy.pipeline_model_parallel_size = 1
    pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = 1
    pretrain.trainer.strategy.context_parallel_size = 1

    pretrain.optim = distributed_fused_adam_with_cosine_annealing(
        max_lr=max_lr,
        warmup_steps=warmup_steps,
        min_lr=max_lr * 0.1,
    )

    precision = run.Config(
        nl.MegatronMixedPrecision,
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=True,
        grad_reduce_in_fp32=False,
        fp8="hybrid",
        fp8_amax_history_len=4,
        fp8_amax_compute_algo="most_recent",
        fp8_params=True,
        fp8_dot_product_attention=False,
    )
    pretrain.trainer.plugins = precision

    pretrain.trainer.max_steps = max_steps
    pretrain.data = data_module

    # Use eval_every (integer steps) directly.  The fractional form used in the
    # single-node script (eval_every / GBS = 0.75) fires Lightning's "fraction of
    # epoch" path at step ~65, which triggers NeMo's logger reinitialization and
    # orphans the MLPerfCallback objects, breaking all subsequent eval_accuracy
    # logging.  An integer val_check_interval fires validation at exactly steps
    # eval_every, 2*eval_every, … without any NeMo rebuild side-effect.
    pretrain.trainer.val_check_interval = eval_every
    pretrain.trainer.limit_val_batches = eval_batches
    pretrain.trainer.limit_test_batches = eval_batches

    pretrain.log.tensorboard = None
    pretrain.log.ckpt.every_n_train_steps = None
    pretrain.log.ckpt.save_top_k = 0
    pretrain.log.ckpt.save_last = False
    pretrain.log.ckpt.always_save_context = False
    pretrain.log.ckpt.save_weights_only = False
    pretrain.log.ckpt.save_optim_on_train_end = False
    pretrain.log.ckpt.save_on_train_epoch_end = False
    pretrain.log.ckpt.monitor = "consumed_samples"
    pretrain.log.ckpt.mode = "max"
    pretrain.trainer.strategy.async_save = False

    return pretrain


def main():
    parser = argparse.ArgumentParser(description="Multi-node Llama 3.1 direct trainer")
    parser.add_argument("--nodes", type=int, required=True, help="Total number of nodes")
    parser.add_argument("--gpus_per_node", type=int, required=True)
    parser.add_argument("--gbs", type=int, default=512)
    parser.add_argument("--mbs", type=int, default=2)
    parser.add_argument("--max_lr", type=float, default=8e-3)
    parser.add_argument("--warmup_steps", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=196608,
                        help="Evaluate at least every N training sequences")
    parser.add_argument("--max_steps", type=int, default=1200000)
    parser.add_argument("--seed", type=int, default=4408)
    parser.add_argument("--target_log_ppl", type=float, default=3.3)
    parser.add_argument("--tokenizer_path", type=str, default="/tokenizer")
    parser.add_argument("--step_time_atol", type=int, default=1600)
    parser.add_argument("--use_full_dataset", action="store_true")
    args = parser.parse_args()

    gbs = args.gbs
    nnodes = args.nodes
    ngpus = args.gpus_per_node
    dp = nnodes * ngpus            # TP=PP=CP=1  →  DP = total GPUs

    eval_every_n_batches = math.ceil(args.eval_every / gbs)   # e.g. 196608/512 = 384
    eval_batches = math.ceil(1024 / gbs)                       # 1024 eval seqs → 2

    data = get_data(
        gbs=gbs,
        mbs=args.mbs,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
        use_full_dataset=args.use_full_dataset,
    )

    pretrain = get_pretrain(
        nnodes=nnodes,
        ngpus_per_node=ngpus,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        data_module=data,
        max_lr=args.max_lr,
        eval_every=eval_every_n_batches,
        eval_batches=eval_batches,
    )

    pretrain.data.seed = args.seed
    pretrain.resume = run.Config(
        nl.AutoResume,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
    )
    pretrain.trainer.enable_checkpointing = False
    pretrain.trainer.num_sanity_val_steps = 0

    mini_batch_size = gbs // dp
    grad_accumulation_steps = mini_batch_size // args.mbs

    configs = {
        constants.GLOBAL_BATCH_SIZE: gbs,
        constants.GRADIENT_ACCUMULATION_STEPS: grad_accumulation_steps,
        constants.MAX_SEQUENCE_LENGTH: 8192,
        constants.EVAL_SAMPLES: 1024,
        constants.OPT_NAME: "adamw",
        constants.OPT_BASE_LR: args.max_lr,
        constants.OPT_ADAMW_BETA_1: 0.9,
        constants.OPT_ADAMW_BETA_2: 0.95,
        constants.OPT_ADAMW_EPSILON: 1e-5,
        constants.OPT_ADAMW_WEIGHT_DECAY: 0.1,
        constants.OPT_GRADIENT_CLIP_NORM: 1.0,
        constants.OPT_END_LR: args.max_lr * 0.1,
        constants.OPT_LR_WARMUP_STEPS: args.warmup_steps,
        constants.OPT_LR_DECAY_STEPS: args.max_steps - args.warmup_steps,
        constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
        constants.SEED: args.seed,
        constants.INIT_CHECKPOINT_STEP: 0,
    }

    original_callbacks = pretrain.trainer.callbacks or []
    pretrain.trainer.callbacks = original_callbacks + [
        run.Config(PreemptiveStop, stop_on_step=args.max_steps),
        run.Config(
            MLPerfCallback,
            global_batch_size=gbs,
            micro_batch_size=args.mbs,
            sequence_length=8192,
            eval_every=eval_every_n_batches,
            init_global_step=0,
            configs=configs,
        ),
    ]

    pretrain.log.extra_loggers = [
        run.Config(
            MetricsLogger,
            init_global_step=0,
            global_batch_size=gbs,
            seq_length=8192,
            target_log_ppl=args.target_log_ppl,
            train_step_time_atol=args.step_time_atol,
        ),
    ]

    print(f"[multinode_wrapper] nodes={nnodes} gpus/node={ngpus} dp={dp} "
          f"gbs={gbs} mbs={args.mbs} mini_bs={mini_batch_size} "
          f"grad_accum={grad_accumulation_steps} eval_every={eval_every_n_batches} steps")

    built = fdl.build(pretrain)
    built()


if __name__ == "__main__":
    main()
