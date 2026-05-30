#!/usr/bin/env python3
"""
Thin wrapper around ft-llm/scripts/train.py for multi-node DeepSpeed.

Injects the --deepspeed flag into sys.argv before the original script
parses arguments, so that HuggingFace TrainingArguments picks up the
DeepSpeed config without modifying the original train.py.

Usage (via torchrun):
  torchrun --nnodes=2 --nproc_per_node=8 ... \
      ft_multinode_wrapper.py --deepspeed /path/to/ds_config.json \
      --dataset_path ... --model_path ... [all other train.py args]
"""
import sys
import os
import runpy

# Parse out --deepspeed before handing off to train.py
ds_config = None
new_argv = [sys.argv[0]]
i = 1
while i < len(sys.argv):
    if sys.argv[i] == "--deepspeed" and i + 1 < len(sys.argv):
        ds_config = sys.argv[i + 1]
        i += 2
        continue
    new_argv.append(sys.argv[i])
    i += 1

# Set the env var that HuggingFace Trainer recognises for deepspeed
if ds_config:
    os.environ["DEEPSPEED_CONFIG_FILE"] = ds_config

# Patch sys.argv for the original train.py's HfArgumentParser
sys.argv = new_argv

# Ensure train.py's sibling modules are importable
os.chdir("/workspace/ft-llm")
for p in ["/workspace/ft-llm", "/workspace/ft-llm/scripts"]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Monkey-patch TrainingArguments to inject deepspeed config
if ds_config:
    from transformers import TrainingArguments
    _original_init = TrainingArguments.__init__
    def _patched_init(self, *args, **kwargs):
        if "deepspeed" not in kwargs or kwargs["deepspeed"] is None:
            kwargs["deepspeed"] = ds_config
        _original_init(self, *args, **kwargs)
    TrainingArguments.__init__ = _patched_init

# Run the original train.py as __main__
runpy.run_path("/workspace/ft-llm/scripts/train.py", run_name="__main__")
