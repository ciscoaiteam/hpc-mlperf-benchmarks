#!/usr/bin/env python3
"""
Patch /workspace/single_stage_detector/ssd/engine.py to:
1. Skip NaN/inf loss batches (continue) instead of sys.exit(1)
2. Add gradient clipping (max_norm=1.0) before scaler.step()
"""

ENGINE = "/workspace/single_stage_detector/ssd/engine.py"

with open(ENGINE) as f:
    src = f.read()

if "_patched_nan_skip" in src:
    print(f"Already patched: {ENGINE}")
else:
    # 1. Replace sys.exit(1) on NaN loss with continue (skip the batch)
    old_exit = (
        "        if not math.isfinite(loss_value):\n"
        "            print(\"Loss is {}, stopping training\".format(loss_value))\n"
        "            print(loss_dict_reduced)\n"
        "            sys.exit(1)"
    )
    new_skip = (
        "        if not math.isfinite(loss_value):  # _patched_nan_skip\n"
        "            print(\"[engine] Skipping NaN/inf loss batch: {}\".format(loss_value), flush=True)\n"
        "            print(loss_dict_reduced)\n"
        "            optimizer.zero_grad()\n"
        "            continue"
    )
    if old_exit in src:
        src = src.replace(old_exit, new_skip)
    else:
        print(f"WARNING: NaN stop pattern not found in {ENGINE} — already patched or changed upstream")

    # 2. Add gradient clipping before scaler.step()
    old_scaler = (
        "        scaler.scale(losses).backward()\n"
        "        scaler.step(optimizer)\n"
        "        scaler.update()"
    )
    new_scaler = (
        "        scaler.scale(losses).backward()\n"
        "        scaler.unscale_(optimizer)\n"
        "        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n"
        "        scaler.step(optimizer)\n"
        "        scaler.update()"
    )
    if old_scaler in src:
        src = src.replace(old_scaler, new_scaler)
    else:
        print(f"WARNING: scaler pattern not found in {ENGINE}")

    with open(ENGINE, "w") as f:
        f.write(src)
    print(f"Patched: {ENGINE}")
