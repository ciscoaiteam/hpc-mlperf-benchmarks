#!/usr/bin/env python3
"""
Patch /workspace/single_stage_detector/ssd/coco_utils.py inside the container to:
1. Enable PIL truncated image tolerance
2. Monkey-patch CocoDetection.__getitem__ to skip corrupt/missing images
"""
import sys

UTILS = "/workspace/single_stage_detector/ssd/coco_utils.py"

prepend = """\
import signal as _signal
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.datasets as _tvds
import PIL as _PIL
_orig_coco_gi = _tvds.CocoDetection.__getitem__

class _ImageTimeout(Exception):
    pass

def _safe_coco_getitem(self, idx):
    for off in range(len(self)):
        def _alarm_handler(s, f):
            raise _ImageTimeout()
        _signal.signal(_signal.SIGALRM, _alarm_handler)
        _signal.alarm(10)
        try:
            result = _orig_coco_gi(self, (idx + off) % len(self))
            _signal.alarm(0)
            return result
        except (_ImageTimeout, _PIL.UnidentifiedImageError, OSError, FileNotFoundError):
            _signal.alarm(0)
            print(f"[coco_utils] skipping bad image idx={(idx+off)%len(self)}", flush=True)
    raise RuntimeError("No valid images found in dataset")

_tvds.CocoDetection.__getitem__ = _safe_coco_getitem
"""

with open(UTILS) as f:
    src = f.read()

if "_safe_coco_getitem" in src:
    print(f"Already patched: {UTILS}")
    sys.exit(0)

with open(UTILS, "w") as f:
    f.write(prepend + src)

print(f"Patched: {UTILS}")
