#!/usr/bin/env python3
"""Scan rank-4 shard for PIL-hanging or unreadable images, with per-image timeout."""
import os, json, signal, sys
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Timeout(Exception):
    pass

def _alarm(s, f):
    raise Timeout()

ann_file = "/data/openimages/train/labels/openimages-mlperf.json"
with open(ann_file) as f:
    images = json.load(f)["images"]

# Scan ALL images (full decode) to find anything that hangs libjpeg
all_images = sorted(images, key=lambda x: x["id"])

# Allow scanning a single rank shard via env var (for parallel execution)
rank = int(os.environ.get("SCAN_RANK", "-1"))
world = int(os.environ.get("SCAN_WORLD", "1"))
if rank >= 0:
    shard = all_images[rank::world]
    label = f"rank{rank}_of{world}"
else:
    shard = all_images
    label = "all"

bad_log = f"/opt/mlperf/bad_images_{label}.txt"
print(f"Scanning {len(shard)} images ({label}) with 8s full-decode timeout...", flush=True)

bad = []
for i, info in enumerate(shard):
    fname = os.path.basename(info["file_name"])
    fpath = f"/data/openimages/train/data/{fname}"
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(8)
    try:
        img = Image.open(fpath)
        img.convert("RGB")   # full pixel decode — exercises libjpeg
        signal.alarm(0)
    except Timeout:
        bad.append((fpath, "TIMEOUT"))
        print(f"TIMEOUT: {fpath}", flush=True)
    except Exception as e:
        signal.alarm(0)
        bad.append((fpath, str(e)[:80]))
    if i % 5000 == 0:
        print(f"  {i}/{len(shard)} checked, {len(bad)} bad so far", flush=True)

print(f"DONE: {len(bad)} bad files found")
with open(bad_log, "w") as bf:
    for f, e in bad:
        print(f"  BAD: {f}: {e}")
        bf.write(f"{f}\t{e}\n")
