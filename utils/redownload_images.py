#!/usr/bin/env python3
"""
Re-download specific corrupted OpenImages training images from the public S3 bucket,
validate them with full RGB decode + timeout, then restore the annotation JSON.
"""
import os, signal, sys, urllib.request, shutil, json, tempfile

# mlperf1: fiftyone symlink target; mlperf2: real directory
DATA_DIR   = "/data/openimages/train/data"
ANN_ORIG   = "/data/openimages/train/labels/openimages-mlperf.json.orig"
ANN_ACTIVE = "/data/openimages/train/labels/openimages-mlperf.json"

# OpenImages V6 public S3 bucket
S3_BASE = "https://open-images-dataset.s3.amazonaws.com/train"

BAD_IDS = [
    "add163e3433bf7fb",   # previously deleted (original corrupt file)
    "0351d19af52fe18c",
    "0352fa9fee20ca5a",
    "02db07cca1ca60e6",
    "02dbae76d89c9284",
    "038d2fd9e7e9556c",
    "038dc16c44629f3c",
    "0385beb820a18c83",
    "03870dc884566282",
]

class Timeout(Exception):
    pass

def _alarm(s, f):
    raise Timeout()

def validate_image(path, timeout=15):
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        img = Image.open(path)
        img.convert("RGB")
        signal.alarm(0)
        return True, None
    except Timeout:
        signal.alarm(0)
        return False, "TIMEOUT during decode"
    except Exception as e:
        signal.alarm(0)
        return False, str(e)

results = {}
for img_id in BAD_IDS:
    fname = f"{img_id}.jpg"
    dest  = os.path.join(DATA_DIR, fname)
    url   = f"{S3_BASE}/{fname}"

    print(f"\n[{img_id}] Downloading...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, \
             tempfile.NamedTemporaryFile(dir=DATA_DIR, delete=False, suffix=".tmp") as tmp:
            shutil.copyfileobj(resp, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, dest)
        size = os.path.getsize(dest)
        print(f"  Downloaded {size//1024} KB -> {dest}", flush=True)
    except Exception as e:
        print(f"  DOWNLOAD FAILED: {e}", flush=True)
        results[img_id] = f"download_failed: {e}"
        continue

    ok, err = validate_image(dest)
    if ok:
        print(f"  Validation OK", flush=True)
        results[img_id] = "ok"
    else:
        print(f"  Validation FAILED: {err} — removing", flush=True)
        os.remove(dest)
        results[img_id] = f"validation_failed: {err}"

print("\n=== Summary ===")
good = [k for k, v in results.items() if v == "ok"]
bad  = [(k, v) for k, v in results.items() if v != "ok"]
print(f"  Good: {len(good)}")
print(f"  Failed: {len(bad)}")
for k, v in bad:
    print(f"    {k}: {v}")

# Restore original annotation if it exists and all files downloaded OK
if not bad and os.path.exists(ANN_ORIG):
    print(f"\nAll downloads succeeded — restoring original annotation from {ANN_ORIG}", flush=True)
    shutil.copy2(ANN_ORIG, ANN_ACTIVE)
    print("Annotation restored.", flush=True)
elif bad:
    print(f"\n{len(bad)} images still bad — keeping filtered annotation.", flush=True)

