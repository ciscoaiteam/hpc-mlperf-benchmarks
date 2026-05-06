#!/usr/bin/env python3
"""
Filter openimages-mlperf.json to only include images that exist on disk.
Writes a new filtered JSON alongside the original.
"""
import os, json, sys

DATA_DIR  = "/data/openimages/train/data"
ANN_FILE  = "/data/openimages/train/labels/openimages-mlperf.json"
OUT_FILE  = "/data/openimages/train/labels/openimages-mlperf-filtered.json"

print(f"Loading {ANN_FILE} ...", flush=True)
with open(ANN_FILE) as f:
    ann = json.load(f)

images_orig = ann["images"]
print(f"Original images in annotation: {len(images_orig)}", flush=True)

# Build set of filenames present on disk (fast path via os.scandir)
print("Scanning disk files ...", flush=True)
on_disk = set()
for entry in os.scandir(DATA_DIR):
    on_disk.add(entry.name)
print(f"Files on disk: {len(on_disk)}", flush=True)

# Filter images and collect valid image IDs
valid_ids = set()
images_kept = []
for img in images_orig:
    fname = os.path.basename(img["file_name"])
    if fname in on_disk:
        valid_ids.add(img["id"])
        images_kept.append(img)

print(f"Images kept: {len(images_kept)} (removed {len(images_orig)-len(images_kept)})", flush=True)

# Filter annotations to only reference kept image IDs
anns_orig = ann.get("annotations", [])
anns_kept = [a for a in anns_orig if a["image_id"] in valid_ids]
print(f"Annotations kept: {len(anns_kept)} of {len(anns_orig)}", flush=True)

out = dict(ann)
out["images"] = images_kept
out["annotations"] = anns_kept

print(f"Writing {OUT_FILE} ...", flush=True)
with open(OUT_FILE, "w") as f:
    json.dump(out, f)
print("Done.", flush=True)
