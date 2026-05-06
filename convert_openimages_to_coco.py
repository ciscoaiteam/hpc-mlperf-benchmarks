#!/usr/bin/env python3
"""
Direct OpenImages CSV -> COCO JSON converter.
Bypasses fiftyone entirely; reads from the fiftyone download cache and writes
COCO-format annotation JSON files compatible with the RetinaNet train.py.

Usage:
    python convert_openimages_to_coco.py \
        --fiftyone-cache ~/fiftyone/open-images-v6 \
        --output-dir    ~/mikcochr/openimages \
        --output-labels openimages-mlperf.json

Output layout:
    <output-dir>/train/labels/openimages-mlperf.json
    <output-dir>/validation/labels/openimages-mlperf.json
    <output-dir>/train/data  -> symlink to fiftyone train/data
    <output-dir>/validation/data -> symlink to fiftyone validation/data
"""

import argparse
import csv
import json
import os
import sys
import time

MLPERF_CLASSES = [
    'Airplane', 'Antelope', 'Apple', 'Backpack', 'Balloon', 'Banana',
    'Barrel', 'Baseball bat', 'Baseball glove', 'Bee', 'Beer', 'Bench',
    'Bicycle', 'Bicycle helmet', 'Bicycle wheel', 'Billboard', 'Book',
    'Bookcase', 'Boot', 'Bottle', 'Bowl', 'Bowling equipment', 'Box', 'Boy',
    'Brassiere', 'Bread', 'Broccoli', 'Bronze sculpture', 'Bull', 'Bus',
    'Bust', 'Butterfly', 'Cabinetry', 'Cake', 'Camel', 'Camera', 'Candle',
    'Candy', 'Cannon', 'Canoe', 'Carrot', 'Cart', 'Castle', 'Cat', 'Cattle',
    'Cello', 'Chair', 'Cheese', 'Chest of drawers', 'Chicken',
    'Christmas tree', 'Coat', 'Cocktail', 'Coffee', 'Coffee cup',
    'Coffee table', 'Coin', 'Common sunflower', 'Computer keyboard',
    'Computer monitor', 'Convenience store', 'Cookie', 'Countertop',
    'Cowboy hat', 'Crab', 'Crocodile', 'Cucumber', 'Cupboard', 'Curtain',
    'Deer', 'Desk', 'Dinosaur', 'Dog', 'Doll', 'Dolphin', 'Door',
    'Dragonfly', 'Drawer', 'Dress', 'Drum', 'Duck', 'Eagle', 'Earrings',
    'Egg (Food)', 'Elephant', 'Falcon', 'Fedora', 'Flag', 'Flowerpot',
    'Football', 'Football helmet', 'Fork', 'Fountain', 'French fries',
    'French horn', 'Frog', 'Giraffe', 'Girl', 'Glasses', 'Goat', 'Goggles',
    'Goldfish', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Guitar',
    'Hamburger', 'Handbag', 'Harbor seal', 'Headphones', 'Helicopter',
    'High heels', 'Hiking equipment', 'Horse', 'House', 'Houseplant',
    'Human arm', 'Human beard', 'Human body', 'Human ear', 'Human eye',
    'Human face', 'Human foot', 'Human hair', 'Human hand', 'Human head',
    'Human leg', 'Human mouth', 'Human nose', 'Ice cream', 'Jacket', 'Jeans',
    'Jellyfish', 'Juice', 'Kitchen & dining room table', 'Kite', 'Lamp',
    'Lantern', 'Laptop', 'Lavender (Plant)', 'Lemon', 'Light bulb',
    'Lighthouse', 'Lily', 'Lion', 'Lipstick', 'Lizard', 'Man', 'Maple',
    'Microphone', 'Mirror', 'Mixing bowl', 'Mobile phone', 'Monkey',
    'Motorcycle', 'Muffin', 'Mug', 'Mule', 'Mushroom', 'Musical keyboard',
    'Necklace', 'Nightstand', 'Office building', 'Orange', 'Owl', 'Oyster',
    'Paddle', 'Palm tree', 'Parachute', 'Parrot', 'Pen', 'Penguin',
    'Personal flotation device', 'Piano', 'Picture frame', 'Pig', 'Pillow',
    'Pizza', 'Plate', 'Platter', 'Porch', 'Poster', 'Pumpkin', 'Rabbit',
    'Rifle', 'Roller skates', 'Rose', 'Salad', 'Sandal', 'Saucer',
    'Saxophone', 'Scarf', 'Sea lion', 'Sea turtle', 'Sheep', 'Shelf',
    'Shirt', 'Shorts', 'Shrimp', 'Sink', 'Skateboard', 'Ski', 'Skull',
    'Skyscraper', 'Snake', 'Sock', 'Sofa bed', 'Sparrow', 'Spider', 'Spoon',
    'Sports uniform', 'Squirrel', 'Stairs', 'Stool', 'Strawberry',
    'Street light', 'Studio couch', 'Suit', 'Sun hat', 'Sunglasses',
    'Surfboard', 'Sushi', 'Swan', 'Swimming pool', 'Swimwear', 'Tank', 'Tap',
    'Taxi', 'Tea', 'Teddy bear', 'Television', 'Tent', 'Tie', 'Tiger',
    'Tin can', 'Tire', 'Toilet', 'Tomato', 'Tortoise', 'Tower',
    'Traffic light', 'Train', 'Tripod', 'Truck', 'Trumpet', 'Umbrella',
    'Van', 'Vase', 'Vehicle registration plate', 'Violin', 'Wall clock',
    'Waste container', 'Watch', 'Whale', 'Wheel', 'Wheelchair', 'Whiteboard',
    'Window', 'Wine', 'Wine glass', 'Woman', 'Zebra', 'Zucchini',
]


def load_class_map(classes_csv):
    """Return {label_code: human_name} and {human_name: coco_cat_id}."""
    code_to_name = {}
    with open(classes_csv) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                code_to_name[row[0].strip()] = row[1].strip()
    mlperf_set = set(MLPERF_CLASSES)
    cat_id = {name: i for i, name in enumerate(MLPERF_CLASSES)}
    code_to_cat = {
        code: cat_id[name]
        for code, name in code_to_name.items()
        if name in mlperf_set
    }
    return code_to_name, code_to_cat, cat_id


def load_image_index(image_ids_csv):
    """Return {image_id: int_index} (1-based)."""
    index = {}
    with open(image_ids_csv) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            index[row['ImageID'].strip()] = i
    return index


def convert_split(split, cache_dir, output_dir, output_labels,
                  code_to_cat, cat_id, code_to_name):
    split_cache = os.path.join(cache_dir, split)
    split_out   = os.path.join(output_dir, split)
    labels_dir  = os.path.join(split_out, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    # Symlink image data directory
    src_data = os.path.join(split_cache, 'data')
    dst_data = os.path.join(split_out, 'data')
    if not os.path.exists(dst_data):
        os.symlink(src_data, dst_data)
        print(f"  Symlinked {dst_data} -> {src_data}")

    print(f"  Loading image index for {split}...")
    image_index = load_image_index(
        os.path.join(split_cache, 'metadata', 'image_ids.csv'))
    print(f"  {len(image_index):,} images in {split}")

    # Build COCO categories list
    categories = [{'id': v, 'name': k, 'supercategory': 'object'}
                  for k, v in sorted(cat_id.items(), key=lambda x: x[1])]

    # Build images list (only those that exist on disk)
    images = []
    for img_id_str, idx in image_index.items():
        images.append({'id': idx, 'file_name': f'{img_id_str}.jpg',
                       'height': 0, 'width': 0})

    print(f"  Parsing detections CSV for {split}...")
    annotations = []
    ann_id = 1
    det_csv = os.path.join(split_cache, 'labels', 'detections.csv')
    t0 = time.time()
    with open(det_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_code = row['LabelName'].strip()
            if label_code not in code_to_cat:
                continue
            img_id_str = row['ImageID'].strip()
            if img_id_str not in image_index:
                continue
            cat = code_to_cat[label_code]
            img_idx = image_index[img_id_str]
            xmin = float(row['XMin'])
            xmax = float(row['XMax'])
            ymin = float(row['YMin'])
            ymax = float(row['YMax'])
            # Stored as fractions; COCO wants pixels but train.py normalises
            # them again — store as fractions (train image sizes vary).
            w = xmax - xmin
            h = ymax - ymin
            annotations.append({
                'id': ann_id,
                'image_id': img_idx,
                'category_id': cat,
                'bbox': [xmin, ymin, w, h],
                'area': w * h,
                'iscrowd': int(row.get('IsGroupOf', 0)),
            })
            ann_id += 1
    print(f"  {ann_id - 1:,} annotations kept ({time.time()-t0:.1f}s)")

    # Drop images with zero annotations — they cause loss spikes (num_foreground=0)
    annotated_ids = set(a['image_id'] for a in annotations)
    before = len(images)
    images = [img for img in images if img['id'] in annotated_ids]
    print(f"  Filtered {before - len(images):,} unannotated images "
          f"({len(images):,} remain)")

    coco = {'images': images, 'annotations': annotations,
            'categories': categories}
    out_path = os.path.join(labels_dir, output_labels)
    print(f"  Writing {out_path}...")
    with open(out_path, 'w') as f:
        json.dump(coco, f)
    print(f"  Done — {os.path.getsize(out_path)/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fiftyone-cache',
                    default=os.path.expanduser('~/fiftyone/open-images-v6'))
    ap.add_argument('--output-dir',
                    default=os.path.expanduser('~/mikcochr/openimages'))
    ap.add_argument('--output-labels', default='openimages-mlperf.json')
    ap.add_argument('--splits', nargs='+', default=['train', 'validation'])
    args = ap.parse_args()

    classes_csv = os.path.join(args.fiftyone_cache, 'train',
                               'metadata', 'classes.csv')
    print(f"Loading class map from {classes_csv}")
    code_to_name, code_to_cat, cat_id = load_class_map(classes_csv)
    print(f"  {len(code_to_cat)} label codes map to {len(MLPERF_CLASSES)} MLPerf classes")

    for split in args.splits:
        print(f"\n=== {split} ===")
        convert_split(split, args.fiftyone_cache, args.output_dir,
                      args.output_labels, code_to_cat, cat_id, code_to_name)

    print("\nAll splits done.")


if __name__ == '__main__':
    main()
