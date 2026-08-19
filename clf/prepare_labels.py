#!/usr/bin/env python3
"""
Extract disturbance labels from the CSV and save as a numpy file,
ordered to match the shard image sequence (sorted filename / lexicographic).

The CSV already provides `image_file` and `cell_order` (1-based ViT token
index), so alignment is explicit — no guessing required.

Output: {out_dir}/disturbance_labels.npz
  - disturb_majority:     (n_images * 256,) in shard+token order
  - disturb_recent_class: (n_images * 256,) in shard+token order (10 classes)
  - shard_filenames:      (n_images,) sorted image filenames (shard order)

Usage:
    python prepare_labels.py [--out_dir /path/to/output]
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np

CSV_PATH = "/fs/ess/PAS2136/SAtrEes/HARV_grid_25m_sub16_disturbance.csv"
DEFAULT_OUT = "/fs/ess/PAS2136/fangxun/FloraPalooza/scripts/output"


def main():
    parser = argparse.ArgumentParser(description="Extract disturbance labels from CSV")
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading {args.csv} ...")
    t0 = time.time()

    # Parse CSV: group by image_file, store labels indexed by cell_order
    # cell_order is 1-based ViT patch token index (1..256)
    images = defaultdict(lambda: {"majority": [None] * 256, "recent_class": [None] * 256})

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        n_rows = 0
        for row in reader:
            n_rows += 1
            img = row["image_file"]
            idx = int(row["cell_order"]) - 1  # 0-based
            images[img]["majority"][idx] = row["disturb_majority"]
            images[img]["recent_class"][idx] = row["disturb_recent_class"]

    print(f"  {n_rows} rows, {len(images)} images ({time.time()-t0:.1f}s)")

    # Shard order = sorted filenames (lexicographic, matching ImgFolder)
    shard_filenames = sorted(images.keys())
    n_images = len(shard_filenames)

    # Verify completeness
    for fn in shard_filenames:
        assert None not in images[fn]["majority"], f"{fn}: missing majority labels"
        assert None not in images[fn]["recent_class"], f"{fn}: missing recent_class labels"

    # Flatten in shard order × token order
    all_majority = []
    all_recent_class = []
    for fn in shard_filenames:
        all_majority.extend(images[fn]["majority"])
        all_recent_class.extend(images[fn]["recent_class"])

    all_majority = np.array(all_majority, dtype=object)
    all_recent_class = np.array(all_recent_class, dtype=object)
    shard_filenames_arr = np.array(shard_filenames, dtype=object)

    assert all_majority.shape[0] == n_images * 256

    # Save
    out_path = os.path.join(args.out_dir, "disturbance_labels.npz")
    np.savez(
        out_path,
        disturb_majority=all_majority,
        disturb_recent_class=all_recent_class,
        shard_filenames=shard_filenames_arr,
    )
    print(f"  Saved to {out_path}")
    print(f"  Images: {n_images}")
    print(f"  disturb_majority unique:     {len(np.unique(all_majority))}")
    print(f"  disturb_recent_class unique: {len(np.unique(all_recent_class))}")
    print(f"  First 3 shard files: {shard_filenames[:3]}")
    print(f"  Last  3 shard files: {shard_filenames[-3:]}")


if __name__ == "__main__":
    main()
