# -*- coding: utf-8 -*-
"""Distribution of annotated object sizes, per class and per split.

The hardening step relies on the dataset containing small objects. This
analysis checks that before any time is invested.

COCO convention, applied at the model input size (640x640):
    small  : area < 32^2 px, i.e. under 1024 px
    medium : 32^2 to 96^2 px
    large  : area > 96^2 px

YOLO labels are normalized, so the area is computed in the 640x640 frame after
letterbox, the way the model sees it, not in the original resolution, which
varies from image to image.

Usage:
    python src/analyze_sizes.py data/Drone-Bird-Detection-3
    python src/analyze_sizes.py data/Drone-Bird-Detection-3 --json results/object_sizes.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

INPUT_SIZE = 640
SMALL_AREA = 32 ** 2
MEDIUM_AREA = 96 ** 2


def read_labels(root, split):
    """Returns (classes, areas_px, sides_px) for every box in a split."""
    folder = os.path.join(root, split, "labels")
    classes, areas, sides = [], [], []
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                c, _, _, bw, bh = int(p[0]), *map(float, p[1:5])
                # Coordinates are normalized: multiplying by 640 gives the box
                # in the model's input frame.
                w, h = bw * INPUT_SIZE, bh * INPUT_SIZE
                classes.append(c)
                areas.append(w * h)
                sides.append(max(w, h))
    return np.array(classes), np.array(areas), np.array(sides)


def bucket(areas):
    small = areas < SMALL_AREA
    large = areas > MEDIUM_AREA
    return small, ~small & ~large, large


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset folder containing data.yaml")
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(os.path.join(args.root, "data.yaml"), encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]

    report = {"thresholds_px": {"small": SMALL_AREA, "medium": MEDIUM_AREA},
              "input_size": INPUT_SIZE, "splits": {}}

    for split in ("train", "valid", "test"):
        c, areas, sides = read_labels(args.root, split)
        if len(areas) == 0:
            continue
        small, medium, large = bucket(areas)
        print(f"=== {split}: {len(areas)} objects ===")
        print(f"  small  (< 32 px side)   : {small.sum():5d}  {100 * small.mean():5.1f} %")
        print(f"  medium (32 to 96 px)    : {medium.sum():5d}  {100 * medium.mean():5.1f} %")
        print(f"  large  (> 96 px side)   : {large.sum():5d}  {100 * large.mean():5.1f} %")
        print(f"  median side             : {np.median(sides):.0f} px")
        print(f"  10th percentile side    : {np.percentile(sides, 10):.0f} px")
        print(f"  smallest side           : {sides.min():.0f} px")

        detail = {"objects": int(len(areas)),
                  "small": int(small.sum()), "medium": int(medium.sum()),
                  "large": int(large.sum()),
                  "median_side_px": round(float(np.median(sides)), 1),
                  "p10_side_px": round(float(np.percentile(sides, 10)), 1),
                  "min_side_px": round(float(sides.min()), 1),
                  "per_class": {}}

        for i, name in enumerate(names):
            m = c == i
            if not m.any():
                continue
            s, md, lg = bucket(areas[m])
            print(f"    {name:6}: {m.sum():5d} objects, "
                  f"{100 * s.mean():4.1f} % small, "
                  f"median side {np.median(sides[m]):.0f} px")
            detail["per_class"][name] = {
                "objects": int(m.sum()), "small": int(s.sum()),
                "medium": int(md.sum()), "large": int(lg.sum()),
                "median_side_px": round(float(np.median(sides[m])), 1)}
        print()
        report["splits"][split] = detail

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Details written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
