# -*- coding: utf-8 -*-
"""Measure how much of the drone / bird separation comes from size alone.

Motivation. The trained model never confuses a drone with a bird on the test
set. Before concluding that it learned to tell them apart visually, the
trivial explanation must be ruled out: the two classes may have sizes so
different that a single threshold is enough.

Method. Train the dumbest possible classifier, a single threshold on the box
side, using NO pixels at all. The threshold is chosen on the training split,
then applied unchanged to the test split. Its accuracy is a lower bound on
what the dataset gives away for free.

Reading. If this classifier reaches 90%, the neural network only contributes
the remaining 10 points, and the advertised performance mostly measures a
dataset bias.

Usage:
    python src/size_bias.py data/Drone-Bird-Detection-3
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

INPUT_SIZE = 640


def load(root, split):
    """Returns (classes, sides_px) for every box in a split."""
    classes, sides = [], []
    for path in glob.glob(os.path.join(root, split, "labels", "*.txt")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                classes.append(int(p[0]))
                sides.append(max(float(p[3]), float(p[4])) * INPUT_SIZE)
    return np.array(classes), np.array(sides)


def best_threshold(y, side):
    """Threshold maximizing accuracy, searched over observed percentiles.

    Rule: side >= threshold predicts the majority class above the threshold.
    """
    candidates = np.unique(np.percentile(side, np.arange(1, 100)))
    best, best_acc, direction = None, -1.0, 1
    for t in candidates:
        for orient in (1, 0):
            pred = np.where(side >= t, orient, 1 - orient)
            acc = (pred == y).mean()
            if acc > best_acc:
                best, best_acc, direction = t, acc, orient
    return best, best_acc, direction


def evaluate(y, side, threshold, direction):
    pred = np.where(side >= threshold, direction, 1 - direction)
    acc = (pred == y).mean()
    per_class = {}
    for c in np.unique(y):
        m = y == c
        per_class[int(c)] = float((pred[m] == y[m]).mean())
    return float(acc), per_class, pred


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(os.path.join(args.root, "data.yaml"), encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]

    y_tr, s_tr = load(args.root, "train")
    y_te, s_te = load(args.root, "test")

    threshold, acc_tr, direction = best_threshold(y_tr, s_tr)
    acc_te, per_class, pred = evaluate(y_te, s_te, threshold, direction)

    above = names[direction]
    print("Single-threshold classifier, no pixels used")
    print(f"  threshold learned on train: side >= {threshold:.0f} px  ->  {above}")
    print(f"  accuracy on train         : {100 * acc_tr:.1f} %")
    print(f"  accuracy on test          : {100 * acc_te:.1f} %")
    print()
    for c, a in sorted(per_class.items()):
        print(f"    {names[c]:6}: {100 * a:5.1f} % correctly classified")
    print()

    # Small-object subset: where the threshold can no longer help.
    small = s_te < 32
    if small.sum():
        acc_small, pc_small, _ = evaluate(y_te[small], s_te[small], threshold, direction)
        print(f"  On the {small.sum()} small test objects (< 32 px):")
        print(f"    threshold accuracy: {100 * acc_small:.1f} %")
        for c in np.unique(y_te[small]):
            print(f"      {names[c]:6}: {int((y_te[small] == c).sum()):3d} objects, "
                  f"{100 * pc_small[int(c)]:5.1f} % correct")
    print()

    ratio = None
    if len(np.unique(y_te)) == 2:
        a = np.median(s_te[y_te == 0])
        b = np.median(s_te[y_te == 1])
        ratio = round(float(max(a, b) / min(a, b)), 2)
        print(f"  Median side {names[0]}: {a:.0f} px")
        print(f"  Median side {names[1]}: {b:.0f} px")
        print(f"  Scale ratio between the two classes: x{ratio}")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"threshold_px": round(float(threshold), 1),
                       "class_above": above,
                       "train_accuracy": round(acc_tr, 4),
                       "test_accuracy": round(acc_te, 4),
                       "per_class_test": {names[c]: round(a, 4)
                                          for c, a in per_class.items()},
                       "scale_ratio": ratio}, f, ensure_ascii=False, indent=2)
        print(f"\nDetails written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
