# -*- coding: utf-8 -*-
"""Bar chart of AP50 per class and per object-size band, from evaluate.py JSONs.

Usage:
    python src/plot_size_bands.py results/eval_small.json results/eval_medium.json \
        results/eval_large.json results/eval_test_full.json \
        --labels "small < 32 px" "medium 32 to 96 px" "large >= 96 px" "full test set" \
        --out results/figures/ap_by_size_band.png
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.labels) != len(args.jsons):
        raise SystemExit("one label per json file")

    runs = [json.load(open(p, encoding="utf-8")) for p in args.jsons]
    classes = list(runs[0]["per_class"].keys())
    x = np.arange(len(runs))
    width = 0.8 / (len(classes) + 1)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    colors = {"Bird": "#4C72B0", "Drone": "#DD8452"}
    for i, c in enumerate(classes):
        vals = [r["per_class"][c]["ap50"] or 0.0 for r in runs]
        counts = [r["per_class"][c]["objects"] for r in runs]
        bars = ax.bar(x + (i - len(classes) / 2 + 0.5) * width, vals, width,
                      label=c, color=colors.get(c))
        for b, v, n in zip(bars, vals, counts):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}\n(n={n})",
                    ha="center", va="bottom", fontsize=7.5)
    maps = [r["map50"] for r in runs]
    ax.plot(x, maps, "k_", markersize=26, markeredgewidth=2, label="mAP50")

    ax.set_xticks(x)
    ax.set_xticklabels(args.labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("AP50")
    ax.set_title("AP50 by object-size band, test split, YOLOv11n")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
