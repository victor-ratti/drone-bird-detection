# -*- coding: utf-8 -*-
"""Evaluation of an ONNX detection model, without PyTorch.

Computes AP50 per class and mAP50, with optional filtering on object size.
Size filtering is the core of the hardening step: it measures performance
separately on small objects, which are the operationally hard case for
counter-drone systems.

Everything happens in the 640x640 letterboxed frame the model sees. Annotations
are mapped into that same frame, which avoids an inverse letterbox and its
rounding errors.

Matching protocol: a prediction is a true positive if it overlaps an
annotation of the same class with IoU >= threshold, and that annotation has not
already been matched to a higher-scoring prediction. AP uses all-point
interpolation, COCO convention.

Usage:
    python src/evaluate.py models/baseline_best.onnx data/Drone-Bird-Detection-3
    python src/evaluate.py models/baseline_best.onnx data/... --max-side 32
    python src/evaluate.py models/baseline_best.onnx data/... --split valid --json r.json
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import letterbox, to_tensor  # noqa: E402


def labels_to_letterbox(path, w0, h0, r, dx, dy):
    """Normalized YOLO labels -> xyxy boxes in the 640 frame."""
    boxes, classes = [], []
    if not os.path.exists(path):
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=int)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            cx, cy, bw, bh = cx * w0, cy * h0, bw * w0, bh * h0
            x1 = cx * r - bw * r / 2 + dx
            y1 = cy * r - bh * r / 2 + dy
            boxes.append([x1, y1, x1 + bw * r, y1 + bh * r])
            classes.append(c)
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=int)


def iou_matrix(a, b):
    """IoU between two sets of xyxy boxes. Returns (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes, scores, threshold=0.45):
    """Non-maximum suppression on boxes of a single class."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < threshold]
    return np.array(keep, dtype=int)


def predict(session, input_name, img, conf=0.001, nms_threshold=0.45):
    """Returns (xyxy boxes in the 640 frame, scores, classes)."""
    raw = session.run(None, {input_name: to_tensor(img)})[0]   # (1, 4+nc, N)
    raw = raw[0].T                                              # (N, 4+nc)
    xywh, class_scores = raw[:, :4], raw[:, 4:]
    cls = class_scores.argmax(1)
    sc = class_scores.max(1)
    m = sc > conf
    xywh, sc, cls = xywh[m], sc[m], cls[m]
    if len(sc) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                      xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        keep.extend(idx[nms(boxes[idx], sc[idx], nms_threshold)])
    k = np.array(sorted(keep), dtype=int)
    return boxes[k], sc[k], cls[k]


def ap_per_class(tp, scores, n_truth):
    """Average precision, all-point interpolation, COCO convention."""
    if n_truth == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = scores.argsort()[::-1]
    tp = tp[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    recall = cum_tp / n_truth
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    # Decreasing envelope of precision, then integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    r = np.concatenate(([0.0], recall))
    p = np.concatenate(([precision[0] if len(precision) else 0.0], precision))
    return float(np.sum(np.diff(r) * p[1:]))


def side_filter(boxes, max_side, min_side):
    """Boolean mask of boxes whose longest side falls in [min_side, max_side)."""
    side = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    keep = np.ones(len(side), dtype=bool)
    if max_side:
        keep &= side < max_side
    if min_side:
        keep &= side >= min_side
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("root")
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-side", type=float, default=None,
                    help="keep only objects whose longest side is below this, in px")
    ap.add_argument("--min-side", type=float, default=None)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--json")
    args = ap.parse_args()

    with open(os.path.join(args.root, "data.yaml"), encoding="utf-8") as f:
        names = yaml.safe_load(f)["names"]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    images = sorted(glob.glob(os.path.join(args.root, args.split, "images", "*")))
    if not images:
        raise SystemExit(f"No images in {args.root}/{args.split}/images")

    filtering = bool(args.max_side or args.min_side)
    stats = {c: {"tp": [], "scores": [], "n": 0} for c in range(len(names))}
    confusion = np.zeros((len(names) + 1, len(names) + 1), dtype=int)
    ignored = 0

    for k, path in enumerate(images, 1):
        img0 = cv2.imread(path)
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        img, r, dx, dy = letterbox(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        label = os.path.join(args.root, args.split, "labels",
                             os.path.splitext(os.path.basename(path))[0] + ".txt")
        gt_b, gt_c = labels_to_letterbox(label, w0, h0, r, dx, dy)

        pr_b, pr_s, pr_c = predict(session, input_name, img, args.conf)

        # The same size filter must apply to both annotations and predictions.
        # Filtering only the ground truth would turn every correct detection of
        # a large object into a false positive, since its truth was removed,
        # and crush precision for a purely methodological reason.
        if filtering:
            if len(gt_b):
                keep = side_filter(gt_b, args.max_side, args.min_side)
                ignored += int((~keep).sum())
                gt_b, gt_c = gt_b[keep], gt_c[keep]
            if len(pr_b):
                keep = side_filter(pr_b, args.max_side, args.min_side)
                pr_b, pr_s, pr_c = pr_b[keep], pr_s[keep], pr_c[keep]

        for c in range(len(names)):
            stats[c]["n"] += int((gt_c == c).sum())

        # Greedy matching by decreasing score, per class.
        for c in range(len(names)):
            ip = np.where(pr_c == c)[0]
            ig = np.where(gt_c == c)[0]
            if len(ip) == 0:
                continue
            ip = ip[pr_s[ip].argsort()[::-1]]
            taken = set()
            ious = iou_matrix(pr_b[ip], gt_b[ig]) if len(ig) else None
            for j, p in enumerate(ip):
                hit = 0
                if ious is not None and len(ig):
                    free = [t for t in range(len(ig)) if t not in taken]
                    if free:
                        best = max(free, key=lambda t: ious[j, t])
                        if ious[j, best] >= args.iou:
                            taken.add(best)
                            hit = 1
                stats[c]["tp"].append(hit)
                stats[c]["scores"].append(float(pr_s[p]))

        # Confusion matrix on predictions above the usual confidence.
        strong = pr_s > 0.25
        pb, pc = pr_b[strong], pr_c[strong]
        if len(gt_b):
            ious = iou_matrix(gt_b, pb) if len(pb) else np.zeros((len(gt_b), 0))
            for t in range(len(gt_b)):
                if ious.shape[1] and ious[t].max() >= args.iou:
                    confusion[gt_c[t], pc[ious[t].argmax()]] += 1
                else:
                    confusion[gt_c[t], len(names)] += 1

        if k % 200 == 0:
            print(f"  {k}/{len(images)} images", flush=True)

    print()
    title = f"{os.path.basename(args.model)} on {args.split}"
    if filtering:
        bounds = []
        if args.min_side:
            bounds.append(f"side >= {args.min_side:.0f} px")
        if args.max_side:
            bounds.append(f"side < {args.max_side:.0f} px")
        title += "  [" + ", ".join(bounds) + "]"
    print(title)
    print("-" * len(title))

    aps, detail = [], {}
    for c, name in enumerate(names):
        a = ap_per_class(np.array(stats[c]["tp"]), np.array(stats[c]["scores"]),
                         stats[c]["n"])
        detail[name] = {"ap50": None if np.isnan(a) else round(a, 4),
                        "objects": stats[c]["n"]}
        if not np.isnan(a):
            aps.append(a)
        val = "n/a" if np.isnan(a) else f"{a:.4f}"
        print(f"  {name:8} AP50 = {val:>8}   ({stats[c]['n']} objects)")
    m = float(np.mean(aps)) if aps else float("nan")
    print(f"  {'mAP50':8}      = {m:.4f}")
    if ignored:
        print(f"\n  {ignored} annotations outside the size filter, ignored")

    print("\n  Confusion matrix (truth in rows, prediction in columns)")
    headers = names + ["missed"]
    print("      " + "".join(f"{h:>10}" for h in headers))
    for i, name in enumerate(names):
        print(f"  {name:>4}" + "".join(f"{confusion[i, j]:>10}"
                                       for j in range(len(names) + 1)))

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"model": os.path.basename(args.model),
                       "split": args.split, "iou": args.iou,
                       "max_side": args.max_side, "min_side": args.min_side,
                       "map50": None if np.isnan(m) else round(m, 4),
                       "per_class": detail,
                       "confusion": confusion[:len(names)].tolist()},
                      f, ensure_ascii=False, indent=2)
        print(f"\nDetails written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
