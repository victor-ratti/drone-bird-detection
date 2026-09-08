# -*- coding: utf-8 -*-
"""Detect and track over a video: ONNX detector + ByteTrack, no PyTorch.

Runs the detector on every frame, links detections across frames with
ByteTrack (from Roboflow's `trackers` package, the successor of the deprecated
`supervision.ByteTrack`), writes an annotated video, a JSON of every track, a
stats JSON, and optionally a GIF for the README.

Evaluate on a continuous shot, never on an edited montage. Cuts, title cards
and close-ups produce spurious tracks that say nothing about the tracker:
use --start and --end to select one shot with the object in view.

Metrics without ground-truth tracks. Public drone footage rarely comes with
annotated trajectories, so this script reports what can be measured from the
output alone:

- coverage: share of frames with a confident detection that also carry an
  active track. Below 1.0, the tracker is dropping objects the detector sees.
- fragmentation: number of track ids created per expected object. A single
  drone crossing the frame should yield one id; five ids mean the track broke
  four times.
- track length distribution, births per minute, detect and end-to-end FPS.

MOTA and IDF1 need annotated sequences. That is what the Anti-UAV step is for.

Usage:
    python src/track.py models/baseline_best.onnx data/videos/clip.mp4 \
        --out results/tracking/clip --expected-objects 1 --gif
"""

import argparse
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import predict  # noqa: E402
from preprocess import letterbox  # noqa: E402

# onnxruntime, supervision and trackers are imported inside main(). The
# statistics below are pure functions over track records: keeping them
# importable without the tracking stack lets the tests, and CI, exercise them
# with numpy alone.

COLORS = {0: (60, 120, 255), 1: (255, 140, 40)}   # BGR: Bird blue-ish, Drone orange
NAMES = {0: "Bird", 1: "Drone"}


def to_original(boxes, r, dx, dy, w, h):
    """Boxes from the 640 letterbox frame back to original frame coordinates."""
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - dx) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - dy) / r
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h - 1)
    return out


def summarize(records, n_frames, frames_with_detection, fps_video,
              expected_objects=None, detect_ms=None, total_ms=None):
    """Tracking statistics from per-frame track records.

    `records` is a list of dicts with keys frame, track_id, class_id. Pure
    function, unit-tested in tests/test_track.py.
    """
    by_id, cls_votes = {}, {}
    for rec in records:
        by_id.setdefault(rec["track_id"], []).append(rec["frame"])
        votes = cls_votes.setdefault(rec["track_id"], {})
        votes[rec["class_id"]] = votes.get(rec["class_id"], 0) + 1
    lengths = sorted(len(set(f)) for f in by_id.values())
    frames_with_track = len({rec["frame"] for rec in records})
    classes = {}
    for tid in by_id:
        # A track's class is its majority vote: the detector can flip the
        # label between frames on the same object.
        cls = max(cls_votes[tid].items(), key=lambda kv: kv[1])[0]
        name = NAMES.get(cls, str(cls))
        classes[name] = classes.get(name, 0) + 1

    minutes = n_frames / fps_video / 60.0 if fps_video else None
    stats = {
        "frames": n_frames,
        "frames_with_detection": frames_with_detection,
        "frames_with_track": frames_with_track,
        "coverage": round(frames_with_track / frames_with_detection, 4)
        if frames_with_detection else None,
        "tracks": len(by_id),
        "tracks_per_class": classes,
        "track_length_frames": {
            "min": lengths[0] if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "track_births_per_minute": round(len(by_id) / minutes, 2) if minutes else None,
    }
    if expected_objects:
        stats["expected_objects"] = expected_objects
        stats["fragmentation"] = round(len(by_id) / expected_objects, 2)
        # Ids beyond the expected count are breaks: each one is a lost-and-
        # re-acquired object, the closest proxy to an id switch without truth.
        stats["track_breaks"] = max(0, len(by_id) - expected_objects)
    if detect_ms is not None:
        stats["detect_ms_per_frame"] = round(detect_ms, 2)
        stats["detect_fps"] = round(1000.0 / detect_ms, 1) if detect_ms else None
    if total_ms is not None:
        stats["end_to_end_ms_per_frame"] = round(total_ms, 2)
        stats["end_to_end_fps"] = round(1000.0 / total_ms, 1) if total_ms else None
    return stats


def draw(frame, boxes, ids, classes, confs):
    for (x1, y1, x2, y2), tid, c, s in zip(boxes.astype(int), ids, classes, confs):
        color = COLORS.get(int(c), (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{tid} {NAMES.get(int(c), c)} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)


GIF_WIDTH = 480
GIF_MAX_FRAMES = 150


def gif_frame(frame, width=GIF_WIDTH):
    """Downscale on the fly: keeping full-resolution frames in memory for a
    long video would need gigabytes."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def write_gif(frames, path, fps=10):
    """Small GIF for the README, from already downscaled frames."""
    from PIL import Image
    if not frames:
        return
    imgs = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=128)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="output prefix, e.g. results/tracking/clip")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--expected-objects", type=int, default=None)
    ap.add_argument("--start", type=int, default=0, help="first frame to process")
    ap.add_argument("--end", type=int, default=None, help="last frame, exclusive")
    ap.add_argument("--activation", type=float, default=0.25,
                    help="confidence needed to start a new track")
    ap.add_argument("--high-conf", type=float, default=0.6,
                    help="confidence above which a detection joins the first matching stage")
    ap.add_argument("--min-frames", type=int, default=2,
                    help="consecutive frames before a track is confirmed")
    ap.add_argument("--lost-buffer", type=int, default=30,
                    help="frames a lost track is kept before deletion")
    ap.add_argument("--min-iou", type=float, default=0.1,
                    help="minimum IoU for a detection to match a track")
    ap.add_argument("--gif", action="store_true", help="also write a README-sized GIF")
    ap.add_argument("--gif-every", type=int, default=3,
                    help="keep one processed frame in N for the GIF")
    args = ap.parse_args()

    import onnxruntime as ort
    import supervision as sv
    from trackers import ByteTrackTracker

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{os.path.basename(args.video)}: {w}x{h}, {fps:.1f} fps, {total} frames")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    session = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    tracker = ByteTrackTracker(lost_track_buffer=args.lost_buffer,
                               frame_rate=fps,
                               track_activation_threshold=args.activation,
                               minimum_consecutive_frames=args.min_frames,
                               minimum_iou_threshold=args.min_iou,
                               high_conf_det_threshold=args.high_conf)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    records, gif_frames = [], []
    frames_with_detection, unconfirmed_frames, n = 0, 0, 0
    detect_ms, total_ms = [], []
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_idx = args.start
    end = args.end if args.end is not None else total

    while frame_idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        img, r, dx, dy = letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, scores, classes = predict(session, input_name, img, conf=args.conf)
        t1 = time.perf_counter()
        boxes = to_original(boxes, r, dx, dy, w, h)
        if len(boxes):
            frames_with_detection += 1
            dets = sv.Detections(xyxy=boxes.astype(np.float32),
                                 confidence=scores.astype(np.float32),
                                 class_id=classes.astype(int))
        else:
            dets = sv.Detections.empty()
        tracked = tracker.update(dets)
        t2 = time.perf_counter()
        detect_ms.append((t1 - t0) * 1000)
        total_ms.append((t2 - t0) * 1000)

        if len(tracked) and tracked.tracker_id is not None:
            # tracker_id -1 marks a tentative track, not yet confirmed over
            # minimum_consecutive_frames. It is a detection the tracker is
            # still deciding about, not an object being followed: counted
            # apart, never as a track.
            confirmed = tracked.tracker_id >= 0
            unconfirmed_frames += int((~confirmed).any())
            tracked = tracked[confirmed]
        if len(tracked):
            for box, tid, c, s in zip(tracked.xyxy, tracked.tracker_id,
                                      tracked.class_id, tracked.confidence):
                records.append({"frame": frame_idx, "track_id": int(tid), "class_id": int(c),
                                "confidence": round(float(s), 4),
                                "box": [round(float(v), 1) for v in box]})
            draw(frame, tracked.xyxy, tracked.tracker_id, tracked.class_id, tracked.confidence)

        cv2.putText(frame, f"frame {frame_idx}  tracks {len(tracked)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        if args.gif and n % args.gif_every == 0 and len(gif_frames) < GIF_MAX_FRAMES:
            gif_frames.append(gif_frame(frame))
        n += 1
        frame_idx += 1
        if n % 100 == 0:
            print(f"  {n}/{end - args.start} frames", flush=True)

    cap.release()
    writer.release()

    stats = summarize(records, n, frames_with_detection, fps, args.expected_objects,
                      statistics.median(detect_ms) if detect_ms else None,
                      statistics.median(total_ms) if total_ms else None)
    stats["video"] = os.path.basename(args.video)
    stats["frames_with_unconfirmed_only"] = unconfirmed_frames
    stats["resolution"] = [w, h]
    stats["fps_video"] = round(fps, 2)
    stats["frame_range"] = [args.start, args.start + n]
    stats["tracker"] = {"library": "trackers.ByteTrackTracker",
                        "activation": args.activation, "high_conf": args.high_conf,
                        "min_frames": args.min_frames, "lost_buffer": args.lost_buffer,
                        "min_iou": args.min_iou, "detector_conf": args.conf}

    with open(args.out + "_tracks.json", "w", encoding="utf-8") as f:
        json.dump(records, f)
    with open(args.out + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    if args.gif and gif_frames:
        write_gif(gif_frames, args.out + ".gif")

    print()
    print(f"frames {stats['frames']}, with detection {stats['frames_with_detection']}, "
          f"with track {stats['frames_with_track']}")
    print(f"coverage      : {stats['coverage']}")
    print(f"tracks        : {stats['tracks']}  {stats['tracks_per_class']}")
    print(f"track length  : {stats['track_length_frames']}")
    if "fragmentation" in stats:
        print(f"fragmentation : {stats['fragmentation']}  (breaks: {stats['track_breaks']})")
    print(f"detect        : {stats.get('detect_ms_per_frame')} ms  "
          f"end-to-end {stats.get('end_to_end_ms_per_frame')} ms  "
          f"({stats.get('end_to_end_fps')} FPS)")
    print(f"written       : {args.out}.mp4, _tracks.json, _stats.json"
          + (", .gif" if args.gif else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
