# -*- coding: utf-8 -*-
"""Unit tests on the tracking statistics and the box back-projection."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from track import summarize, to_original  # noqa: E402


def rec(frame, tid, cls=1):
    return {"frame": frame, "track_id": tid, "class_id": cls}


def test_single_unbroken_track():
    records = [rec(f, 1) for f in range(100)]
    s = summarize(records, n_frames=100, frames_with_detection=100, fps_video=30,
                  expected_objects=1)
    assert s["tracks"] == 1
    assert s["coverage"] == pytest.approx(1.0)
    assert s["fragmentation"] == pytest.approx(1.0)
    assert s["track_breaks"] == 0
    assert s["track_length_frames"] == {"min": 100, "median": 100, "max": 100}
    assert s["tracks_per_class"] == {"Drone": 1}


def test_fragmented_track_counts_breaks():
    # One object, but the id changes three times: 4 ids for 1 object.
    records = ([rec(f, 1) for f in range(0, 25)] + [rec(f, 2) for f in range(25, 50)]
               + [rec(f, 3) for f in range(50, 75)] + [rec(f, 4) for f in range(75, 100)])
    s = summarize(records, 100, 100, 30, expected_objects=1)
    assert s["tracks"] == 4
    assert s["fragmentation"] == pytest.approx(4.0)
    assert s["track_breaks"] == 3


def test_coverage_below_one_when_tracker_drops_frames():
    # Detector fires on 100 frames, tracker only carries 80 of them.
    records = [rec(f, 1) for f in range(80)]
    s = summarize(records, 100, 100, 30)
    assert s["coverage"] == pytest.approx(0.8)


def test_no_records():
    s = summarize([], 50, 0, 30, expected_objects=1)
    assert s["tracks"] == 0
    assert s["coverage"] is None
    assert s["track_length_frames"] == {"min": 0, "median": 0, "max": 0}


def test_births_per_minute():
    records = [rec(f, 1) for f in range(10)] + [rec(f, 2) for f in range(10, 20)]
    # 1800 frames at 30 fps is one minute, 2 tracks -> 2 per minute.
    s = summarize(records, 1800, 20, 30)
    assert s["track_births_per_minute"] == pytest.approx(2.0)


def test_speed_fields():
    s = summarize([rec(0, 1)], 1, 1, 30, detect_ms=25.0, total_ms=27.5)
    assert s["detect_fps"] == pytest.approx(40.0)
    assert s["end_to_end_fps"] == pytest.approx(36.4, abs=0.05)


def test_to_original_inverts_letterbox():
    # A 1280x720 frame letterboxed to 640: ratio 0.5, dy = (640 - 360) / 2 = 140.
    boxes_640 = np.array([[100.0, 240.0, 200.0, 340.0]])
    out = to_original(boxes_640, r=0.5, dx=0, dy=140, w=1280, h=720)
    assert out[0].tolist() == pytest.approx([200.0, 200.0, 400.0, 400.0])


def test_to_original_clips_to_frame():
    boxes_640 = np.array([[-50.0, 0.0, 700.0, 640.0]])
    out = to_original(boxes_640, r=1.0, dx=0, dy=0, w=640, h=640)
    assert out[0].tolist() == pytest.approx([0.0, 0.0, 639.0, 639.0])
