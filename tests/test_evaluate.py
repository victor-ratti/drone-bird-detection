# -*- coding: utf-8 -*-
"""Unit tests on the evaluation core: IoU, NMS, AP, letterbox mapping.

The evaluator is the instrument every hardening claim in this repository rests
on. These tests pin its arithmetic to hand-computed values.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import ap_per_class, iou_matrix, labels_to_letterbox, nms  # noqa: E402
from preprocess import letterbox  # noqa: E402


def test_iou_identical_boxes():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_half_overlap():
    # Intersection 5x10 = 50, union 100 + 100 - 50 = 150.
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == pytest.approx(1 / 3, abs=1e-6)


def test_iou_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_nms_keeps_highest_score_among_overlapping():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                     dtype=np.float32)
    scores = np.array([0.6, 0.9, 0.5])
    keep = nms(boxes, scores, threshold=0.45)
    # Box 1 wins over box 0 (they overlap heavily), box 2 is independent.
    assert set(keep.tolist()) == {1, 2}


def test_ap_perfect_ranking():
    assert ap_per_class(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]), 3) == pytest.approx(1.0)


def test_ap_all_false_positives():
    assert ap_per_class(np.array([0, 0]), np.array([0.9, 0.8]), 2) == pytest.approx(0.0)


def test_ap_no_predictions():
    assert ap_per_class(np.array([]), np.array([]), 5) == pytest.approx(0.0)


def test_ap_no_ground_truth_is_nan():
    assert np.isnan(ap_per_class(np.array([1]), np.array([0.9]), 0))


def test_ap_mixed_hand_computed():
    # Ranked by score: TP, FP, TP with 2 ground truths.
    # precision = [1, 1/2, 2/3], recall = [1/2, 1/2, 1]
    # decreasing envelope of precision = [1, 2/3, 2/3]
    # AP = 0.5 * 1 + 0 + 0.5 * 2/3 = 0.8333
    tp = np.array([1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7])
    assert ap_per_class(tp, scores, 2) == pytest.approx(5 / 6, abs=1e-6)


def test_ap_is_order_independent_of_input_arrays():
    # Same detections given in shuffled order must yield the same AP.
    tp = np.array([0, 1, 1])
    scores = np.array([0.8, 0.9, 0.7])
    assert ap_per_class(tp, scores, 2) == pytest.approx(5 / 6, abs=1e-6)


def test_letterbox_pads_height_for_landscape_image():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas, ratio, dx, dy = letterbox(img, 640)
    assert canvas.shape == (640, 640, 3)
    assert ratio == pytest.approx(1.0)
    assert (dx, dy) == (0, 80)
    # Padding rows are gray 114, content rows are black.
    assert canvas[0, 0].tolist() == [114, 114, 114]
    assert canvas[80, 0].tolist() == [0, 0, 0]


def test_labels_map_into_letterbox_frame(tmp_path):
    # Centered box, half width, half height, on a 640x480 image.
    label = tmp_path / "x.txt"
    label.write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    boxes, classes = labels_to_letterbox(str(label), 640, 480, 1.0, 0, 80)
    assert classes.tolist() == [1]
    # x: 320 +/- 160 -> [160, 480]; y: 240 +/- 120 then +80 -> [200, 440]
    assert boxes[0].tolist() == pytest.approx([160, 200, 480, 440])
