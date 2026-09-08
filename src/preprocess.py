# -*- coding: utf-8 -*-
"""Image preprocessing shared by quantization and evaluation.

Reproduces the ultralytics letterbox exactly: keep the aspect ratio, pad with
gray 114. Calibrating or evaluating on stretched images would measure the wrong
activation ranges and the wrong boxes.
"""

import cv2
import numpy as np

INPUT_SIZE = 640
PAD_VALUE = 114


def letterbox(img, size=INPUT_SIZE):
    """Resize keeping aspect ratio, pad to a square canvas.

    Returns (canvas, ratio, dx, dy). The ratio and offsets let annotations be
    mapped into the same 640x640 frame the model sees.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    dy, dx = (size - nh) // 2, (size - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, dx, dy


def to_tensor(img):
    """HWC uint8 RGB -> NCHW float32 in [0, 1], as at training time."""
    x = img.astype(np.float32) / 255.0
    return np.expand_dims(x.transpose(2, 0, 1), 0)


def load_and_prepare(path, size=INPUT_SIZE):
    """Image on disk -> model input tensor, or None if unreadable."""
    img = cv2.imread(path)
    if img is None:
        return None
    canvas, _, _, _ = letterbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size)
    return to_tensor(canvas)
