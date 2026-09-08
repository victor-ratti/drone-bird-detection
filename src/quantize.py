# -*- coding: utf-8 -*-
"""Int8 quantization of an ONNX detection model.

Two modes, deliberately both, because the comparison is the result:

- `dynamic`: quantizes weights only. Activations stay in floating point and are
  converted on the fly. Needs no data. Designed for MatMul and LSTM, so it is
  poorly suited to a Conv-dominated network like YOLO. Serves as a control.
- `static`: quantizes weights AND activations. Needs a calibration set to
  observe the real range of activations. This is the mode that can pay off on
  a CNN.

Usage:
    python src/quantize.py models/baseline_best.onnx --mode dynamic
    python src/quantize.py models/baseline_best.onnx --mode static \
        --calibration data/Drone-Bird-Detection-3/valid/images --n 200
"""

import argparse
import glob
import os
import sys

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType,
    quantize_dynamic, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_prepare  # noqa: E402


class CalibrationReader(CalibrationDataReader):
    """Feeds calibration images one by one to onnxruntime."""

    def __init__(self, folder, input_name, n=200, seed=0):
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = sorted(f for p in patterns for f in glob.glob(os.path.join(folder, p)))
        if not files:
            raise SystemExit(f"No images in {folder}")
        rng = np.random.default_rng(seed)
        if len(files) > n:
            files = [files[i] for i in rng.choice(len(files), n, replace=False)]
        self.name = input_name
        self.files = files
        self.it = None
        print(f"  calibrating on {len(files)} images from {folder}")

    def get_next(self):
        if self.it is None:
            self.it = iter(self.files)
        for path in self.it:
            x = load_and_prepare(path)
            if x is not None:
                return {self.name: x}
        return None

    def rewind(self):
        self.it = None


def output_path(model, suffix):
    base, _ = os.path.splitext(model)
    return f"{base}_{suffix}.onnx"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--mode", choices=["dynamic", "static"], required=True)
    ap.add_argument("--calibration", help="image folder, required for static")
    ap.add_argument("--n", type=int, default=200, help="calibration images")
    ap.add_argument("--output")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    out = args.output or output_path(args.model, f"int8_{args.mode}")

    # Recommended by onnxruntime before any quantization: fixes shapes and
    # simplifies the graph. Without it, static quantization leaves blocks
    # unquantized.
    prepared = output_path(args.model, "prep")
    print(f"Graph preprocessing -> {os.path.basename(prepared)}")
    quant_pre_process(args.model, prepared, skip_symbolic_shape=False)

    if args.mode == "dynamic":
        print("Dynamic quantization, weights only")
        quantize_dynamic(prepared, out, weight_type=QuantType.QInt8)
    else:
        if not args.calibration:
            raise SystemExit("--calibration is required in static mode")
        import onnxruntime as ort
        input_name = ort.InferenceSession(
            prepared, providers=["CPUExecutionProvider"]).get_inputs()[0].name
        print("Static quantization, weights and activations")
        reader = CalibrationReader(args.calibration, input_name, args.n)
        quantize_static(
            prepared, out, reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

    os.remove(prepared)
    before = os.path.getsize(args.model) / 1024 / 1024
    after = os.path.getsize(out) / 1024 / 1024
    print()
    print(f"  {os.path.basename(args.model):<36} {before:6.2f} MB")
    print(f"  {os.path.basename(out):<36} {after:6.2f} MB   x{before / after:.2f} smaller")
    return 0


if __name__ == "__main__":
    sys.exit(main())
