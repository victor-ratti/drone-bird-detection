# Step 3, int8 quantization: a silent failure, its mechanism, and the fix

## The failure

Static int8 quantization of the YOLOv11n export, calibrated on 200 validation
images, produced a model that **detected nothing at all**.

| Model | mAP50, test split | Bird | Drone | Objects missed |
|---|---|---|---|---|
| fp32 | 0.9814 | 0.9736 | 0.9892 | 23 of 900 |
| int8 static, first attempt | **0.0000** | 0.0000 | 0.0000 | **900 of 900** |

Nothing in the pipeline complained. The file loaded, ran at a plausible speed,
returned tensors of the right shape and dtype. Only an accuracy measurement
revealed it. **A quantized model that is broken looks exactly like a quantized
model that works, until you evaluate it.**

## The mechanism

Comparing raw outputs on the same image told the story immediately:

| Model | Box coordinates | Class scores |
|---|---|---|
| fp32 | min 2.53, max 636.8, mean 187.3 | max 0.894, 10 anchors above 0.25 |
| int8 static, first attempt | min 2.53, max 640.3, mean 188.7 | **max 0.000, exactly zero everywhere** |

Boxes intact, scores annihilated. That points at one specific place in the
graph, and the graph confirms it.

A YOLO ONNX export ends with a `Concat` that merges two branches of
incompatible nature:

```
/model.23/Concat_3
  <- /model.23/Mul_2       box coordinates, in pixels, range 0 to 640
  <- /model.23/Sigmoid     class probabilities, range 0 to 1
```

Per-tensor quantization gives that concatenation **a single scale**, and the
calibrator sets it from the widest branch. The quantized graph carries:

```
output0_QuantizeLinear   scale = 2.530844   zero_point = 0
```

2.530844 x 255 = 645, the pixel range. In uint8 with zero_point 0, the smallest
representable non-zero value is 2.53. **Every class probability, all of them in
[0, 1], rounds to zero.** The boxes, spanning hundreds of pixels, survive
almost untouched.

## The fix

Keep the head tail in floating point. `head_nodes_to_exclude()` in
`src/quantize.py` walks back from each graph output and excludes that `Concat`
plus its direct producers. It is the default; `--quantize-head` restores the
broken behaviour for anyone who wants to reproduce it.

```
keeping the head in float: /model.23/Concat_3, /model.23/Mul_2, /model.23/Sigmoid
```

Cost: three nodes out of a 181-layer network. The file grows from 3.03 to
3.05 MB, still 3.31 times smaller than fp32.

## Accuracy after the fix

| Model | Size | mAP50, full test | Bird | Drone | mAP50, objects under 32 px |
|---|---|---|---|---|---|
| **fp32** | 10.11 MB | **0.9814** | 0.9736 | 0.9892 | **0.6229** |
| int8 static, head in float | 3.05 MB | 0.9565 | 0.9640 | 0.9491 | **0.4811** |
| int8 dynamic | 2.85 MB | 0.9833 | 0.9780 | 0.9886 | not measured |
| int8 static, head quantized | 3.03 MB | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Two things worth stating plainly.

**Static quantization costs 2.5 points overall but 14 points on small objects.**
0.9814 to 0.9565 on the full split looks like a fair trade. On objects under 32
pixels, 0.6229 to 0.4811, a 23 % relative drop. The loss lands exactly where the
model was already weakest, and where the operational case lives. **An accuracy
average hides which part of the distribution paid for the compression.**

**Dynamic quantization costs nothing in accuracy**, 0.9833 against 0.9814, a
difference within noise. It only quantizes weights; activations stay in
floating point, so the head problem cannot occur. Its cost is entirely in
speed, see below.

## Latency

10 intra-op threads, 60 runs per pass, 4 independent passes per model, each in
a fresh process, idle machine. Data in `benchmark_repeatability.json`, protocol
and harness pitfalls in `02_benchmark_protocol.md`.

| Model | Size | Passes (ms) | Median | Dispersion | FPS |
|---|---|---|---|---|---|
| **fp32** | 10.11 MB | 24.6 / 25.0 / 26.2 / 27.6 | **25.61 ms** | x1.12 | **39.1** |
| int8 static, head in float | 3.05 MB | 38.0 / 38.9 / 40.0 / 39.1 | 38.99 ms | x1.05 | 25.6 |
| int8 static, head quantized | 3.03 MB | 40.7 / 38.2 / 38.5 / 39.7 | 39.06 ms | x1.06 | 25.6 |
| int8 dynamic | 2.85 MB | 808.5 / 781.8 / 189.6 / 214.1 | see below | **x4.26** | 2 to 5 |

**Working int8 is 1.52 times slower than fp32.** Same conclusion as before the
fix, now measured on a model that actually detects something.

**Keeping the head in float costs nothing in speed**: 38.99 against 39.06 ms,
inside the dispersion. Three nodes out of 181 stay in floating point and the
model recovers its entire accuracy for free. There was never a trade-off to
make, only a bug to fix.

**The dynamic model's latency is not reproducible on this machine.** Four
passes yesterday gave 207 to 213 ms with dispersion 1.03; four passes today
gave 190 to 810 ms with dispersion 4.26. The distribution is bimodal, roughly
200 ms or 800 ms, depending on nothing the harness controls. No single number
is reported for it, and the range is what goes in the README. That instability
is itself consistent with the diagnosis: the dynamic path falls back on generic
kernels whose cost depends on scheduling, where the fp32 NEON path is stable to
12 %.

The latency figures published before this fix for the static model were
measured on the broken file, the one that returned zero detections. **Timing a
model that detects nothing measures nothing**, and that is the sixth
measurement artifact caught in this project, the only one found by a metric
other than a reproducibility check.

The irony, visible in the table: the broken and the fixed model run at the same
speed, 39.06 against 38.99 ms. The old number happened to be right. It was
still meaningless, because nobody had checked what the model did with that
time.

## What this changes for an embedded deployment

1. **Always evaluate accuracy after quantizing, before benchmarking speed.**
   The failure here was silent, produced a well-formed file, and would have
   passed any shape or smoke test. Only mAP caught it.
2. **Never quantize a tensor that mixes physical units.** Coordinates in pixels
   and probabilities in [0, 1] cannot share a scale. The same trap exists in
   any detection head that concatenates regression and classification outputs,
   and in any model that emits several quantities on one tensor.
3. **Report accuracy per regime, not as an average.** The 2.5-point headline
   loss and the 14-point loss on small objects are the same model.
4. **The choice of backend still precedes the choice of weight format.** Even
   working, int8 does not pay off on this ARM CPU, because onnxruntime's
   default CPU executor has no int8 kernels on par with its NEON fp32 path.
   The gain would exist on the NPU through the QNN execution provider, on
   XNNPACK, or on a Jetson with TensorRT.

## Reproduce

```bash
python src/quantize.py models/baseline_best.onnx --mode static \
    --calibration data/Drone-Bird-Detection-3/valid/images --n 200
python src/evaluate.py models/baseline_best_int8_static.onnx data/Drone-Bird-Detection-3
python src/evaluate.py models/baseline_best_int8_static.onnx data/Drone-Bird-Detection-3 --max-side 32

# reproduce the failure
python src/quantize.py models/baseline_best.onnx --mode static --quantize-head \
    --calibration data/Drone-Bird-Detection-3/valid/images --n 200
```

`models/baseline_best_int8_static_head_quantized.onnx` is the broken file, kept
so the claim can be checked without rerunning the calibration.
