# Step 4, hardening: where the 0.98 lies

## Question asked

The model reports 0.9878 mAP50 and essentially never confuses a drone with a
bird. Before believing that number, two hypotheses to rule out:

1. The two classes may be separable by size alone, in which case the network
   contributes little.
2. The real operational case, a small distant object, may be absent or poorly
   handled, and drowned in an average dominated by easy objects.

Both check out. Both are true.

## 1. The classes are largely separable by size

Distribution of box sides on the test split, in the 640x640 frame the model
sees (`object_sizes.json`):

| Class | Median side | Share of small objects |
|---|---|---|
| Bird | 378 px | 2.6 % |
| Drone | 66 px | 11.7 % |

**Scale ratio between the two classes: 5.72.** A bird in this dataset fills the
frame, a drone is a small object.

Test: a single-threshold classifier on the box side, **using no pixels at all**,
threshold learned on the training split and applied unchanged to the test
split (`size_bias.json`).

| | Accuracy |
|---|---|
| Majority-class guess | 50.7 % |
| Size threshold alone, side >= 152 px means Bird | **76.1 %** |
| The model | ~99 % |

The bias exists and it is substantial: 25 points above chance without looking
at the image. It does not explain everything, the network does contribute the
remaining 23 points. But any performance claim on this dataset must mention
that a quarter of the job is handed over by the size statistics.

## 2. Performance collapses on small objects

Evaluation by size band on the test split. Annotations **and** predictions are
filtered by the same band; otherwise detections of out-of-band objects count
as false positives and crush precision for a purely methodological reason.

| Band | Bird AP50 | Drone AP50 | mAP50 |
|---|---|---|---|
| Small, side < 32 px | **0.3824** (13 objects) | 0.8634 (76) | **0.6229** |
| Medium, 32 to 96 px | 0.8047 (95) | 0.9215 (233) | 0.8631 |
| Large, side >= 96 px | 0.9957 (348) | 0.9478 (135) | 0.9718 |
| Full split | 0.9736 (456) | 0.9892 (444) | 0.9814 |

![AP50 by object-size band](figures/ap_by_size_band.png)

**mAP50 goes from 0.981 to 0.623 on small objects. The Bird class falls to
0.382, 2.6 times lower than on the full split.**

The headline number in the README therefore mostly measures the easy case. On
the case that matters, a distant object under 32 pixels, the model is mediocre,
and nothing flagged it.

The inversion between the two classes is consistent with the bias in point 1:
birds in this dataset are almost always large, so the model never learned to
recognize a small bird. On large objects Bird reaches 0.9957, on small ones it
drops to 0.3824.

## 3. The dataset does not contain the hard case

**13 small birds in the entire test split.** The 0.3824 figure is a signal, not
a reliable measurement: the confidence interval on 13 objects is wide.

This is the most important conclusion of the step. The operational
counter-drone problem, telling a small flying object from a bird at range, **is
not represented in this dataset**. Neither in quantity nor in difficulty.

## What it changes

The hypothesis made on 2026-09-07 when the dataset was chosen ("at 0.979
reference this set is easy and discrimination is not the real challenge") is
now established by three independent measurements: the scale ratio, the
single-threshold classifier, and the drop by size band.

Logical next step: move to **Anti-UAV** (CVPR challenge, RGB and infrared),
which contains sequences of small objects at range, and rerun the same three
measurements on it. The full pipeline, training, export, benchmark, size-band
evaluator, is in place and replays unchanged.

## Method note

The first pass of this evaluation gave 0.0977 mAP50 on small objects, a
spectacular collapse. It was a protocol defect: only the annotations were
filtered by size, not the predictions, so every correct detection of a large
object counted as a false positive. The confusion matrix, computed differently,
showed 12 birds detected out of 13, which raised the flag.

Third time in this project that a spectacular result turns out to be a
measurement artifact. The two previous ones are in `02_benchmark_protocol.md`.

## Tools written for this step

- `src/analyze_sizes.py`: size distribution per class and per split.
- `src/size_bias.py`: single-threshold classifier, lower bound on what the
  dataset gives away for free.
- `src/evaluate.py`: AP50 evaluation without PyTorch, on onnxruntime and
  numpy, with size-band filtering. Calibrated to 0.6 point of the ultralytics
  implementation on the full split, 0.9814 against 0.9878.
- `src/plot_size_bands.py`: the figure above, from the evaluator's JSON output.
