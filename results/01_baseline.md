# Step 1, baseline model `baseline_n`

Raw log of the training run of 2026-09-08. This is the trace; the README
carries the synthesis.

## Conditions

| | |
|---|---|
| Starting weights | `yolo11n.pt`, COCO-pretrained |
| Dataset | Drone-Bird-Detection v3, 5418 / 1547 / 772 |
| Epochs | 50, early stopping not triggered (`patience=15`) |
| Image size | 640 |
| Batch | 32 |
| Optimizer | AdamW, chosen automatically, lr0 = 0.001667, momentum = 0.9 |
| Hardware | Tesla T4 on Google Colab |
| Duration | 1 h 33 |
| Ultralytics | 8.4.143, torch 2.11.0+cu128 |

Architecture: 181 layers, 2,590,230 parameters during training, 2,582,542
after fusion, 6.4 GFLOPs. 451 of 499 pretrained tensors were transferred, the
detection head being relearned for 2 classes instead of 80.

## Results

Test split, 772 images, 900 instances:

| | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| All classes | 0.9878 | 0.7505 | 0.968 | 0.966 |
| Drone (444) | 0.9928 | 0.738 | 0.984 | 0.982 |
| Bird (456) | 0.9829 | 0.763 | 0.953 | 0.950 |

Validation split, 1547 images, 1752 instances: mAP50 0.975, mAP50-95 0.758.

Reference published by the dataset author, same architecture: mAP50 0.979.

## Learning curve

| Epoch | Validation mAP50 |
|---|---|
| 1 | 0.594 |
| 5 | 0.852 |
| 10 | 0.916 |
| 20 | 0.960 |
| 30 | 0.968 |
| 40 | 0.973 |
| 50 | 0.975 |

Plateau around epoch 30. The last 20 epochs bring 0.7 point. Disabling
mosaic for the last 10 epochs (`close_mosaic=10`) drops the training losses
without a noticeable validation gain.

## What these numbers say

**The pipeline is correct.** Beating the published reference by 0.9 point with
the same architecture rules out a configuration, path or class error.

**The dataset is saturated.** At 0.988 mAP50 there is nothing left to gain on
this metric. Optimizing detection further would be wasted time.

**The real signal is the gap between mAP50 and mAP50-95.** 0.988 against 0.751:
the model sees the objects but places its boxes loosely. On a real
counter-drone system, box quality drives range estimation and tracking
stability. It is measurable and improvable.

**Drone / bird discrimination is already solved, and that is bad news.** The
confusion matrix on the test split shows essentially no cross-confusion.
Ultralytics' normalized matrix rounds both off-diagonal cells to 0.00; the
repository's own evaluator (`src/evaluate.py`, confidence 0.25) finds 2 birds
predicted as drones out of 456, and 0 drones predicted as birds out of 444.

|  | True Bird | True Drone | True background |
|---|---|---|---|
| Predicted Bird | 0.96 | 0.00 | 0.66 |
| Predicted Drone | 0.00 | 0.98 | 0.34 |
| Predicted background | 0.04 | 0.02 | - |

The recall deficit on Bird (0.950 vs 0.982) therefore does not come from birds
taken for drones, but from birds simply missed: 4 % fall to background, against
2 % for drones.

The only remaining error mode is the false alarm on empty background, and two
thirds of those phantom detections carry the Bird label.

**What it implies.** The hypothesis made on 2026-09-07 when choosing this
dataset ("at 0.979 reference, the set is easy and discrimination is not the real
challenge") is now verified by measurement. The hardening step is no longer a
comfort option, it is the only way to make this project interesting.

## Next

Do not chase a better mAP50, it is saturated.

1. **Step 3, compression and measurement on ARM CPU.** The result nobody has
   published on this dataset, and the one that speaks to embedded roles.
2. **Step 4, hardening, now mandatory.** Isolate instances under 32 pixels and
   measure on them separately. If cross-confusion stays null even on small
   objects, the dataset must change, to Anti-UAV, which contains the hard case.

## Files

Colab archive `results_baseline_n.zip`, containing weights, curves, confusion
matrices and the validation outputs on the test split. Unpacked into
`results/01_baseline/` (training) and `results/01_baseline_test/` (test-split
validation).
