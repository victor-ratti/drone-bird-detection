# Step 5, tracking: ByteTrack on real footage, and what a tracker cannot do

## Setup

`src/track.py` runs the ONNX detector on every frame and links detections
with ByteTrack from Roboflow's `trackers` package, the successor of the
deprecated `supervision.ByteTrack`. No PyTorch. Boxes are mapped back from the
640 letterbox to the original frame. Per-frame track records, a stats JSON, an
annotated video and an optional GIF are written. Eight unit tests pin the
statistics and the back-projection.

Public drone footage comes without annotated trajectories, so MOTA and IDF1
are out of reach. What can be measured from the output alone:

- **coverage**: frames carrying a confirmed track, over frames where the
  detector fired at the run's threshold;
- **fragmentation**: confirmed track ids per expected object, and **breaks**,
  ids beyond the expected count;
- track length distribution, births per minute, detect and end-to-end FPS.

Tracker settings: activation 0.25, high-confidence stage 0.6, confirmation
after 2 consecutive frames, lost buffer 30 frames, minimum IoU 0.1.

## The footage, and why three of five videos were unusable

Five videos from Wikimedia Commons, all with explicit licenses.

| Video | What it actually is | Verdict |
|---|---|---|
| 173rd Airborne drone operator training | Edited documentary: close-ups of drones on a soldier's lap, operators wearing FPV goggles, cuts every few seconds | Unusable. FPV goggles on a head detected as Drone at 0.61; a drone on a lap split into two tracks; 18 ids in 41 s |
| BLM drone training | Edited documentary with one real 17-frame shot of a drone in the sky, reused six times in the edit (pixel diff about 2 between occurrences, compression noise) | Unusable. 0.7 s of real flight; the end-card logo on black detected as an object |
| Hovering drone, Ystad | Continuous 20 s shot, **top-down**, a DJI Mini filling a 2696x2160 frame | Usable: one object, continuous presence |
| Hawk attacks drone | Continuous 36 s shot **from the drone's own camera**; the drone is never visible, a hawk makes three attack passes | Usable: one real bird, intermittent presence, plus ground structures |
| Hovering drone, slow motion | Downloaded as fallback, not needed | Unused |

**First lesson: an edited montage is not a tracking benchmark.** Cuts, title
cards and close-ups produce tracks that say nothing about the tracker. The
harness gained `--start` and `--end` to select one continuous shot.

**Second lesson: none of the available footage is the canonical case.** A
small drone in the sky, filmed from the ground, in continuous flight, is what
the detector was trained on and what a counter-drone system faces. Commons has
drone point-of-view footage and close-ups instead. That is the same data gap
found in step 4, from another direction, and the same answer: Anti-UAV.

## Results on the two usable shots

### Ystad, one hovering drone, 603 frames

| Detector threshold fed to the tracker | Frames with detection | Frames with track | Coverage | Tracks | Breaks |
|---|---|---|---|---|---|
| 0.25 | 402 | 317 | 0.79 | 2 | 1 |
| **0.10** | 554 | 543 | **0.98** | **1** | **0** |

At 0.25 the object is tracked as id 0 for frames 12 to 207, lost, then as id 1
from 345 to 601: one break, 137 frames without a track. At 0.10 it is a single
id over 543 frames, with 14 frames left unconfirmed.

**This is ByteTrack doing exactly what its paper says.** The drone never leaves
the frame; its confidence dips below 0.25 during gusts, when rotor blur and
motion blur hit a model that already only sees it at 0.38 on average. Feeding
those low-score boxes to the tracker lets the second association stage keep the
identity through the dips.

The mean confidence, 0.38 on an object filling half the frame, is the size bias
of step 4 seen from the other side: the training set taught the model that a
large object is a bird. A large drone is out of distribution.

### Hawk, three attack passes, 855 frames

| Threshold | Frames with detection | Frames with track | Coverage | Tracks | Ids on the hawk | Ids on ground structures |
|---|---|---|---|---|---|---|
| 0.25 | 260 | 186 | 0.72 | 5 | 3 | 2 |
| 0.10 | 417 | 198 | 0.47 | 6 | 3, plus one 1-frame blur at 0.17 | 2 |

Per track at 0.10:

| Id | Class | Frames | Length | Mean conf | Box (px) | What it is |
|---|---|---|---|---|---|---|
| 1 | Bird | 393 to 407 | 15 | 0.54 | 292x302 | hawk, first pass |
| 2 | Bird | 497 to 551 | 55 | 0.60 | 605x503 | hawk, second pass |
| 4 | Bird | 748 to 842 | 95 | 0.78 | 876x654 | hawk, third pass, talons out |
| 3 | Drone | 547 to 575 | 26 | 0.44 | 99x67 | a house roof in the forest |
| 5 | Drone | 844 to 854 | 11 | 0.65 | 274x73 | a ground structure |
| 0 | Bird | 292 | 1 | 0.17 | 606x496 | wing blur, unconfirmed noise |

![Hawk third pass, tracked as one id](figures/tracking_hawk.gif)

**Within each pass, one id and no break.** The three ids are three real
appearances: the hawk leaves the frame between passes. A motion-only tracker
cannot re-identify an object after a true absence, and it should not be
blamed for it. Judged per continuous appearance, fragmentation is 1.0.

**Lowering the threshold buys nothing here.** Detections rise from 260 to 417
frames, tracked frames barely move (186 to 198), the longest track stays at 95,
and one 1-frame blur track appears. The gaps are absences, not low-confidence
detections. The two runs together draw the line: ByteTrack's low-score recovery
helps when the object is present and the detector hesitates, and only then.

**A large bird gets 0.78, a large drone gets 0.38.** Same size bias again: the
hawk filling the frame is exactly what the training set calls a bird.

**The two Drone tracks are false positives on ground structures**, a roof and a
building seen from above. The detector has never seen the ground from a drone.

## Method notes

1. **Tentative track ids.** `trackers` returns id -1 for detections not yet
   confirmed over `minimum_consecutive_frames`. The first version of the
   harness counted them as a track, which turned a bucket of 82 unconfirmed
   detections spread over 840 frames into "the longest track" on the hawk
   video. Now excluded from every statistic and reported separately.
2. **Do not pre-filter at the tracker's activation threshold.** The first
   version cut detections at 0.25 before the tracker saw them, disabling the
   low-score stage that is the whole point of ByteTrack. The detector now
   passes everything above `--conf` (0.10 recommended), and the tracker
   applies its own two thresholds.
3. **Coverage misbehaves at low thresholds.** Its denominator grows with weak
   detections the tracker rightly refuses, which is why the hawk run reads
   0.47 at 0.10. Read coverage together with tracked frames and track lengths,
   never alone.
4. **Class of a track is a majority vote** over its frames: the detector
   flips labels on the same object.

## Speed

On the Snapdragon X Elite, 10 threads: detection 44 ms per 1080p frame and
57 ms per 2696x2160 frame, video decode and letterbox included. Tracking adds
under 1 ms. End to end, 22 FPS at 1080p, 17 FPS at 4K, against 39.5 FPS for
inference alone on a 640 tensor: at this resolution, decoding and resizing
cost as much as the network.

## What is missing

Ground-view footage with annotated trajectories, to compute MOTA and IDF1 and
to measure the tracker on its intended scene. Anti-UAV provides both. The
harness is ready for it.
