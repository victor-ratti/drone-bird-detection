# Step 3, benchmark harness: the protocol, and the method error that almost got through

## The error, because it is more instructive than the result

First sweep of the thread count, every point measured in the same Python
process, one after the other:

| Threads | Median (ms) |
|---|---|
| 1 | 323 |
| 4 | 82 |
| 6 | **65** |
| 12 | 527 |

Conclusion drawn at the time: latency is not monotonic, onnxruntime's default
setting is 8 times slower than the optimum, and the operating point at 4 or 6
threads matches one 4-core cluster of the Snapdragon, whose 36 MB of L2 is
split into 3 blocks of 12 MB.

The explanation was coherent, checkable against the CPU datasheet, and entirely
wrong.

**The control that demolished it.** Five independent passes, each in a fresh
process:

| Threads | Passes | Median |
|---|---|---|
| 4 | 40.92 / 40.60 / 40.67 / 40.57 / 41.29 | 40.8 ms |
| 12 | 24.64 / 25.05 / 26.00 / 26.38 / 27.40 | 25.9 ms |

12 threads is **faster**, and both settings are stable to 2 and 11 %. The
opposite of the previous conclusion.

**Cause.** `onnxruntime` does not release its thread pools when a session is
destroyed. Chaining measurements in one process makes threads pile up: at the
seventh sweep point, some forty surviving threads were fighting over 12 cores.
**The harness was measuring its own leak, not the model.**

Fix applied in `src/benchmark.py`: every measurement point now runs in a fresh
process, via `measure_isolated()`.

What to keep from this, beyond this project: a plausible, documented
explanation is not a proof. The reproducibility check under independent
conditions is not a formality, it is what separates a measurement from an
impression.

## Results, after the fix

Clean sweep, one process per point, 60 runs each:

| Threads | Median (ms) | p90 (ms) | FPS |
|---|---|---|---|
| 1 | 330.92 | 420.10 | 3.0 |
| 2 | 142.28 | 145.96 | 7.0 |
| 4 | 40.52 | 82.77 | 24.7 |
| 6 | 31.48 | 32.24 | 31.8 |
| 8 | 25.89 | 26.34 | 38.6 |
| **10** | **23.93** | 24.63 | **41.8** |
| 12 | 24.42 | 26.13 | 41.0 |

Monotonic curve, saturation from 10 threads, p90 hugging the median on every
point above 4. That is the signature of a healthy measurement.

Going from 2 to 4 threads gains a factor 3.5 for a doubling of resources.
Superlinear, probably the effect of the 4-core cluster and its shared L2. This
time the hypothesis stays a hypothesis: it is not needed to conclude, and it
was not tested.

**Operating point retained: 10 threads.** Every comparative measurement in the
project uses this setting.

## fp32 against int8 dynamic

10 threads, consolidated over 4 independent passes:

| Model | Size | Median | FPS |
|---|---|---|---|
| fp32 | 10.11 MB | 25.29 ms | 39.5 |
| int8 dynamic | 2.85 MB | 209.72 ms | 4.8 |

**Dynamic quantization divides size by 3.55 and speed by 8.3.**

A first isolated measurement gave 2016 ms, an 84x factor. It was polluted and
did not survive the reproducibility check. Details in `03_quantization.md`.

This is not a marginal underperformance, it is a trap. Dynamic quantization
inserts `DynamicQuantizeLinear` operators and moves convolutions onto integer
kernels that have no optimized ARM64 implementation: execution falls back to a
slow reference path, plus the cost of requantizing activations at every
inference.

It is designed for MatMul and LSTM, not for a Conv-dominated network like
YOLO. Kept in the repository as a measured control, because "I tested it and
here is by how much it misses" beats "I did not try".

## Protocol retained

1. One fresh process per measurement point. Never two models in the same one.
2. Thread count set explicitly. Here 10.
3. 10 warm-up runs before the 60 to 100 measured runs.
4. Median and p90 reported together. A p90 far above the median flags a
   polluted measurement, not an irregular model.
5. Machine on mains power, heavy applications closed, load recorded.
6. Any counter-intuitive conclusion is replayed under independent conditions
   before it is written down.

Superseded runs produced by the single-process harness are kept in
`results/superseded/` for the record. None of their numbers is cited.
