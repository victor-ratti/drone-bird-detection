# Step 3, int8 quantization: a negative result, and a clear one

## The result

10 intra-op threads, 60 runs per measurement, 4 independent passes per model,
each in a fresh process. Idle machine. Data in `benchmark_repeatability.json`.

| Model | Size | Passes (ms) | Median | Dispersion | FPS |
|---|---|---|---|---|---|
| **fp32** | 10.11 MB | 24.0 / 24.9 / 25.8 / 26.6 | **25.29 ms** | x1.11 | **39.5** |
| int8 static | 3.03 MB | 37.6 / 38.7 / 40.7 / 42.4 | 39.83 ms | x1.13 | 25.1 |
| int8 dynamic | 2.85 MB | 207.1 / 210.5 / 212.9 / 208.4 | 209.72 ms | x1.03 | 4.8 |

**On this CPU, with this runtime, int8 quantization does not speed anything
up. It slows down.** Static is 1.57 times slower than fp32, dynamic 8.3 times.

The size gain is real: 10.11 MB down to 3.03 MB, a factor 3.34.

## Why

Quantization only makes a model faster if the runtime has integer kernels
optimized for the target architecture. That is not the case here.

- The **fp32** path in onnxruntime goes through MLAS, with NEON kernels written
  and tuned for ARM64. That is fast code.
- The **int8** path of the default CPU executor has no equally mature
  equivalent on ARM64. Quantized convolutions fall back to generic
  implementations, and the cost of quantizing and dequantizing tensors is
  added at every layer.
- **Dynamic** is the worst of both: it requantizes activations at every
  inference, and it targets MatMul and LSTM, not convolutions.

In other words, the limiting factor is not the model, it is the **execution
backend**.

## What it implies for an embedded deployment

This is the point to remember, and it carries over as is to a drone's compute
board.

1. **Quantizing before knowing what the target runtime can execute is wasted
   time.** The choice of backend precedes the choice of weight format.
2. **Size gain and speed gain are two distinct problems.** Here size was
   divided by 3.34 while throughput dropped 37 %. If flash memory is the
   constraint, that is a good trade. If latency is, it is a regression.
3. **fp32 already holds real time** at 39.5 frames per second on an ARM CPU
   with no accelerator. On this hardware there is no speed problem to solve.

## Where the gain would actually be

Three avenues, not explored, in order of expected return:

- **The Snapdragon NPU**, through onnxruntime's QNN execution provider. That is
  precisely the hardware built for integer inference. It needs the
  `onnxruntime-qnn` package, a separate build: the providers available in the
  current install are limited to `CPUExecutionProvider` and
  `AzureExecutionProvider`.
- **The XNNPACK execution provider**, which ships optimized ARM int8 kernels.
- **An NVIDIA Jetson target**, with TensorRT in int8, where the quantization
  gain is well documented. That is the architecture most drones in the sector
  actually carry.

## Method

This result almost came out wrong twice.

- A first sweep concluded that latency was non-monotonic with an optimum at 6
  threads. Artifact: `onnxruntime` does not release its thread pools, and the
  harness was measuring its own leak. See `02_benchmark_protocol.md`.
- A first measurement of the dynamic model gave 2016 ms, an 84x factor.
  Replayed over 4 independent passes, the value is 210 ms, a factor 8.3. The
  first measurement was polluted.

In both cases the reproducibility check decided, not the most attractive
explanation.

## Still open

Measure the mAP of the static quantized model. A precision loss would add to
the speed loss and close the case. Intact precision would make the model
interesting for a memory-constrained target, not a time-constrained one. The
evaluator in `src/evaluate.py` takes any ONNX file, so this is one command
away.
