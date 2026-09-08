# -*- coding: utf-8 -*-
"""Latency benchmark for an ONNX detection model.

Measures pure inference time on CPU, with warm-up and dispersion statistics.
Latency does not depend on image content, only on input size: a random input
gives the same measurement as a photo, which keeps this metric independent of
the dataset.

Usage:
    python src/benchmark.py models/baseline_best.onnx
    python src/benchmark.py models/a.onnx models/b.onnx --threads 10 --json results/bench.json
    python src/benchmark.py models/baseline_best.onnx --sweep 1,2,4,8,12
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort


def machine_info():
    """Description of the machine. Recorded alongside every measurement."""
    info = {
        "processor": platform.processor() or platform.machine(),
        "architecture": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "logical_cores": os.cpu_count(),
        "onnxruntime": ort.__version__,
    }
    if platform.system() == "Windows":
        # platform.processor() only returns a generic identifier on ARM64.
        try:
            name = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                text=True, stderr=subprocess.DEVNULL).strip()
            if name:
                info["processor"] = name
        except Exception:
            pass
    return info


def cpu_load():
    """Instantaneous CPU load, to be recorded: background load skews results."""
    if platform.system() != "Windows":
        return None
    try:
        v = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return int(v)
    except Exception:
        return None


def measure(path, runs=100, warmup=10, threads=None):
    """Measure the inference latency of an ONNX model in this process."""
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1

    session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]

    # Dynamic dimensions (None or a string) fall back to 1 or 640.
    shape = [d if isinstance(d, int) else (1 if i == 0 else 640)
             for i, d in enumerate(inp.shape)]
    batch = np.random.rand(*shape).astype(np.float32)
    feed = {inp.name: batch}

    for _ in range(warmup):
        session.run(None, feed)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return {
        "model": os.path.basename(path),
        "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
        "input_shape": shape,
        "runs": runs,
        "threads": threads or "auto",
        "latency_ms_median": round(statistics.median(times), 2),
        "latency_ms_mean": round(statistics.fmean(times), 2),
        "latency_ms_p90": round(times[int(0.90 * len(times))], 2),
        "latency_ms_min": round(times[0], 2),
        "latency_ms_max": round(times[-1], 2),
        "fps_median": round(1000.0 / statistics.median(times), 1),
    }


def measure_isolated(path, runs, warmup, threads):
    """Measure in a fresh process.

    Required: onnxruntime does not release its thread pools when a session is
    destroyed. Chaining several measurements in one process makes threads pile
    up, and each measurement then reflects the contention left by the previous
    one, not the model. Observed on 2026-09-08: a single-process sweep produced
    a phantom optimum at 4 threads and an 8x penalty at 12 threads, both of
    which vanished once each point ran in isolation.
    """
    code = (
        "import json,sys;sys.path.insert(0,%r);"
        "from benchmark import measure;"
        "print(json.dumps(measure(%r,%d,%d,%s)))"
        % (os.path.dirname(os.path.abspath(__file__)),
           path, runs, warmup, repr(threads))
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    return json.loads(out.strip().splitlines()[-1])


def print_machine(info):
    print("Measurement machine")
    for k, v in info.items():
        print(f"  {k:14}: {v}")


def sweep(path, spec, runs, warmup, out_json=None):
    """Sweep the intra-op thread count, one fresh process per point."""
    threads = [int(x) for x in spec.split(",") if x.strip()]
    info = machine_info()
    print_machine(info)
    print(f"  {'cpu load start':14}: {cpu_load()} %")
    print()
    print(f"Sweep on {os.path.basename(path)}, {runs} runs per point")
    print()
    header = f"{'Threads':>7} {'Median':>10} {'p90':>9} {'FPS':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for t in threads:
        r = measure_isolated(path, runs, warmup, t)
        rows.append(r)
        print(f"{t:>7} {r['latency_ms_median']:>9.2f}m "
              f"{r['latency_ms_p90']:>8.2f}m {r['fps_median']:>8.1f}")

    best = min(rows, key=lambda r: r["latency_ms_median"])
    worst = max(rows, key=lambda r: r["latency_ms_median"])
    print()
    print(f"Best : {best['threads']} threads, "
          f"{best['latency_ms_median']} ms, {best['fps_median']} FPS")
    print(f"Worst: {worst['threads']} threads, {worst['latency_ms_median']} ms, "
          f"x{worst['latency_ms_median'] / best['latency_ms_median']:.1f} slower")
    end = cpu_load()
    print(f"CPU load end: {end} %")

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"machine": info, "cpu_load_end": end,
                       "model": os.path.basename(path), "sweep": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"Details written to {out_json}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help=".onnx files to compare")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None,
                    help="intra-op threads. Default: onnxruntime's choice")
    ap.add_argument("--json", help="write details to this file")
    ap.add_argument("--sweep", metavar="N,N,...",
                    help="sweep several thread counts on the first model")
    args = ap.parse_args()

    if args.sweep:
        return sweep(args.models[0], args.sweep, args.runs, args.warmup, args.json)

    info = machine_info()
    print_machine(info)
    print()

    rows = []
    for path in args.models:
        if not os.path.exists(path):
            print(f"  not found, skipped: {path}", file=sys.stderr)
            continue
        print(f"  measuring {os.path.basename(path)} ...", end="", flush=True)
        # One process per model, for the same reason as in sweep().
        rows.append(measure_isolated(path, args.runs, args.warmup, args.threads))
        print(" done")
    print()

    if not rows:
        return 1

    header = f"{'Model':<36} {'MB':>6} {'Median':>9} {'p90':>8} {'FPS':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<36} {r['size_mb']:>6} "
              f"{r['latency_ms_median']:>8.2f}m {r['latency_ms_p90']:>7.2f}m "
              f"{r['fps_median']:>7.1f}")

    if len(rows) > 1:
        ref = rows[0]
        print()
        print(f"Reference: {ref['model']}")
        for r in rows[1:]:
            speed = ref["latency_ms_median"] / r["latency_ms_median"]
            size = ref["size_mb"] / r["size_mb"]
            print(f"  {r['model']:<34} x{speed:.2f} speed, x{size:.2f} smaller")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"machine": info, "measurements": rows},
                      f, ensure_ascii=False, indent=2)
        print(f"\nDetails written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
