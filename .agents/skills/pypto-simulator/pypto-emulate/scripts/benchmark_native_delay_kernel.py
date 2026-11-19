#!/usr/bin/env python3
"""Measure native A2/A3 device-delay child fidelity with a one-task graph."""

from __future__ import annotations

import argparse
import json
import math
import platform as host_platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from simpler.task_interface import CallConfig, ChipCallable, CoreCallable, TaskArgs
from simpler.worker import Worker
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

SOURCE_DIR = Path(__file__).resolve().parents[1] / "native_delay"
TICKS_PER_US = 50
FUNC_ID = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--duration-us", type=float, action="append", dest="durations")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0 or args.warmup < 1 or args.rounds < 3:
        parser.error("--device must be non-negative, --warmup >= 1, and --rounds >= 3")
    args.durations = args.durations or [0.0, 1.0, 3.0, 10.0, 40.0, 250.0, 1000.0]
    if not args.durations or any(value < 0 or not math.isfinite(value) for value in args.durations):
        parser.error("durations must be finite and non-negative")
    if len(set(args.durations)) != len(args.durations):
        parser.error("durations must be unique")
    return args


def require_free_device(device: int) -> None:
    result = subprocess.run(
        ["npu-smi", "info", "-t", "proc-mem", "-i", str(device)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or "No process in device" not in result.stdout:
        raise RuntimeError(f"NPU {device} is occupied or unavailable")


def kernel_source(duration_us: float) -> str:
    ticks = round(duration_us * TICKS_PER_US)
    delay = ""
    if ticks:
        delay = f"""
    const uint64_t started_at = get_sys_cnt();
    while (get_sys_cnt() - started_at < {ticks}ULL) {{
        __asm__ volatile("" ::: "memory");
    }}
"""
    return f'''#include <cstdint>
#include <pto/pto-inst.hpp>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {{
    (void)args;
{delay}
    pipe_barrier(PIPE_ALL);
}}
'''


def compile_kernel(compiler: KernelCompiler, pto_isa_root: str, duration_us: float) -> bytes:
    with tempfile.TemporaryDirectory(prefix="pypto-native-delay-fidelity-") as temporary:
        root = Path(temporary)
        source = root / "delay.cpp"
        source.write_text(kernel_source(duration_us), encoding="ascii")
        raw = compiler.compile_incore(
            source_path=str(source),
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            extra_include_dirs=compiler.get_orchestration_include_dirs("tensormap_and_ringbuffer"),
            build_dir=str(root),
        )
        return extract_text_section(raw)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values),
        "samples_ms": values,
    }


def fit_line(xs: list[float], ys: list[float]) -> dict[str, float]:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return {
        "slope": slope,
        "intercept_us": intercept,
        "max_abs_residual_us": max(abs(value) for value in residuals),
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    require_free_device(args.device)

    compiler = KernelCompiler(platform="a2a3")
    pto_isa_root = ensure_pto_isa_root()
    orchestration = compiler.compile_orchestration(
        runtime_name="tensormap_and_ringbuffer",
        source_path=str(SOURCE_DIR / "overhead_orchestration.cpp"),
    )
    callables = {}
    binary_sizes = {}
    for duration_us in args.durations:
        binary = compile_kernel(compiler, pto_isa_root, duration_us)
        binary_sizes[str(duration_us)] = len(binary)
        child = CoreCallable.build(signature=[], binary=binary)
        callables[duration_us] = ChipCallable.build(
            signature=[],
            func_name="overhead_single",
            binary=orchestration,
            children=[(FUNC_ID, child)],
            config_name="overhead_benchmark_config",
        )

    worker = Worker(level=2, platform="a2a3", runtime="tensormap_and_ringbuffer", device_id=args.device)
    handles = {duration: worker.register(callable_obj) for duration, callable_obj in callables.items()}
    worker.init()
    results = {}
    try:
        config = CallConfig()
        config.aicpu_thread_num = 4
        task_args = TaskArgs()
        for duration_us in args.durations:
            handle = handles[duration_us]
            for _ in range(args.warmup):
                worker.run(handle, task_args, config)
            samples = []
            for _ in range(args.rounds):
                started_at = time.perf_counter_ns()
                worker.run(handle, task_args, config)
                samples.append((time.perf_counter_ns() - started_at) / 1_000_000.0)
            results[str(duration_us)] = summarize(samples)
    finally:
        worker.close()

    zero_key = str(0.0)
    if zero_key not in results:
        raise RuntimeError("a zero-duration point is required for differential fitting")
    zero_mean_ms = float(results[zero_key]["mean_ms"])
    xs = [value for value in args.durations if value > 0]
    ys = [(float(results[str(value)]["mean_ms"]) - zero_mean_ms) * 1000.0 for value in xs]
    report = {
        "schema_version": 1,
        "benchmark": "pypto-native-device-delay-fidelity",
        "host": host_platform.node(),
        "device": args.device,
        "runtime": "tensormap_and_ringbuffer",
        "warmup": args.warmup,
        "rounds": args.rounds,
        "ticks_per_us": TICKS_PER_US,
        "binary_sizes": binary_sizes,
        "results": results,
        "differential_observed_us": {str(x): y for x, y in zip(xs, ys)},
        "linear_fit": fit_line(xs, ys),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"output": str(output), "linear_fit": report["linear_fit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
