#!/usr/bin/env python3
"""Compare zero-work PyPTO graph latency on a2a3sim and native a2a3."""

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

from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    TaskArgs,
)
from simpler.worker import Worker
from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.runtime_builder import RuntimeBuilder

SOURCE_DIR = Path(__file__).resolve().parents[1] / "native_delay"
CASES = {
    "chain2_aiv": "overhead_chain",
    "spmd8_aiv": "overhead_spmd8",
    "mix_aic_aiv0_aiv1": "overhead_mix",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2a3sim", "a2a3"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runtime", choices=("tensormap_and_ringbuffer",), default="tensormap_and_ringbuffer")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument(
        "--runtime-lib-dir",
        type=Path,
        help="prebuilt runtime lib root; required when the source checkout has no build/lib",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0 or args.warmup < 1 or args.rounds < 3:
        parser.error("--device must be non-negative, --warmup >= 1, and --rounds >= 3")
    return args


def _require_free_native_device(device: int) -> None:
    result = subprocess.run(
        ["npu-smi", "info", "-t", "proc-mem", "-i", str(device)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot query NPU {device}: {result.stderr.strip() or result.stdout.strip()}")
    if "No process in device" not in result.stdout:
        raise RuntimeError(f"NPU {device} is occupied; refusing contaminated native benchmark")


def _compile_sim_noop() -> bytes:
    source = """
#include <cstdint>
extern "C" void kernel_entry(int64_t *args) { (void)args; }
extern "C" __attribute__((visibility("default")))
const char *simpler_cpu_sim_target() { return "a2a3sim"; }
"""
    with tempfile.TemporaryDirectory(prefix="pypto-overhead-noop-") as temporary:
        root = Path(temporary)
        source_path = root / "noop.cpp"
        output_path = root / "noop.so"
        source_path.write_text(source, encoding="ascii")
        result = subprocess.run(
            ["g++", "-shared", "-fPIC", "-O2", "-std=c++17", str(source_path), "-o", str(output_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"fallback sim no-op compilation failed: {result.stderr.strip()}")
        return output_path.read_bytes()


def _incore_binary(compiler: KernelCompiler, core_type: str, runtime: str, pto_isa_root: str) -> bytes:
    if compiler.platform.endswith("sim"):
        return _compile_sim_noop()
    binary = compiler.compile_incore(
        source_path=str(SOURCE_DIR / "noop_kernel.cpp"),
        core_type=core_type,
        pto_isa_root=pto_isa_root,
        extra_include_dirs=compiler.get_orchestration_include_dirs(runtime),
    )
    return extract_text_section(binary)


def _build_callables(platform: str, runtime: str) -> dict[str, ChipCallable]:
    compiler = KernelCompiler(platform=platform)
    pto_isa_root = ensure_pto_isa_root()
    aic = _incore_binary(compiler, "aic", runtime, pto_isa_root)
    aiv = _incore_binary(compiler, "aiv", runtime, pto_isa_root)
    orchestration = compiler.compile_orchestration(
        runtime_name=runtime,
        source_path=str(SOURCE_DIR / "overhead_orchestration.cpp"),
    )
    children = [
        (0, CoreCallable.build(signature=[ArgDirection.OUT], binary=aiv)),
        (1, CoreCallable.build(signature=[ArgDirection.IN], binary=aiv)),
        (2, CoreCallable.build(signature=[], binary=aiv)),
        (3, CoreCallable.build(signature=[], binary=aic)),
        (4, CoreCallable.build(signature=[], binary=aiv)),
        (5, CoreCallable.build(signature=[], binary=aiv)),
    ]
    return {
        name: ChipCallable.build(
            signature=[],
            func_name=function_name,
            binary=orchestration,
            children=children,
            config_name="overhead_benchmark_config",
        )
        for name, function_name in CASES.items()
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": values,
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    if args.platform == "a2a3":
        _require_free_native_device(args.device)
    if args.runtime_lib_dir is not None:
        runtime_lib_dir = args.runtime_lib_dir.resolve()
        if not runtime_lib_dir.is_dir():
            raise SystemExit(f"runtime lib directory does not exist: {runtime_lib_dir}")
        RuntimeBuilder._LIB_DIR = runtime_lib_dir

    callables = _build_callables(args.platform, args.runtime)
    worker = Worker(
        level=2,
        platform=args.platform,
        runtime=args.runtime,
        device_id=args.device,
    )
    handles = {name: worker.register(callable_obj) for name, callable_obj in callables.items()}
    worker.init()
    try:
        config = CallConfig()
        config.aicpu_thread_num = 4
        task_args = TaskArgs()
        results: dict[str, Any] = {}
        for name, handle in handles.items():
            for _ in range(args.warmup):
                worker.run(handle, task_args, config)
            samples = []
            for _ in range(args.rounds):
                started_at = time.perf_counter_ns()
                worker.run(handle, task_args, config)
                samples.append((time.perf_counter_ns() - started_at) / 1_000_000.0)
            results[name] = _summary(samples)
    finally:
        worker.close()

    report = {
        "schema_version": 1,
        "benchmark": "pypto-zero-work-control-plane",
        "platform": args.platform,
        "device": args.device,
        "runtime": args.runtime,
        "runtime_lib_dir": (
            None if args.runtime_lib_dir is None else str(args.runtime_lib_dir.resolve())
        ),
        "warmup": args.warmup,
        "rounds": args.rounds,
        "host": host_platform.node(),
        "kernel_compiler": (
            "system-g++-fallback-for-zero-work-sim-child"
            if args.platform == "a2a3sim"
            else "pinned-ccec"
        ),
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"output": str(output), "platform": args.platform, "cases": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
