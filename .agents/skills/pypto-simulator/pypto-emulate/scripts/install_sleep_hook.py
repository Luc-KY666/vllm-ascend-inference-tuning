#!/usr/bin/env python3
"""Install the PyPTO a2a3sim sleep-kernel replacement hook."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_HEADER_SIZE = 17568  # pinned simpler ChipCallable wire ABI
_COMMON = Path(__file__).resolve().parents[2] / "pypto-common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
_INSTALLED = False
_LOCK = threading.Lock()
_STUB_CACHE: dict[str, bytes] = {}
_MEMORY_PHASE_LOCK = threading.RLock()
_MEMORY_PHASE: str | None = None
_MEMORY_PHASE_BYTES = 0
_EXACT_MEMORY_BASELINES: dict[tuple[int, int], int] = {}
_GRAPH_TIMING_LOCK = threading.Lock()


class SleepHookError(RuntimeError):
    pass


def _strict() -> bool:
    return os.environ.get("PYPTO_SLEEP_STRICT", "1").lower() not in {"0", "false", "off"}


def _sim_aicpu_threads() -> int | None:
    raw = os.environ.get("PYPTO_SLEEP_AICPU_THREADS", "").strip()
    if not raw or raw == "0":
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise SleepHookError("PYPTO_SLEEP_AICPU_THREADS must be an integer") from error
    if value < 2 or value > 16:
        raise SleepHookError("PYPTO_SLEEP_AICPU_THREADS must be 0 or in [2, 16]")
    return value


def _load_plan() -> dict[str, Any]:
    path = os.environ.get("PYPTO_SLEEP_PLAN")
    if not path:
        raise SleepHookError("PYPTO_SLEEP_PLAN is required")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("execution_mode") != "sleep" or data.get("backend") != "a2a3sim":
        raise SleepHookError("simulation plan is not an a2a3sim sleep plan")
    if _strict() and not data.get("coverage", {}).get("runtime_stub_ready", False):
        raise SleepHookError("simulation plan is not runtime-stub-ready")
    return data


def _load_memory_profile() -> dict[str, Any] | None:
    path = os.environ.get("PYPTO_SLEEP_MEMORY_PROFILE", "").strip()
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), list):
        raise SleepHookError("unsupported memory profile schema")
    allow_non_strict = os.environ.get("PYPTO_SLEEP_MEMORY_ALLOW_NON_STRICT", "0").lower() in {
        "1", "true", "on"
    }
    if not data.get("strict_ready", False) and not allow_non_strict:
        raise SleepHookError("memory profile is not strict-ready")
    return data


class _MemoryProfile:
    def __init__(self, manifest: dict[str, Any]):
        self._models = {str(model["operation"]): model for model in manifest["models"]}
        footprint = manifest.get("footprint_model")
        self.reserved_bytes = 0 if footprint is None else int(footprint["reserved_bytes"])
        self.native_total_bytes = (
            None if footprint is None or "native_total_bytes" not in footprint
            else int(footprint["native_total_bytes"])
        )
        self.driver_visible_non_kv_bytes = (
            None if footprint is None or "driver_visible_non_kv_bytes" not in footprint
            else int(footprint["driver_visible_non_kv_bytes"])
        )
        if self.reserved_bytes < 0:
            raise SleepHookError("memory profile has negative runtime reserved bytes")
        if (self.native_total_bytes is None) != (self.driver_visible_non_kv_bytes is None):
            raise SleepHookError("memory profile has an incomplete exact footprint model")
        if self.native_total_bytes is not None and not (
            0 <= self.driver_visible_non_kv_bytes <= self.native_total_bytes
        ):
            raise SleepHookError("memory profile has an invalid exact footprint model")

    def estimate_us(self, operation: str, byte_count: int) -> float:
        model = self._models.get(operation)
        if model is None:
            raise SleepHookError(f"memory profile has no model for {operation}")
        points = sorted(
            (int(point["bytes"]), float(point["duration_us"])) for point in model.get("points", [])
        )
        if not points:
            raise SleepHookError(f"memory profile has no points for {operation}")
        for measured_bytes, duration_us in points:
            if byte_count == measured_bytes:
                return duration_us
        if (
            len(points) == 1
            and model.get("non_strict_reason") == "byte count comes from a rounded serving log"
            and round(byte_count / 1_000_000_000, 2) == points[0][0] / 1_000_000_000
        ):
            return points[0][1]
        if byte_count < points[0][0] or byte_count > points[-1][0]:
            raise SleepHookError(
                f"memory operation {operation} with {byte_count} bytes is outside measured domain "
                f"[{points[0][0]}, {points[-1][0]}]"
            )
        for (lower_bytes, lower_us), (upper_bytes, upper_us) in itertools.pairwise(points):
            if lower_bytes < byte_count < upper_bytes:
                ratio = (byte_count - lower_bytes) / (upper_bytes - lower_bytes)
                return lower_us + ratio * (upper_us - lower_us)
        raise SleepHookError(f"cannot interpolate memory operation {operation} at {byte_count} bytes")


def _compensate_phase_duration(operation: str, target_us: float, started_at: float) -> None:
    elapsed_us = (time.perf_counter() - started_at) * 1_000_000
    remaining_us = target_us - elapsed_us
    if remaining_us < 0:
        raise SleepHookError(
            f"virtual {operation} backend cost {elapsed_us:.3f} us exceeds modeled {target_us:.3f} us"
        )
    if remaining_us:
        time.sleep(remaining_us / 1_000_000)


def _dtype_itemsize(dtype: Any) -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is None:
        raise SleepHookError(f"cannot determine dtype size for {dtype!r}")
    return int(itemsize)


def _shape_nbytes(shape: Any, dtype: Any) -> int:
    elements = math.prod(int(dim) for dim in shape)
    if elements < 0:
        raise SleepHookError(f"invalid tensor shape {shape!r}")
    return elements * _dtype_itemsize(dtype)


def _install_virtual_model_memory(memory_profile: _MemoryProfile, model: dict[str, Any] | None) -> None:
    from model_adapter import load_runner_class
    from pypto.runtime.runtime_base import Worker

    model_runner = load_runner_class(model)

    original_alloc_tensor = Worker.alloc_tensor
    original_materialize_static = model_runner._materialize_static_tensors
    original_alloc_kv_tensor = model_runner._alloc_kv_cache_tensor
    original_alloc_kv_with_retry = model_runner._alloc_kv_cache_with_retry
    try:
        threshold = int(os.environ.get("PYPTO_SLEEP_VIRTUAL_MEMORY_MIN_BYTES", str(1024 * 1024)))
    except ValueError as error:
        raise SleepHookError("PYPTO_SLEEP_VIRTUAL_MEMORY_MIN_BYTES must be an integer") from error
    if threshold <= 0:
        raise SleepHookError("PYPTO_SLEEP_VIRTUAL_MEMORY_MIN_BYTES must be positive")

    def alloc_tensor(self, shape, dtype, *, init=None, worker_id=0):
        global _MEMORY_PHASE_BYTES
        with _MEMORY_PHASE_LOCK:
            init_bytes = 0 if init is None else init.numel() * init.element_size()
            if _MEMORY_PHASE is not None:
                _MEMORY_PHASE_BYTES += init_bytes
            virtualize = _MEMORY_PHASE is not None and init is not None and init_bytes >= threshold
        return original_alloc_tensor(
            self,
            shape,
            dtype,
            init=None if virtualize else init,
            worker_id=worker_id,
        )

    def materialize_static(self):
        global _MEMORY_PHASE, _MEMORY_PHASE_BYTES
        model = memory_profile._models.get("weight_materialize")
        if model is None:
            raise SleepHookError("memory profile has no weight_materialize model")
        points = list(model.get("points", []))
        if len(points) != 1:
            raise SleepHookError("weight_materialize requires one exact aggregate byte point")
        started_at = time.perf_counter()
        with _MEMORY_PHASE_LOCK:
            if _MEMORY_PHASE is not None:
                raise SleepHookError(f"nested virtual memory phase: {_MEMORY_PHASE}")
            _MEMORY_PHASE = "weight_materialize"
            _MEMORY_PHASE_BYTES = 0
        try:
            result = original_materialize_static(self)
        finally:
            with _MEMORY_PHASE_LOCK:
                byte_count = _MEMORY_PHASE_BYTES
                _MEMORY_PHASE = None
                _MEMORY_PHASE_BYTES = 0
        target_us = memory_profile.estimate_us("weight_materialize", byte_count)
        _compensate_phase_duration("weight_materialize", target_us, started_at)
        return result

    def alloc_kv_tensor(self, shape, dtype):
        with _MEMORY_PHASE_LOCK:
            virtualize = _MEMORY_PHASE == "kv_materialize" and _shape_nbytes(shape, dtype) >= threshold
        if not virtualize:
            return original_alloc_kv_tensor(self, shape, dtype)
        return self._shared_l3_worker().alloc_tensor(shape, dtype)

    def alloc_kv_with_retry(self, model_id, config, runtime, num_pages):
        global _MEMORY_PHASE
        started_at = time.perf_counter()
        with _MEMORY_PHASE_LOCK:
            if _MEMORY_PHASE is not None:
                raise SleepHookError(f"nested virtual memory phase: {_MEMORY_PHASE}")
            _MEMORY_PHASE = "kv_materialize"
        try:
            pages = original_alloc_kv_with_retry(self, model_id, config, runtime, num_pages)
        finally:
            with _MEMORY_PHASE_LOCK:
                _MEMORY_PHASE = None
        byte_count = (
            pages
            * config.num_hidden_layers
            * 2
            * config.num_key_value_heads
            * runtime.page_size
            * config.head_dim
            * _dtype_itemsize(getattr(__import__("torch"), runtime.kv_dtype))
        )
        target_us = memory_profile.estimate_us("kv_materialize", byte_count)
        _compensate_phase_duration("kv_materialize", target_us, started_at)
        return pages

    Worker.alloc_tensor = alloc_tensor
    model_runner._materialize_static_tensors = materialize_static
    model_runner._alloc_kv_cache_tensor = alloc_kv_tensor
    model_runner._alloc_kv_cache_with_retry = alloc_kv_with_retry


def _install_graph_timing(path: Path) -> None:
    from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin

    original_run_l3 = L3DispatchMixin._run_l3
    records: list[dict[str, Any]] = []

    def flush_records() -> None:
        with _GRAPH_TIMING_LOCK:
            if not records:
                return
            lines = "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            )
            records.clear()
        with path.open("a", encoding="ascii") as stream:
            stream.write(lines)

    def run_l3(self, callable_spec, *args, config=None):
        started_at = time.perf_counter_ns()
        try:
            return original_run_l3(self, callable_spec, *args, config=config)
        finally:
            record = {
                "event": "sleep_graph_wall",
                "program": str(callable_spec.name),
                "wall_us": (time.perf_counter_ns() - started_at) / 1000.0,
            }
            with _GRAPH_TIMING_LOCK:
                records.append(record)

    atexit.register(flush_records)
    L3DispatchMixin._run_l3 = run_l3


def _program_models(plan: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    programs: dict[str, dict[int, dict[str, Any]]] = {}
    for entry in plan.get("kernel_models", []):
        program = str(entry.get("program", "default"))
        func_id = int(entry["func_id"])
        duration = float(entry["duration_us"])
        if not math.isfinite(duration) or duration < 0:
            raise SleepHookError(f"invalid duration for {program}:{func_id}")
        existing = programs.setdefault(program, {}).get(func_id)
        if existing is not None:
            same_duration = math.isclose(float(existing["duration_us"]), duration, rel_tol=0.01, abs_tol=0.5)
            same_controls = existing.get("control_outputs", []) == entry.get("control_outputs", [])
            if not same_duration or not same_controls:
                raise SleepHookError(
                    f"func_id collision for {program}:{func_id}; task-site models cannot share one stub"
                )
            continue
        programs[program][func_id] = entry
    return programs


def _program_from_path(path: Path, programs: dict[str, Any]) -> str:
    for parent in (path, *path.parents):
        sidecar = parent / "dispatch_program.json"
        if not sidecar.is_file():
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in ("program", "program_name", "name", "dispatch_program"):
            value = str(data.get(key, ""))
            if value in programs:
                return value
    for part in reversed(path.parts):
        if part in programs:
            return part
    if len(programs) == 1:
        return next(iter(programs))
    if "default" in programs:
        return "default"
    raise SleepHookError(f"cannot map artifact {path} to one of programs {sorted(programs)}")


def _control_source(outputs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for output in outputs:
        if output.get("strategy") != "constant_token" or "arg_index" not in output:
            continue
        index = int(output["arg_index"])
        token = int(output["token_id"])
        padding = int(output.get("padding_value", 0))
        lines.extend(
            [
                f"  if (args && args[{index}]) {{",
                f"    auto* tensor = reinterpret_cast<unsigned char*>(args[{index}]);",
                "    const auto address = load_u64(tensor + 0);",
                "    const auto start = load_u64(tensor + 24);",
                "    const auto rows = load_u32(tensor + 44);",
                "    const auto cols = load_u32(tensor + 48);",
                "    const auto stride0 = load_u32(tensor + 72);",
                "    const auto stride1 = load_u32(tensor + 76);",
                "    auto* base = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(address));",
                "    if (base && rows > 0 && cols > 0) {",
                "      for (uint32_t row = 0; row < rows; ++row) {",
                f"        for (uint32_t col = 0; col < cols; ++col) base[start + row * stride0 + col * stride1] = {padding};",
                f"        base[start + row * stride0] = {token};",
                "      }",
                "    }",
                "  }",
            ]
        )
    return "\n".join(lines)


def _telemetry_source(entry: dict[str, Any] | None) -> str:
    unmodeled = entry is None or bool(entry.get("_unmodeled", False))
    record = {
        "event": "unmodeled_stub_execution" if unmodeled else "sleep_stub_first_execution",
        "program": "unmapped" if entry is None else str(entry.get("program", "default")),
        "func_id": -1 if entry is None else int(entry.get("func_id", -1)),
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    return f'''static void record_first_execution() {{
  static std::atomic<bool> recorded{{false}};
  if (recorded.load(std::memory_order_relaxed)) return;
  bool expected = false;
  if (!recorded.compare_exchange_strong(expected, true, std::memory_order_relaxed)) return;
  const char* path = std::getenv("PYPTO_SLEEP_TELEMETRY");
  if (!path || !*path) return;
  const int fd = ::open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0) return;
  static constexpr char line[] = {json.dumps(line)};
  (void)::write(fd, line, sizeof(line) - 1);
  (void)::close(fd);
}}'''


def _stub_source(entry: dict[str, Any] | None) -> str:
    unmodeled = entry is None or bool(entry.get("_unmodeled", False))
    duration_us = 0.0 if unmodeled else float(entry["duration_us"])
    controls = [] if unmodeled else list(entry.get("control_outputs", []))
    duration_ns = max(0, round(duration_us * 1000.0))
    delay_source = ""
    delay = ""
    if duration_ns > 0:
        delay_source = r'''
static inline void cpu_relax() {
#if defined(__aarch64__)
  __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

using RuntimeDelayHook = void (*)(uint64_t);
static RuntimeDelayHook runtime_delay_hook = nullptr;

extern "C" __attribute__((visibility("default"))) void pypto_sim_register_delay_hook(void* hook) {
  runtime_delay_hook = reinterpret_cast<RuntimeDelayHook>(hook);
}

static void fallback_delay_for_ns(uint64_t duration_ns) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(duration_ns);
  constexpr auto sleep_margin = std::chrono::microseconds(100);
  if (duration_ns > 200000ULL) std::this_thread::sleep_until(deadline - sleep_margin);
  while (std::chrono::steady_clock::now() < deadline) cpu_relax();
}

static void delay_for_ns(uint64_t duration_ns) {
  if (runtime_delay_hook != nullptr) {
    runtime_delay_hook(duration_ns);
    return;
  }
  fallback_delay_for_ns(duration_ns);
}
'''
        delay = f"  delay_for_ns({duration_ns}ULL);\n"
    body = (
        "  record_first_execution();\n  std::abort();"
        if unmodeled
        else f'''  record_first_execution();
{delay}{_control_source(controls)}'''
    )
    return f'''#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <thread>
#include <unistd.h>

static uint64_t load_u64(const unsigned char* p) {{ uint64_t v; std::memcpy(&v, p, sizeof(v)); return v; }}
static uint32_t load_u32(const unsigned char* p) {{ uint32_t v; std::memcpy(&v, p, sizeof(v)); return v; }}
{delay_source}
{_telemetry_source(entry)}

extern "C" __attribute__((visibility("default"))) const char* simpler_cpu_sim_target() {{ return "a2a3sim"; }}
extern "C" __attribute__((visibility("default"))) void kernel_entry(int64_t* args) {{
{body}
}}
'''


def _compile_stub(entry: dict[str, Any] | None) -> bytes:
    source = _stub_source(entry)
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()
    with _LOCK:
        cached = _STUB_CACHE.get(key)
        if cached is not None:
            return cached
        cache_root = Path(os.environ.get("PYPTO_SLEEP_STUB_CACHE", tempfile.gettempdir())) / "pypto-sleep-stubs"
        cache_root.mkdir(parents=True, exist_ok=True)
        output = cache_root / f"{key}.so"
        if not output.is_file():
            compiler = os.environ.get("CXX") or shutil.which("g++")
            if not compiler:
                raise SleepHookError("g++ is required to build sleep stubs")
            source_path = cache_root / f"{key}.cpp"
            source_path.write_text(source, encoding="ascii")
            result = subprocess.run(
                [compiler, "-shared", "-fPIC", "-O2", "-std=c++17", "-pthread", str(source_path), "-o", str(output)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise SleepHookError(f"sleep stub compilation failed: {result.stderr.strip()}")
        binary = output.read_bytes()
        _STUB_CACHE[key] = binary
        return binary


def _callable_parent_binary(callable_obj: Any) -> bytes:
    size = int(callable_obj.binary_size)
    return ctypes.string_at(int(callable_obj.buffer_ptr()) + _HEADER_SIZE, size)


def _replace_callable(callable_obj: Any, program: str, models: dict[str, dict[int, dict[str, Any]]]) -> Any:
    from simpler.task_interface import ChipCallable, CoreCallable

    per_func = models.get(program, {})
    children = []
    for index in range(int(callable_obj.child_count)):
        func_id = int(callable_obj.child_func_id(index))
        child = callable_obj.child(index)
        signature = [child.sig(i) for i in range(int(child.sig_count))]
        entry = per_func.get(func_id)
        if entry is None:
            entry = {"program": program, "func_id": func_id, "_unmodeled": True}
        children.append((func_id, CoreCallable.build(signature=signature, binary=_compile_stub(entry))))
    return ChipCallable.build(
        signature=[callable_obj.sig(i) for i in range(int(callable_obj.sig_count))],
        func_name=callable_obj.func_name,
        binary=_callable_parent_binary(callable_obj),
        children=children,
        config_name=callable_obj.config_name,
    )


def _simulator_memory_info(
    worker: Any,
    worker_id: int = 0,
    runtime_reserved_bytes: int = 0,
    *,
    native_total_bytes: int | None = None,
    driver_visible_non_kv_bytes: int | None = None,
) -> tuple[int, int]:
    default_total = 65_450_000_000
    configured = os.environ.get("PYPTO_SLEEP_DEVICE_MEMORY_BYTES")
    if native_total_bytes is not None:
        if driver_visible_non_kv_bytes is None:
            raise SleepHookError("exact native total requires driver-visible footprint bytes")
        if configured is not None:
            try:
                configured_total = int(configured)
            except ValueError as error:
                raise SleepHookError(
                    "PYPTO_SLEEP_DEVICE_MEMORY_BYTES must be an integer"
                ) from error
            if configured_total != native_total_bytes:
                raise SleepHookError(
                    "PYPTO_SLEEP_DEVICE_MEMORY_BYTES cannot override an exact footprint model"
                )
        total = native_total_bytes
        committed = int(worker.committed_device_memory(worker_id))
        baseline_key = (id(worker), worker_id)
        with _MEMORY_PHASE_LOCK:
            baseline = _EXACT_MEMORY_BASELINES.setdefault(baseline_key, committed)
        if baseline < 0 or baseline > driver_visible_non_kv_bytes:
            raise SleepHookError(
                f"simulator non-KV baseline {baseline} exceeds the modeled "
                f"driver-visible footprint {driver_visible_non_kv_bytes}"
            )
        allocation_delta = committed - baseline
        if allocation_delta < 0:
            raise SleepHookError(
                f"simulator committed memory fell below its exact-footprint baseline: "
                f"{committed} < {baseline}"
            )
        modeled_used = driver_visible_non_kv_bytes + allocation_delta
        if modeled_used > total:
            raise SleepHookError(
                f"simulator logical memory {modeled_used} exceeds the modeled total {total}"
            )
        return total - modeled_used, total
    try:
        total = int(configured or default_total)
    except ValueError as error:
        raise SleepHookError("PYPTO_SLEEP_DEVICE_MEMORY_BYTES must be an integer") from error
    if total <= 0:
        raise SleepHookError("PYPTO_SLEEP_DEVICE_MEMORY_BYTES must be positive")
    committed = int(worker.committed_device_memory(worker_id)) + runtime_reserved_bytes
    if committed < 0 or committed > total:
        raise SleepHookError(
            f"simulator committed memory {committed} is outside the configured {total}-byte budget"
        )
    return total - committed, total


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    plan = _load_plan()
    models = _program_models(plan)
    memory_manifest = _load_memory_profile()
    memory_profile = None if memory_manifest is None else _MemoryProfile(memory_manifest)

    from simpler_setup import runtime_builder

    from pypto.runtime import device_runner, distributed_runner

    diagnostic_log_level = os.environ.get("PYPTO_SLEEP_SIMPLER_LOG_LEVEL", "").strip()
    if diagnostic_log_level:
        from simpler_setup.log_config import configure_logging

        configure_logging(diagnostic_log_level)

    runtime_lib_dir = os.environ.get("PYPTO_SLEEP_RUNTIME_LIB_DIR", "").strip()
    sim_aicpu_threads = _sim_aicpu_threads()
    if sim_aicpu_threads is not None and sim_aicpu_threads > 4 and not runtime_lib_dir:
        raise SleepHookError("more than 4 simulator AICPU threads require PYPTO_SLEEP_RUNTIME_LIB_DIR")
    if runtime_lib_dir:
        custom_lib = Path(runtime_lib_dir).resolve()
        expected = custom_lib / "a2a3" / "sim" / "tensormap_and_ringbuffer" / "libhost_runtime.so"
        if not expected.is_file():
            raise SleepHookError(f"custom simulator runtime is incomplete: {expected}")
        runtime_builder.RuntimeBuilder._LIB_DIR = custom_lib

    original_compile_single = device_runner._compile_single_kernel
    original_assemble = device_runner._compile_and_assemble
    original_memory_info = distributed_runner.DistributedWorker.device_memory_info
    original_make_call_config = distributed_runner._make_call_config

    def compile_single(kernel, compiler, platform, pto_isa_root=None, runtime_name="tensormap_and_ringbuffer", **kwargs):
        if platform != "a2a3sim":
            return original_compile_single(
                kernel, compiler, platform, pto_isa_root, runtime_name, **kwargs
            )
        program = _program_from_path(Path(kernel["source"]).resolve(), models)
        func_id = int(kernel.get("func_id", -1))
        entry = models.get(program, {}).get(func_id)
        if entry is None:
            entry = {"program": program, "func_id": func_id, "_unmodeled": True}
        binary = _compile_stub(entry)
        return binary, binary

    def assemble(work_dir, platform, *, save_prebuilt=False):
        callable_obj, runtime_name, runtime_config = original_assemble(
            work_dir, platform, save_prebuilt=save_prebuilt
        )
        if platform != "a2a3sim":
            return callable_obj, runtime_name, runtime_config
        program = _program_from_path(Path(work_dir).resolve(), models)
        return _replace_callable(callable_obj, program, models), runtime_name, runtime_config

    def device_memory_info(worker, worker_id=0):
        if not str(worker._compiled.platform).endswith("sim"):
            return original_memory_info(worker, worker_id)
        return _simulator_memory_info(
            worker,
            worker_id,
            0 if memory_profile is None else memory_profile.reserved_bytes,
            native_total_bytes=(
                None if memory_profile is None else memory_profile.native_total_bytes
            ),
            driver_visible_non_kv_bytes=(
                None
                if memory_profile is None
                else memory_profile.driver_visible_non_kv_bytes
            ),
        )

    def make_call_config(*args, **kwargs):
        config = original_make_call_config(*args, **kwargs)
        if sim_aicpu_threads is not None:
            config.aicpu_thread_num = sim_aicpu_threads
        return config

    device_runner._compile_single_kernel = compile_single
    device_runner._compile_and_assemble = assemble
    distributed_runner._make_call_config = make_call_config
    distributed_runner.DistributedWorker.device_memory_info = device_memory_info
    if memory_profile is not None:
        _install_virtual_model_memory(memory_profile, plan.get("model"))
    graph_timing_path = os.environ.get("PYPTO_SLEEP_GRAPH_TIMING", "").strip()
    if graph_timing_path:
        _install_graph_timing(Path(graph_timing_path).resolve())
    _INSTALLED = True


if __name__ == "__main__":
    install()
