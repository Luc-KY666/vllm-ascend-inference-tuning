#!/usr/bin/env python3
"""Install strict a2a3 CoreCallable replacement with device-side delay kernels."""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_HEADER_SIZE = 17568
_COMMON = Path(__file__).resolve().parents[2] / "pypto-common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
_SYS_CNT_TICKS_PER_US = 50
_INSTALLED = False
_LOCK = threading.Lock()
_MEMORY_LOCK = threading.RLock()
_MEMORY_PHASE: str | None = None
_MEMORY_PHASE_BYTES = 0
_EXACT_MEMORY_BASELINES: dict[tuple[int, int], int] = {}
_VMEM_MAGIC = 0x5059544F564D454D
_VMEM_VERSION = 1
_VMEM_IDLE = 0
_VMEM_PENDING = 1
_VMEM_CONSUMED = 2
_VMEM_FAILED = 3
_VMEM_RECORD = struct.Struct("<QQQQQQQqqq32s")


class NativeDelayHookError(RuntimeError):
    pass


class _MemoryProfile:
    def __init__(self, manifest: dict[str, Any]):
        if manifest.get("schema_version") != 1 or manifest.get("strict_ready") is not True:
            raise NativeDelayHookError("native delay requires a strict-ready schema-v1 memory profile")
        self._models = {str(model["operation"]): model for model in manifest.get("models", [])}
        footprint = manifest.get("footprint_model")
        if not isinstance(footprint, dict) or footprint.get("strict_ready") is not True:
            raise NativeDelayHookError("memory profile has no strict exact footprint model")
        self.native_total_bytes = int(footprint["native_total_bytes"])
        self.driver_visible_non_kv_bytes = int(footprint["driver_visible_non_kv_bytes"])
        self.weight_bytes = self._exact_bytes("weight_materialize")
        self.kv_bytes = self._exact_bytes("kv_materialize")

    def _exact_bytes(self, operation: str) -> int:
        model = self._models.get(operation)
        if model is None or model.get("strict_ready") is not True:
            raise NativeDelayHookError(f"memory profile has no strict model for {operation}")
        points = list(model.get("points", []))
        if len(points) != 1:
            raise NativeDelayHookError(f"{operation} requires one exact byte point")
        byte_count = int(points[0]["bytes"])
        if byte_count <= 0:
            raise NativeDelayHookError(f"{operation} has an invalid byte point")
        return byte_count

    def duration_us(self, operation: str, byte_count: int) -> float:
        expected = self._exact_bytes(operation)
        if byte_count != expected:
            raise NativeDelayHookError(
                f"memory operation {operation} has {byte_count} bytes; expected exact {expected}"
            )
        duration = float(self._models[operation]["points"][0]["duration_us"])
        if not math.isfinite(duration) or duration < 0:
            raise NativeDelayHookError(f"memory operation {operation} has invalid duration")
        return duration


class _VirtualMemoryController:
    def __init__(self, control_path: Path, audit_path: Path):
        self._control_path = control_path
        self._audit_path = audit_path
        self._request_id = 0
        self._lock = threading.Lock()

    @staticmethod
    def _decode(raw: bytes) -> dict[str, Any]:
        if len(raw) != _VMEM_RECORD.size:
            raise NativeDelayHookError("virtual-memory control record has the wrong size")
        values = _VMEM_RECORD.unpack(raw)
        operation = values[10].split(b"\0", 1)[0].decode("ascii")
        return {
            "magic": values[0],
            "version": values[1],
            "state": values[2],
            "request_id": values[3],
            "expected_bytes": values[4],
            "virtual_address": values[5],
            "reserved_bytes": values[6],
            "consumer_pid": values[7],
            "forbidden_pid": values[8],
            "error_code": values[9],
            "operation": operation,
        }

    @staticmethod
    def _encode(record: dict[str, Any]) -> bytes:
        operation = str(record["operation"]).encode("ascii")
        if len(operation) >= 32:
            raise NativeDelayHookError("virtual-memory operation name is too long")
        return _VMEM_RECORD.pack(
            _VMEM_MAGIC,
            _VMEM_VERSION,
            int(record["state"]),
            int(record["request_id"]),
            int(record["expected_bytes"]),
            int(record.get("virtual_address", 0)),
            int(record.get("reserved_bytes", 0)),
            int(record.get("consumer_pid", 0)),
            int(record["forbidden_pid"]),
            int(record.get("error_code", 0)),
            operation.ljust(32, b"\0"),
        )

    def _locked_record(self, stream: Any) -> dict[str, Any]:
        stream.seek(0)
        return self._decode(stream.read(_VMEM_RECORD.size))

    def allocate(self, operation: str, byte_count: int, callback: Any) -> Any:
        if byte_count <= 0:
            raise NativeDelayHookError("virtual allocation byte count must be positive")
        with self._lock:
            with self._control_path.open("r+b", buffering=0) as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                current = self._locked_record(stream)
                if current["magic"] != _VMEM_MAGIC or current["version"] != _VMEM_VERSION:
                    raise NativeDelayHookError("virtual-memory control ABI mismatch")
                if current["state"] == _VMEM_PENDING:
                    raise NativeDelayHookError("a virtual allocation token is already pending")
                self._request_id += 1
                pending = {
                    "state": _VMEM_PENDING,
                    "request_id": self._request_id,
                    "expected_bytes": byte_count,
                    "forbidden_pid": os.getpid(),
                    "operation": operation,
                }
                stream.seek(0)
                stream.write(self._encode(pending))
                stream.flush()
                os.fsync(stream.fileno())
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

            result: Any = None
            callback_error: BaseException | None = None
            try:
                result = callback()
            except BaseException as error:  # noqa: BLE001
                callback_error = error

            with self._control_path.open("r+b", buffering=0) as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                observed = self._locked_record(stream)
                final_state = observed["state"]
                idle = dict(observed)
                idle["state"] = _VMEM_IDLE
                stream.seek(0)
                stream.write(self._encode(idle))
                stream.flush()
                os.fsync(stream.fileno())
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

            if observed["request_id"] != self._request_id:
                raise NativeDelayHookError("virtual allocation consumed the wrong request ID")
            if observed["operation"] != operation or observed["expected_bytes"] != byte_count:
                raise NativeDelayHookError("virtual allocation consumed the wrong operation or extent")
            if final_state not in {_VMEM_CONSUMED, _VMEM_FAILED}:
                raise NativeDelayHookError(
                    f"virtual allocation token was not consumed for {operation}:{byte_count}"
                ) from callback_error
            if observed["consumer_pid"] <= 0 or observed["consumer_pid"] == os.getpid():
                raise NativeDelayHookError("virtual allocation was not consumed by a chip worker")
            if final_state == _VMEM_FAILED or observed["error_code"] != 0:
                raise NativeDelayHookError(
                    f"virtual allocation failed for {operation}:{byte_count}: "
                    f"error={observed['error_code']} consumer_pid={observed['consumer_pid']}"
                ) from callback_error
            if callback_error is not None:
                raise NativeDelayHookError(
                    f"virtual allocation callback failed after token consumption for {operation}:{byte_count}"
                ) from callback_error
            if int(observed["virtual_address"]) <= 0 or int(observed["reserved_bytes"]) < byte_count:
                raise NativeDelayHookError("virtual allocation returned an invalid reserved range")
            audit = {
                "schema_version": 1,
                "event": "native_delay_virtual_allocation",
                "operation": operation,
                "logical_bytes": byte_count,
                "virtual_address": int(observed["virtual_address"]),
                "reserved_address_bytes": int(observed["reserved_bytes"]),
                "consumer_pid": int(observed["consumer_pid"]),
                "physical_backing_bytes": 0,
            }
            line = json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n"
            with self._audit_path.open("a", encoding="ascii") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(line)
                stream.flush()
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return result


def _load_memory_profile() -> _MemoryProfile:
    raw = os.environ.get("PYPTO_NATIVE_DELAY_MEMORY_PROFILE", "").strip()
    if not raw:
        raise NativeDelayHookError("PYPTO_NATIVE_DELAY_MEMORY_PROFILE is required")
    data = json.loads(Path(raw).read_text(encoding="utf-8"))
    return _MemoryProfile(data)


def _compensate_memory_duration(operation: str, target_us: float, started_at: float) -> None:
    elapsed_us = (time.perf_counter() - started_at) * 1_000_000
    remaining_us = target_us - elapsed_us
    if remaining_us < 0:
        raise NativeDelayHookError(
            f"virtual {operation} backend cost {elapsed_us:.3f} us exceeds modeled {target_us:.3f} us"
        )
    if remaining_us:
        time.sleep(remaining_us / 1_000_000)


def _dtype_itemsize(dtype: Any) -> int:
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is None:
        raise NativeDelayHookError(f"cannot determine dtype size for {dtype!r}")
    return int(itemsize)


def _shape_nbytes(shape: Any, dtype: Any) -> int:
    elements = math.prod(int(dim) for dim in shape)
    if elements <= 0:
        raise NativeDelayHookError(f"invalid tensor shape {shape!r}")
    return elements * _dtype_itemsize(dtype)


def _exact_memory_info(
    profile: _MemoryProfile,
    *,
    actual_free: int,
    actual_total: int,
    committed: int,
    baseline: int,
) -> tuple[int, int]:
    if actual_total != profile.native_total_bytes:
        raise NativeDelayHookError(
            f"physical NPU total {actual_total} does not match exact profile "
            f"{profile.native_total_bytes}"
        )
    if not 0 <= actual_free <= actual_total:
        raise NativeDelayHookError("physical NPU returned invalid free/total memory")
    if baseline < 0 or baseline > profile.driver_visible_non_kv_bytes:
        raise NativeDelayHookError(
            f"native-delay non-KV baseline {baseline} exceeds exact driver-visible "
            f"footprint {profile.driver_visible_non_kv_bytes}"
        )
    allocation_delta = committed - baseline
    if allocation_delta < 0:
        raise NativeDelayHookError(
            f"native-delay committed memory fell below baseline: {committed} < {baseline}"
        )
    modeled_used = profile.driver_visible_non_kv_bytes + allocation_delta
    physical_used = actual_total - actual_free
    logical_used = max(modeled_used, physical_used)
    if logical_used > profile.native_total_bytes:
        raise NativeDelayHookError(
            f"native-delay logical memory {logical_used} exceeds exact total "
            f"{profile.native_total_bytes}"
        )
    return profile.native_total_bytes - logical_used, profile.native_total_bytes


def _install_virtual_model_memory(profile: _MemoryProfile, model: dict[str, Any] | None) -> None:
    from model_adapter import load_runner_class
    from pypto.runtime.runtime_base import Worker

    from pypto.runtime import distributed_runner

    model_runner = load_runner_class(model)

    control_raw = os.environ.get("PYPTO_NATIVE_DELAY_VMEM_CONTROL", "").strip()
    audit_raw = os.environ.get("PYPTO_NATIVE_DELAY_MEMORY_AUDIT", "").strip()
    if not control_raw or not audit_raw:
        raise NativeDelayHookError("native virtual memory requires control and audit paths")
    controller = _VirtualMemoryController(Path(control_raw), Path(audit_raw))
    original_alloc_tensor = Worker.alloc_tensor
    original_memory_info = distributed_runner.DistributedWorker.device_memory_info
    original_materialize_static = model_runner._materialize_static_tensors
    original_alloc_kv_with_retry = model_runner._alloc_kv_cache_with_retry

    def device_memory_info(worker, worker_id=0):
        actual_free, actual_total = original_memory_info(worker, worker_id)
        committed = int(worker.committed_device_memory(worker_id))
        baseline_key = (id(worker), int(worker_id))
        with _MEMORY_LOCK:
            baseline = _EXACT_MEMORY_BASELINES.setdefault(baseline_key, committed)
        return _exact_memory_info(
            profile,
            actual_free=actual_free,
            actual_total=actual_total,
            committed=committed,
            baseline=baseline,
        )

    def alloc_tensor(self, shape, dtype, *, init=None, worker_id=0):
        global _MEMORY_PHASE_BYTES
        with _MEMORY_LOCK:
            phase = _MEMORY_PHASE
        if phase is None:
            return original_alloc_tensor(self, shape, dtype, init=init, worker_id=worker_id)
        byte_count = _shape_nbytes(shape, dtype)
        if init is not None and int(init.numel()) * int(init.element_size()) != byte_count:
            raise NativeDelayHookError(f"{phase} init bytes do not match logical tensor bytes")
        result = controller.allocate(
            phase,
            byte_count,
            lambda: original_alloc_tensor(self, shape, dtype, init=None, worker_id=worker_id),
        )
        with _MEMORY_LOCK:
            _MEMORY_PHASE_BYTES += byte_count
        return result

    def materialize_static(self):
        global _MEMORY_PHASE, _MEMORY_PHASE_BYTES
        started_at = time.perf_counter()
        with _MEMORY_LOCK:
            if _MEMORY_PHASE is not None:
                raise NativeDelayHookError(f"nested virtual memory phase: {_MEMORY_PHASE}")
            _MEMORY_PHASE = "weight_materialize"
            _MEMORY_PHASE_BYTES = 0
        try:
            result = original_materialize_static(self)
        finally:
            with _MEMORY_LOCK:
                byte_count = _MEMORY_PHASE_BYTES
                _MEMORY_PHASE = None
                _MEMORY_PHASE_BYTES = 0
        target_us = profile.duration_us("weight_materialize", byte_count)
        _compensate_memory_duration("weight_materialize", target_us, started_at)
        return result

    def alloc_kv_tensor(self, shape, dtype):
        global _MEMORY_PHASE, _MEMORY_PHASE_BYTES
        with _MEMORY_LOCK:
            outer_phase = _MEMORY_PHASE
            if outer_phase is None:
                _MEMORY_PHASE = "kv_scratch"
                _MEMORY_PHASE_BYTES = 0
        try:
            return self._shared_l3_worker().alloc_tensor(shape, dtype)
        finally:
            if outer_phase is None:
                with _MEMORY_LOCK:
                    _MEMORY_PHASE = None
                    _MEMORY_PHASE_BYTES = 0

    def alloc_kv_with_retry(self, model_id, config, runtime, num_pages):
        global _MEMORY_PHASE, _MEMORY_PHASE_BYTES
        started_at = time.perf_counter()
        with _MEMORY_LOCK:
            if _MEMORY_PHASE is not None:
                raise NativeDelayHookError(f"nested virtual memory phase: {_MEMORY_PHASE}")
            _MEMORY_PHASE = "kv_materialize"
            _MEMORY_PHASE_BYTES = 0
        try:
            pages = original_alloc_kv_with_retry(self, model_id, config, runtime, num_pages)
        finally:
            with _MEMORY_LOCK:
                byte_count = _MEMORY_PHASE_BYTES
                _MEMORY_PHASE = None
                _MEMORY_PHASE_BYTES = 0
        target_us = profile.duration_us("kv_materialize", byte_count)
        _compensate_memory_duration("kv_materialize", target_us, started_at)
        return pages

    Worker.alloc_tensor = alloc_tensor
    distributed_runner.DistributedWorker.device_memory_info = device_memory_info
    model_runner._materialize_static_tensors = materialize_static
    model_runner._alloc_kv_cache_tensor = alloc_kv_tensor
    model_runner._alloc_kv_cache_with_retry = alloc_kv_with_retry


def _load_plan() -> tuple[dict[str, Any], str]:
    raw = os.environ.get("PYPTO_NATIVE_DELAY_PLAN", "").strip()
    if not raw:
        raise NativeDelayHookError("PYPTO_NATIVE_DELAY_PLAN is required")
    path = Path(raw).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("execution_mode") != "sleep" or data.get("backend") != "a2a3":
        raise NativeDelayHookError("native delay requires an a2a3 sleep plan")
    if not data.get("coverage", {}).get("runtime_stub_ready", False):
        raise NativeDelayHookError("native delay plan is not runtime-stub-ready")
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


def _program_models(plan: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    programs: dict[str, dict[int, dict[str, Any]]] = {}
    for entry in plan.get("kernel_models", []):
        program = str(entry.get("program", "default"))
        func_id = int(entry["func_id"])
        duration = float(entry["duration_us"])
        if not math.isfinite(duration) or duration < 0:
            raise NativeDelayHookError(f"invalid duration for {program}:{func_id}")
        existing = programs.setdefault(program, {}).get(func_id)
        if existing is not None:
            same_duration = math.isclose(
                float(existing["duration_us"]), duration, rel_tol=0.01, abs_tol=0.5
            )
            same_controls = existing.get("control_outputs", []) == entry.get(
                "control_outputs", []
            )
            if not same_duration or not same_controls:
                raise NativeDelayHookError(
                    f"func_id collision for {program}:{func_id}; one child needs one model"
                )
            continue
        programs[program][func_id] = entry
    if not programs:
        raise NativeDelayHookError("native delay plan contains no kernel models")
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
    raise NativeDelayHookError(f"cannot map artifact {path} to {sorted(programs)}")


def _load_kernel_config(work_dir: Path) -> list[dict[str, Any]]:
    path = work_dir / "kernel_config.py"
    spec = importlib.util.spec_from_file_location(
        f"_native_delay_kernel_config_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}",
        path,
    )
    if spec is None or spec.loader is None:
        raise NativeDelayHookError(f"cannot load kernel config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.KERNELS)


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
                f"    if (args != nullptr && args[{index}] != 0) {{",
                f"        __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[{index}]);",
                "        __gm__ int32_t *base = reinterpret_cast<__gm__ int32_t *>(tensor->buffer.addr);",
                "        if (base != nullptr && tensor->ndims >= 1 && tensor->shapes[0] > 0) {",
                "            const uint32_t rows = tensor->shapes[0];",
                "            const uint32_t cols = tensor->ndims > 1 ? tensor->shapes[1] : 1;",
                "            const uint32_t stride0 = tensor->strides[0];",
                "            const uint32_t stride1 = tensor->ndims > 1 ? tensor->strides[1] : 1;",
                "            for (uint32_t row = 0; row < rows; ++row) {",
                "                for (uint32_t col = 0; col < cols; ++col) {",
                f"                    base[tensor->start_offset + row * stride0 + col * stride1] = {padding};",
                "                }",
                f"                base[tensor->start_offset + row * stride0] = {token};",
                "            }",
                "            dcci(base, SINGLE_CACHE_LINE, CACHELINE_OUT);",
                "        }",
                "    }",
            ]
        )
    return "\n".join(lines)


def _block_delay_source(entry: dict[str, Any], default_ticks: int) -> str:
    raw = entry.get("block_duration_us_by_subslot", {})
    vectors = {
        str(slot): [
            max(0, round(float(duration) * _SYS_CNT_TICKS_PER_US))
            for duration in durations
        ]
        for slot, durations in raw.items()
        if durations
    }
    if not vectors:
        return f"""
    const uint64_t started_at = get_sys_cnt();
    while (get_sys_cnt() - started_at < {default_ticks}ULL) {{
        __asm__ volatile("" ::: "memory");
    }}
"""

    def branches(ticks: list[int], indent: str, low: int = 0) -> list[str]:
        if len(ticks) == 1:
            return [
                f"{indent}if (block_idx == {low}) target_ticks = {ticks[0]}ULL;"
            ]
        middle = len(ticks) // 2
        split = low + middle
        return [
            f"{indent}if (block_idx < {split}) {{",
            *branches(ticks[:middle], indent + "    ", low),
            f"{indent}}} else {{",
            *branches(ticks[middle:], indent + "    ", split),
            f"{indent}}}",
        ]

    lines = [
        f"    uint64_t target_ticks = {default_ticks}ULL;",
        "    const int32_t block_idx = get_block_idx(args);",
    ]
    if set(vectors) == {"AIV0", "AIV1"}:
        lines.extend(
            [
                "    if (get_sub_block_id(args) == 0) {",
                *branches(vectors["AIV0"], "        "),
                "    } else {",
                *branches(vectors["AIV1"], "        "),
                "    }",
            ]
        )
    elif len(vectors) == 1:
        lines.extend(branches(next(iter(vectors.values())), "    "))
    else:
        return _block_delay_source(
            {"block_duration_us_by_subslot": {}}, default_ticks
        )
    lines.extend(
        [
            "    const uint64_t started_at = get_sys_cnt();",
            "    while (get_sys_cnt() - started_at < target_ticks) {",
            '        __asm__ volatile("" ::: "memory");',
            "    }",
        ]
    )
    return "\n".join(lines) + "\n"


def _stub_source(entry: dict[str, Any], *, zero_duration: bool) -> str:
    duration_us = 0.0 if zero_duration else float(entry["duration_us"])
    ticks = max(0, round(duration_us * _SYS_CNT_TICKS_PER_US))
    delay = _block_delay_source(entry, ticks) if ticks else ""
    controls = _control_source(list(entry.get("control_outputs", [])))
    body = f"{delay}{controls}"
    return f'''#include <cstdint>
#include <pto/pto-inst.hpp>
#include "intrinsic.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {{
{body}
    pipe_barrier(PIPE_ALL);
}}
'''


def _compile_stub(
    entry: dict[str, Any],
    core_type: str,
    compiler: Any,
    pto_isa_root: str,
    runtime_name: str,
    *,
    zero_duration: bool,
) -> bytes:
    from pypto.runtime.elf_parser import extract_text_section

    source = _stub_source(entry, zero_duration=zero_duration)
    identity = hashlib.sha256(
        (source + "\0" + core_type + "\0" + runtime_name).encode("utf-8")
    ).hexdigest()
    root = Path(
        os.environ.get(
            "PYPTO_NATIVE_DELAY_STUB_CACHE",
            str(Path(tempfile.gettempdir()) / "pypto-native-delay-stubs"),
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / f"{identity}.cpp"
    binary_path = root / f"{identity}.bin"
    with _LOCK:
        if binary_path.is_file():
            return binary_path.read_bytes()
        source_path.write_text(source, encoding="ascii")
    raw = compiler.compile_incore(
        str(source_path),
        core_type=core_type,
        pto_isa_root=pto_isa_root,
        runtime_name=runtime_name,
        build_dir=str(root),
    )
    binary = extract_text_section(raw)
    with _LOCK:
        if not binary_path.is_file():
            temporary = binary_path.with_name(
                f"{binary_path.name}.{threading.get_ident()}.tmp"
            )
            temporary.write_bytes(binary)
            os.replace(temporary, binary_path)
    return binary


def _install_graph_timing(path: Path) -> None:
    from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin

    original_run_l3 = L3DispatchMixin._run_l3
    records: list[dict[str, Any]] = []

    def flush_records() -> None:
        with _LOCK:
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
            records.append(
                {
                    "event": "native_delay_graph_wall",
                    "program": str(callable_spec.name),
                    "wall_us": (time.perf_counter_ns() - started_at) / 1000.0,
                }
            )

    atexit.register(flush_records)
    L3DispatchMixin._run_l3 = run_l3


def _parent_binary(callable_obj: Any) -> bytes:
    size = int(callable_obj.binary_size)
    return ctypes.string_at(int(callable_obj.buffer_ptr()) + _HEADER_SIZE, size)


def _record_replacement(
    path: Path,
    program: str,
    plan_sha256: str,
    replacements: list[dict[str, Any]],
    omitted_func_ids: list[int],
    zero: bool,
) -> None:
    record = {
        "schema_version": 1,
        "event": "native_delay_callable_replaced",
        "program": program,
        "plan_sha256": plan_sha256,
        "zero_duration": zero,
        "replacements": replacements,
        "unregistered_unmodeled_func_ids": omitted_func_ids,
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with _LOCK, path.open("a", encoding="ascii") as stream:
        stream.write(line)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("PYPTO_NATIVE_DELAY_ENABLE", "") != "1":
        raise NativeDelayHookError("PYPTO_NATIVE_DELAY_ENABLE=1 is required")
    plan, plan_sha256 = _load_plan()
    memory_profile = _load_memory_profile()
    models = _program_models(plan)
    audit_raw = os.environ.get("PYPTO_NATIVE_DELAY_AUDIT", "").strip()
    if not audit_raw:
        raise NativeDelayHookError("PYPTO_NATIVE_DELAY_AUDIT is required")
    audit_path = Path(audit_raw).resolve()
    zero_duration = os.environ.get("PYPTO_NATIVE_DELAY_ZERO", "0") == "1"

    from pypto.runtime.kernel_compiler import KernelCompiler
    from pypto.runtime.pto_isa import ensure_pto_isa_root
    from pypto.runtime.task_interface import ChipCallable, CoreCallable

    from pypto.runtime import device_runner

    original_assemble = device_runner._compile_and_assemble

    def assemble(work_dir, platform, *, save_prebuilt=False):
        callable_obj, runtime_name, runtime_config = original_assemble(
            work_dir, platform, save_prebuilt=save_prebuilt
        )
        if platform != "a2a3":
            return callable_obj, runtime_name, runtime_config
        program = _program_from_path(Path(work_dir).resolve(), models)
        per_func = models.get(program, {})
        kernels = _load_kernel_config(Path(work_dir))
        core_types: dict[int, str] = {}
        for kernel in kernels:
            func_id = int(kernel["func_id"])
            core_type = str(kernel["core_type"])
            previous = core_types.setdefault(func_id, core_type)
            if previous != core_type:
                raise NativeDelayHookError(
                    f"{program}:{func_id} maps to multiple core types: {previous}, {core_type}"
                )
        child_ids = [int(callable_obj.child_func_id(i)) for i in range(int(callable_obj.child_count))]
        unexpected = sorted(set(per_func) - set(child_ids))
        if unexpected:
            raise NativeDelayHookError(
                f"strict callable mismatch for {program}: plan_only={unexpected}"
            )
        omitted_func_ids = sorted(set(child_ids) - set(per_func))
        modeled_indices = [
            index for index, func_id in enumerate(child_ids) if func_id in per_func
        ]
        compiler = KernelCompiler(platform="a2a3")
        pto_isa_root = ensure_pto_isa_root()

        def compile_child(index: int) -> tuple[int, Any, dict[str, Any]]:
            func_id = child_ids[index]
            child = callable_obj.child(index)
            signature = [child.sig(i) for i in range(int(child.sig_count))]
            core_type = core_types.get(func_id)
            if core_type not in {"aic", "aiv"}:
                raise NativeDelayHookError(f"missing core type for {program}:{func_id}")
            entry = per_func[func_id]
            binary = _compile_stub(
                entry,
                core_type,
                compiler,
                pto_isa_root,
                runtime_name,
                zero_duration=zero_duration,
            )
            replacement = CoreCallable.build(signature=signature, binary=binary)
            audit = {
                "func_id": func_id,
                "core_type": core_type,
                "duration_us": 0.0 if zero_duration else float(entry["duration_us"]),
                "block_duration_counts_by_subslot": (
                    {}
                    if zero_duration
                    else {
                        str(slot): len(values)
                        for slot, values in entry.get(
                            "block_duration_us_by_subslot", {}
                        ).items()
                    }
                ),
                "modeled": True,
                "binary_sha256": hashlib.sha256(binary).hexdigest(),
            }
            return func_id, replacement, audit

        with ThreadPoolExecutor(max_workers=min(16, len(modeled_indices) or 1)) as executor:
            compiled = list(executor.map(compile_child, modeled_indices))
        replacement_callable = ChipCallable.build(
            signature=[callable_obj.sig(i) for i in range(int(callable_obj.sig_count))],
            func_name=callable_obj.func_name,
            binary=_parent_binary(callable_obj),
            children=[(func_id, child) for func_id, child, _ in compiled],
            config_name=callable_obj.config_name,
        )
        _record_replacement(
            audit_path,
            program,
            plan_sha256,
            [audit for _, _, audit in compiled],
            omitted_func_ids,
            zero_duration,
        )
        return replacement_callable, runtime_name, runtime_config

    device_runner._compile_and_assemble = assemble
    graph_timing_raw = os.environ.get("PYPTO_NATIVE_DELAY_GRAPH_TIMING", "").strip()
    if graph_timing_raw:
        _install_graph_timing(Path(graph_timing_raw).resolve())
    _install_virtual_model_memory(memory_profile, plan.get("model"))
    _INSTALLED = True


if __name__ == "__main__":
    install()
