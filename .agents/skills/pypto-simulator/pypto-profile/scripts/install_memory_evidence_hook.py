#!/usr/bin/env python3
"""Install exact, read-only model memory evidence collection on native PyPTO."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

_INSTALLED = False
_COMMON = Path(__file__).resolve().parents[2] / "pypto-common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
_LOCK = threading.Lock()
_PHASE: str | None = None
_WEIGHT_BYTES = 0
_MEMORY_INFO: list[tuple[int, int]] = []


class MemoryEvidenceError(RuntimeError):
    pass


def _output_path() -> Path:
    raw = os.environ.get("PYPTO_MEMORY_EVIDENCE_OUTPUT", "").strip()
    if not raw:
        raise MemoryEvidenceError("PYPTO_MEMORY_EVIDENCE_OUTPUT is required")
    return Path(raw).resolve()


def _record(event: dict[str, Any]) -> None:
    record = {
        "schema_version": 1,
        "workload_id": os.environ.get("PYPTO_MEMORY_EVIDENCE_WORKLOAD_ID", "").strip(),
        **event,
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    path = _output_path()
    with _LOCK, path.open("a", encoding="ascii") as stream:
        stream.write(line)


def _require_native(worker: Any) -> None:
    platform = str(worker._compiled.platform)
    if platform != "a2a3":
        raise MemoryEvidenceError(f"memory evidence requires native a2a3, got {platform}")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _output_path()

    from model_adapter import load_runner_class
    from pypto.runtime.runtime_base import Worker

    from pypto.runtime import distributed_runner

    model_runner = load_runner_class()

    original_alloc_tensor = Worker.alloc_tensor
    original_memory_info = distributed_runner.DistributedWorker.device_memory_info
    original_materialize_static = model_runner._materialize_static_tensors
    original_compute_kv_pages = model_runner._compute_kv_cache_pages
    original_alloc_kv_with_retry = model_runner._alloc_kv_cache_with_retry

    def alloc_tensor(self, shape, dtype, *, init=None, worker_id=0):
        global _WEIGHT_BYTES
        if init is not None:
            with _LOCK:
                if _PHASE == "weight_materialize":
                    _WEIGHT_BYTES += int(init.numel()) * int(init.element_size())
        return original_alloc_tensor(self, shape, dtype, init=init, worker_id=worker_id)

    def memory_info(self, worker_id=0):
        result = original_memory_info(self, worker_id)
        with _LOCK:
            if _PHASE == "kv_sizing":
                _MEMORY_INFO.append((int(result[0]), int(result[1])))
        return result

    def materialize_static(self):
        global _PHASE, _WEIGHT_BYTES
        worker = self._shared_l3_worker()
        _require_native(worker)
        with _LOCK:
            if _PHASE is not None:
                raise MemoryEvidenceError(f"nested memory evidence phase: {_PHASE}")
            _PHASE = "weight_materialize"
            _WEIGHT_BYTES = 0
        started_at = time.perf_counter_ns()
        try:
            return original_materialize_static(self)
        finally:
            duration_us = (time.perf_counter_ns() - started_at) / 1000.0
            with _LOCK:
                byte_count = _WEIGHT_BYTES
                _WEIGHT_BYTES = 0
                _PHASE = None
            _record(
                {
                    "event": "weight_materialize",
                    "bytes": byte_count,
                    "bytes_evidence": "native_runtime_tensor_sum",
                    "duration_us": duration_us,
                }
            )

    def compute_kv_pages(config, runtime, worker, simpler_committed=0):
        global _PHASE
        _require_native(worker)
        with _LOCK:
            if _PHASE is not None:
                raise MemoryEvidenceError(f"nested memory evidence phase: {_PHASE}")
            _PHASE = "kv_sizing"
            _MEMORY_INFO.clear()
        try:
            pages = original_compute_kv_pages(
                config,
                runtime,
                worker,
                simpler_committed=simpler_committed,
            )
        finally:
            with _LOCK:
                observations = list(_MEMORY_INFO)
                _MEMORY_INFO.clear()
                _PHASE = None
        if not observations:
            raise MemoryEvidenceError("native KV sizing produced no device_memory_info observation")
        free_bytes, total_bytes = observations[-1]
        dtype = getattr(__import__("torch"), runtime.kv_dtype)
        bytes_per_page = (
            config.num_hidden_layers
            * 2
            * config.num_key_value_heads
            * runtime.page_size
            * config.head_dim
            * int(dtype.itemsize)
        )
        _record(
            {
                "event": "kv_sizing",
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "driver_visible_non_kv_bytes": total_bytes - free_bytes,
                "simpler_committed_bytes": int(simpler_committed),
                "runtime_reserved_bytes": max(total_bytes - free_bytes - int(simpler_committed), 0),
                "requested_pages": int(pages),
                "bytes_per_page": bytes_per_page,
            }
        )
        return pages

    def alloc_kv_with_retry(self, model_id, config, runtime, num_pages):
        worker = self._shared_l3_worker()
        _require_native(worker)
        started_at = time.perf_counter_ns()
        pages = original_alloc_kv_with_retry(self, model_id, config, runtime, num_pages)
        duration_us = (time.perf_counter_ns() - started_at) / 1000.0
        dtype = getattr(__import__("torch"), runtime.kv_dtype)
        bytes_per_page = (
            config.num_hidden_layers
            * 2
            * config.num_key_value_heads
            * runtime.page_size
            * config.head_dim
            * int(dtype.itemsize)
        )
        _record(
            {
                "event": "kv_materialize",
                "bytes": int(pages) * bytes_per_page,
                "bytes_evidence": "native_runtime_page_count",
                "duration_us": duration_us,
                "page_count": int(pages),
                "page_bytes": bytes_per_page,
            }
        )
        return pages

    Worker.alloc_tensor = alloc_tensor
    distributed_runner.DistributedWorker.device_memory_info = memory_info
    model_runner._materialize_static_tensors = materialize_static
    model_runner._compute_kv_cache_pages = staticmethod(compute_kv_pages)
    model_runner._alloc_kv_cache_with_retry = alloc_kv_with_retry
    _INSTALLED = True


if __name__ == "__main__":
    install()
