#!/usr/bin/env python3
"""Install the isolated PyPTO a2a3sim timewarp hook."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SLEEP_HOOK = ROOT / "pypto-emulate" / "scripts" / "install_sleep_hook.py"
TIMEKEEPER = Path(__file__).with_name("timekeeper.py")
_INSTALLED = False


class TimewarpHookError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise TimewarpHookError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_plan() -> dict[str, Any]:
    raw = os.environ.get("PYPTO_TIMEWARP_PLAN")
    if not raw:
        raise TimewarpHookError("PYPTO_TIMEWARP_PLAN is required")
    plan = json.loads(Path(raw).read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise TimewarpHookError("timewarp plan must use schema_version 1")
    if plan.get("execution_mode") != "timewarp" or plan.get("backend") != "a2a3sim":
        raise TimewarpHookError("plan is not an a2a3sim timewarp plan")
    if not plan.get("coverage", {}).get("runtime_stub_ready", False):
        raise TimewarpHookError("timewarp plan is not runtime-stub-ready")
    graph_models = plan.get("timewarp", {}).get("graph_models")
    if not isinstance(graph_models, list) or not graph_models:
        raise TimewarpHookError("timewarp plan has no graph models")
    programs = set()
    for model in graph_models:
        program = str(model.get("program", ""))
        duration = model.get("duration_us")
        if (
            not program
            or program in programs
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            raise TimewarpHookError(f"invalid graph model: {model!r}")
        programs.add(program)
    return plan


def _no_wait_stub_source(sleep_hook: ModuleType, entry: dict[str, Any] | None) -> str:
    unmodeled = entry is None or bool(entry.get("_unmodeled", False))
    controls = [] if unmodeled else list(entry.get("control_outputs", []))
    telemetry = sleep_hook._telemetry_source(entry).replace(
        "sleep_stub_first_execution", "timewarp_stub_first_execution"
    )
    body = (
        "  record_first_execution();\n  std::abort();"
        if unmodeled
        else f"  record_first_execution();\n{sleep_hook._control_source(controls)}"
    )
    return f'''#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

static uint64_t load_u64(const unsigned char* p) {{ uint64_t v; std::memcpy(&v, p, sizeof(v)); return v; }}
static uint32_t load_u32(const unsigned char* p) {{ uint32_t v; std::memcpy(&v, p, sizeof(v)); return v; }}
{telemetry}

extern "C" __attribute__((visibility("default"))) const char* simpler_cpu_sim_target() {{ return "a2a3sim"; }}
extern "C" __attribute__((visibility("default"))) void kernel_entry(int64_t* args) {{
{body}
}}
'''


def _map_shared_environment() -> None:
    mappings = {
        "PYPTO_TIMEWARP_MEMORY_PROFILE": "PYPTO_SLEEP_MEMORY_PROFILE",
        "PYPTO_TIMEWARP_MEMORY_ALLOW_NON_STRICT": "PYPTO_SLEEP_MEMORY_ALLOW_NON_STRICT",
        "PYPTO_TIMEWARP_TELEMETRY": "PYPTO_SLEEP_TELEMETRY",
        "PYPTO_TIMEWARP_DEVICE_MEMORY_BYTES": "PYPTO_SLEEP_DEVICE_MEMORY_BYTES",
        "PYPTO_TIMEWARP_RUNTIME_LIB_DIR": "PYPTO_SLEEP_RUNTIME_LIB_DIR",
        "PYPTO_TIMEWARP_SIMPLER_LOG_LEVEL": "PYPTO_SLEEP_SIMPLER_LOG_LEVEL",
        "PYPTO_TIMEWARP_AICPU_THREADS": "PYPTO_SLEEP_AICPU_THREADS",
    }
    for source, target in mappings.items():
        if source in os.environ:
            os.environ[target] = os.environ[source]
        else:
            os.environ.pop(target, None)
    os.environ["PYPTO_SLEEP_STRICT"] = os.environ.get("PYPTO_TIMEWARP_STRICT", "1")
    os.environ.pop("PYPTO_SLEEP_GRAPH_TIMING", None)


def _install_serving_bridge(plan: dict[str, Any], timekeeper: ModuleType) -> None:
    from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin
    from pypto_serving.serving.engine.async_engine import ReplicaEngineCore
    from pypto_serving.serving.server.ipc import decode_result
    from pypto_serving.serving.server.serving_worker import WorkerProcess

    graph_durations_ns = {
        str(model["program"]): round(float(model["duration_us"]) * 1000.0)
        for model in plan["timewarp"]["graph_models"]
    }
    client = timekeeper.TimewarpClient()

    def report_graph(program: str) -> None:
        try:
            duration_ns = graph_durations_ns[program]
        except KeyError as error:
            raise TimewarpHookError(f"no timewarp graph model for {program}") from error
        client.graph(program, duration_ns)

    original_run_l3 = L3DispatchMixin._run_l3
    original_submit_l3 = L3DispatchMixin._submit_l3

    def run_l3(self, callable_spec, *args, config=None):
        report_graph(str(callable_spec.name))
        return original_run_l3(self, callable_spec, *args, config=config)

    def submit_l3(self, callable_spec, *args, config=None):
        report_graph(str(callable_spec.name))
        return original_submit_l3(self, callable_spec, *args, config=config)

    L3DispatchMixin._run_l3 = run_l3
    L3DispatchMixin._submit_l3 = submit_l3

    original_worker_init = WorkerProcess.__init__
    original_lifecycle = WorkerProcess._apply_command_lifecycle
    original_reclaim_decode = WorkerProcess._reclaim_pending_decode
    original_reclaim_prefill = WorkerProcess._reclaim_pending_prefill

    class OutputQueueProxy:
        def __init__(self, queue: Any, replica: int) -> None:
            self._queue = queue
            self._replica = replica

        def put(self, value: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(value, bytes):
                result = decode_result(value)
                key = client.step_key(self._replica, int(result.step_id))
                client.seal_step(key)
            return self._queue.put(value, *args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._queue, name)

    def worker_init(self, *args, **kwargs):
        original_worker_init(self, *args, **kwargs)
        replica = int(getattr(self.config, "dp_rank", 0))
        self.output_queue = OutputQueueProxy(self.output_queue, replica)

    def lifecycle(self, cmd, *args, **kwargs):
        replica = int(getattr(self.config, "dp_rank", 0))
        key = client.step_key(replica, int(cmd.step_id))
        client.begin_step(key)
        return original_lifecycle(self, cmd, *args, **kwargs)

    def reclaim_with_context(original, self, work):
        replica = int(getattr(self.config, "dp_rank", 0))
        key = client.step_key(replica, int(work.cmd.step_id))
        previous = client.current_step()
        client.set_current_step(key)
        try:
            return original(self, work)
        finally:
            client.set_current_step(previous)

    def reclaim_decode(self, work):
        return reclaim_with_context(original_reclaim_decode, self, work)

    def reclaim_prefill(self, work):
        return reclaim_with_context(original_reclaim_prefill, self, work)

    WorkerProcess.__init__ = worker_init
    WorkerProcess._apply_command_lifecycle = lifecycle
    WorkerProcess._reclaim_pending_decode = reclaim_decode
    WorkerProcess._reclaim_pending_prefill = reclaim_prefill

    original_get_live_result = ReplicaEngineCore._get_live_result
    original_publish_outputs = ReplicaEngineCore._publish_outputs
    original_iterate_output = ReplicaEngineCore._iterate_request_output

    class RequestQueueProxy:
        def __init__(self, queue: Any) -> None:
            self._queue = queue

        @staticmethod
        def _expects_ack(value: Any) -> bool:
            return hasattr(value, "finished") and not bool(value.finished)

        def put_nowait(self, value: Any) -> None:
            if self._expects_ack(value):
                client.output_expected()
            self._queue.put_nowait(value)

        async def put(self, value: Any) -> None:
            if self._expects_ack(value):
                client.output_expected()
            await self._queue.put(value)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._queue, name)

    def publish_outputs(self, request_outputs, *, release_worker=True):
        for context in self._request_contexts.values():
            if not isinstance(context.queue, RequestQueueProxy):
                context.queue = RequestQueueProxy(context.queue)
        return original_publish_outputs(
            self, request_outputs, release_worker=release_worker
        )

    async def iterate_output(self, context):
        async for output in original_iterate_output(self, context):
            try:
                yield output
            finally:
                if not output.finished:
                    await asyncio.to_thread(client.output_delivered)

    async def get_live_result(self):
        raw = await original_get_live_result(self)
        result = decode_result(raw)
        replica = int(getattr(self.config, "dp_rank", 0))
        key = client.step_key(replica, int(result.step_id))
        await asyncio.to_thread(client.wait_step, key)
        return raw

    ReplicaEngineCore._publish_outputs = publish_outputs
    ReplicaEngineCore._iterate_request_output = iterate_output
    ReplicaEngineCore._get_live_result = get_live_result


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    plan = _load_plan()
    if not os.environ.get("PYPTO_TIMEWARP_SOCKET") or not os.environ.get("PYPTO_TIMEWARP_CLOCK"):
        raise TimewarpHookError("PYPTO_TIMEWARP_SOCKET and PYPTO_TIMEWARP_CLOCK are required")
    _map_shared_environment()
    sleep_hook = _load_module(SLEEP_HOOK, "pypto_timewarp_sleep_base")
    timekeeper = _load_module(TIMEKEEPER, "pypto_timewarp_clock")
    client = timekeeper.TimewarpClient()

    def load_timewarp_as_sleep_plan() -> dict[str, Any]:
        translated = dict(plan)
        translated["execution_mode"] = "sleep"
        return translated

    def compensate_phase_duration(operation: str, target_us: float, started_at: float) -> None:
        elapsed_us = (time.perf_counter() - started_at) * 1_000_000
        remaining_us = target_us - elapsed_us
        if remaining_us < 0:
            raise TimewarpHookError(
                f"virtual {operation} backend cost {elapsed_us:.3f} us exceeds modeled {target_us:.3f} us"
            )
        client.startup_duration(operation, round(remaining_us * 1000.0))

    sleep_hook._load_plan = load_timewarp_as_sleep_plan
    sleep_hook._stub_source = lambda entry: _no_wait_stub_source(sleep_hook, entry)
    sleep_hook._compensate_phase_duration = compensate_phase_duration
    sleep_hook.install()
    _install_serving_bridge(plan, timekeeper)
    _INSTALLED = True


if __name__ == "__main__":
    install()
