from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "pypto-timewarp" / "scripts"
BENCHMARK = ROOT / "pypto-emulate" / "scripts" / "run_streaming_benchmark.py"


def load(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TimewarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = load("build_timewarp_plan.py")
        cls.clock_module = load("timekeeper.py")
        cls.hook = load("install_timewarp_hook.py")
        cls.launcher = load("run_timewarp_simulator.py")
        spec = importlib.util.spec_from_file_location("test_streaming_benchmark_clock", BENCHMARK)
        assert spec is not None and spec.loader is not None
        cls.benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.benchmark)

    def test_task_dag_scheduler_models_dependencies_and_shared_resources(self) -> None:
        models = {
            ("p", 1): {
                "program": "p",
                "func_id": 1,
                "duration_us": 5.0,
                "block_duration_us_by_subslot": {"AIC": [5.0, 7.0]},
            },
            ("p", 2): {
                "program": "p",
                "func_id": 2,
                "duration_us": 3.0,
                "block_duration_us_by_subslot": {"AIV0": [3.0]},
            },
        }
        tasks = [
            {
                "task_id": "1",
                "dependencies": [],
                "kernel_ids": [1, -1, -1],
                "block_num": 2,
            },
            {
                "task_id": "2",
                "dependencies": ["1"],
                "kernel_ids": [-1, 2, -1],
                "block_num": 1,
            },
        ]
        graph = self.builder._schedule_graph("p", tasks, models, clusters=1)
        self.assertEqual(graph["duration_us"], 15.0)
        self.assertEqual(graph["modeled_block_count"], 3)

    def test_plan_builder_keeps_sleep_input_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "task_id": "1",
                                "program": "p",
                                "features": {"capture_role": "request"},
                                "dependencies": [],
                                "kernel_ids": [1, -1, -1],
                                "block_num": 1,
                                "resource_shape": "AIC",
                            }
                        ]
                    }
                ),
                encoding="ascii",
            )
            sleep_plan = {
                "schema_version": 1,
                "execution_mode": "sleep",
                "backend": "a2a3sim",
                "coverage": {"runtime_stub_ready": True},
                "preferred_capture_roles": ["request"],
                "source_profile_manifests": [str(profile)],
                "kernel_models": [
                    {
                        "program": "p",
                        "func_id": 1,
                        "duration_us": 4.0,
                        "block_duration_us_by_subslot": {"AIC": [4.0]},
                        "selected_capture_roles": ["request"],
                    }
                ],
            }
            result = self.builder.build_timewarp_plan(sleep_plan, 1)
        self.assertEqual(sleep_plan["execution_mode"], "sleep")
        self.assertEqual(result["execution_mode"], "timewarp")
        self.assertEqual(result["timewarp"]["graph_models"][0]["duration_us"], 4.0)

    def test_shared_clock_advances_only_at_step_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clock = self.clock_module.SharedClock(Path(temporary) / "clock", create=True)
            try:
                keeper = self.clock_module.TimeKeeper(clock)
                before = clock.now_ns()
                keeper.begin_step("0:1")
                deadline = keeper.graph("0:1", "decode_fwd", 20_000_000)
                self.assertLess(clock.now_ns(), deadline)
                keeper.seal_step("0:1")
                delivered = keeper.wait_step("0:1", 1.0)
                self.assertGreaterEqual(delivered, deadline)
                self.assertGreaterEqual(delivered - before, 20_000_000)
                self.assertEqual(keeper.snapshot()["completed_steps"], 1)
            finally:
                clock.close()

    def test_deadline_overrun_is_reported_without_rewriting_clock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clock = self.clock_module.SharedClock(Path(temporary) / "clock", create=True)
            try:
                keeper = self.clock_module.TimeKeeper(clock)
                keeper.begin_step("0:1")
                keeper.graph("0:1", "decode_fwd", 0)
                keeper.seal_step("0:1")
                time.sleep(0.002)
                keeper.wait_step("0:1", 1.0)
                snapshot = keeper.snapshot()
                self.assertEqual(snapshot["deadline_overrun_count"], 1)
                self.assertGreater(snapshot["deadline_overrun_total_ns"], 0)
                self.assertEqual(clock.offset_ns(), 0)
            finally:
                clock.close()

    def test_device_frontier_preserves_pipelined_step_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clock = self.clock_module.SharedClock(Path(temporary) / "clock", create=True)
            try:
                keeper = self.clock_module.TimeKeeper(clock)
                keeper.begin_step("0:1")
                first = keeper.graph("0:1", "decode_fwd", 5_000_000)
                keeper.begin_step("0:2")
                second = keeper.graph("0:2", "decode_fwd", 7_000_000)
                self.assertGreaterEqual(second - first, 7_000_000)
                keeper.seal_step("0:1")
                keeper.seal_step("0:2")
                self.assertGreaterEqual(keeper.wait_step("0:1", 1.0), first)
                self.assertGreaterEqual(keeper.wait_step("0:2", 1.0), second)
            finally:
                clock.close()

    def test_output_observer_gate_prevents_next_jump(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clock = self.clock_module.SharedClock(Path(temporary) / "clock", create=True)
            try:
                keeper = self.clock_module.TimeKeeper(clock)
                keeper.output_expected()
                keeper.begin_step("0:2")
                deadline = keeper.graph("0:2", "decode_fwd", 200_000_000)
                keeper.seal_step("0:2")
                delivered: list[int] = []
                waiter = threading.Thread(
                    target=lambda: delivered.append(keeper.wait_step("0:2", 1.0))
                )
                waiter.start()
                time.sleep(0.02)
                self.assertTrue(waiter.is_alive())
                self.assertLess(clock.now_ns(), deadline)
                keeper.output_delivered()
                waiter.join(timeout=1.0)
                self.assertFalse(waiter.is_alive())
                self.assertGreaterEqual(delivered[0], deadline)
                snapshot = keeper.snapshot()
                self.assertEqual(snapshot["output_pending"], 0)
                self.assertEqual(snapshot["delivered_outputs"], 1)
            finally:
                clock.close()

    def test_streaming_benchmark_clock_is_explicit(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            wall_delta = abs(self.benchmark.benchmark_now() - time.perf_counter())
        self.assertLess(wall_delta, 0.01)
        with tempfile.TemporaryDirectory() as temporary:
            clock = self.clock_module.SharedClock(Path(temporary) / "clock", create=True)
            try:
                clock.write_offset_ns(50_000_000)
                self.benchmark._TIMEWARP_CLOCK = None
                with mock.patch.dict(
                    os.environ, {"PYPTO_TIMEWARP_CLOCK": clock.path}, clear=True
                ):
                    virtual_delta = self.benchmark.benchmark_now() - time.monotonic()
                self.assertGreater(virtual_delta, 0.04)
                self.assertLess(virtual_delta, 0.06)
            finally:
                if self.benchmark._TIMEWARP_CLOCK is not None:
                    self.benchmark._TIMEWARP_CLOCK.close()
                    self.benchmark._TIMEWARP_CLOCK = None
                clock.close()

    def test_timewarp_stub_has_no_wall_delay_and_keeps_control_output(self) -> None:
        fake_sleep = SimpleNamespace(
            _telemetry_source=lambda entry: "static void record_first_execution() {}",
            _control_source=lambda outputs: "  write_token();" if outputs else "",
        )
        source = self.hook._no_wait_stub_source(
            fake_sleep,
            {
                "program": "topk_select_fwd",
                "func_id": 1,
                "duration_us": 1000.0,
                "control_outputs": [{"strategy": "constant_token"}],
            },
        )
        self.assertIn("write_token()", source)
        self.assertNotIn("sleep_until", source)
        self.assertNotIn("runtime_delay_hook", source)

    def test_runtime_lib_dir_accepts_build_root_and_lib_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_root = root / "lib"
            runtime = library_root / "a2a3/sim/tensormap_and_ringbuffer/libhost_runtime.so"
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            self.assertEqual(self.launcher._resolve_runtime_lib_dir(root), library_root)
            self.assertEqual(self.launcher._resolve_runtime_lib_dir(library_root), library_root)
            with self.assertRaisesRegex(ValueError, "custom simulator runtime"):
                self.launcher._resolve_runtime_lib_dir(root / "missing")

    def test_timewarp_plan_validation_rejects_sleep_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "execution_mode": "sleep",
                        "backend": "a2a3sim",
                    }
                ),
                encoding="ascii",
            )
            with (
                mock.patch.dict(os.environ, {"PYPTO_TIMEWARP_PLAN": str(path)}, clear=True),
                self.assertRaisesRegex(self.hook.TimewarpHookError, "not an a2a3sim timewarp"),
            ):
                self.hook._load_plan()


if __name__ == "__main__":
    unittest.main()
