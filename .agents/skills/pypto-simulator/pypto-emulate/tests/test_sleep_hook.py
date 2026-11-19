from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "pypto-emulate" / "scripts" / "install_sleep_hook.py"


class SleepHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("install_sleep_hook", HOOK)
        assert spec is not None and spec.loader is not None
        cls.hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.hook)

    def test_simulator_memory_info_subtracts_committed_allocations(self) -> None:
        class Worker:
            @staticmethod
            def committed_device_memory(worker_id: int) -> int:
                return 256 if worker_id == 3 else 0

        with mock.patch.dict(os.environ, {"PYPTO_SLEEP_DEVICE_MEMORY_BYTES": "1024"}):
            self.assertEqual(self.hook._simulator_memory_info(Worker(), 3), (768, 1024))

    def test_simulator_memory_info_uses_exact_native_footprint(self) -> None:
        class Worker:
            committed = 600

            def committed_device_memory(self, worker_id: int) -> int:
                return self.committed

        worker = Worker()
        self.hook._EXACT_MEMORY_BASELINES.clear()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                self.hook._simulator_memory_info(
                    worker,
                    native_total_bytes=2048,
                    driver_visible_non_kv_bytes=768,
                ),
                (1280, 2048),
            )
            worker.committed = 800
            self.assertEqual(
                self.hook._simulator_memory_info(
                    worker,
                    native_total_bytes=2048,
                    driver_visible_non_kv_bytes=768,
                ),
                (1080, 2048),
            )
        with (
            mock.patch.dict(
                os.environ, {"PYPTO_SLEEP_DEVICE_MEMORY_BYTES": "4096"}
            ),
            self.assertRaisesRegex(self.hook.SleepHookError, "cannot override"),
        ):
            self.hook._simulator_memory_info(
                Worker(),
                native_total_bytes=2048,
                driver_visible_non_kv_bytes=768,
            )

    def test_simulator_memory_info_rejects_overcommit(self) -> None:
        class Worker:
            @staticmethod
            def committed_device_memory(worker_id: int) -> int:
                return 2048

        with (
            mock.patch.dict(os.environ, {"PYPTO_SLEEP_DEVICE_MEMORY_BYTES": "1024"}),
            self.assertRaisesRegex(self.hook.SleepHookError, "outside"),
        ):
            self.hook._simulator_memory_info(Worker())

    def test_sim_aicpu_thread_override_is_optional_and_bounded(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.hook._sim_aicpu_threads())
        with mock.patch.dict(os.environ, {"PYPTO_SLEEP_AICPU_THREADS": "16"}):
            self.assertEqual(self.hook._sim_aicpu_threads(), 16)
        with (
            mock.patch.dict(os.environ, {"PYPTO_SLEEP_AICPU_THREADS": "17"}),
            self.assertRaisesRegex(self.hook.SleepHookError, r"\[2, 16\]"),
        ):
            self.hook._sim_aicpu_threads()

    def test_memory_profile_interpolates_only_inside_measured_domain(self) -> None:
        profile = self.hook._MemoryProfile(
            {
                "models": [
                    {
                        "operation": "kv_materialize",
                        "points": [
                            {"bytes": 100, "duration_us": 10.0},
                            {"bytes": 200, "duration_us": 30.0},
                        ],
                    }
                ]
            }
        )
        self.assertEqual(profile.estimate_us("kv_materialize", 100), 10.0)
        self.assertEqual(profile.estimate_us("kv_materialize", 150), 20.0)
        with self.assertRaisesRegex(self.hook.SleepHookError, "outside measured domain"):
            profile.estimate_us("kv_materialize", 201)

    def test_memory_profile_allows_only_matching_rounded_weight_bytes_in_diagnostic_model(self) -> None:
        profile = self.hook._MemoryProfile(
            {
                "models": [
                    {
                        "operation": "weight_materialize",
                        "non_strict_reason": "byte count comes from a rounded serving log",
                        "points": [{"bytes": 27_980_000_000, "duration_us": 10.0}],
                    }
                ]
            }
        )
        self.assertEqual(profile.estimate_us("weight_materialize", 27_979_400_000), 10.0)
        with self.assertRaisesRegex(self.hook.SleepHookError, "outside measured domain"):
            profile.estimate_us("weight_materialize", 27_970_000_000)

    def test_memory_profile_rejects_non_strict_manifest_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.json"
            path.write_text(
                json.dumps({"schema_version": 1, "models": [], "strict_ready": False}),
                encoding="ascii",
            )
            with (
                mock.patch.dict(
                    os.environ,
                    {"PYPTO_SLEEP_MEMORY_PROFILE": str(path)},
                    clear=True,
                ),
                self.assertRaisesRegex(self.hook.SleepHookError, "not strict-ready"),
            ):
                self.hook._load_memory_profile()
            with mock.patch.dict(
                os.environ,
                {
                    "PYPTO_SLEEP_MEMORY_PROFILE": str(path),
                    "PYPTO_SLEEP_MEMORY_ALLOW_NON_STRICT": "1",
                },
                clear=True,
            ):
                self.assertEqual(self.hook._load_memory_profile()["schema_version"], 1)

    def test_unmodeled_stub_fails_closed(self) -> None:
        self.assertIn("std::abort()", self.hook._stub_source(None))

    def test_zero_duration_stub_uses_read_only_telemetry_fast_path(self) -> None:
        source = self.hook._stub_source(
            {
                "program": "decode_fwd",
                "func_id": 17,
                "duration_us": 0.0,
                "control_outputs": [],
            }
        )
        self.assertIn("recorded.load(std::memory_order_relaxed)", source)
        self.assertIn("compare_exchange_strong", source)
        self.assertNotIn("test_and_set", source)
        self.assertNotIn("sleep_until", source)

    def test_positive_duration_stub_uses_hybrid_deadline_delay(self) -> None:
        source = self.hook._stub_source(
            {
                "program": "decode_fwd",
                "func_id": 17,
                "duration_us": 250.0,
                "control_outputs": [],
            }
        )
        self.assertIn("delay_for_ns(250000ULL)", source)
        self.assertIn("sleep_until(deadline - sleep_margin)", source)
        self.assertIn("cpu_relax()", source)

    def test_graph_timing_buffers_records_until_process_exit(self) -> None:
        class FakeDispatchMixin:
            def _run_l3(self, callable_spec, *args, config=None):
                return callable_spec.name

        modules = {
            name: ModuleType(name)
            for name in (
                "pypto_serving",
                "pypto_serving.model",
                "pypto_serving.model.common",
                "pypto_serving.model.common.runner",
                "pypto_serving.model.common.runner.l3_dispatch",
            )
        }
        modules["pypto_serving.model.common.runner.l3_dispatch"].L3DispatchMixin = FakeDispatchMixin
        callbacks = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graphs.jsonl"
            with (
                mock.patch.dict(sys.modules, modules),
                mock.patch.object(self.hook.atexit, "register", side_effect=callbacks.append),
            ):
                self.hook._install_graph_timing(path)
            self.assertEqual(FakeDispatchMixin()._run_l3(SimpleNamespace(name="decode_fwd")), "decode_fwd")
            self.assertFalse(path.exists())
            self.assertEqual(len(callbacks), 1)
            callbacks[0]()
            record = json.loads(path.read_text(encoding="ascii"))
        self.assertEqual(record["event"], "sleep_graph_wall")
        self.assertEqual(record["program"], "decode_fwd")
        self.assertGreaterEqual(record["wall_us"], 0)

    def test_stub_records_first_execution_once(self) -> None:
        entry = {
            "program": "decode_fwd",
            "func_id": 17,
            "duration_us": 0.0,
            "control_outputs": [],
        }
        binary = self.hook._compile_stub(entry)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library_path = root / "stub.so"
            telemetry = root / "telemetry.jsonl"
            library_path.write_bytes(binary)
            with mock.patch.dict(os.environ, {"PYPTO_SLEEP_TELEMETRY": str(telemetry)}):
                library = ctypes.CDLL(str(library_path))
                library.kernel_entry.argtypes = [ctypes.POINTER(ctypes.c_int64)]
                library.kernel_entry.restype = None
                library.kernel_entry(None)
                library.kernel_entry(None)
            records = telemetry.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(records), 1)
        self.assertIn('"func_id":17', records[0])
        self.assertIn('"program":"decode_fwd"', records[0])

    def test_stub_sleeps_and_materializes_qwen_token(self) -> None:
        binary = self.hook._compile_stub(
            {
                "program": "topk_select_fwd",
                "func_id": 3,
                "duration_us": 2_000.0,
                "control_outputs": [
                    {
                        "strategy": "constant_token",
                        "arg_index": 1,
                        "token_id": 9,
                        "padding_value": 0,
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            library_path = Path(temporary) / "stub.so"
            library_path.write_bytes(binary)
            library = ctypes.CDLL(str(library_path))
            library.kernel_entry.argtypes = [ctypes.POINTER(ctypes.c_int64)]
            library.kernel_entry.restype = None

            output = (ctypes.c_int32 * 8)()
            tensor = (ctypes.c_ubyte * 128)()
            address = ctypes.addressof(output)
            struct.pack_into("=Q", tensor, 0, address)
            struct.pack_into("=Q", tensor, 24, 0)
            struct.pack_into("=I", tensor, 44, 2)
            struct.pack_into("=I", tensor, 48, 4)
            struct.pack_into("=I", tensor, 72, 4)
            struct.pack_into("=I", tensor, 76, 1)
            call_args = (ctypes.c_int64 * 2)(0, ctypes.addressof(tensor))

            started = time.perf_counter()
            library.kernel_entry(call_args)
            elapsed = time.perf_counter() - started

        self.assertGreaterEqual(elapsed, 0.001)
        self.assertEqual(list(output), [9, 0, 0, 0, 9, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
