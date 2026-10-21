from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import npu_resource_probe  # noqa: E402


class NpuResourceProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {
            "GOAL_PLUS_DEVICE_POOLS": json.dumps(
                {
                    "ascend-npu": {
                        "visibility_env": "ASCEND_RT_VISIBLE_DEVICES",
                        "devices": ["0", "1"],
                    }
                }
            ),
            "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        }

    def runner(self, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "NPU 0: No running processes found\n"
                "NPU 1: No running processes found\n"
            ),
            stderr="",
        )

    def test_selects_only_explicitly_idle_devices(self) -> None:
        result = npu_resource_probe.probe_authorized_devices(
            environment=self.environment,
            required_count=2,
            runner=self.runner,
        )
        self.assertEqual(result["selected_devices"], ["0", "1"])
        self.assertEqual(result["probe"]["busy_devices"], [])

    def test_busy_device_reduces_available_count(self) -> None:
        runner = lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="NPU 0: No running processes found\nNPU 1: Process ID: 1234",
            stderr="",
        )
        with self.assertRaises(npu_resource_probe.NpuProbeError) as context:
            npu_resource_probe.probe_authorized_devices(
                environment=self.environment,
                required_count=2,
                runner=runner,
            )
        self.assertEqual(context.exception.failure_type, "insufficient_available_devices")

    def test_pool_or_visibility_mismatch_fails_closed(self) -> None:
        environment = dict(self.environment)
        environment["ASCEND_RT_VISIBLE_DEVICES"] = "2"
        with self.assertRaises(npu_resource_probe.NpuProbeError) as context:
            npu_resource_probe.probe_authorized_devices(
                environment=environment,
                runner=self.runner,
            )
        self.assertEqual(context.exception.failure_type, "device_visibility_mismatch")

    def test_explicit_visible_devices_must_match_environment(self) -> None:
        with self.assertRaises(npu_resource_probe.NpuProbeError) as context:
            npu_resource_probe.probe_authorized_devices(
                environment=self.environment,
                visible_devices=["0"],
                runner=self.runner,
            )
        self.assertEqual(context.exception.failure_type, "device_visibility_mismatch")

    def test_unparseable_npu_smi_output_fails_closed(self) -> None:
        runner = lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="NPU inventory only",
            stderr="",
        )
        with self.assertRaises(npu_resource_probe.NpuProbeError) as context:
            npu_resource_probe.probe_authorized_devices(
                environment=self.environment,
                runner=runner,
            )
        self.assertEqual(context.exception.failure_type, "device_probe_failed")

    def test_missing_pool_fails_closed(self) -> None:
        with self.assertRaises(npu_resource_probe.NpuProbeError) as context:
            npu_resource_probe.probe_authorized_devices(
                environment={},
                runner=self.runner,
            )
        self.assertEqual(context.exception.failure_type, "device_pool_missing")


if __name__ == "__main__":
    unittest.main()
