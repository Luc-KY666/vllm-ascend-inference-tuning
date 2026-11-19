from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pypto-profile" / "scripts" / "export_runtime_profile.py"


class ExportRuntimeProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("export_runtime_profile", SCRIPT)
        assert spec and spec.loader
        cls.exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.exporter)

    def test_tensormap_topology_distinguishes_vector_lanes(self) -> None:
        metadata = {
            "metadata": {
                "runtime": "tensormap_and_ringbuffer",
                "core_types": ["aic"] * 20 + ["aiv"] * 40,
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "chip_swimlane_records.json"
            capture.write_text(json.dumps(metadata), encoding="utf-8")
            slots = self.exporter._physical_slot_map(
                capture, {"runtime": "tensormap_and_ringbuffer"}
            )

        self.assertEqual(slots[0], "AIC")
        self.assertEqual(slots[20], "AIV0")
        self.assertEqual(slots[21], "AIV1")
        self.assertEqual(slots[58], "AIV0")
        self.assertEqual(slots[59], "AIV1")
        active = ["AIC", "AIV0", "AIV1"]
        self.assertEqual(
            self.exporter._slot({"core_type": "aiv", "core_id": 20}, active, slots),
            "AIV0",
        )
        self.assertEqual(
            self.exporter._slot({"core_type": "aiv", "core_id": 21}, active, slots),
            "AIV1",
        )

    def test_uses_timing_dispatch_as_independent_workload_identity(self) -> None:
        identity = {
            "owner_instance_id": "owner-a",
            "run_id": 7,
            "endpoint_dispatch_id": 9,
            "callable_digest": "0123456789abcdef" * 4,
        }
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary)
            (capture / "dispatch_identity.timing.json").write_text(
                json.dumps(identity), encoding="utf-8"
            )
            workload_id, metadata = self.exporter._workload_identity(capture, None)

        self.assertEqual(workload_id, "owner-a:7:9:0123456789abcdef")
        self.assertEqual(metadata["run_id"], 7)

    def test_single_vector_slot_does_not_depend_on_physical_lane(self) -> None:
        row = {"core_type": "aiv", "core_id": 59}
        self.assertEqual(self.exporter._slot(row, ["AIV0"], {}), "AIV0")

    def test_rejects_unvalidated_tensormap_topology(self) -> None:
        metadata = {"metadata": {"core_types": ["aic", "aiv", "aiv", "aic"]}}
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / "chip_swimlane_records.json"
            capture.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(self.exporter.ContractError, "topology"):
                self.exporter._physical_slot_map(
                    capture, {"runtime": "tensormap_and_ringbuffer"}
                )


if __name__ == "__main__":
    unittest.main()
