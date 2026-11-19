from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "pypto-common"
sys.path.insert(0, str(COMMON))

from pypto_contract import should_intercept

SCRIPT = ROOT / "pypto-profile" / "scripts" / "normalize_task_profile.py"
EXAMPLE = ROOT / "examples" / "qwen-decode-task-profile.json"


class NormalizeTaskProfileTests(unittest.TestCase):
    def test_interception_boundary_is_narrow(self) -> None:
        self.assertTrue(should_intercept({"kind": "compute"}))
        self.assertTrue(should_intercept({"kind": "communication"}))
        self.assertTrue(
            should_intercept({"kind": "memory", "tensor_roles": ["weight"]})
        )
        self.assertTrue(
            should_intercept({"kind": "memory", "tensor_roles": ["kv_cache"]})
        )
        self.assertFalse(
            should_intercept({"kind": "memory", "tensor_roles": ["activation"]})
        )
        self.assertFalse(should_intercept({"kind": "synchronization"}))

    def test_normalizes_qwen_fixture_without_npu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "profile"
            subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--input",
                    str(EXAMPLE),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "profile_manifest.json").read_text())
            coverage = json.loads((output / "coverage.json").read_text())

        self.assertEqual(manifest["model"]["family"], "qwen")
        self.assertEqual(coverage["task_count"], 4)
        self.assertEqual(coverage["measured_task_count"], 3)
        self.assertEqual(coverage["intercepted_task_count"], 3)
        self.assertFalse(coverage["npu_accessed"])

    def test_rejects_unknown_dependency(self) -> None:
        raw = {
            "model": {"family": "qwen", "model_id": "test"},
            "tasks": [
                {
                    "task_id": "t0",
                    "task_name": "test",
                    "kind": "compute",
                    "dependencies": ["missing"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.json"
            source.write_text(json.dumps(raw), encoding="utf-8")
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output",
                    str(root / "out"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown dependencies", result.stderr)


if __name__ == "__main__":
    unittest.main()
