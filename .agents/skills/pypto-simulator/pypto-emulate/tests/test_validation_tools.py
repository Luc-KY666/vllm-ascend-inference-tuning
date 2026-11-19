from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "references" / "qwen" / "run_validation.py"


class ValidationToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("run_qwen_validation", RUNNER)
        assert spec and spec.loader
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_summarizes_measured_requests_only(self) -> None:
        rows = [
            {"phase": "warmup", "wall_ms": 1000.0},
            {"phase": "measure", "wall_ms": 10.0},
            {"phase": "measure", "wall_ms": 20.0},
            {"phase": "measure", "wall_ms": 30.0},
        ]
        summary = self.runner.summarize(rows)
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["mean_ms"], 20.0)
        self.assertEqual(summary["p50_ms"], 20.0)
        self.assertEqual(summary["p95_ms"], 30.0)

    def test_rejects_wrong_simulator_control_token(self) -> None:
        args = SimpleNamespace(warmup_requests=0, requests=1, max_new_tokens=2)
        rows = [
            {
                "phase": "measure",
                "token_count": 2,
                "finish_reason": "length",
                "token_ids": [7, 8],
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "unexpected control token"):
            self.runner.validate_rows(rows, args, expected_token=7)
    def test_acceptance_runner_rejects_scheduler_topology_override(self) -> None:
        argv = [
            str(RUNNER),
            "--model",
            "/tmp/model",
            "--plan",
            "/tmp/plan.json",
            "--output",
            "/tmp/output",
            "--device",
            "0",
            "--sim-aicpu-threads",
            "6",
        ]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
            self.runner.parse_args()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
