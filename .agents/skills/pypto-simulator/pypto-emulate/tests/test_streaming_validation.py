from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "pypto-emulate" / "scripts" / "run_streaming_benchmark.py"
VALIDATION = ROOT / "pypto-emulate" / "scripts" / "run_streaming_validation.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class StreamingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workload = load_module("run_streaming_benchmark", WORKLOAD)
        cls.validation = load_module("run_streaming_validation", VALIDATION)

    def test_build_prompt_has_exact_token_count(self) -> None:
        prompt = self.workload.build_prompt(FakeTokenizer(), 17, 42)
        self.assertEqual(len(FakeTokenizer().encode(prompt)), 17)

    def test_manifest_rejects_output_too_short_for_tpot(self) -> None:
        manifest = {
            "schema_version": 1,
            "seed": 1,
            "warmup_rounds": 0,
            "measure_rounds": 1,
            "cases": [
                {
                    "id": "bad",
                    "requests": 1,
                    "concurrency": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least two output tokens"):
                self.workload.load_manifest(path)

    def test_summary_uses_measured_rows_and_nearest_rank_percentiles(self) -> None:
        rows = [
            {
                "phase": "warmup",
                "case": "one",
                "ttft_ms": 1000.0,
                "tpot_ms": 1000.0,
                "e2el_ms": 1000.0,
            },
            *[
                {
                    "phase": "measure",
                    "case": "one",
                    "ttft_ms": float(value),
                    "tpot_ms": float(value * 2),
                    "e2el_ms": float(value * 3),
                }
                for value in range(1, 5)
            ],
        ]
        summary = self.validation.summarize(rows)
        self.assertEqual(summary["request_count"], 4)
        self.assertEqual(summary["overall"]["ttft_ms"]["mean"], 2.5)
        self.assertEqual(summary["overall"]["ttft_ms"]["p50"], 2.0)
        self.assertEqual(summary["overall"]["ttft_ms"]["p95"], 4.0)

    def test_serving_command_preserves_venv_python_symlink_path(self) -> None:
        args = SimpleNamespace(
            ptoas_root=Path("/tmp/ptoas"),
            manifest=Path("/tmp/manifest.json"),
            model=Path("/tmp/model"),
            device=2,
            max_model_len=512,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            ring_dep_pool=262144,
            ring_task_window=262144,
            ring_heap=1073741824,
        )
        command = self.validation.serving_command(args, Path("/tmp/results.jsonl"))
        self.assertEqual(command[0], "/tmp/ptoas/bin/python")

    def test_threshold_checks_each_case_and_each_statistic(self) -> None:
        metric = {
            stat: {
                "native_a_ms": 100.0,
                "simulator_ms": 100.0,
                "native_b_ms": 100.0,
                "native_midpoint_ms": 100.0,
                "simulator_error_pct": 0.0,
                "native_drift_pct": 0.0,
            }
            for stat in self.validation.STATS
        }
        scope = {name: json.loads(json.dumps(metric)) for name in self.validation.METRICS}
        comparison = {"overall": scope, "cases": {"one": json.loads(json.dumps(scope))}}
        self.assertTrue(self.validation.all_within_threshold(comparison))
        comparison["cases"]["one"]["tpot_ms"]["p95"]["simulator_error_pct"] = 5.01
        self.assertFalse(self.validation.all_within_threshold(comparison))


if __name__ == "__main__":
    unittest.main()
