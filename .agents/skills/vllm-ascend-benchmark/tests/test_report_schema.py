from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_report_schema", ROOT / "scripts" / "report_schema.py"
)
assert SPEC and SPEC.loader
report_schema = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_schema)


def template() -> dict:
    return json.loads(
        (ROOT / "references" / "benchmark-report-template.json").read_text(encoding="utf-8")
    )


class ReportSchemaTests(unittest.TestCase):
    def test_template_is_final_report_shape(self) -> None:
        report = template()
        self.assertEqual(report_schema.validate_report(report), [])
        self.assertEqual(report["schema_version"], report_schema.SCHEMA_VERSION)
        self.assertNotIn("timings", report)
        self.assertNotIn("preflight", report)
        self.assertEqual(report["parameters"]["seed"], None)
        self.assertEqual(report["parameters"]["percentile_metrics"], ["ttft", "tpot", "e2el"])
        self.assertEqual(report["parameters"]["top_p"], None)

    def test_workload_and_result_contract(self) -> None:
        report = template()
        report["model"]["name"] = "example-model"
        report["load"].update({"mode": "stress", "selection": "preset", "scenario": "random"})
        report["dataset"].update({"mode": "random", "name": "synthetic"})
        report["arrival"].update(
            {
                "mode": "closed_loop",
                "requested_count": 100,
                "warmups": 10,
                "max_concurrency": 8,
                "rounds": 1,
            }
        )
        report["parameters"].update(
            {
                "backend": "openai-chat",
                "tokenizer": "/models/example",
                "base_url": "http://127.0.0.1:8117",
                "endpoint": "/v1/chat/completions",
                "seed": 42,
                "seed_source": "contract",
                "temperature": 0.0,
                "temperature_source": "provider_default",
                "top_p": 1.0,
                "stream": True,
                "include_usage": True,
                "num_prompts": 100,
                "max_concurrency": 8,
                "num_warmups": 10,
                "request_rate": 0.0,
                "burstiness": 1.0,
            }
        )
        report["result"].update(
            {
                "requested": 100,
                "completed": 100,
                "failed": 0,
                "ttft_ms": {"p50": 10.0, "p95": 20.0},
                "tpot_ms": {"p50": 2.0, "p95": 4.0},
                "e2el_ms": {"p50": 100.0, "p95": 150.0},
            }
        )
        self.assertEqual(report_schema.validate_report(report), [])

    def test_goodput_is_only_valid_when_requested(self) -> None:
        report = template()
        report["result"]["goodput"] = {
            "status": "reported",
            "slo_ms": {"ttft": 1000},
            "request_goodput": 0.9,
        }
        report["parameters"]["goodput"] = {"ttft": 1000}
        self.assertEqual(report_schema.validate_report(report), [])
        report["result"]["goodput"] = {
            "status": "reported",
            "slo_ms": {},
            "request_goodput": 0.9,
        }
        self.assertTrue(any("slo_ms" in error for error in report_schema.validate_report(report)))

    def test_failure_and_error_contract(self) -> None:
        report = template()
        report["status"] = "error"
        report["failure"] = {
            "stage": "benchmark",
            "type": "runtime_oom",
            "reason": "runtime out of memory",
        }
        report["errors"].append(
            {
                "stage": "benchmark",
                "type": "runtime_oom",
                "message": "runtime out of memory",
                "recoverable": False,
                "resolved": False,
            }
        )
        self.assertEqual(report_schema.validate_report(report), [])

    def test_old_intermediate_sections_are_rejected(self) -> None:
        report = template()
        report["timings"] = {}
        errors = report_schema.validate_report(report)
        self.assertTrue(any("timings" in error for error in errors))

    def test_validation_does_not_mutate_report(self) -> None:
        report = template()
        before = copy.deepcopy(report)
        report_schema.validate_report(report)
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
