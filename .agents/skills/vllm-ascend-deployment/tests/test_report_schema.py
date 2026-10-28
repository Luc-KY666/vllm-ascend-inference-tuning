from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deployment_report_schema", ROOT / "scripts" / "report_schema.py"
)
assert SPEC and SPEC.loader
report_schema = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_schema)


def canonical_report() -> dict:
    return json.loads(
        (ROOT / "references" / "deployment-report-template.json").read_text(encoding="utf-8")
    )


class ReportSchemaTests(unittest.TestCase):
    def test_template_is_final_report_shape(self) -> None:
        report = canonical_report()
        self.assertEqual(report_schema.validate_report(report), [])
        self.assertEqual(report["schema_version"], report_schema.SCHEMA_VERSION)
        self.assertNotIn("timings", report)
        self.assertNotIn("stages", report)
        self.assertEqual(set(report["launch"]), {"parameters"})

    def test_failure_and_error_contract(self) -> None:
        report = canonical_report()
        report["status"] = "failed"
        report["failure"] = {
            "stage": "service",
            "type": "startup_timeout",
            "reason": "service did not become ready",
        }
        report["errors"].append(
            {
                "stage": "service",
                "type": "startup_timeout",
                "message": "service did not become ready",
                "recoverable": False,
                "resolved": False,
            }
        )
        self.assertEqual(report_schema.validate_report(report), [])

    def test_old_intermediate_sections_are_rejected(self) -> None:
        report = canonical_report()
        report["timings"] = {}
        errors = report_schema.validate_report(report)
        self.assertTrue(any("timings" in error for error in errors))

    def test_invalid_service_endpoint_is_rejected(self) -> None:
        report = canonical_report()
        report["service"]["endpoint"] = "http://127.0.0.1:8117/v1/chat/completions"
        errors = report_schema.validate_report(report)
        self.assertTrue(any("service.endpoint" in error for error in errors))

    def test_launch_parameters_and_resources_are_checked(self) -> None:
        report = canonical_report()
        report["resources"] = {"npu": [0, 1], "port": 8117}
        report["launch"]["parameters"] = {
            "--tensor-parallel-size": 2,
            "--data-parallel-size": 1,
            "--max-num-seqs": 32,
            "--max-num-batched-tokens": 4096,
        }
        self.assertEqual(report_schema.validate_report(report), [])
        report["resources"]["port"] = 0
        self.assertTrue(any("resources.port" in error for error in report_schema.validate_report(report)))

    def test_validation_does_not_mutate_report(self) -> None:
        report = canonical_report()
        before = copy.deepcopy(report)
        report_schema.validate_report(report)
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
