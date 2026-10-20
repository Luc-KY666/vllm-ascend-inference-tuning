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
    def test_template_is_canonical(self) -> None:
        report = template()
        self.assertEqual(report_schema.validate_report(report), [])
        self.assertEqual(report["schema_version"], report_schema.SCHEMA_VERSION)

    def test_success_report_accepts_canonical_artifact_paths(self) -> None:
        report = template()
        report["status"] = "success"
        report["preflight"].update(
            {
                "status": "pass",
                "provider_path": "managed_deployment",
                "execution_context": "current",
                "environment_probe": "/attempt/environment_probe.json",
                "deployment_report": "/attempt/deployment_report.json",
            }
        )
        report["source"].update(
            {
                "raw_result": "/attempt/raw_benchmark_result.json",
                "raw_result_sha256": "a" * 64,
            }
        )
        self.assertEqual(report_schema.validate_report(report), [])

    def test_duplicate_paths_are_rejected(self) -> None:
        report = template()
        report["preflight"]["environment_probe"] = "/attempt/same.json"
        report["preflight"]["deployment_report"] = "/attempt/same.json"
        errors = report_schema.validate_report(report)
        self.assertTrue(any("duplicated" in error for error in errors))

    def test_legacy_fields_and_large_output_are_rejected(self) -> None:
        report = template()
        report["inputs"] = {}
        report["errors"].append({"stdout": "large output"})
        report["errors"].append("stderr dump")
        errors = report_schema.validate_report(report)
        self.assertTrue(any("legacy" in error for error in errors))
        self.assertTrue(any("large output" in error for error in errors))

    def test_old_report_name_and_wrong_raw_filename_are_rejected(self) -> None:
        report = template()
        report["source"]["raw_result"] = "/attempt/benchmark_result.json"
        errors = report_schema.validate_report(report)
        self.assertTrue(any("raw_benchmark_result.json" in error for error in errors))
        self.assertTrue(any("forbidden output marker" in error for error in errors))

    def test_raw_result_requires_sha256(self) -> None:
        report = template()
        report["source"]["raw_result"] = "/attempt/raw_benchmark_result.json"
        errors = report_schema.validate_report(report)
        self.assertTrue(any("64-character hexadecimal" in error for error in errors))

    def test_failure_artifacts_cannot_duplicate_absolute_paths(self) -> None:
        report = template()
        report["failure"] = {"stage": "benchmark", "type": "error", "artifacts": ["/tmp/log"]}
        errors = report_schema.validate_report(report)
        self.assertTrue(any("logical references" in error for error in errors))

    def test_validation_does_not_mutate_report(self) -> None:
        report = template()
        before = copy.deepcopy(report)
        report_schema.validate_report(report)
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
