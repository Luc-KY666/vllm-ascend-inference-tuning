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
    def test_template_has_all_required_sections(self) -> None:
        report = canonical_report()
        self.assertEqual(report_schema.validate_report(report), [])
        self.assertEqual(set(report["stages"]), report_schema.STAGE_KEYS)
        self.assertEqual(set(report["artifacts"]["probe"]), set(report_schema.PROBE_ARTIFACTS))

    def test_environment_paths_are_rejected(self) -> None:
        report = canonical_report()
        report["environment"]["probe_path"] = "/attempt/environment_probe.json"
        errors = report_schema.validate_report(report)
        self.assertTrue(any("environment.probe_path" in error for error in errors))

    def test_environment_path_like_fields_are_rejected(self) -> None:
        report = canonical_report()
        report["environment"]["diagnostic_path"] = "/attempt/environment_probe.json"
        errors = report_schema.validate_report(report)
        self.assertTrue(any("environment.diagnostic_path" in error for error in errors))

    def test_probe_paths_have_one_canonical_owner(self) -> None:
        report = canonical_report()
        report["artifacts"]["probe"]["environment_probe"] = "/attempt/environment_probe.json"
        report["artifacts"]["probe"]["sandbox_detection"] = "/attempt/sandbox-detection.json"
        self.assertEqual(report_schema.validate_report(report), [])
        paths = report_schema.artifact_path_index(report)
        self.assertEqual(paths["/attempt/environment_probe.json"], ["artifacts.probe.environment_probe"])
        self.assertEqual(paths["/attempt/sandbox-detection.json"], ["artifacts.probe.sandbox_detection"])

    def test_duplicate_artifact_paths_are_rejected(self) -> None:
        report = canonical_report()
        report["artifacts"]["probe"]["sandbox_detection"] = "/attempt/same"
        report["artifacts"]["raw_logs"]["server_log"] = "/attempt/same"
        errors = report_schema.validate_report(report)
        self.assertTrue(any("artifact 路径重复" in error for error in errors))

    def test_probe_path_cannot_be_duplicated_in_provider_payload(self) -> None:
        report = canonical_report()
        report["artifacts"]["probe"]["environment_probe"] = "/attempt/environment_probe.json"
        report["input"]["provider_payload"] = {"environment_probe": "/attempt/environment_probe.json"}
        errors = report_schema.validate_report(report)
        self.assertTrue(any("artifact 路径重复" in error for error in errors))
        self.assertTrue(any("input 不得保存" in error for error in errors))

    def test_log_metadata_is_not_mistaken_for_a_second_path(self) -> None:
        report = canonical_report()
        report["artifacts"]["raw_logs"]["server_log"] = {
            "path": "/attempt/server.log",
            "sha256": "same-digest",
        }
        report["artifacts"]["raw_logs"]["command_transcript"] = {
            "path": "/attempt/command.txt",
            "sha256": "same-digest",
        }
        self.assertEqual(report_schema.validate_report(report), [])

    def test_old_aliases_and_intermediate_paths_are_rejected(self) -> None:
        report = canonical_report()
        report["inputs"] = {"environment_probe": "/attempt/environment_probe.json"}
        report["stages"]["static_oom"]["result"] = {"path": "static-oom-report.json"}
        errors = report_schema.validate_report(report)
        self.assertTrue(any("旧字段" in error for error in errors))
        self.assertTrue(any("static-oom-report.json" in error for error in errors))

    def test_failure_uses_logical_artifact_references(self) -> None:
        report = canonical_report()
        report["failure"] = {"artifacts": ["/attempt/server.log"]}
        errors = report_schema.validate_report(report)
        self.assertTrue(any("逻辑引用" in error for error in errors))

    def test_report_has_no_mutation_from_validation(self) -> None:
        report = canonical_report()
        before = copy.deepcopy(report)
        report_schema.validate_report(report)
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
