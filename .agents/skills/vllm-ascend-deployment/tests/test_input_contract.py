from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import input_contract  # noqa: E402


class InputContractTests(unittest.TestCase):
    def payload(self, root: Path, **runtime: object) -> dict[str, object]:
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"num_attention_heads": 8, "hidden_size": 16}),
            encoding="utf-8",
        )
        return {
            "model": {
                "model_path": str(model),
                "served_model_name": "test-model",
            },
            "runtime": runtime,
            "resources": {
                "visible_devices": ["0", "1"],
                "port_candidates": [18117],
            },
            "outputs": {"attempt_dir": str(root / "attempt")},
        }

    def test_missing_tp_gets_defaults_and_recommendation_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = input_contract.validate_and_build(self.payload(Path(directory)))
        self.assertEqual(result["input_status"], "missing_parameters")
        self.assertIsNone(result["runtime"]["tp"])
        self.assertEqual(result["runtime"]["dp"], 1)
        self.assertEqual(result["runtime"]["max_num_seqs"], 32)
        self.assertEqual(result["runtime"]["max_num_batched_tokens"], 4096)
        self.assertEqual(result["runtime"]["allowed_tp"], [1, 2])

    def test_invalid_model_path_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.payload(Path(directory))
            payload["model"]["model_path"] = str(Path(directory) / "missing")
            result = input_contract.validate_and_build(payload)
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(result["diagnostics"]["errors"])

    def test_missing_config_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.payload(root)
            model = root / "model"
            (model / "config.json").unlink()
            result = input_contract.validate_and_build(payload)
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("config.json" in error for error in result["diagnostics"]["errors"]))

    def test_invalid_batch_relationship_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = input_contract.validate_and_build(
                self.payload(Path(directory), tp=1, dp=1, max_num_seqs=64, max_num_batched_tokens=32)
            )
        self.assertEqual(result["input_status"], "invalid")

    def test_recommended_tp_is_written_back_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment = input_contract.validate_and_build(self.payload(root))
            path = root / "deployment.json"
            path.write_text(json.dumps(deployment), encoding="utf-8")
            result = input_contract.resolve_recommended_tp(path, 2, "static-oom-report.json")
            stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["runtime"]["tp"], 2)
        self.assertEqual(stored["runtime"]["tp"], 2)
        self.assertEqual(stored["resolution"]["mode"], "auto_recommended")
        self.assertEqual(stored["resolution"]["original_status"], "missing_parameters")

    def test_recommended_tp_is_revalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment = input_contract.validate_and_build(self.payload(root))
            path = root / "deployment.json"
            path.write_text(json.dumps(deployment), encoding="utf-8")
            result = input_contract.resolve_recommended_tp(path, 3, "static-oom-report.json")
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(result["diagnostics"]["errors"])

    def test_cli_builds_contract_from_provider_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"num_attention_heads": 8}), encoding="utf-8"
            )
            attempt = root / "attempt"
            code = input_contract.main(
                [
                    "--model-path",
                    str(model),
                    "--served-model-name",
                    "field-model",
                    "--visible-devices",
                    "0",
                    "1",
                    "--port-candidates",
                    "18117",
                    "--attempt-dir",
                    str(attempt),
                ]
            )
            stored = json.loads((attempt / "deployment.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(stored["input_status"], "missing_parameters")
        self.assertNotIn("python", stored["resources"])
        self.assertNotIn("vllm", stored["resources"])
        self.assertNotIn("atb_probe", stored["resources"])

    def test_unsupported_tool_fields_are_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = self.payload(Path(directory))
            payload["resources"].update(
                {
                    "python": "/tmp/python",
                    "vllm": "vllm",
                    "atb_probe": "/tmp/probe.py",
                    "activation_script": "/tmp/activate.sh",
                }
            )
            result = input_contract.validate_and_build(payload)
        self.assertEqual(result["input_status"], "missing_parameters")
        self.assertNotIn("python", result["resources"])
        self.assertIn("python", result["resolution"]["ignored_fields"])

    def test_resolution_also_removes_unsupported_tool_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment = input_contract.validate_and_build(self.payload(root))
            deployment["resources"]["vllm"] = "vllm"
            path = root / "deployment.json"
            path.write_text(json.dumps(deployment), encoding="utf-8")
            result = input_contract.resolve_recommended_tp(path, 2, "static-oom-report.json")
        self.assertEqual(result["runtime"]["tp"], 2)
        self.assertNotIn("vllm", result["resources"])


if __name__ == "__main__":
    unittest.main()
