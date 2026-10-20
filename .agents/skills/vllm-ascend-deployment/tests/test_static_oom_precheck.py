from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import static_oom_precheck as precheck_module  # noqa: E402


class StaticOomPrecheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "schema_version": "vllm-ascend-static-oom-input-v1",
            "model": {
                "weight_bytes": 100,
                "config": {
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 2,
                    "intermediate_size": 16,
                    "tie_word_embeddings": False,
                    "torch_dtype": "float16",
                },
            },
            "hardware": {
                "card_memory_bytes": 1000,
                "device_count": 4,
            },
            "runtime": {
                "tp": 1,
                "dp": 1,
                "max_num_seqs": 1,
                "gpu_memory_utilization": 0.9,
                "min_required_context_len": 4,
                "block_size": 2,
                "dtype": "float16",
                "quantization": None,
                "allowed_tp": [1, 2, 4],
                "official_min_tp": 1,
            },
        }

    def result(self, **changes: object) -> dict[str, object]:
        payload = copy.deepcopy(self.payload)
        for path, value in changes.items():
            target: object = payload
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]  # type: ignore[index]
            target[parts[-1]] = value  # type: ignore[index]
        return precheck_module.precheck(payload)

    def test_current_configuration_passes(self) -> None:
        result = self.result()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["current"]["status"], "pass")
        self.assertEqual(result["recommendation"]["recommended_tp"], 1)

    def test_weight_at_budget_is_rejected(self) -> None:
        result = self.result(**{"model.weight_bytes": 900})
        self.assertEqual(result["status"], "reject")
        self.assertEqual(result["current"]["risk_level"], "capacity_exceeded")

    def test_minimum_context_kv_capacity_is_rejected(self) -> None:
        result = self.result(
            **{
                "model.weight_bytes": 800,
                "runtime.max_num_seqs": 2,
                "runtime.min_required_context_len": 100,
            }
        )
        self.assertEqual(result["status"], "reject")
        self.assertIn("KV Cache", result["message"])

    def test_device_count_rejects_current_tp_dp(self) -> None:
        result = self.result(
            **{
                "runtime.tp": 2,
                "runtime.dp": 2,
                "hardware.device_count": 3,
            }
        )
        self.assertEqual(result["status"], "reject")
        self.assertEqual(result["current"]["risk_level"], "resource_exceeded")

    def test_missing_tp_recommends_smallest_passing_tp(self) -> None:
        result = self.result(**{"model.weight_bytes": 880, "runtime.tp": None})
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["recommendation"]["status"], "available")
        self.assertEqual(result["recommendation"]["recommended_tp"], 2)
        self.assertEqual(
            [item["tp"] for item in result["recommendation"]["candidates"]],
            [1, 2],
        )

    def test_recommendation_never_invents_missing_runtime_values(self) -> None:
        for field in (
            "dp",
            "max_num_seqs",
            "gpu_memory_utilization",
            "min_required_context_len",
            "block_size",
            "allowed_tp",
            "official_min_tp",
        ):
            with self.subTest(field=field):
                payload = copy.deepcopy(self.payload)
                payload["runtime"].pop(field)
                payload["runtime"]["tp"] = None
                result = precheck_module.precheck(payload)
                self.assertEqual(result["recommendation"]["status"], "unknown")

    def test_missing_card_memory_is_unknown(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["hardware"].pop("card_memory_bytes")
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("card_memory_bytes", result["message"])

    def test_missing_device_count_is_unknown(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["hardware"].pop("device_count")
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("device_count", result["message"])

    def test_missing_runtime_parameter_is_unknown(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["runtime"].pop("max_num_seqs")
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("max_num_seqs", result["message"])

    def test_all_tp_candidates_can_be_unavailable(self) -> None:
        result = self.result(
            **{
                "model.weight_bytes": 10_000,
                "runtime.tp": None,
                "runtime.allowed_tp": [1, 2],
            }
        )
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["recommendation"]["status"], "unavailable")
        self.assertIsNone(result["recommendation"]["recommended_tp"])

    def test_device_limit_can_make_all_candidates_unavailable(self) -> None:
        result = self.result(
            **{
                "model.weight_bytes": 880,
                "runtime.tp": None,
                "runtime.dp": 2,
                "hardware.device_count": 2,
                "runtime.allowed_tp": [1, 2],
            }
        )
        self.assertEqual(result["recommendation"]["status"], "unavailable")
        self.assertEqual(
            [item["status"] for item in result["recommendation"]["candidates"]],
            ["reject", "reject"],
        )

    def test_explicit_weight_gib_is_converted_upward(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["model"].pop("weight_bytes")
        payload["model"]["weight_gib"] = 1 / (1024**3)
        result = precheck_module.precheck(payload)
        self.assertEqual(result["inputs"]["weight_method"], "explicit_weight_gib")
        self.assertEqual(result["current"]["weight_bytes_total"], 1)

    def test_config_can_derive_dense_weight_requirement(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["model"].pop("weight_bytes")
        payload["hardware"]["card_memory_bytes"] = 10_000
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["inputs"]["weight_method"], "config_derived")

    def test_text_config_is_used_as_official_architecture(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["model"].pop("weight_bytes")
        payload["hardware"]["card_memory_bytes"] = 10_000
        config = payload["model"].pop("config")
        payload["model"]["config"] = {"text_config": config}
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["inputs"]["weight_method"], "config_derived")

    def test_quantization_without_explicit_weight_is_unknown(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["model"].pop("weight_bytes")
        payload["runtime"]["quantization"] = "awq"
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("量化", result["message"])

    def test_quantized_model_with_explicit_weight_can_be_checked(self) -> None:
        result = self.result(**{"runtime.quantization": "awq"})
        self.assertEqual(result["status"], "pass")

    def test_quantization_config_without_explicit_weight_is_unknown(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["model"].pop("weight_bytes")
        payload["model"]["config"]["quantization_config"] = {"bits": 4}
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("量化", result["message"])

    def test_complex_moe_config_requires_explicit_weight(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["model"].pop("weight_bytes")
        payload["model"]["config"]["num_experts"] = 2
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("num_experts", result["message"])

    def test_missing_architecture_is_unknown(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["model"]["config"].pop("num_attention_heads")
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("num_attention_heads", result["message"])

    def test_attention_heads_must_be_divisible_by_tp(self) -> None:
        result = self.result(**{"runtime.tp": 3})
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["current"]["risk_level"], "incompatible_configuration")
        self.assertIn("attention heads", result["message"])

    def test_kv_heads_cannot_exceed_attention_heads(self) -> None:
        result = self.result(**{"model.config.num_key_value_heads": 8})
        self.assertEqual(result["status"], "unknown")
        self.assertIn("不能大于", result["message"])

    def test_inconsistent_head_dimension_does_not_block_oom_check(self) -> None:
        result = self.result(**{"model.config.head_dim": 3})
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["current"]["status"], "pass")
        self.assertEqual(result["current"]["kv_bytes_per_token_per_card"], 48)

    def test_unknown_dtype_is_unknown_even_with_explicit_weight(self) -> None:
        result = self.result(**{"runtime.dtype": "int4"})
        self.assertEqual(result["status"], "unknown")
        self.assertIn("dtype", result["message"])

    def test_missing_current_tp_keeps_top_level_unknown(self) -> None:
        result = self.result(**{"runtime.tp": None})
        self.assertEqual(result["status"], "unknown")
        self.assertNotEqual(result["status"], "pass")
        self.assertEqual(result["current"]["status"], "missing_parameter")

    def test_recommendation_mode_supplies_explicit_runtime_defaults(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["hardware"]["card_memory_bytes"] = 10000
        payload["recommendation_mode"] = True
        payload["runtime"]["tp"] = None
        payload["runtime"].pop("dp")
        payload["runtime"].pop("max_num_seqs")
        payload["runtime"]["max_num_batched_tokens"] = None
        result = precheck_module.precheck(payload)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["recommendation"]["status"], "available")
        self.assertEqual(result["inputs"]["max_num_batched_tokens"], 4096)
        self.assertTrue(result["inputs"]["recommendation_mode"])

    def test_reject_is_not_rewritten_as_pass(self) -> None:
        result = self.result(**{"model.weight_bytes": 900})
        self.assertEqual(result["status"], "reject")
        self.assertNotEqual(result["status"], "pass")

    def test_main_accepts_bom_and_writes_stdout_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "static-oom-input.json"
            input_path.write_text(
                "\ufeff" + json.dumps(self.payload),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = precheck_module.main(
                    ["--input", str(input_path)]
                )
            self.assertEqual(exit_code, 0)
            output = json.loads(stdout.getvalue())
            self.assertEqual(output["status"], "pass")
            self.assertFalse((Path(temp_dir) / "static-oom-report.json").exists())

    def test_main_returns_reject_exit_code_and_preserves_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.json"
            input_path.write_text(json.dumps({**self.payload, "model": {"weight_bytes": 900, "config": self.payload["model"]["config"]}}), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = precheck_module.main(
                    ["--input", str(input_path)]
                )
            self.assertEqual(exit_code, 3)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "reject")

    def test_main_reports_invalid_json_with_exit_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "invalid.json"
            input_path.write_text("{", encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = precheck_module.main(["--input", str(input_path)])
            self.assertEqual(exit_code, 2)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "unknown")
            self.assertIn("输入无效", result["message"])

    def test_main_accepts_stdin(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(self.payload))), redirect_stdout(stdout):
            exit_code = precheck_module.main(["--input", "-"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "pass")

    def test_stdin_does_not_create_input_or_report_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_artifact = Path(temp_dir) / "static-oom-input.json"
            output_path = Path(temp_dir) / "static-oom-report.json"
            stdout = io.StringIO()
            with patch("sys.stdin", io.StringIO(json.dumps(self.payload))), redirect_stdout(stdout):
                exit_code = precheck_module.main(["--input", "-"])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "pass")
            self.assertFalse(input_artifact.exists())
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
