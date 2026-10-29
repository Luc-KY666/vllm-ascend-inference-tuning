from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import input_contract  # noqa: E402


HELP_TEXT = """INFO plugin output
usage: vllm serve [model_tag] [options]

options:
  --help                 show this help message and exit
  --config CONFIG        Read options from a config file.
  --model MODEL          Alternate model source.
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
  --tokenizer TOKENIZER
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
  --data-parallel-size DATA_PARALLEL_SIZE, -dp DATA_PARALLEL_SIZE
  --data-parallel-rank DATA_PARALLEL_RANK
  --data-parallel-address DATA_PARALLEL_ADDRESS
  --data-parallel-rpc-port DATA_PARALLEL_RPC_PORT
  --max-num-seqs MAX_NUM_SEQS
  --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
  --max-model-len MAX_MODEL_LEN
  --block-size BLOCK_SIZE
  --dtype {auto,bfloat16,float16}
  --quantization QUANTIZATION, -q QUANTIZATION
  --port PORT
  --enable-expert-parallel, --no-enable-expert-parallel
  --enable-prefix-caching, --no-enable-prefix-caching
  --async-scheduling, --no-async-scheduling
  --distributed-executor-backend BACKEND
  --compilation-config COMPILATION_CONFIG
  --additional-config ADDITIONAL_CONFIG
  --api-key API_KEY [API_KEY ...]
  --hf-token [HF_TOKEN]
"""


class InputContractTests(unittest.TestCase):
    def payload(self, root: Path, argv: list[str] | None = None, **runtime: object) -> dict[str, object]:
        model = root / "model"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text(
            json.dumps({"num_attention_heads": 8, "hidden_size": 16}),
            encoding="utf-8",
        )
        if argv is None:
            argv = self.argv(model)
        return {
            "serve": {"argv": argv},
            "runtime": runtime,
            "resources": {
                "visible_devices": ["0", "1"],
                "port_candidates": [18117],
            },
            "outputs": {"attempt_dir": str(root / "attempt")},
        }

    def argv(self, model: Path, *, include_tp: bool = True) -> list[str]:
        argv = [str(model), "--served-model-name", "test-model"]
        if include_tp:
            argv += ["--tensor-parallel-size", "2"]
        argv += [
            "--data-parallel-size", "1",
            "--max-num-seqs", "32",
            "--max-num-batched-tokens", "4096",
            "--gpu-memory-utilization", "0.9",
        ]
        return argv

    def validate(self, payload: dict[str, object]) -> dict[str, object]:
        return input_contract.validate_and_build(
            payload,
            help_text=HELP_TEXT,
            help_digest="test-help-digest",
        )

    def test_complete_json_argv_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.validate(self.payload(Path(directory)))
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["schema_version"], "vllm-ascend-deployment-input-v3")
        self.assertEqual(result["runtime"]["tp"], 2)
        self.assertEqual(result["runtime"]["dp"], 1)
        self.assertEqual(result["runtime"]["max_num_seqs"], 32)
        self.assertEqual(result["runtime"]["max_num_batched_tokens"], 4096)
        self.assertEqual(result["serve"]["help_digest"], "test-help-digest")

    def test_user_extra_launch_flags_are_validated_by_help(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"num_attention_heads": 8, "hidden_size": 16}),
                encoding="utf-8",
            )
            argv = self.argv(model) + [
                "--enable-expert-parallel",
                "--distributed_executor_backend", "mp",
                "--no-enable-prefix-caching",
                "--async-scheduling",
                "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
                "--additional-config", '{"enable_flashcomm1":true,"weight_nz_mode":2}',
            ]
            result = self.validate(self.payload(root, argv=argv))
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["serve"]["argv"][-1], '{"enable_flashcomm1":true,"weight_nz_mode":2}')

    def test_help_option_aliases_and_inline_values_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"num_attention_heads": 8}), encoding="utf-8"
            )
            argv = [
                str(model), "--served-model-name=test-model", "-tp=2", "-dp", "1",
                "--max-num-seqs=32", "--max-num-batched-tokens=4096",
                "--gpu-memory-utilization=0.95",
            ]
            result = self.validate(self.payload(root, argv=argv))
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["runtime"]["tp"], 2)
        self.assertEqual(result["runtime"]["gpu_memory_utilization"], 0.95)

    def test_unknown_option_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.validate(self.payload(root, argv=self.argv(root / "model") + ["--not-in-help"]))
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("不支持参数" in error for error in result["diagnostics"]["errors"]))

    def test_missing_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self.argv(root / "model")[:-1] + ["--gpu-memory-utilization"]
            result = self.validate(self.payload(root, argv=argv))
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("缺少 value" in error for error in result["diagnostics"]["errors"]))

    def test_non_hard_option_value_is_not_interpreted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.validate(
                self.payload(
                    root,
                    argv=self.argv(root / "model")
                    + [
                        "--enable-expert-parallel=true",
                        "--dtype",
                        "not-a-vllm-dtype",
                        "--max-model-len",
                        "auto",
                        "--additional-config",
                        "--compilation-config",
                        "not-json",
                    ],
                )
            )
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["runtime"]["dtype"], "not-a-vllm-dtype")
        self.assertEqual(result["runtime"]["max_model_len"], "auto")

    def test_hidden_parameter_sources_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.validate(self.payload(root, argv=self.argv(root / "model") + ["--config", "config.yaml"]))
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("第二份启动参数来源" in error for error in result["diagnostics"]["errors"]))

    def test_duplicate_hard_parameter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.validate(self.payload(root, argv=self.argv(root / "model") + ["-tp", "2"]))
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("runtime.tp 不能重复" in error for error in result["diagnostics"]["errors"]))

    def test_original_hard_constraints_are_preserved(self) -> None:
        cases = (
            ("batch", "--max-num-batched-tokens", "16"),
            ("devices", "--tensor-parallel-size", "4"),
            ("heads", "--tensor-parallel-size", "3"),
            ("utilization", "--gpu-memory-utilization", "1.1"),
        )
        for label, flag, value in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = self.argv(root / "model")
                argv[argv.index(flag) + 1] = value
                result = self.validate(self.payload(root, argv=argv))
                self.assertEqual(result["input_status"], "invalid")

    def test_model_config_is_read_once_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_read_bytes = Path.read_bytes
            with patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=lambda path: original_read_bytes(path),
            ) as read_bytes:
                result = self.validate(self.payload(root))
            model = root / "model"
            config_digest = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(read_bytes.call_count, 1)
        self.assertEqual(result["model"]["config_digest"], config_digest)
        self.assertEqual(result["model"]["config"]["num_attention_heads"], 8)

    def test_missing_tp_keeps_recommendation_status_and_resolve_updates_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"num_attention_heads": 8}), encoding="utf-8"
            )
            normalized = self.validate(
                self.payload(root, argv=self.argv(model, include_tp=False))
            )
            result = input_contract.resolve_recommended_tp(normalized, 2, "static_oom")
        self.assertEqual(normalized["input_status"], "missing_parameters")
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["runtime"]["tp"], 2)
        self.assertEqual(result["serve"]["argv"][-2:], ["--tensor-parallel-size", "2"])

    def test_role_device_count_allows_role_dp_with_local_devices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            argv = self.argv(model)
            argv[argv.index("--tensor-parallel-size") + 1] = "1"
            argv[argv.index("--data-parallel-size") + 1] = "2"
            payload = self.payload(root, argv=argv)
            payload["resources"] = {
                "visible_devices": ["0"],
                "role_device_count": 2,
                "port_candidates": [18117],
            }
            result = self.validate(payload)
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["resources"]["role_device_count"], 2)

    def test_role_device_count_rejects_excessive_role_dp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            argv = self.argv(model)
            argv[argv.index("--tensor-parallel-size") + 1] = "1"
            argv[argv.index("--data-parallel-size") + 1] = "2"
            payload = self.payload(root, argv=argv)
            payload["resources"] = {
                "visible_devices": ["0"],
                "role_device_count": 1,
                "port_candidates": [18117],
            }
            result = self.validate(payload)
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("角色可用设备数量" in error for error in result["diagnostics"]["errors"]))

    def test_local_tp_is_checked_separately_from_role_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            argv = self.argv(model)
            argv[argv.index("--data-parallel-size") + 1] = "2"
            payload = self.payload(root, argv=argv)
            payload["resources"] = {
                "visible_devices": ["0"],
                "role_device_count": 2,
                "port_candidates": [18117],
            }
            result = self.validate(payload)
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("当前实例可见设备数量" in error for error in result["diagnostics"]["errors"]))

    def test_resolve_recommended_tp_uses_role_device_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            argv = self.argv(model, include_tp=False)
            argv[argv.index("--data-parallel-size") + 1] = "2"
            payload = self.payload(root, argv=argv)
            payload["resources"] = {
                "visible_devices": ["0"],
                "role_device_count": 2,
                "port_candidates": [18117],
            }
            normalized = self.validate(payload)
            result = input_contract.resolve_recommended_tp(normalized, 1, "static_oom")
        self.assertEqual(normalized["input_status"], "missing_parameters")
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["runtime"]["tp"], 1)

    def test_missing_role_device_count_keeps_ordinary_device_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            argv = self.argv(model)
            argv[argv.index("--tensor-parallel-size") + 1] = "1"
            argv[argv.index("--data-parallel-size") + 1] = "2"
            payload = self.payload(root, argv=argv)
            payload["resources"] = {
                "visible_devices": ["0"],
                "port_candidates": [18117],
            }
            result = self.validate(payload)
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("角色可用设备数量" in error for error in result["diagnostics"]["errors"]))

    def test_invalid_model_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.validate(self.payload(root, argv=self.argv(root / "missing")))
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(result["diagnostics"]["errors"])

    def test_invalid_help_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(input_contract.ContractError):
                input_contract.validate_and_build(self.payload(root), help_text="not a vllm help")

    def test_cli_reads_json_and_help_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.payload(root)
            payload["model"] = {"weight_bytes": 123}
            input_path = root / "deployment-input.json"
            help_path = root / "vllm-serve-help-all.txt"
            runtime_path = root / ".work" / "runtime.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            help_path.write_text(HELP_TEXT, encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = input_contract.main(
                    [
                        "--input",
                        str(input_path),
                        "--serve-help-file",
                        str(help_path),
                        "--runtime-output",
                        str(runtime_path),
                    ]
                )
            result = json.loads(output.getvalue())
            runtime_result = json.loads(runtime_path.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["serve"]["help_digest"], hashlib.sha256(HELP_TEXT.encode()).hexdigest())
        self.assertEqual(runtime_result["schema_version"], "vllm-ascend-runtime-v1")
        self.assertEqual(runtime_result["runtime"]["tp"], 2)
        self.assertEqual(runtime_result["resources"]["visible_devices"], ["0", "1"])
        self.assertEqual(runtime_result["model"]["weight_bytes"], 123)
        self.assertNotIn("serve", runtime_result)

    def test_cli_writes_runtime_for_missing_tp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.payload(root, argv=self.argv(root / "model", include_tp=False))
            input_path = root / "deployment-input.json"
            help_path = root / "vllm-serve-help-all.txt"
            runtime_path = root / ".work" / "runtime.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            help_path.write_text(HELP_TEXT, encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = input_contract.main(
                    [
                        "--input",
                        str(input_path),
                        "--serve-help-file",
                        str(help_path),
                        "--runtime-output",
                        str(runtime_path),
                    ]
                )
            result = json.loads(output.getvalue())
            runtime_result = json.loads(runtime_path.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(result["input_status"], "missing_parameters")
        self.assertIsNone(runtime_result["runtime"]["tp"])

    def test_invalid_cli_input_does_not_write_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.payload(root)
            payload["serve"]["argv"].append("--not-in-help")
            input_path = root / "deployment-input.json"
            help_path = root / "vllm-serve-help-all.txt"
            runtime_path = root / ".work" / "runtime.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            help_path.write_text(HELP_TEXT, encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = input_contract.main(
                    [
                        "--input",
                        str(input_path),
                        "--serve-help-file",
                        str(help_path),
                        "--runtime-output",
                        str(runtime_path),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertFalse(runtime_path.exists())

    def test_cli_requires_help_for_normal_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "deployment-input.json"
            input_path.write_text(json.dumps(self.payload(root)), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = input_contract.main(["--input", str(input_path)])
            result = json.loads(output.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(result["input_status"], "invalid")
        self.assertIn("--serve-help-file", result["diagnostics"]["errors"][0])


if __name__ == "__main__":
    unittest.main()
