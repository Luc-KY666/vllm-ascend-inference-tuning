from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import input_contract  # noqa: E402
from static_oom_precheck import precheck  # noqa: E402


HELP_TEXT = """usage: vllm serve [model_tag] [options]

options:
  --help                 show this help message and exit
  --host HOST
  --port PORT
  --served-model-name SERVED_MODEL_NAME
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
  --data-parallel-size DATA_PARALLEL_SIZE, -dp DATA_PARALLEL_SIZE
  --data-parallel-rank DATA_PARALLEL_RANK
  --data-parallel-address DATA_PARALLEL_ADDRESS
  --data-parallel-rpc-port DATA_PARALLEL_RPC_PORT
  --max-num-seqs MAX_NUM_SEQS
  --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
  --max-model-len MAX_MODEL_LEN
  --trust-remote-code
  --kv-transfer-config KV_TRANSFER_CONFIG
"""


class PdDeploymentTests(unittest.TestCase):
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 32,
        "intermediate_size": 16,
        "torch_dtype": "float16",
    }

    def _node_payload(
        self,
        root: Path,
        *,
        role: str,
        rank: int,
        devices: list[str],
        role_device_count: int,
        api_port: int,
        tp: int | None = None,
    ) -> dict[str, object]:
        model = root / "model"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        role_dp_size = role_device_count // len(devices)
        argv = [str(model), "--host", "0.0.0.0", "--port", str(api_port)]
        if tp is not None:
            argv += ["--tensor-parallel-size", str(tp)]
        argv += [
            "--data-parallel-size",
            str(role_dp_size),
            "--served-model-name",
            "pd-model",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "1",
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "32",
            "--trust-remote-code",
            "--kv-transfer-config",
            '{"kv_role":"kv_producer"}'
            if role == "prefill"
            else '{"kv_role":"kv_consumer"}',
        ]
        if role_dp_size > 1:
            argv += [
                "--data-parallel-rank",
                str(rank),
                "--data-parallel-address",
                "192.168.0.171",
                "--data-parallel-rpc-port",
                "12321",
            ]
        return {
            "serve": {"argv": argv},
            "model": {"config_path": str(model / "config.json")},
            "runtime": {"min_required_context_len": 4, "official_min_tp": 1},
            "resources": {
                "visible_devices": devices,
                "role_device_count": role_device_count,
                "port_candidates": [api_port, 30000 + rank],
                "port": api_port,
            },
            "outputs": {"attempt_dir": str(root / "attempt")},
        }

    def _validate(self, payload: dict[str, object]) -> dict[str, object]:
        return input_contract.validate_and_build(
            payload,
            help_text=HELP_TEXT,
            help_digest="pd-test-help-digest",
        )

    def _oom_payload(
        self,
        normalized: dict[str, object],
        *,
        card_memory_bytes: int,
        allowed_tp: list[int],
    ) -> dict[str, object]:
        runtime = normalized["runtime"]
        resources = normalized["resources"]
        return {
            "recommendation_mode": runtime["tp"] is None,
            "model": {
                "config": normalized["model"]["config"],
                "weight_bytes": 800,
            },
            "hardware": {
                "card_memory_bytes": card_memory_bytes,
                "device_count": len(resources["visible_devices"]),
            },
            "runtime": {
                "tp": runtime["tp"],
                "dp": 1,
                "max_num_seqs": runtime["max_num_seqs"],
                "max_num_batched_tokens": runtime["max_num_batched_tokens"],
                "gpu_memory_utilization": runtime["gpu_memory_utilization"],
                "min_required_context_len": 4,
                "block_size": 1,
                "dtype": "float16",
                "allowed_tp": allowed_tp,
                "official_min_tp": 1,
            },
        }

    def test_each_pd_node_uses_ordinary_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = {
                "p1": self._node_payload(
                    root,
                    role="prefill",
                    rank=0,
                    devices=["0", "1"],
                    role_device_count=4,
                    api_port=18080,
                ),
                "p2": self._node_payload(
                    root,
                    role="prefill",
                    rank=1,
                    devices=["2", "3"],
                    role_device_count=4,
                    api_port=18081,
                ),
                "d1": self._node_payload(
                    root,
                    role="decode",
                    rank=0,
                    devices=["4", "5"],
                    role_device_count=2,
                    api_port=19080,
                ),
            }
            normalized = {name: self._validate(payload) for name, payload in nodes.items()}

        self.assertEqual(set(normalized), {"p1", "p2", "d1"})
        self.assertEqual(normalized["p1"]["input_status"], "missing_parameters")
        self.assertEqual(normalized["p2"]["input_status"], "missing_parameters")
        self.assertEqual(normalized["d1"]["input_status"], "missing_parameters")
        self.assertEqual(normalized["p1"]["resources"]["role_device_count"], 4)
        self.assertEqual(normalized["p2"]["resources"]["visible_devices"], ["2", "3"])
        self.assertEqual(normalized["d1"]["resources"]["role_device_count"], 2)

    def test_each_node_gets_an_independent_static_oom_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p1 = self._validate(
                self._node_payload(
                    root,
                    role="prefill",
                    rank=0,
                    devices=["0", "1"],
                    role_device_count=4,
                    api_port=18080,
                )
            )
            p2 = self._validate(
                self._node_payload(
                    root,
                    role="prefill",
                    rank=1,
                    devices=["2", "3"],
                    role_device_count=4,
                    api_port=18081,
                )
            )
            p1_result = precheck(self._oom_payload(p1, card_memory_bytes=1200, allowed_tp=[1, 2]))
            p2_result = precheck(self._oom_payload(p2, card_memory_bytes=600, allowed_tp=[1, 2]))
            p1_resolved = input_contract.resolve_recommended_tp(
                p1, p1_result["recommendation"]["recommended_tp"], "static_oom"
            )
            p2_resolved = input_contract.resolve_recommended_tp(
                p2, p2_result["recommendation"]["recommended_tp"], "static_oom"
            )

        self.assertEqual(p1_result["recommendation"]["recommended_tp"], 1)
        self.assertEqual(p2_result["recommendation"]["recommended_tp"], 2)
        self.assertEqual(p1_resolved["runtime"]["tp"], 1)
        self.assertEqual(p2_resolved["runtime"]["tp"], 2)

    def test_rejected_node_blocks_pd_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = self._validate(
                self._node_payload(
                    root,
                    role="prefill",
                    rank=0,
                    devices=["0", "1"],
                    role_device_count=2,
                    api_port=18080,
                    tp=1,
                )
            )
            result = precheck(self._oom_payload(node, card_memory_bytes=500, allowed_tp=[1]))
        self.assertEqual(result["status"], "reject")

    def test_unavailable_recommendation_blocks_pd_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = self._validate(
                self._node_payload(
                    root,
                    role="prefill",
                    rank=0,
                    devices=["0", "1"],
                    role_device_count=2,
                    api_port=18080,
                )
            )
            result = precheck(self._oom_payload(node, card_memory_bytes=500, allowed_tp=[1]))
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["recommendation"]["status"], "unavailable")

    def test_role_tp_mismatch_blocks_launch(self) -> None:
        recommendations = {"p1": 1, "p2": 2}
        self.assertNotEqual(len(set(recommendations.values())), 1)


if __name__ == "__main__":
    unittest.main()
