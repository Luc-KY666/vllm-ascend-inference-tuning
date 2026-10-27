import json
import tempfile
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from pd_input_contract import validate_pd  # noqa: E402
from pd_static_oom_precheck import precheck_pd  # noqa: E402


class PdDeploymentTests(unittest.TestCase):
    def _payload(self, root: Path, **pd_overrides):
        (root / "config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "vocab_size": 4,
                    "intermediate_size": 4,
                    "torch_dtype": "float16",
                }
            ),
            encoding="utf-8",
        )
        pd = {
            "prefill": {"devices": [0], "tp": 1, "api_port": 18080, "kv_port": 30000},
            "decode": {"devices": [1], "tp": 1, "api_port": 18081, "kv_port": 30200},
            "proxy": {"port": 18000},
        }
        pd.update(pd_overrides)
        return {
            "deployment": {"mode": "single_node_pd", "connector": "MooncakeConnectorV1"},
            "model": {"model_path": str(root), "served_model_name": "model"},
            "resources": {
                "visible_devices": [0, 1],
                "port_candidates": [18000, 18080, 18081, 30000, 30200],
            },
            "pd": pd,
            "outputs": {"attempt_dir": str(root / "attempt")},
        }

    def test_pd_contract_accepts_disjoint_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            result = validate_pd(self._payload(Path(directory)))
        self.assertEqual(result["input_status"], "pass")
        self.assertEqual(result["pd"]["prefill"]["dp"], 1)

    def test_pd_contract_rejects_device_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self._payload(
                Path(directory),
                decode={"devices": [0], "tp": 1, "api_port": 18081, "kv_port": 30200},
            )
            result = validate_pd(payload)
        self.assertEqual(result["input_status"], "invalid")
        self.assertTrue(any("设备重叠" in error for error in result["diagnostics"]["errors"]))

    def test_pd_oom_requires_both_roles(self):
        normalized = {
            "model": {
                "config": {
                    "hidden_size": 4,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "vocab_size": 4,
                    "intermediate_size": 4,
                    "torch_dtype": "float16",
                },
                "weight_bytes": 1,
            },
            "runtime": {
                "max_num_seqs": 1,
                "max_num_batched_tokens": 1,
                "gpu_memory_utilization": 0.9,
                "min_required_context_len": 1,
                "block_size": 1,
            },
            "pd": {
                "prefill": {"devices": ["0"], "tp": None, "allowed_tp": [1]},
                "decode": {"devices": ["1"], "tp": None, "allowed_tp": [1]},
            },
        }
        result = precheck_pd(
            {
                "normalized": normalized,
                "hardware": {
                    "prefill": {"device_count": 1, "card_memory_bytes": 1024**3},
                    "decode": {"device_count": 1, "card_memory_bytes": 1024**3},
                },
            }
        )
        self.assertEqual(result["aggregate"]["status"], "pass")
        self.assertEqual(result["effective"]["prefill"]["tp"], 1)
        self.assertEqual(result["effective"]["decode"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
