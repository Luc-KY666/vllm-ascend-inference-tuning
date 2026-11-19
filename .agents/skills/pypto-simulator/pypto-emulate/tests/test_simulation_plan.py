from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NORMALIZE = ROOT / "pypto-profile" / "scripts" / "normalize_task_profile.py"
FIT = ROOT / "pypto-latency-model" / "scripts" / "fit_latency_models.py"
PLAN = ROOT / "pypto-emulate" / "scripts" / "build_simulation_plan.py"
CONTROL = ROOT / "references" / "qwen" / "qwen_control.py"
EXAMPLE = ROOT / "examples" / "qwen-decode-task-profile.json"


class SimulationPlanTests(unittest.TestCase):
    def test_preferred_capture_role_isolated_per_callable(self) -> None:
        spec = importlib.util.spec_from_file_location("simulation_plan", PLAN)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidates = [
            {"capture_role": "initialization_warmup_prefill", "duration_us": 100.0},
            {"capture_role": "request_prefill", "duration_us": 20.0},
        ]

        selected, used_preference = module._select_kernel_candidates(
            candidates, {"request_prefill"}
        )
        fallback, used_fallback_preference = module._select_kernel_candidates(
            candidates[:1], {"request_prefill"}
        )

        self.assertTrue(used_preference)
        self.assertEqual(selected, candidates[1:])
        self.assertFalse(used_fallback_preference)
        self.assertEqual(fallback, candidates[:1])

    def test_qwen_plan_has_constant_sampled_id_and_strict_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profile"
            model = root / "model"
            plan = root / "plan"
            subprocess.run(
                [
                    "python3",
                    str(NORMALIZE),
                    "--input",
                    str(EXAMPLE),
                    "--output",
                    str(profile),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "python3",
                    str(FIT),
                    "--manifest",
                    str(profile / "profile_manifest.json"),
                    "--min-workloads",
                    "1",
                    "--output",
                    str(model),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "python3",
                    str(PLAN),
                    "--profile-manifest",
                    str(profile / "profile_manifest.json"),
                    "--model-manifest",
                    str(model / "model_manifest.json"),
                    "--sampled-token-id",
                    "7",
                    "--strict",
                    "--require-baseline-confidence",
                    "--output",
                    str(plan),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            generated = json.loads((plan / "simulation_plan.json").read_text())
            native_plan = root / "native-plan"
            subprocess.run(
                [
                    "python3",
                    str(PLAN),
                    "--profile-manifest",
                    str(profile / "profile_manifest.json"),
                    "--model-manifest",
                    str(model / "model_manifest.json"),
                    "--sampled-token-id",
                    "7",
                    "--backend",
                    "a2a3",
                    "--strict",
                    "--require-baseline-confidence",
                    "--output",
                    str(native_plan),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            native_generated = json.loads(
                (native_plan / "simulation_plan.json").read_text()
            )

        self.assertTrue(generated["coverage"]["strict_ready"])
        self.assertEqual(generated["backend"], "a2a3sim")
        self.assertEqual(native_generated["backend"], "a2a3")
        self.assertEqual(
            native_generated["interception_layer"],
            "simpler_native_aicore_child_dispatch",
        )
        self.assertEqual(native_generated["coverage"], generated["coverage"])
        self.assertEqual(generated["coverage"]["runtime_callable_count"], 5)
        self.assertEqual(generated["coverage"]["runtime_callable_conflicts"], [])
        self.assertFalse(generated["coverage"]["npu_accessed"])
        self.assertEqual(generated["coverage"]["simulated_task_count"], 3)
        greedy = next(
            task for task in generated["tasks"] if task["task_id"] == "t2"
        )
        attention = next(
            task for task in generated["tasks"] if task["task_id"] == "t1"
        )
        self.assertEqual(greedy["control_outputs"][0]["token_id"], 7)
        self.assertEqual(
            attention["predicted_subslot_durations_us"],
            {"AIC": 41.0, "AIV0": 36.0, "AIV1": 35.0},
        )

    def test_materializes_only_first_sampled_id_column(self) -> None:
        spec = importlib.util.spec_from_file_location("qwen_control", CONTROL)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            module.materialize_sampled_ids(batch_size=2, width=4, token_id=9),
            [[9, 0, 0, 0], [9, 0, 0, 0]],
        )


if __name__ == "__main__":
    unittest.main()
