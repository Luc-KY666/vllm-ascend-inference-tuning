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
EXAMPLE = ROOT / "examples" / "qwen-decode-task-profile.json"


class FitLatencyModelsTests(unittest.TestCase):
    def test_fits_block_rank_medians_independently(self) -> None:
        spec = importlib.util.spec_from_file_location("fit_latency_models", FIT)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        record = module._point_record(
            [8.0, 9.0, 10.0],
            {"AIC": [8.0, 9.0, 10.0]},
            {"AIC": [[2.0, 8.0], [3.0, 9.0], [4.0, 10.0]]},
        )

        self.assertEqual(
            record["subslot_models"]["AIC"][
                "block_rank_median_durations_us"
            ],
            [3.0, 9.0],
        )
        self.assertEqual(
            record["subslot_models"]["AIC"][
                "block_distribution_sample_count"
            ],
            3,
        )

    def test_builds_exact_median_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profile"
            model = root / "model"
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
            manifest = json.loads((model / "model_manifest.json").read_text())
            lookup = json.loads((model / "latency_lookup.json").read_text())

        self.assertEqual(manifest["status"], "baseline")
        self.assertEqual(manifest["minimum_independent_workloads_per_point"], 1)
        self.assertEqual(manifest["group_count"], 3)
        self.assertEqual(len(lookup["durations_us"]), 3)
        self.assertEqual(len(lookup["subslot_durations_us"]), 3)
        medians = sorted(
            group["median_duration_us"] for group in manifest["groups"]
        )
        self.assertEqual(medians, [3.0, 7.0, 41.0])
        mix_group = next(
            group
            for group in manifest["groups"]
            if group["signature"]["resource_shape"] == "MIX"
        )
        self.assertEqual(
            {
                slot: model["median_duration_us"]
                for slot, model in mix_group["subslot_models"].items()
            },
            {"AIC": 41.0, "AIV0": 36.0, "AIV1": 35.0},
        )


if __name__ == "__main__":
    unittest.main()
