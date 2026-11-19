#!/usr/bin/env python3
"""Fit per-feature-point median latency models for PyPTO tasks."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

COMMON = Path(__file__).resolve().parents[2] / "pypto-common"
sys.path.insert(0, str(COMMON))

from pypto_contract import (
    ContractError,
    canonical_json,
    latency_group_key,
    latency_group_signature,
    latency_key,
    read_json,
    validate_profile_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument(
        "--min-workloads",
        type=int,
        default=2,
        help="minimum independent capture/workload identities per feature point",
    )
    return parser.parse_args()


def _point_record(
    samples: list[float],
    slot_samples: dict[str, list[float]],
    block_samples: dict[str, list[list[float]]],
) -> dict[str, Any]:
    subslot_models = {}
    for slot, values in sorted(slot_samples.items()):
        if not values:
            continue
        model = {
            "sample_count": len(values),
            "median_duration_us": float(statistics.median(values)),
            "min_duration_us": min(values),
            "max_duration_us": max(values),
        }
        vectors = block_samples.get(slot, [])
        if vectors and len({len(vector) for vector in vectors}) == 1:
            model["block_rank_median_durations_us"] = [
                float(statistics.median(vector[index] for vector in vectors))
                for index in range(len(vectors[0]))
            ]
            model["block_distribution_sample_count"] = len(vectors)
        subslot_models[slot] = model
    return {
        "sample_count": len(samples),
        "median_duration_us": float(statistics.median(samples)),
        "min_duration_us": min(samples),
        "max_duration_us": max(samples),
        "subslot_models": subslot_models,
    }


def main() -> int:
    args = parse_args()
    if args.min_samples < 1 or args.min_workloads < 1:
        print("error: --min-samples and --min-workloads must be positive", file=sys.stderr)
        return 2

    groups: dict[str, dict[str, Any]] = {}
    point_samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    point_slots: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    point_blocks: dict[tuple[str, str], dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    point_features: dict[tuple[str, str], dict[str, Any]] = {}
    point_latency_keys: dict[tuple[str, str], str] = {}
    point_workloads: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_manifests: list[str] = []

    try:
        for path in args.manifest:
            manifest = read_json(path)
            validate_profile_manifest(manifest)
            source_manifests.append(str(path.resolve()))
            source = manifest.get("source", {})
            workload_id = str(source.get("workload_id", "")).strip()
            if source.get("npu_measured", False) and not workload_id:
                raise ContractError(f"NPU manifest has no workload_id: {path}")
            if not workload_id:
                workload_id = f"manifest:{path.resolve()}"
            for task in manifest["tasks"]:
                samples = [float(value) for value in task["duration_samples_us"]]
                if not samples:
                    continue
                group_key = latency_group_key(task, manifest["model"])
                signature = latency_group_signature(task, manifest["model"])
                if group_key in groups and groups[group_key]["signature"] != signature:
                    raise ContractError(f"latency-group-key collision for {group_key}")
                groups[group_key] = {"signature": signature}
                features = task.get("features", {})
                feature_key = canonical_json(features)
                point_id = (group_key, feature_key)
                point_features[point_id] = features
                point_latency_keys[point_id] = latency_key(task, manifest["model"])
                point_workloads[point_id].add(workload_id)
                point_samples[point_id].extend(samples)
                for slot, values in task["subslot_duration_samples_us"].items():
                    point_slots[point_id][slot].extend(float(value) for value in values)
                for slot, epochs in task.get(
                    "subslot_block_duration_samples_us", {}
                ).items():
                    point_blocks[point_id][slot].extend(
                        [[float(value) for value in epoch] for epoch in epochs]
                    )
    except (OSError, ValueError, ContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    output_groups: list[dict[str, Any]] = []
    exact_lookup: dict[str, float] = {}
    exact_slot_lookup: dict[str, dict[str, float]] = {}
    low_confidence_points = 0
    for group_key in sorted(groups):
        points = []
        for point_id in sorted((item for item in point_samples if item[0] == group_key), key=lambda item: item[1]):
            record = _point_record(
                point_samples[point_id],
                point_slots[point_id],
                point_blocks[point_id],
            )
            workload_ids = sorted(point_workloads[point_id])
            confidence = (
                "measured"
                if record["sample_count"] >= args.min_samples
                and len(workload_ids) >= args.min_workloads
                else "low_confidence"
            )
            record.update(
                {
                    "latency_key": point_latency_keys[point_id],
                    "features": point_features[point_id],
                    "workload_count": len(workload_ids),
                    "workload_ids": workload_ids,
                    "confidence": confidence,
                }
            )
            low_confidence_points += record["confidence"] == "low_confidence"
            points.append(record)
            exact_lookup[record["latency_key"]] = record["median_duration_us"]
            exact_slot_lookup[record["latency_key"]] = {
                slot: model["median_duration_us"] for slot, model in record["subslot_models"].items()
            }

        all_samples = [sample for point_id, samples in point_samples.items() if point_id[0] == group_key for sample in samples]
        representative = points[0]
        output_groups.append(
            {
                "latency_group_key": group_key,
                "signature": groups[group_key]["signature"],
                "feature_point_count": len(points),
                "sample_count": len(all_samples),
                "median_duration_us": float(statistics.median(all_samples)),
                "min_duration_us": min(all_samples),
                "max_duration_us": max(all_samples),
                "subslot_models": representative["subslot_models"] if len(points) == 1 else {},
                "confidence": (
                    "measured" if all(point["confidence"] == "measured" for point in points) else "low_confidence"
                ),
                "estimator": "feature_point_median_idw",
                "points": points,
            }
        )

    model_manifest = {
        "schema_version": 1,
        "producer": "pypto-latency-model",
        "estimator": "feature_point_median_idw",
        "interpolation": "inverse_distance_within_measured_bounds",
        "extrapolation": "reject",
        "status": "baseline" if output_groups and low_confidence_points == 0 else "incomplete",
        "source_manifests": source_manifests,
        "group_count": len(output_groups),
        "feature_point_count": len(point_samples),
        "low_confidence_point_count": low_confidence_points,
        "low_confidence_group_count": sum(group["confidence"] == "low_confidence" for group in output_groups),
        "minimum_samples_per_point": args.min_samples,
        "minimum_independent_workloads_per_point": args.min_workloads,
        "groups": output_groups,
    }
    write_json(args.output / "model_manifest.json", model_manifest)
    write_json(
        args.output / "latency_lookup.json",
        {
            "schema_version": 1,
            "lookup_type": "exact_feature_point_median",
            "durations_us": exact_lookup,
            "subslot_durations_us": exact_slot_lookup,
        },
    )
    print(args.output / "model_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
