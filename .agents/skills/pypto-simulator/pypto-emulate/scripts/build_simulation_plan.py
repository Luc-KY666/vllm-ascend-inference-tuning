#!/usr/bin/env python3
"""Build a sleep-only PyPTO task simulation and strict coverage plan."""

from __future__ import annotations

import argparse
import math
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
    latency_key,
    read_json,
    sampled_id_outputs,
    should_intercept,
    validate_profile_manifest,
    write_json,
)

SLOT_ORDER = ("AIC", "AIV0", "AIV1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-manifest", type=Path, action="append", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampled-token-id", type=int, default=1)
    parser.add_argument(
        "--prefer-capture-role",
        action="append",
        default=[],
        help=(
            "prefer candidates with this exact features.capture_role when a callable "
            "also appears in other lifecycle phases; repeat for multiple roles"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("a2a3sim", "a2a3"),
        default="a2a3sim",
        help="target CPU-sim child stubs or native AIC/AIV device delay stubs",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-baseline-confidence", action="store_true")
    return parser.parse_args()


def _model_groups(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("groups"), list):
        raise ContractError("invalid model manifest")
    groups = {}
    for group in manifest["groups"]:
        key = str(group.get("latency_group_key", ""))
        if not key or key in groups:
            raise ContractError("model groups require unique latency_group_key values")
        if not isinstance(group.get("points"), list) or not group["points"]:
            raise ContractError(f"model group {key} has no feature points")
        groups[key] = group
    return groups


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _weighted_value(points: list[dict[str, Any]], weights: list[float], field: str) -> float:
    return sum(float(point[field]) * weight for point, weight in zip(points, weights)) / sum(weights)


def _predict(group: dict[str, Any], features: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    points = group["points"]
    feature_key = canonical_json(features)
    exact = next((point for point in points if canonical_json(point.get("features", {})) == feature_key), None)
    if exact is not None:
        return {
            "duration_us": float(exact["median_duration_us"]),
            "subslots": {
                slot: float(model["median_duration_us"])
                for slot, model in exact.get("subslot_models", {}).items()
            },
            "block_ranks": {
                slot: [
                    float(value)
                    for value in model.get("block_rank_median_durations_us", [])
                ]
                for slot, model in exact.get("subslot_models", {}).items()
                if model.get("block_rank_median_durations_us")
            },
            "low_confidence": exact.get("confidence") == "low_confidence",
            "source_points": [exact.get("latency_key")],
        }, "exact"

    if not points or any(set(point.get("features", {})) != set(features) for point in points):
        return None, "feature_schema_mismatch"

    numeric_names = [name for name, value in features.items() if _numeric(value)]
    categorical_names = [name for name in features if name not in numeric_names]
    candidates = [
        point
        for point in points
        if all(point["features"].get(name) == features[name] for name in categorical_names)
        and all(_numeric(point["features"].get(name)) for name in numeric_names)
    ]
    if not candidates:
        return None, "categorical_feature_miss"

    ranges: dict[str, tuple[float, float]] = {}
    for name in numeric_names:
        values = [float(point["features"][name]) for point in candidates]
        low, high = min(values), max(values)
        requested = float(features[name])
        if requested < low or requested > high:
            return None, "extrapolation_rejected"
        if low == high and requested != low:
            return None, "extrapolation_rejected"
        ranges[name] = (low, high)

    if not numeric_names:
        return None, "no_numeric_interpolation_axis"

    ranked: list[tuple[float, dict[str, Any]]] = []
    for point in candidates:
        distance_sq = 0.0
        for name in numeric_names:
            low, high = ranges[name]
            scale = high - low
            if scale:
                distance_sq += ((float(point["features"][name]) - float(features[name])) / scale) ** 2
        ranked.append((math.sqrt(distance_sq), point))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[: min(4, len(ranked))]
    if not selected or selected[0][0] == 0:
        return None, "ambiguous_feature_point"
    selected_points = [point for _, point in selected]
    weights = [1.0 / max(distance, 1e-12) for distance, _ in selected]
    common_slots = set.intersection(
        *(set(point.get("subslot_models", {})) for point in selected_points)
    ) if selected_points else set()
    block_ranks = {}
    for slot in sorted(common_slots):
        vectors = [
            point["subslot_models"][slot].get(
                "block_rank_median_durations_us", []
            )
            for point in selected_points
        ]
        if vectors and vectors[0] and len({len(vector) for vector in vectors}) == 1:
            block_ranks[slot] = [
                sum(float(vector[index]) * weight for vector, weight in zip(vectors, weights))
                / sum(weights)
                for index in range(len(vectors[0]))
            ]
    return {
        "duration_us": _weighted_value(selected_points, weights, "median_duration_us"),
        "subslots": {
            slot: sum(
                float(point["subslot_models"][slot]["median_duration_us"]) * weight
                for point, weight in zip(selected_points, weights)
            ) / sum(weights)
            for slot in sorted(common_slots)
        },
        "block_ranks": block_ranks,
        "low_confidence": any(point.get("confidence") == "low_confidence" for point in selected_points),
        "source_points": [point.get("latency_key") for point in selected_points],
    }, "interpolated"


def _control_directives(task: dict[str, Any], token_id: int) -> list[dict[str, Any]]:
    directives = []
    for output in sampled_id_outputs(task):
        directive = {
            "output": output["name"],
            "strategy": "constant_token",
            "token_id": token_id,
            "semantic_columns": 1,
            "padding_value": 0,
        }
        if "arg_index" in output:
            directive["arg_index"] = int(output["arg_index"])
        directives.append(directive)
    return directives


def _select_kernel_candidates(
    candidates: list[dict[str, Any]], preferred_roles: set[str]
) -> tuple[list[dict[str, Any]], bool]:
    preferred = [
        candidate
        for candidate in candidates
        if candidate.get("capture_role") in preferred_roles
    ]
    return (preferred, True) if preferred else (candidates, False)


def main() -> int:
    args = parse_args()
    if args.sampled_token_id < 0:
        print("error: --sampled-token-id must be non-negative", file=sys.stderr)
        return 2
    try:
        profiles = [read_json(path) for path in args.profile_manifest]
        model = read_json(args.model_manifest)
        for profile in profiles:
            validate_profile_manifest(profile)
        if any(profile["model"] != profiles[0]["model"] for profile in profiles[1:]):
            raise ContractError("profile manifests must describe the same model")
        groups = _model_groups(model)
    except (OSError, ValueError, ContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    plan_tasks = []
    kernel_candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    low_confidence: list[str] = []
    extrapolation_rejections: list[str] = []
    control_directive_count = 0
    runtime_unmapped: list[str] = []

    for profile in profiles:
        for task in profile["tasks"]:
            program = task.get("program", "default")
            task_ref = f"{program}:{task['task_id']}"
            intercept = should_intercept(task)
            group_key = latency_group_key(task, profile["model"])
            group = groups.get(group_key) if intercept else None
            prediction = None
            prediction_kind = "preserved"
            action = "preserve"
            if intercept and group is None:
                action = "missing_model"
                missing.append(task_ref)
            elif intercept:
                prediction, prediction_kind = _predict(group, task.get("features", {}))
                if prediction is None:
                    action = "missing_model"
                    missing.append(task_ref)
                    if prediction_kind == "extrapolation_rejected":
                        extrapolation_rejections.append(task_ref)
                else:
                    action = "simulate"
                    if prediction["low_confidence"]:
                        low_confidence.append(task_ref)

            directives = _control_directives(task, args.sampled_token_id)
            control_directive_count += len(directives)
            predicted = None if prediction is None else float(prediction["duration_us"])
            predicted_subslots = {} if prediction is None else prediction["subslots"]
            if action == "simulate":
                mapped = False
                for slot, func_id in zip(SLOT_ORDER, task.get("kernel_ids", [-1, -1, -1])):
                    if int(func_id) < 0:
                        continue
                    mapped = True
                    controls = [directive for directive in directives if "arg_index" in directive]
                    kernel_candidates[(program, int(func_id))].append(
                        {
                            "task_id": task["task_id"],
                            "task_name": task["task_name"],
                            "subslot": slot,
                            "duration_us": float(predicted_subslots.get(slot, predicted)),
                            "block_durations_us": [
                                float(value)
                                for value in prediction.get("block_ranks", {}).get(slot, [])
                            ],
                            "capture_role": str(
                                task.get("features", {}).get("capture_role", "")
                            ),
                            "control_outputs": controls,
                        }
                    )
                if not mapped:
                    runtime_unmapped.append(task_ref)

            plan_tasks.append(
                {
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "program": program,
                    "action": action,
                    "latency_key": latency_key(task, profile["model"]) if intercept else None,
                    "latency_group_key": group_key if intercept else None,
                    "prediction_kind": prediction_kind,
                    "prediction_source_points": [] if prediction is None else prediction["source_points"],
                    "predicted_duration_us": predicted,
                    "predicted_subslot_durations_us": predicted_subslots,
                    "control_outputs": directives,
                }
            )

    kernel_models: list[dict[str, Any]] = []
    runtime_conflicts: list[str] = []
    preferred_roles = set(args.prefer_capture_role)
    preferred_candidate_count = 0
    for (program, func_id), all_candidates in sorted(kernel_candidates.items()):
        candidates, used_preference = _select_kernel_candidates(
            all_candidates, preferred_roles
        )
        if used_preference:
            preferred_candidate_count += len(candidates)
        controls_by_value = {
            canonical_json(candidate["control_outputs"]): candidate["control_outputs"]
            for candidate in candidates
        }
        if len(controls_by_value) != 1:
            runtime_conflicts.append(f"{program}:{func_id}:control_outputs")
            continue
        durations = [float(candidate["duration_us"]) for candidate in candidates]
        block_durations_by_subslot = {}
        for slot in sorted(
            {str(candidate["subslot"]) for candidate in candidates},
            key=SLOT_ORDER.index,
        ):
            slot_candidates = [
                candidate for candidate in candidates if candidate["subslot"] == slot
            ]
            vectors = [candidate["block_durations_us"] for candidate in slot_candidates]
            if (
                vectors
                and all(vectors)
                and len({len(vector) for vector in vectors}) == 1
            ):
                block_durations_by_subslot[slot] = [
                    float(statistics.median(vector[index] for vector in vectors))
                    for index in range(len(vectors[0]))
                ]
        kernel_models.append(
            {
                "program": program,
                "func_id": func_id,
                "duration_us": float(statistics.median(durations)),
                "min_duration_us": min(durations),
                "max_duration_us": max(durations),
                "candidate_count": len(candidates),
                "input_candidate_count": len(all_candidates),
                "aggregation": (
                    "median_preferred_capture_role"
                    if used_preference
                    else "median_same_callable_feature"
                ),
                "selected_capture_roles": sorted(
                    {str(candidate.get("capture_role", "")) for candidate in candidates}
                ),
                "excluded_capture_roles": sorted(
                    {
                        str(candidate.get("capture_role", ""))
                        for candidate in all_candidates
                    }
                    - {str(candidate.get("capture_role", "")) for candidate in candidates}
                ),
                "block_duration_us_by_subslot": block_durations_by_subslot,
                "task_ids": sorted({str(candidate["task_id"]) for candidate in candidates}),
                "task_names": sorted({str(candidate["task_name"]) for candidate in candidates}),
                "subslots": sorted(
                    {str(candidate["subslot"]) for candidate in candidates},
                    key=SLOT_ORDER.index,
                ),
                "control_outputs": next(iter(controls_by_value.values())),
            }
        )

    strict_ready = (
        not missing
        and not runtime_conflicts
        and (not args.require_baseline_confidence or not low_confidence)
    )
    runtime_stub_ready = strict_ready and not runtime_unmapped and all(
        not task["control_outputs"] or all("arg_index" in item for item in task["control_outputs"])
        for task in plan_tasks
        if task["action"] == "simulate"
    )
    coverage = {
        "schema_version": 1,
        "task_count": len(plan_tasks),
        "simulated_task_count": sum(task["action"] == "simulate" for task in plan_tasks),
        "preserved_task_count": sum(task["action"] == "preserve" for task in plan_tasks),
        "missing_model_task_ids": missing,
        "low_confidence_task_ids": low_confidence,
        "extrapolation_rejected_task_ids": extrapolation_rejections,
        "runtime_unmapped_task_ids": runtime_unmapped,
        "runtime_callable_conflicts": runtime_conflicts,
        "runtime_callable_count": len(kernel_models),
        "collapsed_kernel_candidate_count": sum(len(values) for values in kernel_candidates.values()),
        "preferred_kernel_candidate_count": preferred_candidate_count,
        "control_directive_count": control_directive_count,
        "strict_ready": strict_ready,
        "runtime_stub_ready": runtime_stub_ready,
        "npu_accessed": False,
    }
    plan = {
        "schema_version": 1,
        "producer": "pypto-emulate",
        "execution_mode": "sleep",
        "backend": args.backend,
        "interception_layer": (
            "simpler_cpu_sim_kernel_dispatch"
            if args.backend == "a2a3sim"
            else "simpler_native_aicore_child_dispatch"
        ),
        "model": profiles[0]["model"],
        "source_profile_manifests": [str(path.resolve()) for path in args.profile_manifest],
        "sampled_token_id": args.sampled_token_id,
        "preferred_capture_roles": sorted(preferred_roles),
        "tasks": plan_tasks,
        "kernel_models": kernel_models,
        "coverage": coverage,
    }
    write_json(args.output / "simulation_plan.json", plan)
    write_json(args.output / "coverage.json", coverage)
    if args.strict and not strict_ready:
        print("error: simulation plan is not strict-ready", file=sys.stderr)
        return 3
    print(args.output / "simulation_plan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
