"""Shared manifest and classification contract for PyPTO simulator skills."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TASK_KINDS = {
    "compute",
    "communication",
    "memory",
    "synchronization",
    "control",
    "metadata",
}
RESOURCE_SHAPES = {"AIC", "AIV", "MIX", "HOST", "NONE"}
SUBSLOTS = {"AIC", "AIV0", "AIV1"}
VIRTUAL_TENSOR_ROLES = {"weight", "kv_cache"}


class ContractError(ValueError):
    """Raised when a simulator manifest violates the shared contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContractError(f"{field} must be a list of strings")
    return list(value)


def _kernel_ids(raw: Any, field: str) -> list[int]:
    if raw is None:
        return [-1, -1, -1]
    if not isinstance(raw, list) or len(raw) != 3:
        raise ContractError(f"{field} must be [aic, aiv0, aiv1]")
    try:
        values = [int(value) for value in raw]
    except (TypeError, ValueError) as error:
        raise ContractError(f"{field} must contain integers") from error
    if any(value < -1 for value in values):
        raise ContractError(f"{field} values must be -1 or non-negative")
    return values


def _duration_values(raw: Any, field: str) -> list[float]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ContractError(f"{field} must be a list")
    samples: list[float] = []
    for value in raw:
        try:
            sample = float(value)
        except (TypeError, ValueError) as error:
            raise ContractError(f"invalid {field} sample: {value!r}") from error
        if not math.isfinite(sample) or sample < 0:
            raise ContractError(
                f"{field} sample must be finite and non-negative: {value!r}"
            )
        samples.append(sample)
    return samples


def _duration_samples(task: dict[str, Any]) -> list[float]:
    raw = task.get("duration_samples_us")
    if raw is None and task.get("duration_us") is not None:
        raw = [task["duration_us"]]
    return _duration_values(raw, "duration_samples_us")


def normalize_task(task: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(task, dict):
        raise ContractError(f"tasks[{index}] must be an object")
    task_id = str(task.get("task_id", "")).strip()
    task_name = str(task.get("task_name", task.get("name", ""))).strip()
    if not task_id:
        raise ContractError(f"tasks[{index}].task_id is required")
    if not task_name:
        raise ContractError(f"tasks[{index}].task_name is required")

    kind = str(task.get("kind", "")).strip().lower()
    if kind not in TASK_KINDS:
        raise ContractError(f"tasks[{index}].kind must be one of {sorted(TASK_KINDS)}")
    resource_shape = str(task.get("resource_shape", "NONE")).strip().upper()
    if resource_shape not in RESOURCE_SHAPES:
        raise ContractError(
            f"tasks[{index}].resource_shape must be one of {sorted(RESOURCE_SHAPES)}"
        )
    try:
        block_num = int(task.get("block_num", 1))
    except (TypeError, ValueError) as error:
        raise ContractError(f"tasks[{index}].block_num must be an integer") from error
    if block_num < 1:
        raise ContractError(f"tasks[{index}].block_num must be positive")

    subslots = _string_list(task.get("active_subslots", []), "active_subslots")
    subslots = [slot.upper() for slot in subslots]
    unknown_subslots = set(subslots) - SUBSLOTS
    if unknown_subslots:
        raise ContractError(
            f"tasks[{index}] has unknown subslots: {sorted(unknown_subslots)}"
        )
    active_subslots = sorted(set(subslots))
    raw_subslot_samples = task.get("subslot_duration_samples_us", {})
    if not isinstance(raw_subslot_samples, dict):
        raise ContractError(
            f"tasks[{index}].subslot_duration_samples_us must be an object"
        )
    subslot_samples = {}
    for raw_slot, values in raw_subslot_samples.items():
        slot = str(raw_slot).upper()
        if slot not in active_subslots:
            raise ContractError(
                f"tasks[{index}] has duration samples for inactive subslot {slot!r}"
            )
        subslot_samples[slot] = _duration_values(
            values, f"subslot_duration_samples_us.{slot}"
        )

    raw_block_samples = task.get("subslot_block_duration_samples_us", {})
    if not isinstance(raw_block_samples, dict):
        raise ContractError(
            f"tasks[{index}].subslot_block_duration_samples_us must be an object"
        )
    block_samples = {}
    for raw_slot, epochs in raw_block_samples.items():
        slot = str(raw_slot).upper()
        if slot not in active_subslots:
            raise ContractError(
                f"tasks[{index}] has block samples for inactive subslot {slot!r}"
            )
        if not isinstance(epochs, list):
            raise ContractError(
                f"subslot_block_duration_samples_us.{slot} must be a list"
            )
        normalized_epochs = []
        for epoch_index, values in enumerate(epochs):
            normalized = _duration_values(
                values,
                f"subslot_block_duration_samples_us.{slot}[{epoch_index}]",
            )
            if len(normalized) != block_num:
                raise ContractError(
                    f"subslot_block_duration_samples_us.{slot}[{epoch_index}] "
                    f"has {len(normalized)} blocks, expected {block_num}"
                )
            if normalized != sorted(normalized):
                raise ContractError(
                    f"subslot_block_duration_samples_us.{slot}[{epoch_index}] "
                    "must be sorted"
                )
            normalized_epochs.append(normalized)
        block_samples[slot] = normalized_epochs

    dependencies = _string_list(task.get("dependencies", []), "dependencies")
    tensor_roles = [
        role.lower()
        for role in _string_list(task.get("tensor_roles", []), "tensor_roles")
    ]
    kernel_names = _string_list(task.get("kernel_names", []), "kernel_names")
    kernel_ids = _kernel_ids(task.get("kernel_ids"), f"tasks[{index}].kernel_ids")
    expected_slots = [slot for slot, value in zip(("AIC", "AIV0", "AIV1"), kernel_ids) if value >= 0]
    if expected_slots and sorted(expected_slots) != active_subslots:
        raise ContractError(
            f"tasks[{index}].kernel_ids active slots {sorted(expected_slots)} "
            f"do not match active_subslots {active_subslots}"
        )
    features = task.get("features", {})
    outputs = task.get("outputs", [])
    metadata = task.get("metadata", {})
    if not isinstance(features, dict):
        raise ContractError(f"tasks[{index}].features must be an object")
    if not isinstance(outputs, list) or any(
        not isinstance(item, dict) for item in outputs
    ):
        raise ContractError(f"tasks[{index}].outputs must be a list of objects")
    if not isinstance(metadata, dict):
        raise ContractError(f"tasks[{index}].metadata must be an object")

    return {
        "task_id": task_id,
        "task_name": task_name,
        "program": str(task.get("program", "default")).strip() or "default",
        "phase": str(task.get("phase", "unknown")).strip().lower() or "unknown",
        "kind": kind,
        "resource_shape": resource_shape,
        "block_num": block_num,
        "active_subslots": active_subslots,
        "dependencies": dependencies,
        "tensor_roles": sorted(set(tensor_roles)),
        "kernel_names": sorted(set(kernel_names)),
        "kernel_ids": kernel_ids,
        "features": features,
        "duration_samples_us": _duration_samples(task),
        "subslot_duration_samples_us": dict(sorted(subslot_samples.items())),
        "subslot_block_duration_samples_us": dict(sorted(block_samples.items())),
        "outputs": outputs,
        "metadata": metadata,
    }


def normalize_profile_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError("profile input must be an object")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ContractError("model must be an object")
    family = str(model.get("family", "")).strip().lower()
    model_id = str(model.get("model_id", "")).strip()
    if not family or not model_id:
        raise ContractError("model.family and model.model_id are required")
    raw_tasks = raw.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ContractError("tasks must be a list")
    tasks = [normalize_task(task, index) for index, task in enumerate(raw_tasks)]

    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ContractError("task_id values must be unique")
    known_ids = set(task_ids)
    for task in tasks:
        missing = set(task["dependencies"]) - known_ids
        if missing:
            raise ContractError(
                f"task {task['task_id']!r} has unknown dependencies: {sorted(missing)}"
            )

    source = raw.get("source", {})
    metadata = raw.get("metadata", {})
    if not isinstance(source, dict) or not isinstance(metadata, dict):
        raise ContractError("source and metadata must be objects")
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": "pypto-profile",
        "model": {**model, "family": family, "model_id": model_id},
        "source": source,
        "metadata": metadata,
        "tasks": tasks,
    }


def validate_profile_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(
            f"unsupported schema_version {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    normalize_profile_manifest(manifest)


def should_intercept(task: dict[str, Any]) -> bool:
    kind = str(task.get("kind", "")).lower()
    roles = {str(role).lower() for role in task.get("tensor_roles", [])}
    return kind in {"compute", "communication"} or bool(
        roles & VIRTUAL_TENSOR_ROLES
    )


def latency_group_signature(task: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Stable identity shared by all measured feature points for one task site."""
    return {
        "model_family": str(model.get("family", "")).lower(),
        "model_id": str(model.get("model_id", "")),
        "program": task.get("program", "default"),
        "task_name": task.get("task_name", ""),
        "phase": task.get("phase", "unknown"),
        "kind": task.get("kind", ""),
        "resource_shape": task.get("resource_shape", "NONE"),
        "block_num": int(task.get("block_num", 1)),
        "active_subslots": sorted(task.get("active_subslots", [])),
        "kernel_names": sorted(task.get("kernel_names", [])),
        "kernel_ids": [int(value) for value in task.get("kernel_ids", [-1, -1, -1])],
    }


def latency_group_key(task: dict[str, Any], model: dict[str, Any]) -> str:
    payload = canonical_json(latency_group_signature(task, model)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def latency_signature(task: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    return {
        **latency_group_signature(task, model),
        "features": task.get("features", {}),
    }


def latency_key(task: dict[str, Any], model: dict[str, Any]) -> str:
    payload = canonical_json(latency_signature(task, model)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def sampled_id_outputs(task: dict[str, Any]) -> list[dict[str, Any]]:
    names = {"sampled_ids", "sampled_ids_out", "sampled_token_ids"}
    return [
        output
        for output in task.get("outputs", [])
        if str(output.get("name", "")).lower() in names
    ]
