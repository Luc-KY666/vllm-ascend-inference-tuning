#!/usr/bin/env python3
"""Export a PyPTO profile manifest from native deps and chip-swimlane artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

COMMON = Path(__file__).resolve().parents[2] / "pypto-common"
sys.path.insert(0, str(COMMON))

from pypto_contract import ContractError, normalize_profile_manifest, write_json

SLOTS = ("AIC", "AIV0", "AIV1")
COMM_RE = re.compile(
    r"(?:allreduce|allgather|reduce_scatter|alltoall|send|recv|window|hccs|comm)",
    re.IGNORECASE,
)
SAMPLE_RE = re.compile(r"(?:greedy|sample)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--swimlane", type=Path)
    parser.add_argument("--name-map", type=Path)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--program")
    parser.add_argument("--workload-id", help="stable identity for one independent native capture")
    parser.add_argument("--features-json", default="{}", help="JSON object or path to one")
    parser.add_argument("--require-level4", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _one(path: Path, pattern: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ContractError(f"expected one {pattern!r} under {path}, found {len(matches)}")
    return matches[0].resolve()


def _features(raw: str) -> dict[str, Any]:
    candidate = Path(raw)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else raw
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ContractError("--features-json must decode to an object")
    return value


def _load_converter():
    try:
        from simpler_setup.tools import swimlane_converter
    except ImportError as error:
        raise ContractError("simpler_setup is required to decode chip-swimlane artifacts") from error
    return swimlane_converter


def _name_map(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("callable_id_to_name", data)
    if not isinstance(raw, dict):
        raise ContractError(f"invalid name map: {path}")
    return {int(key): str(value) for key, value in raw.items()}


def _program_name(capture_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    sidecar = capture_dir / "dispatch_program.json"
    if sidecar.is_file():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        for key in ("program", "program_name", "name", "dispatch_program"):
            if data.get(key):
                return str(data[key])
    return capture_dir.name or "default"


def _workload_identity(capture_dir: Path, explicit: str | None) -> tuple[str, dict[str, Any]]:
    if explicit:
        value = explicit.strip()
        if not value:
            raise ContractError("--workload-id cannot be empty")
        return value, {"workload_id_source": "explicit"}

    timing_path = capture_dir / "dispatch_identity.timing.json"
    if not timing_path.is_file():
        raise ContractError("native capture requires --workload-id or dispatch_identity.timing.json")
    data = json.loads(timing_path.read_text(encoding="utf-8"))
    required = ("owner_instance_id", "run_id", "endpoint_dispatch_id", "callable_digest")
    if any(data.get(field) in (None, "") for field in required):
        raise ContractError(f"incomplete timing dispatch identity: {timing_path}")
    identity = {
        "owner_instance_id": str(data["owner_instance_id"]),
        "run_id": int(data["run_id"]),
        "endpoint_dispatch_id": int(data["endpoint_dispatch_id"]),
        "callable_digest": str(data["callable_digest"]),
    }
    workload_id = (
        f"{identity['owner_instance_id']}:{identity['run_id']}:"
        f"{identity['endpoint_dispatch_id']}:{identity['callable_digest'][:16]}"
    )
    return workload_id, {"workload_id_source": str(timing_path.resolve()), **identity}


def _task_id(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"invalid task id {value!r}") from error


def _physical_slot_map(swimlane_path: Path, decoded: dict[str, Any]) -> dict[int, str]:
    if decoded.get("runtime") != "tensormap_and_ringbuffer":
        return {}

    data = json.loads(swimlane_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    core_types = metadata.get("core_types")
    if not isinstance(core_types, list) or not core_types or len(core_types) % 3:
        raise ContractError("invalid tensormap_and_ringbuffer core topology metadata")
    aic_count = len(core_types) // 3
    expected = ["aic"] * aic_count + ["aiv"] * (2 * aic_count)
    if core_types != expected:
        raise ContractError("unsupported tensormap_and_ringbuffer core topology")

    slots = {core_id: "AIC" for core_id in range(aic_count)}
    for core_id in range(aic_count, len(core_types)):
        slots[core_id] = "AIV0" if (core_id - aic_count) % 2 == 0 else "AIV1"
    return slots


def _slot(row: dict[str, Any], active: list[str], physical_slots: dict[int, str]) -> str:
    if "subtask_slot" in row:
        value = int(row["subtask_slot"])
        if 0 <= value < len(SLOTS):
            return SLOTS[value]
    if row.get("core_type") == "aic":
        return "AIC"
    aiv = [slot for slot in active if slot.startswith("AIV")]
    if len(aiv) == 1:
        return aiv[0]
    physical_slot = physical_slots.get(int(row.get("core_id", -1)))
    if physical_slot in aiv:
        return physical_slot
    raise ContractError(
        "capture cannot distinguish AIV0/AIV1; collect HBG execution identities "
        "or provide validated tensormap_and_ringbuffer topology metadata"
    )


def _control_outputs(task: dict[str, Any], task_name: str) -> list[dict[str, Any]]:
    if not SAMPLE_RE.search(task_name):
        return []
    outputs = []
    for arg in task.get("args", []):
        if str(arg.get("type", "")).upper() not in {"OUTPUT", "OUTPUT_EXISTING", "INOUT"}:
            continue
        if str(arg.get("dtype", "")).upper() != "INT32":
            continue
        shape = arg.get("shape", [])
        if not isinstance(shape, list) or len(shape) < 2:
            continue
        outputs.append(
            {
                "name": "sampled_ids",
                "role": "control",
                "dtype": "int32",
                "shape": shape,
                "arg_index": int(arg["idx"]),
            }
        )
    return outputs


def main() -> int:
    args = parse_args()
    try:
        capture_dir = args.capture_dir.resolve()
        deps_path = _one(capture_dir, "deps.json", args.deps)
        swimlane_path = _one(capture_dir, "chip_swimlane_records.json", args.swimlane)
        name_path = _one(capture_dir, "name_map*.json", args.name_map)
        deps = json.loads(deps_path.read_text(encoding="utf-8"))
        converter = _load_converter()
        decoded = converter.read_perf_data(swimlane_path)
        level = int(decoded.get("chip_swimlane_level", 0))
        if args.require_level4 and level != 4:
            raise ContractError(f"level-4 capture required, got level {level}")
        names = _name_map(name_path)
        features = _features(args.features_json)
        program = _program_name(capture_dir, args.program)
        workload_id, workload_identity = _workload_identity(capture_dir, args.workload_id)
        physical_slots = _physical_slot_map(swimlane_path, decoded)

        dep_tasks = deps.get("tasks")
        if not isinstance(dep_tasks, list):
            raise ContractError("deps.json has no tasks array")
        dep_by_id = {_task_id(task.get("task_id")): task for task in dep_tasks}
        dependencies: dict[int, set[int]] = defaultdict(set)
        external_dependencies: dict[int, set[int]] = defaultdict(set)
        for edge in deps.get("edges", []):
            predecessor = _task_id(edge.get("pred"))
            successor = _task_id(edge.get("succ"))
            if successor not in dep_by_id:
                raise ContractError(f"dependency successor {successor} is absent from deps tasks")
            target = dependencies if predecessor in dep_by_id else external_dependencies
            target[successor].add(predecessor)

        timed: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in decoded.get("tasks", []):
            timed[_task_id(row["task_id"])][int(row.get("run_epoch", 0))].append(row)

        normalized_tasks = []
        incomplete_rows: list[str] = []
        for task_id, task in sorted(dep_by_id.items()):
            kernel_ids = [int(value) for value in task.get("kernel_ids", [-1, -1, -1])]
            if len(kernel_ids) != 3:
                raise ContractError(f"task {task_id} has invalid kernel_ids")
            active = [slot for slot, func_id in zip(SLOTS, kernel_ids) if func_id >= 0]
            kernel_names = [names.get(func_id, f"func_{func_id}") for func_id in kernel_ids if func_id >= 0]
            task_name = "+".join(kernel_names) if kernel_names else f"task_{task_id}"
            block_num = max(1, int(task.get("block_num", 1)))
            duration_samples: list[float] = []
            slot_samples: dict[str, list[float]] = defaultdict(list)
            slot_block_samples: dict[str, list[list[float]]] = defaultdict(list)
            for epoch, rows in sorted(timed.get(task_id, {}).items()):
                by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    if row.get("executed", True):
                        by_slot[_slot(row, active, physical_slots)].append(row)
                expected = block_num * len(active)
                if len(rows) != expected:
                    incomplete_rows.append(f"{task_id}@{epoch}:{len(rows)}/{expected}")
                    continue
                epoch_slot_values = {}
                for slot in active:
                    values = [float(row["duration_us"]) for row in by_slot.get(slot, [])]
                    if len(values) != block_num:
                        incomplete_rows.append(f"{task_id}@{epoch}:{slot}:{len(values)}/{block_num}")
                        epoch_slot_values = {}
                        break
                    # Keep the sorted physical-block distribution without claiming
                    # an unavailable block_idx identity from the swimlane schema.
                    sorted_values = sorted(values)
                    slot_block_samples[slot].append(sorted_values)
                    epoch_slot_values[slot] = sorted_values[-1]
                if not epoch_slot_values and active:
                    continue
                for slot, value in epoch_slot_values.items():
                    slot_samples[slot].append(value)
                if epoch_slot_values:
                    duration_samples.append(max(epoch_slot_values.values()))

            outputs = _control_outputs(task, task_name)
            roles = ["control"] if outputs else []
            if not active:
                kind = "synchronization"
                resource_shape = "NONE"
            else:
                kind = "communication" if any(COMM_RE.search(name) for name in kernel_names) else "compute"
                resource_shape = "MIX" if len(active) > 1 else ("AIC" if active == ["AIC"] else "AIV")
            normalized_tasks.append(
                {
                    "task_id": str(task_id),
                    "task_name": task_name,
                    "program": program,
                    "phase": args.phase,
                    "kind": kind,
                    "resource_shape": resource_shape,
                    "block_num": block_num,
                    "active_subslots": active,
                    "kernel_ids": kernel_ids,
                    "kernel_names": kernel_names,
                    "dependencies": [str(value) for value in sorted(dependencies.get(task_id, set()))],
                    "tensor_roles": roles,
                    "features": features,
                    "duration_samples_us": duration_samples,
                    "subslot_duration_samples_us": dict(slot_samples),
                    "subslot_block_duration_samples_us": dict(slot_block_samples),
                    "outputs": outputs,
                    "metadata": {
                        "scope": task.get("scope"),
                        "args": task.get("args", []),
                        "external_dependencies": [
                            str(value) for value in sorted(external_dependencies.get(task_id, set()))
                        ],
                    },
                }
            )

        raw_manifest = {
            "model": {"family": args.model_family, "model_id": args.model_id},
            "source": {
                "kind": "npu_chip_swimlane",
                "npu_measured": True,
                "capture_dir": str(capture_dir),
                "deps": str(deps_path),
                "swimlane": str(swimlane_path),
                "name_map": str(name_path),
                "chip_swimlane_level": level,
                "workload_id": workload_id,
                "workload_identity": workload_identity,
            },
            "metadata": {
                "runtime": decoded.get("runtime"),
                "program": program,
                "incomplete_physical_rows": incomplete_rows,
                "external_predecessor_task_ids": [
                    str(value)
                    for value in sorted(
                        {value for task_values in external_dependencies.values() for value in task_values}
                    )
                ],
            },
            "tasks": normalized_tasks,
        }
        manifest = normalize_profile_manifest(raw_manifest)
    except (OSError, ValueError, ImportError, ContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    measured = sum(bool(task["duration_samples_us"]) for task in manifest["tasks"])
    coverage = {
        "schema_version": 1,
        "task_count": len(manifest["tasks"]),
        "measured_task_count": measured,
        "unmeasured_task_count": len(manifest["tasks"]) - measured,
        "incomplete_physical_rows": incomplete_rows,
        "complete": not incomplete_rows and measured > 0,
        "npu_accessed": False,
        "source_npu_measured": True,
    }
    write_json(args.output / "profile_manifest.json", manifest)
    write_json(args.output / "coverage.json", coverage)
    print(args.output / "profile_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
