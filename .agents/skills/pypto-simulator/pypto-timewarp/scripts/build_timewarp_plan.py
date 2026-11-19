#!/usr/bin/env python3
"""Compile task-level profile DAGs into an a2a3sim timewarp plan."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SLOTS = ("AIC", "AIV0", "AIV1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aic-clusters", type=int, default=20)
    parser.add_argument(
        "--prefer-capture-role",
        action="append",
        default=[],
        help="override capture-role priority for legacy sleep plans; repeat as needed",
    )
    return parser.parse_args()


def _task_sort_key(task_id: str) -> tuple[int, str]:
    try:
        return int(task_id), task_id
    except ValueError:
        return sys.maxsize, task_id


def _graph_signature(tasks: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(task["task_id"]),
            tuple(sorted(str(value) for value in task.get("dependencies", []))),
            tuple(int(value) for value in task.get("kernel_ids", [-1, -1, -1])),
            int(task.get("block_num", 1)),
            str(task.get("resource_shape", "DUMMY")),
        )
        for task in tasks
    )


def _select_graphs(
    plan: dict[str, Any], preferred_roles: list[str] | None = None
) -> dict[str, list[dict[str, Any]]]:
    preferred = [
        str(value)
        for value in (
            preferred_roles
            if preferred_roles
            else (plan.get("preferred_capture_roles") or [])
        )
    ]
    by_program_role: dict[tuple[str, str], list[list[dict[str, Any]]]] = defaultdict(list)
    for raw_path in plan.get("source_profile_manifests", []):
        manifest = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for task in manifest.get("tasks", []):
            program = str(task.get("program", "default"))
            role = str(task.get("features", {}).get("capture_role", ""))
            grouped[(program, role)].append(task)
        for key, tasks in grouped.items():
            by_program_role[key].append(tasks)

    programs = sorted({str(model["program"]) for model in plan.get("kernel_models", [])})
    selected: dict[str, list[dict[str, Any]]] = {}
    for program in programs:
        roles = sorted(role for candidate, role in by_program_role if candidate == program)
        role = next((candidate for candidate in preferred if candidate in roles), None)
        if role is None:
            model_roles = {
                str(value)
                for model in plan["kernel_models"]
                if str(model["program"]) == program
                for value in model.get("selected_capture_roles", [])
                if str(value) in roles
            }
            if len(model_roles) == 1:
                role = next(iter(model_roles))
            elif not model_roles and len(roles) == 1:
                role = roles[0]
            else:
                raise ValueError(
                    f"cannot select one capture role for {program}: "
                    f"roles={roles}, model_roles={sorted(model_roles)}"
                )
        candidates = by_program_role[(program, role)]
        signature = _graph_signature(candidates[0])
        if any(_graph_signature(tasks) != signature for tasks in candidates[1:]):
            raise ValueError(f"independent {program}/{role} captures have different DAGs")
        selected[program] = candidates[0]
    return selected


def _duration_vector(
    model: dict[str, Any], slot: str, block_num: int, program: str, task_id: str
) -> list[float]:
    values = model.get("block_duration_us_by_subslot", {}).get(slot)
    if values:
        result = [float(value) for value in values]
        if len(result) != block_num:
            raise ValueError(
                f"block model mismatch for {program}:{task_id}:{slot}: "
                f"{len(result)} != {block_num}"
            )
    else:
        result = [float(model["duration_us"])] * block_num
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError(f"invalid block duration for {program}:{task_id}:{slot}")
    return result


def _schedule_graph(
    program: str,
    tasks: list[dict[str, Any]],
    models: dict[tuple[str, int], dict[str, Any]],
    clusters: int,
) -> dict[str, Any]:
    if clusters <= 0:
        raise ValueError("aic-clusters must be positive")
    task_by_id = {str(task["task_id"]): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError(f"duplicate task id in {program}")
    dependencies: dict[str, list[str]] = {}
    consumers: dict[str, list[str]] = defaultdict(list)
    pending: dict[str, int] = {}
    for task_id, task in task_by_id.items():
        deps = [str(value) for value in task.get("dependencies", [])]
        missing = [value for value in deps if value not in task_by_id]
        if missing:
            raise ValueError(f"{program}:{task_id} has missing local dependencies {missing[:4]}")
        dependencies[task_id] = deps
        pending[task_id] = len(deps)
        for predecessor in deps:
            consumers[predecessor].append(task_id)

    aic_available = [0.0] * clusters
    aiv_available = [0.0] * (2 * clusters)
    finishes: dict[str, float] = {}
    ready: list[tuple[float, tuple[int, str], str]] = []
    for task_id, count in pending.items():
        if count == 0:
            heapq.heappush(ready, (0.0, _task_sort_key(task_id), task_id))

    modeled_tasks = 0
    modeled_blocks = 0
    while ready:
        dependency_ready, _, task_id = heapq.heappop(ready)
        task = task_by_id[task_id]
        block_num = int(task.get("block_num", 1))
        if block_num <= 0:
            raise ValueError(f"{program}:{task_id} has invalid block_num")
        kernel_ids = [int(value) for value in task.get("kernel_ids", [-1, -1, -1])]
        active = [(slot, func_id) for slot, func_id in zip(SLOTS, kernel_ids) if func_id >= 0]
        if not active:
            task_finish = dependency_ready
        else:
            vectors = {}
            for slot, func_id in active:
                model = models.get((program, func_id))
                if model is None:
                    raise ValueError(
                        f"no runtime model for {program}:{task_id}:{slot}/func_id={func_id}"
                    )
                vectors[slot] = _duration_vector(model, slot, block_num, program, task_id)
            block_finishes = []
            slots = {slot for slot, _ in active}
            for block_idx in range(block_num):
                if slots == {"AIC"}:
                    core = min(range(clusters), key=lambda index: (aic_available[index], index))
                    start = max(dependency_ready, aic_available[core])
                    finish = start + vectors["AIC"][block_idx]
                    aic_available[core] = finish
                elif slots <= {"AIV0", "AIV1"} and len(slots) == 1:
                    core = min(
                        range(2 * clusters), key=lambda index: (aiv_available[index], index)
                    )
                    slot = next(iter(slots))
                    start = max(dependency_ready, aiv_available[core])
                    finish = start + vectors[slot][block_idx]
                    aiv_available[core] = finish
                else:
                    cluster = min(
                        range(clusters),
                        key=lambda index: (
                            max(
                                aic_available[index] if "AIC" in slots else 0.0,
                                aiv_available[2 * index] if "AIV0" in slots else 0.0,
                                aiv_available[2 * index + 1] if "AIV1" in slots else 0.0,
                            ),
                            index,
                        ),
                    )
                    start = max(
                        dependency_ready,
                        aic_available[cluster] if "AIC" in slots else 0.0,
                        aiv_available[2 * cluster] if "AIV0" in slots else 0.0,
                        aiv_available[2 * cluster + 1] if "AIV1" in slots else 0.0,
                    )
                    lane_finishes = []
                    if "AIC" in slots:
                        aic_available[cluster] = start + vectors["AIC"][block_idx]
                        lane_finishes.append(aic_available[cluster])
                    if "AIV0" in slots:
                        aiv_available[2 * cluster] = start + vectors["AIV0"][block_idx]
                        lane_finishes.append(aiv_available[2 * cluster])
                    if "AIV1" in slots:
                        aiv_available[2 * cluster + 1] = start + vectors["AIV1"][block_idx]
                        lane_finishes.append(aiv_available[2 * cluster + 1])
                    finish = max(lane_finishes)
                block_finishes.append(finish)
            task_finish = max(block_finishes)
            modeled_tasks += 1
            modeled_blocks += block_num * len(active)
        finishes[task_id] = task_finish
        for consumer in consumers[task_id]:
            pending[consumer] -= 1
            if pending[consumer] == 0:
                consumer_ready = max(finishes[value] for value in dependencies[consumer])
                heapq.heappush(ready, (consumer_ready, _task_sort_key(consumer), consumer))

    if len(finishes) != len(tasks):
        blocked = sorted(set(task_by_id) - set(finishes), key=_task_sort_key)
        raise ValueError(f"cycle or unresolved graph in {program}: {blocked[:8]}")
    return {
        "program": program,
        "duration_us": max(finishes.values(), default=0.0),
        "task_count": len(tasks),
        "modeled_task_count": modeled_tasks,
        "modeled_block_count": modeled_blocks,
        "aic_clusters": clusters,
        "scheduler": "deterministic_task_dag_resource_list_v1",
    }


def build_timewarp_plan(
    plan: dict[str, Any], clusters: int, preferred_roles: list[str] | None = None
) -> dict[str, Any]:
    if plan.get("schema_version") != 1:
        raise ValueError("sleep plan must use schema_version 1")
    if plan.get("execution_mode") != "sleep" or plan.get("backend") != "a2a3sim":
        raise ValueError("timewarp requires an a2a3sim sleep plan")
    if not plan.get("coverage", {}).get("runtime_stub_ready", False):
        raise ValueError("sleep plan is not runtime-stub-ready")
    models = {
        (str(model["program"]), int(model["func_id"])): model
        for model in plan.get("kernel_models", [])
    }
    graphs = _select_graphs(plan, preferred_roles)
    result = dict(plan)
    result["execution_mode"] = "timewarp"
    result["timewarp"] = {
        "clock": "CLOCK_MONOTONIC+non_decreasing_global_offset",
        "sleep_oracle_required": True,
        "graph_models": [
            _schedule_graph(program, tasks, models, clusters)
            for program, tasks in sorted(graphs.items())
        ],
    }
    return result


def main() -> int:
    args = parse_args()
    try:
        plan = json.loads(args.sleep_plan.read_text(encoding="utf-8"))
        result = build_timewarp_plan(plan, args.aic_clusters, args.prefer_capture_role)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
