#!/usr/bin/env python3
"""Compare native-delay block targets with level-4 observed durations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SLOTS = ("AIC", "AIV0", "AIV1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--capture",
        action="append",
        required=True,
        help="PROGRAM=level-4 capture directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _capture_specs(raw_values: list[str]) -> dict[str, Path]:
    result = {}
    for raw in raw_values:
        program, separator, path = raw.partition("=")
        if not separator or not program or not path:
            raise ValueError(f"invalid --capture {raw!r}; expected PROGRAM=PATH")
        if program in result:
            raise ValueError(f"duplicate capture program {program!r}")
        result[program] = Path(path).resolve()
    return result


def _physical_slots(swimlane: dict[str, Any]) -> dict[int, str]:
    metadata = swimlane.get("metadata", {})
    core_types = metadata.get("core_types")
    if not isinstance(core_types, list) or not core_types or len(core_types) % 3:
        raise ValueError("invalid blocked AIC/AIV topology metadata")
    aic_count = len(core_types) // 3
    if core_types != ["aic"] * aic_count + ["aiv"] * (2 * aic_count):
        raise ValueError("unsupported AIC/AIV topology")
    result = {core_id: "AIC" for core_id in range(aic_count)}
    for core_id in range(aic_count, len(core_types)):
        result[core_id] = "AIV0" if (core_id - aic_count) % 2 == 0 else "AIV1"
    return result


def _slot(row: dict[str, Any], active: list[str], physical: dict[int, str]) -> str:
    if row.get("core_type") == "aic":
        return "AIC"
    aiv = [slot for slot in active if slot.startswith("AIV")]
    if len(aiv) == 1:
        return aiv[0]
    result = physical.get(int(row.get("core_id", -1)))
    if result not in aiv:
        raise ValueError(f"cannot map physical AIV row to {active}: {row}")
    return result


def _summary(errors: list[tuple[float, float]]) -> dict[str, float | int]:
    absolute = [abs(observed - target) for target, observed in errors]
    relative = [100.0 * abs(observed - target) / target for target, observed in errors if target > 0]
    signed = [observed - target for target, observed in errors]
    return {
        "sample_count": len(errors),
        "mean_absolute_error_us": statistics.mean(absolute),
        "p95_absolute_error_us": _percentile(absolute, 0.95),
        "max_absolute_error_us": max(absolute),
        "mean_absolute_percent_error": statistics.mean(relative),
        "p95_absolute_percent_error": _percentile(relative, 0.95),
        "mean_signed_error_us": statistics.mean(signed),
    }


def main() -> int:
    args = parse_args()
    try:
        from simpler_setup.tools import swimlane_converter

        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        captures = _capture_specs(args.capture)
        models = {
            (str(entry["program"]), int(entry["func_id"])): entry
            for entry in plan.get("kernel_models", [])
        }
        errors: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
        missing: list[str] = []

        for program, capture in sorted(captures.items()):
            deps = json.loads((capture / "deps.json").read_text(encoding="utf-8"))
            swimlane_path = capture / "chip_swimlane_records.json"
            swimlane = json.loads(swimlane_path.read_text(encoding="utf-8"))
            decoded = swimlane_converter.read_perf_data(swimlane_path)
            physical = _physical_slots(swimlane)
            rows_by_task_epoch: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for row in decoded.get("tasks", []):
                if row.get("executed", True):
                    rows_by_task_epoch[
                        (int(row["task_id"]), int(row.get("run_epoch", 0)))
                    ].append(row)

            for task in deps.get("tasks", []):
                task_id = int(task["task_id"])
                kernel_ids = [int(value) for value in task.get("kernel_ids", [-1, -1, -1])]
                active = [slot for slot, func_id in zip(SLOTS, kernel_ids) if func_id >= 0]
                block_num = int(task.get("block_num", 1))
                epochs = sorted(
                    (epoch, rows)
                    for (row_task_id, epoch), rows in rows_by_task_epoch.items()
                    if row_task_id == task_id
                )
                for epoch, rows in epochs:
                    by_slot: dict[str, list[float]] = defaultdict(list)
                    for row in rows:
                        by_slot[_slot(row, active, physical)].append(float(row["duration_us"]))
                    for slot, func_id in zip(SLOTS, kernel_ids):
                        if func_id < 0:
                            continue
                        model = models.get((program, func_id))
                        if model is None:
                            missing.append(f"{program}:{task_id}@{epoch}:{slot}:model")
                            continue
                        target = model.get("block_duration_us_by_subslot", {}).get(slot)
                        if not target:
                            target = [float(model["duration_us"])] * block_num
                        observed = sorted(by_slot.get(slot, []))
                        if len(target) != block_num or len(observed) != block_num:
                            missing.append(
                                f"{program}:{task_id}@{epoch}:{slot}:"
                                f"target={len(target)}/observed={len(observed)}/expected={block_num}"
                            )
                            continue
                        errors[(program, func_id, slot)].extend(
                            (float(expected), actual)
                            for expected, actual in zip(target, observed)
                        )

        callable_rows = []
        for (program, func_id, slot), values in sorted(errors.items()):
            summary = _summary(values)
            if any(not math.isfinite(float(value)) for value in summary.values()):
                raise ValueError(f"non-finite summary for {program}:{func_id}:{slot}")
            callable_rows.append(
                {"program": program, "func_id": func_id, "subslot": slot, **summary}
            )
        program_rows = []
        for program in sorted(captures):
            values = [
                pair
                for (candidate, _, _), pairs in errors.items()
                if candidate == program
                for pair in pairs
            ]
            if values:
                program_rows.append({"program": program, **_summary(values)})
        all_values = [pair for pairs in errors.values() for pair in pairs]
        if missing or not all_values:
            raise ValueError(f"incomplete comparison: missing={missing[:20]}")
        result = {
            "schema_version": 1,
            "plan": str(args.plan.resolve()),
            "captures": {program: str(path) for program, path in sorted(captures.items())},
            "overall": _summary(all_values),
            "programs": program_rows,
            "callables": callable_rows,
            "missing": missing,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, ImportError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
