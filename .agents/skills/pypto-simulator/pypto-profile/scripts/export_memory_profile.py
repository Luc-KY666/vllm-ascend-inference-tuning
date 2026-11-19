#!/usr/bin/env python3
"""Export measured model-weight and KV materialization phases from serving logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TIMESTAMP = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})")
_WEIGHT_BYTES = re.compile(r"weights \(estimated\):\s+(?P<gb>[0-9.]+) GB")
_KV_START = re.compile(r"num_pages=(?P<pages>\d+), allocating")
_KV_END = re.compile(r"allocated (?P<pages>\d+) pages.*: (?P<gb>[0-9.]+) GB KV cache")
_KV_SIZING = re.compile(
    r"peak_non_kv=(?P<peak>[0-9.]+) GB, simpler_committed=(?P<committed>[0-9.]+) GB"
)

# Legacy compatibility default; new models must pass --kv-page-bytes explicitly.
QWEN3_14B_KV_PAGE_BYTES = 2 * 40 * 8 * 128 * 128 * 2


@dataclass(frozen=True)
class Mark:
    line: int
    seconds: float


def _seconds(raw: str) -> float:
    hours, minutes, seconds = raw.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _mark(line: str, number: int) -> Mark | None:
    match = _TIMESTAMP.match(line)
    return None if match is None else Mark(number, _seconds(match.group("time")))


def _elapsed_us(start: Mark, end: Mark) -> float:
    elapsed = end.seconds - start.seconds
    if elapsed < 0:
        elapsed += 24 * 3600
    if elapsed <= 0:
        raise ValueError("memory phase has non-positive elapsed time")
    return elapsed * 1_000_000


def _workload_id(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]


def parse_log(path: Path, *, kv_page_bytes: int = QWEN3_14B_KV_PAGE_BYTES) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    weight_start: Mark | None = None
    weight_end: Mark | None = None
    weight_gb: float | None = None
    kv_start: Mark | None = None
    kv_end: Mark | None = None
    kv_start_pages: int | None = None
    kv_end_pages: int | None = None
    kv_reported_gb: float | None = None
    reserved_bytes: int | None = None

    for number, line in enumerate(lines, start=1):
        mark = _mark(line, number)
        if mark is None:
            continue
        if "[init_kv_cache] uploading static tensors" in line:
            weight_start = mark
        elif weight_start is not None and weight_end is None and "[init_kv_cache] profile warmup" in line:
            weight_end = mark

        weight_match = _WEIGHT_BYTES.search(line)
        if weight_match is not None:
            weight_gb = float(weight_match.group("gb"))

        kv_start_match = _KV_START.search(line)
        if kv_start_match is not None:
            kv_start = mark
            kv_start_pages = int(kv_start_match.group("pages"))
        kv_end_match = _KV_END.search(line)
        if kv_end_match is not None:
            kv_end = mark
            kv_end_pages = int(kv_end_match.group("pages"))
            kv_reported_gb = float(kv_end_match.group("gb"))

        kv_sizing_match = _KV_SIZING.search(line)
        if kv_sizing_match is not None:
            peak_gb = float(kv_sizing_match.group("peak"))
            committed_gb = float(kv_sizing_match.group("committed"))
            reserved_bytes = round(max(peak_gb - committed_gb, 0.0) * 1_000_000_000)

    workload_id = _workload_id(path)
    records: list[dict[str, Any]] = []
    if weight_start is not None and weight_end is not None and weight_gb is not None:
        records.append(
            {
                "operation": "weight_materialize",
                "bytes": round(weight_gb * 1_000_000_000),
                "bytes_evidence": "rounded_serving_log",
                "duration_us": _elapsed_us(weight_start, weight_end),
                "start_line": weight_start.line,
                "end_line": weight_end.line,
                "source_log": str(path.resolve()),
                "workload_id": workload_id,
            }
        )
    if kv_start is not None and kv_end is not None:
        if kv_start_pages != kv_end_pages:
            raise ValueError(f"KV page count changed within allocation phase: {path}")
        assert kv_start_pages is not None
        records.append(
            {
                "operation": "kv_materialize",
                "bytes": kv_start_pages * kv_page_bytes,
                "bytes_evidence": "exact_page_count",
                "duration_us": _elapsed_us(kv_start, kv_end),
                "page_count": kv_start_pages,
                "page_bytes": kv_page_bytes,
                "reported_gb": kv_reported_gb,
                "start_line": kv_start.line,
                "end_line": kv_end.line,
                "source_log": str(path.resolve()),
                "workload_id": workload_id,
            }
        )
    if reserved_bytes is not None:
        records.append(
            {
                "operation": "runtime_reserved_memory",
                "bytes": reserved_bytes,
                "bytes_evidence": "difference_of_rounded_serving_log_values",
                "source_log": str(path.resolve()),
                "workload_id": workload_id,
            }
        )
    return records


def parse_evidence(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    by_event = {str(record.get("event")): record for record in records}
    required = {"weight_materialize", "kv_sizing", "kv_materialize"}
    if not required <= by_event.keys():
        raise ValueError(f"incomplete memory evidence {path}: missing {sorted(required - by_event.keys())}")
    workload_ids = {str(record.get("workload_id", "")) for record in records}
    if len(workload_ids) != 1 or not next(iter(workload_ids)):
        raise ValueError(f"memory evidence must contain one non-empty workload_id: {path}")
    weight_bytes = int(by_event["weight_materialize"]["bytes"])
    sizing = by_event["kv_sizing"]
    reserved_bytes = int(sizing["runtime_reserved_bytes"])
    free_bytes = int(sizing["free_bytes"])
    total_bytes = int(sizing["total_bytes"])
    driver_visible_non_kv_bytes = int(sizing["driver_visible_non_kv_bytes"])
    simpler_committed_bytes = int(sizing["simpler_committed_bytes"])
    page_count = int(by_event["kv_materialize"]["page_count"])
    page_bytes = int(by_event["kv_materialize"]["page_bytes"])
    kv_bytes = int(by_event["kv_materialize"]["bytes"])
    if min(weight_bytes, page_count, page_bytes, kv_bytes, total_bytes) <= 0 or min(
        reserved_bytes,
        free_bytes,
        driver_visible_non_kv_bytes,
        simpler_committed_bytes,
    ) < 0:
        raise ValueError(f"memory evidence contains invalid integer values: {path}")
    if total_bytes - free_bytes != driver_visible_non_kv_bytes:
        raise ValueError(f"memory evidence driver-visible bytes are inconsistent: {path}")
    if max(driver_visible_non_kv_bytes - simpler_committed_bytes, 0) != reserved_bytes:
        raise ValueError(f"memory evidence reserved bytes are inconsistent: {path}")
    if page_count * page_bytes != kv_bytes:
        raise ValueError(f"memory evidence KV bytes do not match page_count * page_bytes: {path}")
    return {
        "source_evidence": str(path.resolve()),
        "workload_id": next(iter(workload_ids)),
        "weight_bytes": weight_bytes,
        "runtime_reserved_bytes": reserved_bytes,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "driver_visible_non_kv_bytes": driver_visible_non_kv_bytes,
        "simpler_committed_bytes": simpler_committed_bytes,
        "page_count": page_count,
        "page_bytes": page_bytes,
        "kv_bytes": kv_bytes,
    }


def _model(operation: str, samples: list[dict[str, Any]], min_workloads: int) -> dict[str, Any]:
    workload_ids = sorted({str(sample["workload_id"]) for sample in samples})
    points: list[dict[str, Any]] = []
    by_bytes: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_bytes.setdefault(int(sample["bytes"]), []).append(sample)
    for byte_count, same_size in sorted(by_bytes.items()):
        durations = [float(sample["duration_us"]) for sample in same_size]
        point_workloads = sorted({str(sample["workload_id"]) for sample in same_size})
        points.append(
            {
                "bytes": byte_count,
                "duration_us": statistics.median(durations),
                "min_duration_us": min(durations),
                "max_duration_us": max(durations),
                "sample_count": len(durations),
                "workload_count": len(point_workloads),
                "workload_ids": point_workloads,
            }
        )
    exact_evidence = {
        "exact_page_count",
        "explicit_runtime_tensor_sum",
        "native_runtime_tensor_sum",
    }
    exact_bytes = all(sample["bytes_evidence"] in exact_evidence for sample in samples)
    points_ready = all(point["workload_count"] >= min_workloads for point in points)
    if not exact_bytes:
        non_strict_reason = "byte count comes from a rounded serving log"
    elif not points_ready:
        non_strict_reason = "one or more byte points lack independent workload coverage"
    else:
        non_strict_reason = None
    return {
        "operation": operation,
        "estimator": "exact_median_or_linear_interpolation_within_measured_bytes",
        "extrapolation": "reject",
        "min_bytes": min(by_bytes),
        "max_bytes": max(by_bytes),
        "workload_count": len(workload_ids),
        "workload_ids": workload_ids,
        "points": points,
        "strict_ready": exact_bytes and points_ready,
        "non_strict_reason": non_strict_reason,
    }


def _footprint_model(samples: list[dict[str, Any]], min_workloads: int) -> dict[str, Any]:
    workload_ids = sorted({str(sample["workload_id"]) for sample in samples})
    values = [int(sample["bytes"]) for sample in samples]
    exact_bytes = all(sample["bytes_evidence"] == "native_memory_info_exact" for sample in samples)
    workloads_ready = len(workload_ids) >= min_workloads
    if not exact_bytes:
        non_strict_reason = "reserved bytes are a difference of rounded serving log values"
    elif not workloads_ready:
        non_strict_reason = "reserved bytes lack independent workload coverage"
    else:
        non_strict_reason = None
    result = {
        "operation": "runtime_reserved_memory",
        "estimator": "median",
        "reserved_bytes": round(statistics.median(values)),
        "min_reserved_bytes": min(values),
        "max_reserved_bytes": max(values),
        "workload_count": len(workload_ids),
        "minimum_independent_workloads": min_workloads,
        "workload_ids": workload_ids,
        "strict_ready": exact_bytes and workloads_ready,
        "non_strict_reason": non_strict_reason,
    }
    exact_fields = (
        "total_bytes",
        "free_bytes",
        "driver_visible_non_kv_bytes",
        "simpler_committed_bytes",
    )
    if exact_bytes and all(all(field in sample for field in exact_fields) for sample in samples):
        totals = {int(sample["total_bytes"]) for sample in samples}
        if len(totals) != 1:
            result["strict_ready"] = False
            result["non_strict_reason"] = "native total memory differs across footprint workloads"
        result.update(
            {
                "native_total_bytes": round(
                    statistics.median(int(sample["total_bytes"]) for sample in samples)
                ),
                "native_free_bytes": round(
                    statistics.median(int(sample["free_bytes"]) for sample in samples)
                ),
                "driver_visible_non_kv_bytes": round(
                    statistics.median(
                        int(sample["driver_visible_non_kv_bytes"]) for sample in samples
                    )
                ),
                "native_simpler_committed_bytes": round(
                    statistics.median(
                        int(sample["simpler_committed_bytes"]) for sample in samples
                    )
                ),
            }
        )
    return result


def build_manifest(
    paths: list[Path],
    min_workloads: int,
    *,
    exact_weight_bytes: int | None = None,
    evidence_paths: list[Path] | None = None,
    kv_page_bytes: int = QWEN3_14B_KV_PAGE_BYTES,
) -> dict[str, Any]:
    samples = [record for path in paths for record in parse_log(path, kv_page_bytes=kv_page_bytes)]
    evidence = [parse_evidence(path) for path in (evidence_paths or [])]
    if evidence:
        weight_values = {int(item["weight_bytes"]) for item in evidence}
        if len(weight_values) != 1:
            raise ValueError("memory evidence disagrees on exact static weight bytes")
        evidence_weight_bytes = next(iter(weight_values))
        if exact_weight_bytes is not None and exact_weight_bytes != evidence_weight_bytes:
            raise ValueError("--weight-bytes disagrees with native memory evidence")
        exact_weight_bytes = evidence_weight_bytes
    if exact_weight_bytes is not None:
        if exact_weight_bytes <= 0:
            raise ValueError("exact weight bytes must be positive")
        for sample in samples:
            if sample["operation"] == "weight_materialize":
                sample["bytes"] = exact_weight_bytes
                sample["bytes_evidence"] = (
                    "native_runtime_tensor_sum" if evidence else "explicit_runtime_tensor_sum"
                )
    if evidence:
        samples = [sample for sample in samples if sample["operation"] != "runtime_reserved_memory"]
        samples.extend(
            {
                "operation": "runtime_reserved_memory",
                "bytes": int(item["runtime_reserved_bytes"]),
                "bytes_evidence": "native_memory_info_exact",
                "source_evidence": str(item["source_evidence"]),
                "workload_id": str(item["workload_id"]),
                "total_bytes": int(item["total_bytes"]),
                "free_bytes": int(item["free_bytes"]),
                "driver_visible_non_kv_bytes": int(
                    item["driver_visible_non_kv_bytes"]
                ),
                "simpler_committed_bytes": int(item["simpler_committed_bytes"]),
            }
            for item in evidence
        )
    footprint_samples = [
        sample for sample in samples if sample["operation"] == "runtime_reserved_memory"
    ]
    duration_samples = [
        sample for sample in samples if sample["operation"] != "runtime_reserved_memory"
    ]
    by_operation: dict[str, list[dict[str, Any]]] = {}
    for sample in duration_samples:
        by_operation.setdefault(str(sample["operation"]), []).append(sample)
    models = [
        _model(operation, records, min_workloads)
        for operation, records in sorted(by_operation.items())
    ]
    footprint_model = (
        _footprint_model(footprint_samples, min_workloads) if footprint_samples else None
    )
    return {
        "schema_version": 1,
        "producer": "pypto-profile/export_memory_profile.py",
        "minimum_independent_workloads": min_workloads,
        "samples": samples,
        "memory_evidence": evidence,
        "models": models,
        "footprint_model": footprint_model,
        "strict_ready": (
            bool(models)
            and all(model["strict_ready"] for model in models)
            and (footprint_model is None or footprint_model["strict_ready"])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=[],
        help="exact native memory evidence JSONL; repeat for independent workloads",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-workloads", type=int, default=3)
    parser.add_argument(
        "--kv-page-bytes",
        type=int,
        default=QWEN3_14B_KV_PAGE_BYTES,
        help="exact bytes per KV page for this model; default preserves the legacy Qwen fixture",
    )
    parser.add_argument(
        "--weight-bytes",
        type=int,
        help="exact sum of tensors uploaded by the pinned model's static materialization phase",
    )
    args = parser.parse_args()
    if args.min_workloads < 1:
        parser.error("--min-workloads must be positive")
    if args.kv_page_bytes <= 0:
        parser.error("--kv-page-bytes must be positive")
    if args.weight_bytes is not None and args.weight_bytes <= 0:
        parser.error("--weight-bytes must be positive")
    return args


def main() -> int:
    args = parse_args()
    missing = [path for path in [*args.log, *args.evidence] if not path.is_file()]
    if missing:
        raise SystemExit(f"missing log: {missing[0]}")
    manifest = build_manifest(
        args.log,
        args.min_workloads,
        exact_weight_bytes=args.weight_bytes,
        evidence_paths=args.evidence,
        kv_page_bytes=args.kv_page_bytes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"output": str(args.output), "strict_ready": manifest["strict_ready"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
