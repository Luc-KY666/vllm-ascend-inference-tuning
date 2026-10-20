"""Validate the canonical vLLM-Ascend benchmark report contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vllm-ascend-benchmark-report-v2"
STATUS_VALUES = {"running", "success", "invalid", "blocked", "error"}
TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "attempt",
    "preflight",
    "service",
    "workload",
    "counts",
    "metrics",
    "goodput",
    "throughput",
    "source",
    "failure",
    "oom",
    "cleanup",
    "timings",
    "errors",
}
REQUIRED_SECTIONS = {
    "attempt",
    "preflight",
    "service",
    "workload",
    "counts",
    "metrics",
    "goodput",
    "throughput",
    "source",
    "oom",
    "cleanup",
    "timings",
    "errors",
}
FORBIDDEN_TOP_LEVEL_KEYS = {
    "inputs",
    "model",
    "deployment",
    "resources",
    "checks",
    "service_identity_valid",
    "benchmark_result",
}
FORBIDDEN_OUTPUT_MARKERS = {
    "benchmark_result.json",
    "stdout",
    "stderr",
    "command_transcript",
    "server_log",
}


def _require_mapping(report: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return {}
    return value


def _walk(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, str(key), child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            if isinstance(child, (Mapping, list)):
                yield from _walk(child, child_path)
            else:
                yield child_path, "", child


def _path_value(value: Any, label: str, errors: list[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty path or null")
        return None
    return value


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return contract violations; an empty list means the report is canonical."""

    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("status") not in STATUS_VALUES:
        errors.append("status must be running, success, invalid, blocked, or error")

    unknown = set(report) - TOP_LEVEL_KEYS
    if unknown:
        errors.append("unknown top-level fields: " + ", ".join(sorted(unknown)))
    missing = REQUIRED_SECTIONS - set(report)
    if missing:
        errors.append("missing top-level sections: " + ", ".join(sorted(missing)))
    old = sorted(set(report) & FORBIDDEN_TOP_LEVEL_KEYS)
    if old:
        errors.append("legacy top-level fields are forbidden: " + ", ".join(old))

    for key in REQUIRED_SECTIONS:
        _require_mapping(report, key, errors) if key != "errors" else None
    if not isinstance(report.get("errors"), list):
        errors.append("errors must be an array")

    preflight = _require_mapping(report, "preflight", errors)
    environment_probe = _path_value(
        preflight.get("environment_probe"), "preflight.environment_probe", errors
    )
    deployment_report = _path_value(
        preflight.get("deployment_report"), "preflight.deployment_report", errors
    )

    source = _require_mapping(report, "source", errors)
    raw_result = _path_value(source.get("raw_result"), "source.raw_result", errors)
    if raw_result is not None and Path(raw_result).name != "raw_benchmark_result.json":
        errors.append("source.raw_result must point to raw_benchmark_result.json")
    raw_digest = source.get("raw_result_sha256")
    if raw_result is None:
        if raw_digest is not None:
            errors.append("source.raw_result_sha256 must be null when raw_result is null")
    elif (
        not isinstance(raw_digest, str)
        or len(raw_digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in raw_digest)
    ):
        errors.append("source.raw_result_sha256 must be a 64-character hexadecimal digest")

    paths = [
        ("preflight.environment_probe", environment_probe),
        ("preflight.deployment_report", deployment_report),
        ("source.raw_result", raw_result),
    ]
    seen: dict[str, str] = {}
    for owner, path in paths:
        if path is None:
            continue
        previous = seen.setdefault(path, owner)
        if previous != owner:
            errors.append(f"artifact path is duplicated: {path} ({previous}, {owner})")

    failure = report.get("failure")
    if failure is not None:
        if not isinstance(failure, Mapping):
            errors.append("failure must be an object or null")
        else:
            refs = failure.get("artifacts", [])
            if isinstance(refs, str):
                refs = [refs]
            if not isinstance(refs, list) or any(
                not isinstance(ref, str) or not ref or "/" in ref or "\\" in ref
                for ref in refs
            ):
                errors.append("failure.artifacts must be an array of logical references")

    for path, key, value in _walk(report):
        lowered_key = key.lower()
        lowered_value = value.lower() if isinstance(value, str) else ""
        if lowered_key in FORBIDDEN_OUTPUT_MARKERS:
            errors.append(f"large output field is forbidden: {path}")
        value_for_scan = lowered_value.replace("raw_benchmark_result.json", "")
        if any(marker in value_for_scan for marker in FORBIDDEN_OUTPUT_MARKERS):
            errors.append(f"forbidden output marker in report: {path}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="benchmark_report.json path")
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"cannot read report: {exc}")
        return 2
    if not isinstance(value, Mapping):
        print("report must be a JSON object")
        return 2
    errors = validate_report(value)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("benchmark report valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
