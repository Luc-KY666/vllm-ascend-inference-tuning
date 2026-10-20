"""Validate the canonical vllm-ascend deployment report contract."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vllm-ascend-deployment-report-v2"
STATUS_VALUES = {"running", "preflight_failed", "service_ready", "failed"}
TOP_LEVEL_KEYS = {
    "schema_version", "status", "attempt", "input", "environment", "stages",
    "artifacts", "failure", "timings",
}
STAGE_KEYS = {"input_validation", "static_oom", "tp_resolution", "resources", "service", "cleanup"}
PROBE_ARTIFACTS = ("environment_probe", "sandbox_detection", "serve_help", "bench_help")
FORBIDDEN_ENVIRONMENT_KEYS = {
    "probe_path", "sandbox_path", "environment_probe", "sandbox_detection", "serve_help", "bench_help",
}
FORBIDDEN_INPUT_KEYS = {
    "environment_probe", "sandbox_detection", "serve_help", "bench_help", "probe_path", "sandbox_path",
}
FORBIDDEN_ALIASES = {
    "inputs", "model", "deployment", "resources", "preflight", "service_identity_valid", "checks", "oom", "error",
}
INTERMEDIATE_NAMES = (
    "deployment.json", "static-oom-input.json", "static-oom-report.json", "npu-resource-probe.json",
    "port-resource-probe.json", "acceptance-checks.json", "cleanup.json",
)


def _find_keys(value: Any, forbidden: set[str], prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in forbidden:
                found.append(key_path)
            found.extend(_find_keys(child, forbidden, key_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_keys(child, forbidden, f"{prefix}[{index}]"))
    return found


def _strings(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value


def _environment_path_keys(value: Any, prefix: str = "environment") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_path = f"{prefix}.{key}"
            lowered = str(key).lower()
            if "path" in lowered or lowered in FORBIDDEN_ENVIRONMENT_KEYS:
                found.append(key_path)
            found.extend(_environment_path_keys(child, key_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_environment_path_keys(child, f"{prefix}[{index}]"))
    return found


def artifact_path_index(report: Mapping[str, Any]) -> dict[str, list[str]]:
    """Return every non-null path-like value under artifacts and its owners."""
    locations: dict[str, list[str]] = defaultdict(list)

    def visit(value: Any, path: str, path_value: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_name = str(key)
                child_is_path = (
                    key_name in PROBE_ARTIFACTS
                    or key_name in {"server_log", "command_transcript", "path", "file", "filename"}
                    or key_name.endswith("_path")
                )
                visit(child, f"{path}.{key}", child_is_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", path_value)
        elif path_value and isinstance(value, str) and value:
            locations[value].append(path)

    artifacts = report.get("artifacts")
    if isinstance(artifacts, Mapping):
        visit(artifacts, "artifacts")
    return dict(locations)


def _value_locations(value: Any, target: str, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_value_locations(child, target, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_value_locations(child, target, f"{prefix}[{index}]"))
    elif value == target:
        found.append(prefix)
    return found


def _require_mapping(report: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = report.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} 必须是 object")
        return {}
    return value


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return contract violations; an empty list means the report is canonical."""
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {SCHEMA_VERSION}")
    if report.get("status") not in STATUS_VALUES:
        errors.append("status 必须是 running、preflight_failed、service_ready 或 failed")
    unknown = set(report) - TOP_LEVEL_KEYS
    if unknown:
        errors.append("报告包含未知顶层字段: " + ", ".join(sorted(unknown)))
    missing_top = TOP_LEVEL_KEYS - set(report)
    if missing_top:
        errors.append("报告缺少顶层字段: " + ", ".join(sorted(missing_top)))
    old = sorted(set(report) & FORBIDDEN_ALIASES)
    if old:
        errors.append("v2 报告不得包含旧字段: " + ", ".join(old))

    _require_mapping(report, "attempt", errors)
    input_section = _require_mapping(report, "input", errors)
    input_paths = _find_keys(input_section, FORBIDDEN_INPUT_KEYS, "input")
    errors.extend(f"input 不得保存 probe/sandbox 路径: {path}" for path in input_paths)
    _require_mapping(report, "timings", errors)
    environment = _require_mapping(report, "environment", errors)
    forbidden = _find_keys(environment, FORBIDDEN_ENVIRONMENT_KEYS, "environment")
    forbidden.extend(_environment_path_keys(environment))
    forbidden = sorted(set(forbidden))
    errors.extend(f"environment 不得保存路径字段: {path}" for path in forbidden)
    for value in _strings(environment):
        if any(marker in value.lower() for marker in ("environment_probe", "sandbox-detection", "serve-help", "bench-help")):
            errors.append("environment 不得保存 probe/help 文件路径")
            break

    stages = _require_mapping(report, "stages", errors)
    errors.extend(f"stages 缺少 {key}" for key in sorted(STAGE_KEYS - set(stages)))
    for key in sorted(STAGE_KEYS & set(stages)):
        if not isinstance(stages[key], Mapping):
            errors.append(f"stages.{key} 必须是 object")

    artifacts = _require_mapping(report, "artifacts", errors)
    probe = artifacts.get("probe")
    if not isinstance(probe, Mapping):
        errors.append("artifacts.probe 必须是 object")
    else:
        unknown_probe = set(probe) - set(PROBE_ARTIFACTS)
        if unknown_probe:
            errors.append("artifacts.probe 包含未知字段: " + ", ".join(sorted(unknown_probe)))
        missing_probe = set(PROBE_ARTIFACTS) - set(probe)
        if missing_probe:
            errors.append("artifacts.probe 缺少字段: " + ", ".join(sorted(missing_probe)))
        for key in PROBE_ARTIFACTS:
            value = probe.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                errors.append(f"artifacts.probe.{key} 必须是非空路径或 null")
    raw_logs = artifacts.get("raw_logs")
    if not isinstance(raw_logs, Mapping):
        errors.append("artifacts.raw_logs 必须是 object")
    else:
        unknown_logs = set(raw_logs) - {"server_log", "command_transcript"}
        missing_logs = {"server_log", "command_transcript"} - set(raw_logs)
        if unknown_logs:
            errors.append("artifacts.raw_logs 只能包含 server_log 和 command_transcript")
        if missing_logs:
            errors.append("artifacts.raw_logs 缺少字段: " + ", ".join(sorted(missing_logs)))
    for key in set(artifacts) - {"probe", "raw_logs"}:
        errors.append(f"artifacts 不得包含旧分组: {key}")

    for path, locations in artifact_path_index(report).items():
        if len(locations) > 1:
            errors.append(f"artifact 路径重复: {path} ({', '.join(locations)})")
        all_locations = _value_locations(report, path)
        canonical = set(locations)
        outside = [location for location in all_locations if location not in canonical]
        if outside:
            errors.append(f"artifact 路径重复: {path} ({', '.join(locations + outside)})")

    failure = report.get("failure")
    if failure is not None and not isinstance(failure, Mapping):
        errors.append("failure 必须是 object 或 null")
    elif isinstance(failure, Mapping) and failure.get("artifacts") is not None:
        refs = failure["artifacts"]
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or not ref.startswith("artifacts.") for ref in refs
        ):
            errors.append("failure.artifacts 必须使用 artifacts.* 逻辑引用")

    for value in _strings(report):
        lowered = value.lower()
        for marker in INTERMEDIATE_NAMES:
            if marker in lowered:
                errors.append(f"报告不得引用部署侧中间文件: {marker}")
                break
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="deployment_report.json 路径")
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"无法读取报告: {exc}")
        return 2
    if not isinstance(value, Mapping):
        print("报告必须是 JSON object")
        return 2
    errors = validate_report(value)
    if errors:
        for error in errors:
            print(error)
        return 1
    print("deployment report valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
