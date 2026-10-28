"""Validate the final vllm-ascend deployment report contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vllm-ascend-deployment-report-v3"
STATUS_VALUES = {"success", "failed", "preflight_failed"}
TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "model",
    "resources",
    "service",
    "launch",
    "failure",
    "errors",
}
MODEL_KEYS = {"name", "weights_path"}
RESOURCE_KEYS = {"npu", "port"}
SERVICE_KEYS = {
    "backend",
    "base_url",
    "endpoint",
    "health_endpoint",
    "models_endpoint",
    "tokenizer",
}
LAUNCH_KEYS = {"parameters"}
FAILURE_KEYS = {"stage", "type", "reason"}
ERROR_FIELDS = {"stage", "type", "message", "recoverable", "resolved"}
FORBIDDEN_KEYS = {
    "attempt",
    "input",
    "environment",
    "stages",
    "artifacts",
    "timings",
    "cleanup",
    "preflight",
    "oom",
    "source",
    "metrics",
    "throughput",
}


def _require_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} 必须是 object")
        return {}
    return value


def _check_exact_keys(value: Mapping[str, Any], expected: set[str], label: str, errors: list[str]) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        errors.append(f"{label} 包含未知字段: {', '.join(sorted(unknown))}")
    if missing:
        errors.append(f"{label} 缺少字段: {', '.join(sorted(missing))}")


def _find_forbidden_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_KEYS:
                found.append(path)
            found.extend(_find_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def _optional_string(value: Any, label: str, errors: list[str]) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        errors.append(f"{label} 必须是非空字符串或 null")


def _path(value: Any, label: str, errors: list[str]) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        errors.append(f"{label} 必须是非空路径或 null")


def _errors(report: Mapping[str, Any], errors: list[str]) -> None:
    value = report.get("errors")
    if not isinstance(value, list):
        errors.append("errors 必须是 array")
        return
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"errors[{index}] 必须是 object")
            continue
        missing = ERROR_FIELDS - set(item)
        unknown = set(item) - ERROR_FIELDS - {"repair_action"}
        if missing:
            errors.append(f"errors[{index}] 缺少字段: {', '.join(sorted(missing))}")
        if unknown:
            errors.append(f"errors[{index}] 包含未知字段: {', '.join(sorted(unknown))}")
        for field in ("stage", "type", "message"):
            if not isinstance(item.get(field), str) or not item.get(field):
                errors.append(f"errors[{index}].{field} 必须是非空字符串")
        for field in ("recoverable", "resolved"):
            if not isinstance(item.get(field), bool):
                errors.append(f"errors[{index}].{field} 必须是 boolean")
        if "repair_action" in item and item["repair_action"] is not None and not isinstance(item["repair_action"], str):
            errors.append(f"errors[{index}].repair_action 必须是字符串或 null")


def _failure(report: Mapping[str, Any], errors: list[str]) -> None:
    value = report.get("failure")
    if value is None:
        return
    failure = _require_mapping(value, "failure", errors)
    _check_exact_keys(failure, FAILURE_KEYS, "failure", errors)
    for field in FAILURE_KEYS:
        if not isinstance(failure.get(field), str) or not failure.get(field):
            errors.append(f"failure.{field} 必须是非空字符串")


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return final-report contract violations; an empty list means valid."""
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {SCHEMA_VERSION}")
    if report.get("status") not in STATUS_VALUES:
        errors.append("status 必须是 success、failed 或 preflight_failed")
    _check_exact_keys(report, TOP_LEVEL_KEYS, "报告", errors)
    errors.extend(f"最终报告不得包含中间字段: {path}" for path in _find_forbidden_keys(report))

    model = _require_mapping(report.get("model"), "model", errors)
    _check_exact_keys(model, MODEL_KEYS, "model", errors)
    _optional_string(model.get("name"), "model.name", errors)
    _path(model.get("weights_path"), "model.weights_path", errors)

    resources = _require_mapping(report.get("resources"), "resources", errors)
    _check_exact_keys(resources, RESOURCE_KEYS, "resources", errors)
    npu = resources.get("npu")
    if not isinstance(npu, list) or any(not isinstance(value, (int, str)) or isinstance(value, bool) for value in npu):
        errors.append("resources.npu 必须是 NPU ID 数组")
    port = resources.get("port")
    if port is not None and (not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535):
        errors.append("resources.port 必须是 1-65535 的整数或 null")

    service = _require_mapping(report.get("service"), "service", errors)
    _check_exact_keys(service, SERVICE_KEYS, "service", errors)
    _optional_string(service.get("backend"), "service.backend", errors)
    _optional_string(service.get("base_url"), "service.base_url", errors)
    _path(service.get("tokenizer"), "service.tokenizer", errors)
    for field in ("endpoint", "health_endpoint", "models_endpoint"):
        value = service.get(field)
        if not isinstance(value, str) or not value.startswith("/") or value.startswith("//") or "://" in value:
            errors.append(f"service.{field} 必须是 HTTP 路径")

    launch = _require_mapping(report.get("launch"), "launch", errors)
    _check_exact_keys(launch, LAUNCH_KEYS, "launch", errors)
    parameters = launch.get("parameters")
    if not isinstance(parameters, Mapping):
        errors.append("launch.parameters 必须是 object")

    _failure(report, errors)
    if report.get("status") == "success" and report.get("failure") is not None:
        errors.append("success 报告的 failure 必须为 null")
    if report.get("status") in {"failed", "preflight_failed"} and report.get("failure") is None:
        errors.append("失败报告必须提供 failure")
    _errors(report, errors)
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
    violations = validate_report(value)
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print("deployment report valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
