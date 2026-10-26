"""Validate the final vllm-ascend benchmark report contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vllm-ascend-benchmark-report-v3"
STATUS_VALUES = {"success", "invalid", "blocked", "error"}
TOP_LEVEL_KEYS = {
    "schema_version",
    "status",
    "model",
    "load",
    "dataset",
    "arrival",
    "parameters",
    "result",
    "failure",
    "errors",
}
MODEL_KEYS = {"name"}
LOAD_KEYS = {"mode", "selection", "scenario"}
DATASET_KEYS = {
    "mode",
    "name",
    "path",
    "split",
    "subset",
    "revision_or_hash",
    "output_len",
    "text_column",
    "skip_chat_template",
}
ARRIVAL_KEYS = {
    "mode",
    "requested_count",
    "warmups",
    "max_concurrency",
    "request_rate",
    "burstiness",
    "rounds",
}
PARAMETER_KEYS = {
    "backend",
    "tokenizer",
    "base_url",
    "endpoint",
    "seed",
    "seed_source",
    "temperature",
    "temperature_source",
    "top_p",
    "stream",
    "include_usage",
    "num_prompts",
    "max_concurrency",
    "num_warmups",
    "request_rate",
    "burstiness",
    "percentile_metrics",
    "metric_percentiles",
    "goodput",
    "additional",
}
RESULT_KEYS = {
    "requested",
    "completed",
    "failed",
    "ttft_ms",
    "tpot_ms",
    "e2el_ms",
    "requests_per_second",
    "output_tokens_per_second",
    "goodput",
}
RESULT_REQUIRED_KEYS = RESULT_KEYS - {"goodput"}
PERCENTILE_KEYS = {"p50", "p95"}
GOODPUT_KEYS = {"status", "slo_ms", "request_goodput"}
FAILURE_KEYS = {"stage", "type", "reason"}
ERROR_FIELDS = {"stage", "type", "message", "recoverable", "resolved"}
FORBIDDEN_KEYS = {
    "attempt",
    "preflight",
    "service",
    "workload",
    "counts",
    "metrics",
    "throughput",
    "source",
    "oom",
    "cleanup",
    "timings",
    "artifacts",
    "stages",
    "environment",
    "command",
    "raw_result",
    "raw_result_sha256",
    "stdout",
    "stderr",
    "server_log",
}


def _require_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _check_exact_keys(value: Mapping[str, Any], expected: set[str], label: str, errors: list[str]) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        errors.append(f"{label} has unknown fields: {', '.join(sorted(unknown))}")
    if missing:
        errors.append(f"{label} is missing fields: {', '.join(sorted(missing))}")


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
        errors.append(f"{label} must be a non-empty string or null")


def _errors(report: Mapping[str, Any], errors: list[str]) -> None:
    value = report.get("errors")
    if not isinstance(value, list):
        errors.append("errors must be an array")
        return
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"errors[{index}] must be an object")
            continue
        missing = ERROR_FIELDS - set(item)
        unknown = set(item) - ERROR_FIELDS - {"repair_action"}
        if missing:
            errors.append(f"errors[{index}] is missing fields: {', '.join(sorted(missing))}")
        if unknown:
            errors.append(f"errors[{index}] has unknown fields: {', '.join(sorted(unknown))}")
        for field in ("stage", "type", "message"):
            if not isinstance(item.get(field), str) or not item.get(field):
                errors.append(f"errors[{index}].{field} must be a non-empty string")
        for field in ("recoverable", "resolved"):
            if not isinstance(item.get(field), bool):
                errors.append(f"errors[{index}].{field} must be boolean")
        if "repair_action" in item and item["repair_action"] is not None and not isinstance(item["repair_action"], str):
            errors.append(f"errors[{index}].repair_action must be a string or null")


def _failure(report: Mapping[str, Any], errors: list[str]) -> None:
    value = report.get("failure")
    if value is None:
        return
    failure = _require_mapping(value, "failure", errors)
    _check_exact_keys(failure, FAILURE_KEYS, "failure", errors)
    for field in FAILURE_KEYS:
        if not isinstance(failure.get(field), str) or not failure.get(field):
            errors.append(f"failure.{field} must be a non-empty string")


def _percentiles(value: Any, label: str, errors: list[str]) -> None:
    metrics = _require_mapping(value, label, errors)
    _check_exact_keys(metrics, PERCENTILE_KEYS, label, errors)
    for field in PERCENTILE_KEYS:
        metric = metrics.get(field)
        if metric is not None and (not isinstance(metric, (int, float)) or isinstance(metric, bool)):
            errors.append(f"{label}.{field} must be a number or null")


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Return final-report contract violations; an empty list means valid."""
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("status") not in STATUS_VALUES:
        errors.append("status must be success, invalid, blocked, or error")
    _check_exact_keys(report, TOP_LEVEL_KEYS, "report", errors)
    errors.extend(f"intermediate field is forbidden: {path}" for path in _find_forbidden_keys(report))

    model = _require_mapping(report.get("model"), "model", errors)
    _check_exact_keys(model, MODEL_KEYS, "model", errors)
    _optional_string(model.get("name"), "model.name", errors)

    load = _require_mapping(report.get("load"), "load", errors)
    _check_exact_keys(load, LOAD_KEYS, "load", errors)
    _optional_string(load.get("mode"), "load.mode", errors)
    _optional_string(load.get("selection"), "load.selection", errors)
    _optional_string(load.get("scenario"), "load.scenario", errors)

    dataset = _require_mapping(report.get("dataset"), "dataset", errors)
    _check_exact_keys(dataset, DATASET_KEYS, "dataset", errors)
    for field in ("mode", "name", "path", "split", "subset", "revision_or_hash", "text_column"):
        _optional_string(dataset.get(field), f"dataset.{field}", errors)
    for field in ("output_len",):
        value = dataset.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            errors.append(f"dataset.{field} must be a non-negative integer or null")
    skip_chat_template = dataset.get("skip_chat_template")
    if skip_chat_template is not None and not isinstance(skip_chat_template, bool):
        errors.append("dataset.skip_chat_template must be boolean or null")

    arrival = _require_mapping(report.get("arrival"), "arrival", errors)
    _check_exact_keys(arrival, ARRIVAL_KEYS, "arrival", errors)
    _optional_string(arrival.get("mode"), "arrival.mode", errors)
    for field in ("requested_count", "warmups", "max_concurrency", "rounds"):
        value = arrival.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            errors.append(f"arrival.{field} must be a non-negative integer or null")
    for field in ("request_rate", "burstiness"):
        value = arrival.get(field)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0):
            errors.append(f"arrival.{field} must be a non-negative number or null")

    parameters = _require_mapping(report.get("parameters"), "parameters", errors)
    _check_exact_keys(parameters, PARAMETER_KEYS, "parameters", errors)
    for field in ("backend", "tokenizer", "base_url", "seed_source", "temperature_source"):
        _optional_string(parameters.get(field), f"parameters.{field}", errors)
    endpoint = parameters.get("endpoint")
    if endpoint is not None and (
        not isinstance(endpoint, str) or not endpoint.startswith("/") or endpoint.startswith("//") or "://" in endpoint
    ):
        errors.append("parameters.endpoint must be an HTTP path or null")
    for field in ("seed", "num_prompts", "max_concurrency", "num_warmups"):
        value = parameters.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            errors.append(f"parameters.{field} must be a non-negative integer or null")
    for field in ("temperature", "request_rate", "burstiness"):
        value = parameters.get(field)
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
        ):
            errors.append(f"parameters.{field} must be a non-negative number or null")
    top_p = parameters.get("top_p")
    if top_p is not None and (
        not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not 0 <= top_p <= 1
    ):
        errors.append("parameters.top_p must be a number in [0, 1] or null")
    for field in ("stream", "include_usage"):
        value = parameters.get(field)
        if value is not None and not isinstance(value, bool):
            errors.append(f"parameters.{field} must be boolean or null")
    percentile_metrics = parameters.get("percentile_metrics")
    if not isinstance(percentile_metrics, list) or any(
        not isinstance(value, str) or not value for value in percentile_metrics
    ):
        errors.append("parameters.percentile_metrics must be a string array")
    metric_percentiles = parameters.get("metric_percentiles")
    if not isinstance(metric_percentiles, list) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 100
        for value in metric_percentiles
    ):
        errors.append("parameters.metric_percentiles must be a number array in [0, 100]")
    parameter_goodput = parameters.get("goodput")
    if parameter_goodput is not None:
        if not isinstance(parameter_goodput, Mapping) or not parameter_goodput:
            errors.append("parameters.goodput must be a non-empty object or null")
        elif any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            for value in parameter_goodput.values()
        ):
            errors.append("parameters.goodput values must be non-negative numbers")
    if not isinstance(parameters.get("additional"), Mapping):
        errors.append("parameters.additional must be an object")

    result = _require_mapping(report.get("result"), "result", errors)
    unknown_result = set(result) - RESULT_KEYS
    missing_result = RESULT_REQUIRED_KEYS - set(result)
    if unknown_result:
        errors.append(f"result has unknown fields: {', '.join(sorted(unknown_result))}")
    if missing_result:
        errors.append(f"result is missing fields: {', '.join(sorted(missing_result))}")
    for field in ("requested", "completed", "failed"):
        value = result.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            errors.append(f"result.{field} must be a non-negative integer or null")
    for field in ("ttft_ms", "tpot_ms", "e2el_ms"):
        _percentiles(result.get(field), f"result.{field}", errors)
    for field in ("requests_per_second", "output_tokens_per_second"):
        value = result.get(field)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            errors.append(f"result.{field} must be a number or null")
    if "goodput" in result:
        goodput = _require_mapping(result.get("goodput"), "result.goodput", errors)
        _check_exact_keys(goodput, GOODPUT_KEYS, "result.goodput", errors)
        status = goodput.get("status")
        if status not in {"reported", "missing"}:
            errors.append("result.goodput.status must be reported or missing")
        slo = goodput.get("slo_ms")
        if not isinstance(slo, Mapping):
            errors.append("result.goodput.slo_ms must be an object")
        elif not slo:
            errors.append("result.goodput.slo_ms must be non-empty when goodput is present")
        elif any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            for value in slo.values()
        ):
            errors.append("result.goodput.slo_ms values must be non-negative numbers")
        request_goodput = goodput.get("request_goodput")
        if request_goodput is not None and (
            not isinstance(request_goodput, (int, float)) or isinstance(request_goodput, bool)
        ):
            errors.append("result.goodput.request_goodput must be a number or null")
        if status == "reported" and request_goodput is None:
            errors.append("reported goodput must include request_goodput")
        if status == "missing" and report.get("status") == "success":
            errors.append("success reports cannot use missing goodput")
        if parameter_goodput is None:
            errors.append("result.goodput requires parameters.goodput")

    _failure(report, errors)
    if report.get("status") == "success" and report.get("failure") is not None:
        errors.append("success reports must set failure to null")
    if report.get("status") != "success" and report.get("failure") is None:
        errors.append("non-success reports must provide failure")
    _errors(report, errors)
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
    violations = validate_report(value)
    if violations:
        for violation in violations:
            print(violation)
        return 1
    print("benchmark report valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
