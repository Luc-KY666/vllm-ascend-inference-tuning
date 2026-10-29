"""Validate a complete vllm serve argv and emit normalized JSON on stdout."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vllm-ascend-deployment-input-v3"
RUNTIME_SCHEMA_VERSION = "vllm-ascend-runtime-v1"
PD_SCHEMA_VERSION = "vllm-ascend-deployment-input-v2"
DEFAULTS = {
    "dp": 1,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096,
    "gpu_memory_utilization": 0.9,
    "min_required_context_len": 4096,
    "block_size": 128,
}
SERVICE_DEFAULTS = {
    "backend": "vllm",
    "endpoint": "/v1/chat/completions",
}
RESOURCE_FIELDS = {
    "visible_devices",
    "role_device_count",
    "port_candidates",
    "preferred_port",
    "port",
    "max_tp_dp",
    "card_memory_bytes",
    "device_count",
}
CONFIG_SUMMARY_FIELDS = {
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "intermediate_size",
    "num_experts",
    "torch_dtype",
    "tie_word_embeddings",
    "has_quantization_config",
}
HIDDEN_ARGUMENTS = {"--config", "--model"}
# Only these options feed normalized fields; all other option values stay opaque.
EXTRACTED_VALUE_ARGUMENTS = {
    "--served-model-name",
    "--tokenizer",
    "--tensor-parallel-size",
    "-tp",
    "--data-parallel-size",
    "-dp",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--gpu-memory-utilization",
    "--max-model-len",
    "--block-size",
    "--dtype",
    "--quantization",
    "-q",
    "--port",
}
OPTION_RE = re.compile(
    r"(?<![\w-])(?:--[A-Za-z0-9][A-Za-z0-9_-]*|-[A-Za-z0-9][A-Za-z0-9_-]*)"
)
NEGATIVE_NUMBER_RE = re.compile(r"^-\d+(?:\.\d+)?$")


class ContractError(ValueError):
    """An invalid Provider contract."""


@dataclass(frozen=True)
class ParsedOption:
    """One option occurrence from the supplied serve argv."""

    name: str
    values: tuple[str, ...]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int(value: Any, field: str) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _device_ids(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    result = [str(item) for item in value]
    if any(not item.isdigit() for item in result) or len(set(result)) != len(result):
        return None
    return result


def _ports(value: Any) -> list[int] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    result: list[int] = []
    for item in value:
        try:
            port = int(item)
        except (TypeError, ValueError):
            return None
        if isinstance(item, bool) or not 1 <= port <= 65535:
            return None
        result.append(port)
    return result if len(set(result)) == len(result) else None


def _config_with_digest(
    model_path: Path, explicit: Any
) -> tuple[Path | None, dict[str, Any] | None, str | None, str | None]:
    config_path = Path(explicit) if isinstance(explicit, str) and explicit else model_path / "config.json"
    if not config_path.is_file() or not os.access(config_path, os.R_OK):
        return config_path, None, f"config.json 不可读: {config_path}", None
    try:
        # Read once: the same bytes provide both the JSON value and its report digest.
        raw = config_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return config_path, None, f"无法读取 config.json: {exc}", locals().get("digest")
    if not isinstance(value, dict):
        return config_path, None, "config.json 必须是 JSON object", digest
    return config_path, value, None, digest


def _config(
    model_path: Path, explicit: Any
) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    """Backward-compatible config loader for callers that do not need a digest."""
    path, value, error, _digest = _config_with_digest(model_path, explicit)
    return path, value, error


def _head_count(config: Mapping[str, Any]) -> int | None:
    source = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    value = source.get("num_attention_heads")
    return value if _positive_int(value, "num_attention_heads") else None


def _config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only architecture facts needed by static OOM and service startup."""
    source: dict[str, Any] = dict(config)
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping):
        source.update(text_config)
    summary = {
        key: source[key]
        for key in CONFIG_SUMMARY_FIELDS
        if key in source and isinstance(source[key], (str, int, float, bool))
    }
    if config.get("quantization_config") not in (None, {}, False, ""):
        summary["has_quantization_config"] = True
    return summary


def _help_declaration(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("-"):
        return None
    # The first two spaces separate argparse's declaration from its description.
    return re.split(r"\s{2,}", stripped, maxsplit=1)[0]


def parse_serve_help(help_text: str) -> set[str]:
    """Extract option names and aliases from ``vllm serve --help=all``."""

    if not re.search(r"usage:\s*vllm\s+serve\b", help_text):
        raise ContractError("vllm serve help 缺少有效 usage")

    options: set[str] = set()
    declarations = 0
    for line in help_text.splitlines():
        declaration = _help_declaration(line)
        if declaration is None:
            continue
        names = tuple(dict.fromkeys(OPTION_RE.findall(declaration)))
        if not names:
            continue
        declarations += 1
        options.update(_canonical_option_name(name) for name in names)
    if declarations == 0 or "--help" not in options:
        raise ContractError("vllm serve help 未找到可用选项")
    return options


def _canonical_option_name(name: str) -> str:
    if name.startswith("--"):
        return "--" + name[2:].replace("_", "-")
    return name


def _split_option_token(token: str) -> tuple[str, str | None] | None:
    if not token.startswith("-") or token == "-" or NEGATIVE_NUMBER_RE.fullmatch(token):
        return None
    if "=" in token:
        name, value = token.split("=", 1)
        return _canonical_option_name(name), value
    return _canonical_option_name(token), None


def _parse_serve_argv(
    argv: Any, options: set[str]
) -> tuple[str, list[ParsedOption], list[str]]:
    if not isinstance(argv, list) or not argv:
        raise ContractError("serve.argv 必须是非空字符串数组")
    if any(not isinstance(item, str) or not item for item in argv):
        raise ContractError("serve.argv 中的 token 必须是非空字符串")
    model_path = argv[0]
    if model_path.startswith("-"):
        raise ContractError("serve.argv 第一项必须是模型路径 positional 参数")

    parsed: list[ParsedOption] = []
    errors: list[str] = []
    for index, token in enumerate(argv[1:], start=1):
        split = _split_option_token(token)
        if split is None:
            continue
        name, inline_value = split
        if name not in options:
            errors.append(f"vllm serve 不支持参数: {name}")
            continue
        if name in HIDDEN_ARGUMENTS:
            errors.append(f"暂不允许通过 {name} 引入第二份启动参数来源")
        values: list[str] = []
        if inline_value is not None:
            values.append(inline_value)
        elif name in EXTRACTED_VALUE_ARGUMENTS and index + 1 < len(argv):
            next_split = _split_option_token(argv[index + 1])
            if next_split is None:
                values.append(argv[index + 1])
        parsed.append(ParsedOption(name, tuple(values)))
    return model_path, parsed, errors


def _option_occurrences(parsed: list[ParsedOption], names: tuple[str, ...]) -> list[ParsedOption]:
    return [item for item in parsed if item.name in names]


def _single_value(
    parsed: list[ParsedOption],
    names: tuple[str, ...],
    field: str,
    errors: list[str],
    *,
    unique: bool = True,
    require_value: bool = True,
) -> str | None:
    matches = _option_occurrences(parsed, names)
    if unique and len(matches) > 1:
        errors.append(f"{field} 不能重复指定: " + ", ".join(item.name for item in matches))
        for match in matches:
            if require_value and not match.values:
                errors.append(f"参数缺少 value: {match.name}")
        return None
    if not matches:
        return None
    values = matches[0].values
    if not values:
        if require_value:
            errors.append(f"参数缺少 value: {matches[0].name}")
        return None
    if len(values) != 1:
        errors.append(f"{field} 必须只有一个 value")
        return None
    return values[0]


def _single_int(
    parsed: list[ParsedOption],
    names: tuple[str, ...],
    field: str,
    errors: list[str],
    *,
    unique: bool = True,
) -> int | None:
    raw = _single_value(parsed, names, field, errors, unique=unique)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        errors.append(f"{field} 必须是整数")
        return None


def _single_float(
    parsed: list[ParsedOption],
    names: tuple[str, ...],
    field: str,
    errors: list[str],
    *,
    unique: bool = True,
) -> float | None:
    raw = _single_value(parsed, names, field, errors, unique=unique)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        errors.append(f"{field} 必须是数字")
        return None


def _serve_fields(
    model_path_value: str,
    parsed: list[ParsedOption],
    errors: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"model_path": model_path_value}
    result["served_model_name"] = _single_value(
        parsed, ("--served-model-name",), "served_model_name", errors, unique=False
    )
    result["tokenizer"] = _single_value(
        parsed, ("--tokenizer",), "tokenizer", errors, unique=False, require_value=False
    )
    result["tp"] = _single_int(parsed, ("--tensor-parallel-size", "-tp"), "runtime.tp", errors)
    result["dp"] = _single_int(parsed, ("--data-parallel-size", "-dp"), "runtime.dp", errors)
    result["max_num_seqs"] = _single_int(parsed, ("--max-num-seqs",), "runtime.max_num_seqs", errors)
    result["max_num_batched_tokens"] = _single_int(
        parsed, ("--max-num-batched-tokens",), "runtime.max_num_batched_tokens", errors
    )
    result["gpu_memory_utilization"] = _single_float(
        parsed, ("--gpu-memory-utilization",), "runtime.gpu_memory_utilization", errors
    )
    # vLLM accepts integer lengths as well as values such as ``auto`` and
    # human-readable suffixes (for example ``25.6k``); preserve this token and
    # leave its complete semantic validation to vLLM startup.
    result["max_model_len"] = _single_value(
        parsed,
        ("--max-model-len",),
        "runtime.max_model_len",
        errors,
        unique=False,
        require_value=False,
    )
    result["block_size"] = _single_int(parsed, ("--block-size",), "runtime.block_size", errors)
    result["dtype"] = _single_value(
        parsed, ("--dtype",), "runtime.dtype", errors, unique=False, require_value=False
    )
    result["quantization"] = _single_value(
        parsed,
        ("--quantization", "-q"),
        "runtime.quantization",
        errors,
        unique=False,
        require_value=False,
    )
    result["port"] = _single_int(parsed, ("--port",), "resources.port", errors)
    return result


def _validate_serve_input(
    payload: Mapping[str, Any], help_text: str
) -> tuple[dict[str, Any], list[str], list[str]]:
    options = parse_serve_help(help_text)
    serve = _mapping(payload.get("serve"))
    model_path_value, parsed, command_errors = _parse_serve_argv(serve.get("argv"), options)
    errors = list(command_errors)
    warnings: list[str] = []
    fields = _serve_fields(model_path_value, parsed, errors)
    if not isinstance(fields.get("served_model_name"), str) or not fields["served_model_name"].strip():
        errors.append("serve.argv 必须包含非空 --served-model-name")
    if fields.get("tokenizer") is not None and not str(fields["tokenizer"]).strip():
        errors.append("--tokenizer 必须是非空字符串")
    return {"fields": fields, "parsed": parsed, "options": options}, errors, warnings


def _config_and_service(
    payload: Mapping[str, Any], fields: Mapping[str, Any], errors: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None, Path | None]:
    model_in = _mapping(payload.get("model"))
    service_in = _mapping(payload.get("service"))
    outputs_in = _mapping(payload.get("outputs"))
    model_path_value = fields.get("model_path")
    model_path = Path(model_path_value) if isinstance(model_path_value, str) else Path(".")
    if not isinstance(model_path_value, str) or not model_path_value.strip():
        errors.append("serve.argv 第一项模型路径必须是非空字符串")
    elif not model_path.is_dir() or not os.access(model_path, os.R_OK):
        errors.append(f"模型路径不可读: {model_path}")

    served_name = fields.get("served_model_name")
    backend = service_in.get("backend", SERVICE_DEFAULTS["backend"])
    endpoint = service_in.get("endpoint", SERVICE_DEFAULTS["endpoint"])
    if backend is None:
        backend = SERVICE_DEFAULTS["backend"]
    if endpoint is None:
        endpoint = SERVICE_DEFAULTS["endpoint"]
    tokenizer = fields.get("tokenizer") or service_in.get("tokenizer")
    if tokenizer is None:
        tokenizer = model_path_value if isinstance(model_path_value, str) else None
    if not isinstance(backend, str) or not backend.strip():
        errors.append("service.backend 必须是非空字符串")
    if (
        not isinstance(endpoint, str)
        or not endpoint.startswith("/")
        or endpoint.startswith("//")
        or "://" in endpoint
    ):
        errors.append("service.endpoint 必须是以 / 开头的 HTTP 路径")
    if not isinstance(tokenizer, str) or not tokenizer.strip():
        errors.append("service.tokenizer 必须是非空字符串")

    config_path, config, config_error, config_digest = _config_with_digest(
        model_path, model_in.get("config_path")
    )
    if config_error:
        errors.append(config_error)
    model = {
        "model_path": str(model_path_value) if isinstance(model_path_value, str) else None,
        "served_model_name": served_name if isinstance(served_name, str) else None,
        "config_path": str(config_path) if config_path else None,
        "config_digest": config_digest,
        "config": _config_summary(config or {}),
    }
    service = {
        "backend": backend if isinstance(backend, str) else None,
        "endpoint": endpoint if isinstance(endpoint, str) else None,
        "tokenizer": tokenizer if isinstance(tokenizer, str) else None,
    }
    return model, service, outputs_in, config, config_path


def _validate_resources(
    payload: Mapping[str, Any],
    fields: Mapping[str, Any],
    runtime: dict[str, Any],
    config: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], list[str]]:
    resources_in = _mapping(payload.get("resources"))
    ignored = sorted(set(resources_in) - RESOURCE_FIELDS)
    visible = _device_ids(resources_in.get("visible_devices", payload.get("visible_devices")))
    if visible is None:
        errors.append("resources.visible_devices 必须是非空且不重复的 NPU ID 数组")
        visible = []

    candidates = resources_in.get("port_candidates", payload.get("port_candidates"))
    ports = _ports(candidates)
    if ports is None or not ports:
        errors.append("resources.port_candidates 必须是非空端口数组")
        ports = []

    preferred = resources_in.get("preferred_port", payload.get("preferred_port"))
    if preferred is not None:
        try:
            preferred = int(preferred)
        except (TypeError, ValueError):
            preferred = None
            errors.append("preferred_port 必须是有效端口")
        if preferred is not None and preferred not in ports:
            errors.append("preferred_port 不在 port_candidates 中")

    command_port = fields.get("port")
    selected_port = resources_in.get("port", payload.get("port"))
    if command_port is not None:
        if selected_port is not None:
            try:
                if int(selected_port) != command_port:
                    errors.append("serve.argv 的 --port 与 resources.port 不一致")
            except (TypeError, ValueError):
                errors.append("resources.port 必须是有效端口")
        selected_port = command_port
    if selected_port is not None:
        try:
            selected_port = int(selected_port)
        except (TypeError, ValueError):
            selected_port = None
            errors.append("port 必须是有效端口")
        if selected_port is not None and selected_port not in ports:
            errors.append("port 不在 port_candidates 中")

    role_device_count = resources_in.get("role_device_count")
    if role_device_count is not None and not _positive_int(
        role_device_count, "role_device_count"
    ):
        errors.append("resources.role_device_count 必须是正整数")
        role_device_count = None
    elif role_device_count is not None and role_device_count < len(visible):
        errors.append("resources.role_device_count 不能小于当前实例可见设备数量")

    max_tp_dp = resources_in.get("max_tp_dp")
    if max_tp_dp is not None and not _positive_int(max_tp_dp, "max_tp_dp"):
        errors.append("resources.max_tp_dp 必须是正整数")
    tp = runtime.get("tp")
    dp = runtime.get("dp")
    heads = _head_count(config)
    if tp is not None and _positive_int(tp, "tp") and _positive_int(dp, "dp"):
        required = tp * dp
        if tp > len(visible):
            errors.append("tp 超过当前实例可见设备数量")
        role_capacity = role_device_count if role_device_count is not None else len(visible)
        if required > role_capacity:
            errors.append("tp * dp 超过角色可用设备数量")
        if _positive_int(max_tp_dp, "max_tp_dp") and required > max_tp_dp:
            errors.append("tp * dp 超过 Provider max_tp_dp")
        if heads is not None and heads % tp:
            errors.append("num_attention_heads 必须能被 tp 整除")

    resources = {
        **{key: value for key, value in resources_in.items() if key in RESOURCE_FIELDS},
        "visible_devices": visible,
        "role_device_count": role_device_count,
        "port_candidates": ports,
        "preferred_port": preferred,
        "port": selected_port,
    }
    return resources, ignored


def validate_and_build(
    payload: Mapping[str, Any],
    *,
    help_text: str,
    help_digest: str | None = None,
    check_unique: bool = True,
) -> dict[str, Any]:
    """Validate the JSON command contract and return an in-memory normalized object."""

    del check_unique  # Kept in the signature for callers that used the old helper.
    if not isinstance(payload, Mapping):
        raise ContractError("输入必须是 JSON object")
    errors: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    required_missing: list[str] = []
    command, command_errors, command_warnings = _validate_serve_input(payload, help_text)
    errors.extend(command_errors)
    warnings.extend(command_warnings)
    fields = command["fields"]
    model, service, outputs_in, config, _config_path = _config_and_service(payload, fields, errors)
    heads = _head_count(config or {})
    runtime_in = _mapping(payload.get("runtime"))
    command_runtime_fields = {
        "tp",
        "dp",
        "max_num_seqs",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
        "max_model_len",
        "block_size",
        "dtype",
        "quantization",
    }
    duplicated_runtime = sorted(command_runtime_fields.intersection(runtime_in))
    if duplicated_runtime:
        errors.append(
            "runtime 中不得重复提供 serve.argv 已负责的字段: " + ", ".join(duplicated_runtime)
        )
    runtime: dict[str, Any] = {
        key: value for key, value in runtime_in.items() if key not in command_runtime_fields
    }
    for field, default in DEFAULTS.items():
        value = fields.get(field)
        if value is None:
            runtime[field] = default
            missing.append(field)
            if field in {"dp", "max_num_seqs", "max_num_batched_tokens"}:
                required_missing.append(field)
        else:
            runtime[field] = value
    runtime["tp"] = fields.get("tp")
    if runtime["tp"] is None:
        missing.append("tp")
        required_missing.append("tp")
    for field in ("max_model_len", "dtype", "quantization"):
        if fields.get(field) is not None:
            runtime[field] = fields[field]

    for field in ("dp", "max_num_seqs", "max_num_batched_tokens"):
        if not _positive_int(runtime.get(field), field):
            errors.append(f"runtime.{field} 必须是正整数")
    if runtime.get("tp") is not None and not _positive_int(runtime["tp"], "tp"):
        errors.append("runtime.tp 必须是正整数或 null")
    if runtime.get("max_num_batched_tokens", 0) < runtime.get("max_num_seqs", 0):
        errors.append("max_num_batched_tokens 不能小于 max_num_seqs")
    utilization = runtime.get("gpu_memory_utilization")
    if (
        not isinstance(utilization, (int, float))
        or isinstance(utilization, bool)
        or not math.isfinite(utilization)
        or not 0 < utilization <= 1
    ):
        errors.append("runtime.gpu_memory_utilization 必须位于 (0, 1] 且为有限数")

    if "official_min_tp" not in runtime:
        runtime["official_min_tp"] = 1
    if "allowed_tp" not in runtime and heads is not None:
        resources_in = _mapping(payload.get("resources"))
        visible_for_tp = _device_ids(resources_in.get("visible_devices", payload.get("visible_devices")))
        if visible_for_tp:
            runtime["allowed_tp"] = [
                candidate
                for candidate in range(1, len(visible_for_tp) + 1)
                if heads % candidate == 0
            ]
    if "allowed_tp" in runtime:
        allowed = runtime["allowed_tp"]
        if (
            not isinstance(allowed, list)
            or not allowed
            or any(not _positive_int(value, "allowed_tp") for value in allowed)
            or len(set(allowed)) != len(allowed)
        ):
            errors.append("runtime.allowed_tp 必须是不重复的正整数数组")
    if not _positive_int(runtime.get("official_min_tp"), "official_min_tp"):
        errors.append("runtime.official_min_tp 必须是正整数")

    resources, ignored_resource_fields = _validate_resources(
        payload, fields, runtime, config or {}, errors
    )
    if ignored_resource_fields:
        warnings.append(
            "忽略不属于 Provider 资源契约的字段: "
            + ", ".join(ignored_resource_fields)
        )
    if runtime.get("tp") is None:
        warnings.append("缺少 tp，将先执行最小 TP 推荐")

    attempt_dir_value = outputs_in.get("attempt_dir", payload.get("output_dir"))
    attempt_dir = Path(attempt_dir_value) if isinstance(attempt_dir_value, str) else Path(".")
    if not isinstance(attempt_dir_value, str) or not attempt_dir.is_absolute():
        errors.append("outputs.attempt_dir 必须是绝对路径")

    input_status = "invalid" if errors else ("missing_parameters" if required_missing else "pass")
    serve = {
        "argv": list(_mapping(payload.get("serve")).get("argv", [])),
        "help_digest": help_digest,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "input_status": input_status,
        "serve": serve,
        "model": model,
        "runtime": runtime,
        "service": service,
        "resources": resources,
        "outputs": {**outputs_in, "attempt_dir": str(attempt_dir)},
        "resolution": {
            "mode": "input",
            "missing_fields": sorted(set(missing)),
            "defaulted_fields": sorted(field for field in missing if field != "tp"),
            "ignored_fields": ignored_resource_fields,
        },
        "diagnostics": {"errors": errors, "warnings": warnings},
    }


def _runtime_artifact(
    payload: Mapping[str, Any], normalized: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the temporary, validated runtime view consumed by static OOM."""

    model = _mapping(normalized.get("model"))
    raw_model = _mapping(payload.get("model"))
    model = dict(model)
    for field in ("weight_bytes", "weight_gib"):
        if field in raw_model:
            model[field] = raw_model[field]
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "model": model,
        "runtime": _mapping(normalized.get("runtime")),
        "resources": _mapping(normalized.get("resources")),
    }


def _write_runtime_artifact(
    path: str, payload: Mapping[str, Any], normalized: Mapping[str, Any]
) -> None:
    runtime_path = Path(path)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = runtime_path.with_name(runtime_path.name + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(
                _runtime_artifact(payload, normalized),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(runtime_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _replace_or_append_tp(argv: list[str], tp: int) -> list[str]:
    result = list(argv)
    names = {"--tensor-parallel-size", "-tp"}
    for index, token in enumerate(result):
        split = _split_option_token(token)
        if split is None or split[0] not in names:
            continue
        name, inline = split
        if inline is not None:
            result[index] = f"{name}={tp}"
        elif index + 1 < len(result):
            result[index + 1] = str(tp)
        else:
            result.append(str(tp))
        return result
    result.extend(["--tensor-parallel-size", str(tp)])
    return result


def resolve_recommended_tp(
    normalized: Mapping[str, Any], tp: int, source: str
) -> dict[str, Any]:
    """Merge a TP recommendation without repeating the input contract."""

    if not _positive_int(tp, "recommended_tp"):
        raise ContractError("recommended_tp 必须是正整数")
    if not isinstance(normalized, Mapping):
        raise ContractError("规范化输入必须是 JSON object")
    deployment = copy.deepcopy(dict(normalized))
    original_status = deployment.get("input_status", "missing_parameters")
    model_obj = _mapping(deployment.get("model"))
    runtime = _mapping(deployment.get("runtime"))
    resources = _mapping(deployment.get("resources"))
    resources = {key: value for key, value in resources.items() if key in RESOURCE_FIELDS}
    deployment["resources"] = resources
    allowed_tp = runtime.get("allowed_tp")
    if isinstance(allowed_tp, list) and tp not in allowed_tp:
        raise ContractError("recommended_tp 不在已校验的 allowed_tp 中")
    dp = runtime.get("dp")
    if _positive_int(dp, "dp"):
        if isinstance(resources.get("visible_devices"), list) and tp > len(
            resources["visible_devices"]
        ):
            raise ContractError("recommended_tp 超过已校验的当前实例可见设备数量")
        required = tp * dp
        visible = resources.get("visible_devices")
        role_device_count = resources.get("role_device_count")
        role_capacity = (
            role_device_count
            if _positive_int(role_device_count, "role_device_count")
            else len(visible)
            if isinstance(visible, list)
            else None
        )
        if role_capacity is not None and required > role_capacity:
            raise ContractError("recommended_tp * dp 超过已校验的角色可用设备数量")
        max_tp_dp = resources.get("max_tp_dp")
        if _positive_int(max_tp_dp, "max_tp_dp") and required > max_tp_dp:
            raise ContractError("recommended_tp * dp 超过已校验的 max_tp_dp")
    heads = _head_count(model_obj.get("config", {}))
    if heads is not None and heads % tp:
        raise ContractError("recommended_tp 不能整除已缓存的 num_attention_heads")
    runtime["tp"] = tp
    deployment["runtime"] = runtime
    serve = _mapping(deployment.get("serve"))
    serve["argv"] = _replace_or_append_tp(list(serve.get("argv", [])), tp)
    deployment["serve"] = serve
    if original_status == "missing_parameters":
        deployment["input_status"] = "pass"
    deployment["resolution"] = {
        "mode": "auto_recommended",
        "recommended_tp": tp,
        "source": source,
        "original_status": original_status,
    }
    return deployment


def _load_json(path: str) -> Mapping[str, Any]:
    if path == "-":
        value = json.load(sys.stdin)
    else:
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ContractError("input 必须是 JSON object")
    return value


def _read_help(path: str) -> tuple[str, str]:
    help_path = Path(path)
    if not help_path.is_file() or not os.access(help_path, os.R_OK):
        raise ContractError(f"vllm serve help 不可读: {help_path}")
    try:
        raw = help_path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
    except OSError as exc:
        raise ContractError(f"无法读取 vllm serve help: {exc}") from exc
    return text, hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="-", help="JSON input file, or - for stdin")
    parser.add_argument("--serve-help-file")
    parser.add_argument(
        "--runtime-output",
        help="写出供 static_oom_precheck.py 使用的临时 runtime.json",
    )
    parser.add_argument("--resolve-tp", type=int)
    parser.add_argument("--resolution-source", default="static_oom")
    args = parser.parse_args(argv)
    try:
        if args.resolve_tp is not None:
            if args.runtime_output:
                raise ContractError("--resolve-tp 不能同时写出 runtime.json")
            result = resolve_recommended_tp(
                _load_json(args.input), args.resolve_tp, args.resolution_source
            )
        else:
            if not args.serve_help_file:
                raise ContractError("普通输入校验必须提供 --serve-help-file")
            payload = _load_json(args.input)
            help_text, help_digest = _read_help(args.serve_help_file)
            result = validate_and_build(payload, help_text=help_text, help_digest=help_digest)
            if args.runtime_output:
                if result.get("input_status") not in {"pass", "missing_parameters"}:
                    raise ContractError("输入无效，不能生成 runtime.json")
                _write_runtime_artifact(args.runtime_output, payload, result)
    except (OSError, UnicodeError, json.JSONDecodeError, ContractError, TypeError, ValueError) as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "input_status": "invalid",
            "diagnostics": {"errors": [str(exc)], "warnings": []},
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("input_status") in {"pass", "missing_parameters"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
