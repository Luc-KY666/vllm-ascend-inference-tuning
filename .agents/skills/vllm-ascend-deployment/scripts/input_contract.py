"""Validate a Provider deployment request and write deployment.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vllm-ascend-deployment-v1"
DEFAULTS = {
    "dp": 1,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096,
    "gpu_memory_utilization": 0.9,
    "min_required_context_len": 4096,
    "block_size": 128,
}
RESOURCE_FIELDS = {
    "visible_devices",
    "port_candidates",
    "preferred_port",
    "port",
    "max_tp_dp",
    "card_memory_bytes",
    "device_count",
}


class ContractError(ValueError):
    """An invalid Provider contract."""


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


def _config(model_path: Path, explicit: Any) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    config_path = Path(explicit) if isinstance(explicit, str) and explicit else model_path / "config.json"
    if not config_path.is_file() or not os.access(config_path, os.R_OK):
        return config_path, None, f"config.json 不可读: {config_path}"
    try:
        value = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return config_path, None, f"无法读取 config.json: {exc}"
    if not isinstance(value, dict):
        return config_path, None, "config.json 必须是 JSON object"
    return config_path, value, None


def _head_count(config: Mapping[str, Any]) -> int | None:
    source = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    value = source.get("num_attention_heads")
    return value if _positive_int(value, "num_attention_heads") else None


def validate_and_build(
    payload: Mapping[str, Any], *, check_unique: bool = True
) -> dict[str, Any]:
    model = _mapping(payload.get("model"))
    runtime_in = _mapping(payload.get("runtime"))
    resources_in = _mapping(payload.get("resources"))
    outputs_in = _mapping(payload.get("outputs"))
    errors: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    required_missing: list[str] = []
    ignored_resource_fields = sorted(set(resources_in) - RESOURCE_FIELDS)
    if ignored_resource_fields:
        warnings.append(
            "忽略不属于 Provider 资源契约的字段: "
            + ", ".join(ignored_resource_fields)
        )

    model_path_value = model.get("model_path", payload.get("model_path"))
    served_name = model.get("served_model_name", payload.get("served_model_name"))
    model_path = Path(model_path_value) if isinstance(model_path_value, str) else Path(".")
    if not isinstance(model_path_value, str) or not model_path_value.strip():
        errors.append("model.model_path 必须是非空字符串")
    elif not model_path.is_dir() or not os.access(model_path, os.R_OK):
        errors.append(f"模型路径不可读: {model_path}")
    if not isinstance(served_name, str) or not served_name.strip():
        errors.append("model.served_model_name 必须是非空字符串")

    config_path, config, config_error = _config(model_path, model.get("config_path"))
    if config_error:
        errors.append(config_error)
    heads = _head_count(config or {})

    runtime: dict[str, Any] = dict(runtime_in)
    for field, default in DEFAULTS.items():
        if field not in runtime or runtime[field] is None:
            runtime[field] = default
            missing.append(field)
            if field in {"dp", "max_num_seqs", "max_num_batched_tokens"}:
                required_missing.append(field)
    if "tp" not in runtime or runtime["tp"] is None:
        runtime["tp"] = None
        missing.append("tp")
        required_missing.append("tp")

    for field in ("dp", "max_num_seqs", "max_num_batched_tokens"):
        if not _positive_int(runtime.get(field), field):
            errors.append(f"runtime.{field} 必须是正整数")
    if runtime.get("tp") is not None and not _positive_int(runtime["tp"], "tp"):
        errors.append("runtime.tp 必须是正整数或 null")
    if runtime.get("max_num_batched_tokens", 0) < runtime.get("max_num_seqs", 0):
        errors.append("max_num_batched_tokens 不能小于 max_num_seqs")
    utilization = runtime.get("gpu_memory_utilization")
    if not isinstance(utilization, (int, float)) or isinstance(utilization, bool) or not math.isfinite(utilization) or not 0 < utilization <= 1:
        errors.append("runtime.gpu_memory_utilization 必须位于 (0, 1] 且为有限数")

    visible = _device_ids(resources_in.get("visible_devices", payload.get("visible_devices")))
    if visible is None:
        errors.append("resources.visible_devices 必须是非空且不重复的 NPU ID 数组")
        visible = []
    if "official_min_tp" not in runtime:
        runtime["official_min_tp"] = 1
    if "allowed_tp" not in runtime and heads is not None and visible:
        runtime["allowed_tp"] = [
            candidate for candidate in range(1, len(visible) + 1) if heads % candidate == 0
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
    selected_port = resources_in.get("port", payload.get("port"))
    if selected_port is not None:
        try:
            selected_port = int(selected_port)
        except (TypeError, ValueError):
            selected_port = None
            errors.append("port 必须是有效端口")
        if selected_port is not None and selected_port not in ports:
            errors.append("port 不在 port_candidates 中")

    tp = runtime.get("tp")
    dp = runtime.get("dp")
    max_tp_dp = resources_in.get("max_tp_dp")
    if max_tp_dp is not None and not _positive_int(max_tp_dp, "max_tp_dp"):
        errors.append("resources.max_tp_dp 必须是正整数")
    if tp is not None and _positive_int(tp, "tp") and _positive_int(dp, "dp"):
        required = tp * dp
        if required > len(visible):
            errors.append("tp * dp 超过 Provider 可见设备数量")
        if _positive_int(max_tp_dp, "max_tp_dp"):
            if required > max_tp_dp:
                errors.append("tp * dp 超过 Provider max_tp_dp")
        if heads is not None and heads % tp:
            errors.append("num_attention_heads 必须能被 tp 整除")
    elif tp is None:
        warnings.append("缺少 tp，将先执行最小 TP 推荐")

    attempt_dir_value = outputs_in.get("attempt_dir", payload.get("output_dir"))
    attempt_dir = Path(attempt_dir_value) if isinstance(attempt_dir_value, str) else Path(".")
    if not isinstance(attempt_dir_value, str) or not attempt_dir.is_absolute():
        errors.append("outputs.attempt_dir 必须是绝对路径")
    elif not attempt_dir.exists():
        try:
            attempt_dir.mkdir(parents=True)
        except OSError as exc:
            errors.append(f"无法创建输出目录: {exc}")
    deployment_path = attempt_dir / "deployment.json"
    if check_unique and deployment_path.exists():
        errors.append(f"输出目录已经包含 deployment.json: {deployment_path}")

    input_status = "invalid" if errors else ("missing_parameters" if required_missing else "pass")
    digest = None
    if config_path and config_path.is_file():
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    deployment = {
        "schema_version": SCHEMA_VERSION,
        "input_status": input_status,
        "model": {
            "model_path": str(model_path_value) if isinstance(model_path_value, str) else None,
            "served_model_name": served_name if isinstance(served_name, str) else None,
            "config_path": str(config_path) if config_path else None,
            "config_digest": digest,
        },
        "runtime": runtime,
        "resources": {
            **{
                key: value
                for key, value in resources_in.items()
                if key in RESOURCE_FIELDS
            },
            "visible_devices": visible,
            "port_candidates": ports,
            "preferred_port": preferred,
            "port": selected_port,
        },
        "outputs": {
            **outputs_in,
            "attempt_dir": str(attempt_dir),
            "deployment": str(deployment_path),
        },
        "resolution": {
            "mode": "input",
            "missing_fields": sorted(set(missing)),
            "defaulted_fields": sorted(field for field in missing if field != "tp"),
            "ignored_fields": ignored_resource_fields,
        },
        "diagnostics": {"errors": errors, "warnings": warnings},
    }
    return deployment


def resolve_recommended_tp(path: str | Path, tp: int, source: str) -> dict[str, Any]:
    if not _positive_int(tp, "recommended_tp"):
        raise ContractError("recommended_tp 必须是正整数")
    target = Path(path)
    deployment = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(deployment, dict):
        raise ContractError("deployment.json 必须是 JSON object")
    original_status = deployment.get("input_status", "missing_parameters")
    runtime = deployment.setdefault("runtime", {})
    runtime["tp"] = tp
    check = validate_and_build(
        {
            "model": deployment.get("model", {}),
            "runtime": runtime,
            "resources": deployment.get("resources", {}),
            "outputs": deployment.get("outputs", {}),
        },
        check_unique=False,
    )
    deployment["input_status"] = check["input_status"]
    deployment["resources"] = check["resources"]
    deployment["diagnostics"] = check["diagnostics"]
    deployment["resolution"] = {
        "mode": "auto_recommended",
        "recommended_tp": tp,
        "source": source,
        "original_status": original_status,
    }
    diagnostics = deployment.setdefault("diagnostics", {})
    diagnostics.setdefault("errors", [])
    diagnostics.setdefault("warnings", [])
    deployment_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    deployment_path.write_text(json.dumps(deployment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(deployment_path, target)
    return deployment


def _write(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="仅供内部读取已有 deployment.json")
    parser.add_argument("--output")
    parser.add_argument("--resolve-tp", type=int)
    parser.add_argument("--resolution-source", default="static-oom-report.json")
    parser.add_argument("--model-path")
    parser.add_argument("--served-model-name")
    parser.add_argument("--config-path")
    parser.add_argument("--visible-devices", nargs="+")
    parser.add_argument("--port-candidates", nargs="+", type=int)
    parser.add_argument("--preferred-port", type=int)
    parser.add_argument("--attempt-dir")
    parser.add_argument("--port", type=int)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--dp", type=int)
    parser.add_argument("--official-min-tp", type=int)
    parser.add_argument("--allowed-tp", nargs="+", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--min-required-context-len", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--dtype")
    parser.add_argument("--quantization")
    parser.add_argument("--extra-arg", action="append", dest="extra_args")
    parser.add_argument("--card-memory-bytes", type=int)
    parser.add_argument("--device-count", type=int)
    parser.add_argument("--max-tp-dp", type=int)
    args = parser.parse_args(argv)
    try:
        if args.resolve_tp is not None:
            if not args.input:
                raise ContractError("--resolve-tp 必须配合内部 deployment.json 的 --input")
            result = resolve_recommended_tp(args.input, args.resolve_tp, args.resolution_source)
        else:
            if args.input:
                raise ContractError("--input 仅用于配合 --resolve-tp 读取已有 deployment.json")
            required = {
                "model_path": args.model_path,
                "served_model_name": args.served_model_name,
                "visible_devices": args.visible_devices,
                "port_candidates": args.port_candidates,
                "attempt_dir": args.attempt_dir,
            }
            missing = [name for name, value in required.items() if value in (None, "")]
            if missing:
                raise ContractError("缺少 Provider 字段: " + ", ".join(missing))
            runtime = {
                "tp": args.tp,
                "dp": args.dp,
                "official_min_tp": args.official_min_tp,
                "allowed_tp": args.allowed_tp,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "min_required_context_len": args.min_required_context_len,
                "max_model_len": args.max_model_len,
                "block_size": args.block_size,
                "dtype": args.dtype,
                "quantization": args.quantization,
                "extra_args": args.extra_args,
            }
            runtime = {key: value for key, value in runtime.items() if value is not None}
            payload = {
                "model": {
                    "model_path": args.model_path,
                    "served_model_name": args.served_model_name,
                    "config_path": args.config_path,
                },
                "runtime": runtime,
                "resources": {
                    "visible_devices": args.visible_devices,
                    "port_candidates": args.port_candidates,
                    "preferred_port": args.preferred_port,
                    "port": args.port,
                    "card_memory_bytes": args.card_memory_bytes,
                    "device_count": args.device_count,
                    "max_tp_dp": args.max_tp_dp,
                },
                "outputs": {"attempt_dir": args.attempt_dir},
            }
            result = validate_and_build(payload)
            output = args.output or str(Path(result["outputs"]["attempt_dir"]) / "deployment.json")
            if Path(output).exists():
                raise ContractError(f"输出文件已存在，拒绝覆盖: {output}")
            _write(output, result)
    except (OSError, json.JSONDecodeError, ContractError, TypeError, ValueError) as exc:
        result = {"schema_version": SCHEMA_VERSION, "input_status": "invalid", "diagnostics": {"errors": [str(exc)]}}
        if args.output and not Path(args.output).exists():
            _write(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("input_status") in {"pass", "missing_parameters"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
