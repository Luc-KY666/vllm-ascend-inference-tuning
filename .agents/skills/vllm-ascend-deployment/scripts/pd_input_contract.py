"""Validate a single-host Prefill/Decode deployment payload."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from input_contract import (
    DEFAULTS,
    SCHEMA_VERSION,
    _config_summary,
    _config_with_digest,
    _device_ids,
    _head_count,
    _mapping,
    _positive_int,
    _ports,
)


def _error_result(errors: list[str], warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "input_status": "invalid" if errors else "pass",
        "diagnostics": {"errors": errors, "warnings": warnings or []},
    }


def validate_pd(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    deployment_in = _mapping(payload.get("deployment"))
    model_in = _mapping(payload.get("model"))
    resources_in = _mapping(payload.get("resources"))
    pd_in = _mapping(payload.get("pd"))
    outputs_in = _mapping(payload.get("outputs"))

    if deployment_in.get("mode") != "single_node_pd":
        errors.append("deployment.mode 必须是 single_node_pd")
    connector = deployment_in.get("connector")
    if connector is not None and (not isinstance(connector, str) or not connector.strip()):
        errors.append("deployment.connector 必须是非空字符串")

    model_path_value = model_in.get("model_path", payload.get("model_path"))
    served_name = model_in.get("served_model_name", payload.get("served_model_name"))
    model_path = Path(model_path_value) if isinstance(model_path_value, str) else Path(".")
    if not isinstance(model_path_value, str) or not model_path_value.strip():
        errors.append("model.model_path 必须是非空字符串")
    elif not model_path.is_dir() or not os.access(model_path, os.R_OK):
        errors.append(f"模型路径不可读: {model_path}")
    if not isinstance(served_name, str) or not served_name.strip():
        errors.append("model.served_model_name 必须是非空字符串")

    config_path, config, config_error, config_digest = _config_with_digest(
        model_path, model_in.get("config_path")
    )
    if config_error:
        errors.append(config_error)
    heads = _head_count(config or {})

    visible = _device_ids(resources_in.get("visible_devices", payload.get("visible_devices")))
    if visible is None:
        errors.append("resources.visible_devices 必须是非空且不重复的 NPU ID 数组")
        visible = []
    visible_set = set(visible)

    candidates = _ports(resources_in.get("port_candidates", payload.get("port_candidates")))
    if candidates is None or not candidates:
        errors.append("resources.port_candidates 必须是非空端口数组")
        candidates = []
    candidate_set = set(candidates)

    roles: dict[str, dict[str, Any]] = {}
    all_devices: set[str] = set()
    all_ports: list[int] = []
    for role in ("prefill", "decode"):
        role_in = _mapping(pd_in.get(role))
        devices = _device_ids(role_in.get("devices"))
        if devices is None:
            errors.append(f"pd.{role}.devices 必须是非空且不重复的 NPU ID 数组")
            devices = []
        if any(device not in visible_set for device in devices):
            errors.append(f"pd.{role}.devices 必须属于 resources.visible_devices")
        overlap = all_devices.intersection(devices)
        if overlap:
            errors.append(f"Prefill/Decode 设备重叠: {sorted(overlap)}")
        all_devices.update(devices)

        tp = role_in.get("tp")
        if tp is not None:
            if not _positive_int(tp, f"pd.{role}.tp"):
                errors.append(f"pd.{role}.tp 必须是正整数")
                tp = None
            elif tp > len(devices):
                errors.append(f"pd.{role}.tp 不能超过该角色设备数")
            elif heads is not None and heads % tp:
                errors.append(f"num_attention_heads 必须能被 pd.{role}.tp 整除")
        allowed_tp = None
        if heads is not None and devices:
            allowed_tp = [candidate for candidate in range(1, len(devices) + 1) if heads % candidate == 0]
        if tp is None and allowed_tp == []:
            errors.append(f"pd.{role} 没有能整除 attention heads 的 TP 候选")

        api_port = role_in.get("api_port")
        kv_port = role_in.get("kv_port")
        normalized_ports: dict[str, int | None] = {}
        for name, value in (("api_port", api_port), ("kv_port", kv_port)):
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = None
            if value is None or not 1 <= value <= 65535:
                errors.append(f"pd.{role}.{name} 必须是有效端口")
            elif value not in candidate_set:
                errors.append(f"pd.{role}.{name} 不在 resources.port_candidates 中")
            else:
                all_ports.append(value)
            normalized_ports[name] = value

        engine_id = role_in.get("engine_id", f"{role}-0")
        if not isinstance(engine_id, str) or not engine_id.strip():
            errors.append(f"pd.{role}.engine_id 必须是非空字符串")
            engine_id = f"{role}-0"
        role_runtime = _mapping(role_in.get("runtime"))
        roles[role] = {
            "devices": devices,
            "tp": tp,
            "allowed_tp": allowed_tp,
            "tp_source": "explicit" if tp is not None else "static_oom_minimum",
            "dp": 1,
            "api_port": normalized_ports["api_port"],
            "kv_port": normalized_ports["kv_port"],
            "engine_id": engine_id,
            "runtime": role_runtime,
        }

    proxy_in = _mapping(pd_in.get("proxy"))
    proxy_port = proxy_in.get("port")
    try:
        proxy_port = int(proxy_port)
    except (TypeError, ValueError):
        proxy_port = None
    if proxy_port is None or not 1 <= proxy_port <= 65535:
        errors.append("pd.proxy.port 必须是有效端口")
    elif proxy_port not in candidate_set:
        errors.append("pd.proxy.port 不在 resources.port_candidates 中")
    else:
        all_ports.append(proxy_port)

    mooncake = _mapping(pd_in.get("mooncake"))
    needs_mooncake = connector == "MultiConnector"
    mooncake_port = mooncake.get("master_port", 50088)
    if needs_mooncake:
        try:
            mooncake_port = int(mooncake_port)
        except (TypeError, ValueError):
            mooncake_port = None
        if mooncake_port is None or not 1 <= mooncake_port <= 65535:
            errors.append("pd.mooncake.master_port 必须是有效端口")
        elif mooncake_port not in candidate_set:
            errors.append("pd.mooncake.master_port 不在 resources.port_candidates 中")
        else:
            all_ports.append(mooncake_port)
    if len(all_ports) != len(set(all_ports)):
        errors.append("PD API、KV、Proxy 和 Mooncake 端口必须互不冲突")

    attempt_dir_value = outputs_in.get("attempt_dir", payload.get("output_dir"))
    attempt_dir = Path(attempt_dir_value) if isinstance(attempt_dir_value, str) else Path(".")
    if not isinstance(attempt_dir_value, str) or not attempt_dir.is_absolute():
        errors.append("outputs.attempt_dir 必须是绝对路径")

    runtime = dict(_mapping(payload.get("runtime")))
    for field, default in DEFAULTS.items():
        runtime.setdefault(field, default)
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "input_status": "invalid" if errors else "pass",
        "deployment": {
            "mode": "single_node_pd",
            "connector": connector,
            "needs_mooncake": needs_mooncake,
            "proxy_script": deployment_in.get("proxy_script"),
        },
        "model": {
            "model_path": str(model_path_value) if isinstance(model_path_value, str) else None,
            "served_model_name": served_name if isinstance(served_name, str) else None,
            "config_path": str(config_path) if config_path else None,
            "config_digest": config_digest,
            "config": _config_summary(config or {}),
        },
        "runtime": runtime,
        "resources": {
            "visible_devices": visible,
            "port_candidates": candidates,
            "network_ip": resources_in.get("network_ip"),
            "network_interface": resources_in.get("network_interface"),
        },
        "pd": {
            "prefill": roles.get("prefill", {}),
            "decode": roles.get("decode", {}),
            "proxy": {"port": proxy_port},
            "mooncake": {**mooncake, "master_port": mooncake_port},
        },
        "outputs": {**outputs_in, "attempt_dir": str(attempt_dir)},
        "resolution": {
            "mode": "input",
            "defaulted_fields": [
                f"pd.{role}.tp" for role in ("prefill", "decode") if roles[role]["tp"] is None
            ],
            "ignored_fields": [],
        },
        "diagnostics": {"errors": errors, "warnings": warnings},
    }
    return normalized


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, Mapping):
            raise ValueError("stdin 必须是 JSON object")
        result = validate_pd(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        result = _error_result([str(exc)])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("input_status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
