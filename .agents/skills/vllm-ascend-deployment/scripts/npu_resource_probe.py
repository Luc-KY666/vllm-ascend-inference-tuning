"""Fail-closed probe for the authorized and provider-visible Ascend NPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "vllm-ascend-npu-resource-probe-v1"
DEVICE_ID_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
NO_RUNNING_RE = re.compile(r"no\s+running\s+processes\s+found", re.IGNORECASE)
DEVICE_RE = re.compile(
    r"\b(?:npu|device)\s*(?:id)?\s*(?:[:=#]|\s)\s*(?P<id>[0-9]+)\b",
    re.IGNORECASE,
)
DEVICE_PROCESS_RE = re.compile(
    r"\b(?:npu|device)\s*(?:id)?\s*(?:[:=#]|\s)\s*(?P<device>[0-9]+)"
    r".*?\b(?:pid|process(?:\s+id)?)\s*(?:[:=#]|\s)\s*(?P<pid>[0-9]+)\b",
    re.IGNORECASE,
)
PROCESS_DEVICE_RE = re.compile(
    r"\b(?:pid|process(?:\s+id)?)\s*(?:[:=#]|\s)\s*(?P<pid>[0-9]+)"
    r".*?\b(?:npu|device)\s*(?:id)?\s*(?:[:=#]|\s)\s*(?P<device>[0-9]+)\b",
    re.IGNORECASE,
)


class NpuProbeError(ValueError):
    """A fail-closed NPU probe error."""

    def __init__(self, failure_type: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message
        self.details = details


def _device_ids(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise NpuProbeError(
            "resource_input_invalid",
            f"{field} 必须是非空设备 ID 数组",
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, (str, int)) or isinstance(item, bool):
            raise NpuProbeError(
                "resource_input_invalid",
                f"{field} 中的设备 ID 必须是字符串或整数",
            )
        normalized = str(item)
        if not DEVICE_ID_RE.fullmatch(normalized):
            raise NpuProbeError(
                "resource_input_invalid",
                f"{field} 中包含非法设备 ID: {normalized!r}",
            )
        result.append(normalized)
    if len(set(result)) != len(result):
        raise NpuProbeError(
            "resource_input_invalid",
            f"{field} 不能包含重复设备",
        )
    return result


def _visibility_device_ids(value: str, field: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise NpuProbeError(
            "device_visibility_mismatch",
            f"{field} 必须是非空的设备可见性字符串",
        )
    return _device_ids(
        [item for item in re.split(r"[,\s]+", value.strip()) if item],
        field,
    )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NpuProbeError("resource_input_invalid", f"{field} 必须是正整数")
    return value


def parse_pool_environment(raw: str | None) -> tuple[list[str], str]:
    if not raw:
        raise NpuProbeError("device_pool_missing", "缺少 GOAL_PLUS_DEVICE_POOLS")
    try:
        pools = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NpuProbeError(
            "device_pool_invalid",
            "GOAL_PLUS_DEVICE_POOLS 不是有效 JSON",
        ) from exc
    if not isinstance(pools, Mapping):
        raise NpuProbeError(
            "device_pool_invalid",
            "GOAL_PLUS_DEVICE_POOLS 必须是 JSON object",
        )
    pool = pools.get("ascend-npu")
    if not isinstance(pool, Mapping):
        raise NpuProbeError(
            "device_pool_missing",
            "GOAL_PLUS_DEVICE_POOLS 缺少 ascend-npu pool",
        )
    visibility_env = pool.get("visibility_env")
    if visibility_env != "ASCEND_RT_VISIBLE_DEVICES":
        raise NpuProbeError(
            "device_pool_invalid",
            "ascend-npu pool 必须使用 ASCEND_RT_VISIBLE_DEVICES",
        )
    return (
        _device_ids(
            pool.get("devices"),
            "GOAL_PLUS_DEVICE_POOLS.ascend-npu.devices",
        ),
        visibility_env,
    )


def _pipe_cells(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _line_device_id(line: str, known_devices: set[str]) -> str | None:
    match = DEVICE_RE.search(line)
    if match and match.group("id") in known_devices:
        return match.group("id")
    cells = _pipe_cells(line)
    if cells and cells[0] in known_devices:
        return cells[0]
    return None


def parse_npu_smi_output(
    output: str,
    device_ids_to_check: Sequence[str],
) -> dict[str, Any]:
    """Accept only explicit free/busy evidence for every checked device."""

    known = set(device_ids_to_check)
    states: dict[str, dict[str, Any]] = {
        device: {"status": None, "processes": []}
        for device in device_ids_to_check
    }
    current_device: str | None = None
    process_table_seen = False

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        explicit_device = _line_device_id(line, known)
        if explicit_device is not None:
            current_device = explicit_device

        if NO_RUNNING_RE.search(line):
            device = explicit_device or current_device
            if device is not None:
                states[device]["status"] = "free"
            continue

        explicit_process = DEVICE_PROCESS_RE.search(line) or PROCESS_DEVICE_RE.search(line)
        if explicit_process is not None:
            groups = explicit_process.groupdict()
            device = groups.get("device")
            if device in known:
                states[device]["status"] = "busy"
                states[device]["processes"].append({"pid": int(groups["pid"])})
            continue

        lowered = line.lower()
        if "process id" in lowered or "process name" in lowered:
            process_table_seen = True
            continue
        if process_table_seen:
            cells = _pipe_cells(line)
            if len(cells) >= 2 and cells[0] in known and cells[1].isdigit():
                pid = int(cells[1])
                if pid > 0:
                    states[cells[0]]["status"] = "busy"
                    states[cells[0]]["processes"].append({"pid": pid})

    unresolved = [
        device for device, state in states.items() if state["status"] is None
    ]
    if unresolved:
        raise NpuProbeError(
            "device_probe_failed",
            "npu-smi 输出没有为所有目标设备提供明确空闲/占用状态",
            {"unresolved_devices": unresolved},
        )
    return {
        "devices": states,
        "available_devices": [
            device for device, state in states.items() if state["status"] == "free"
        ],
        "busy_devices": [
            device for device, state in states.items() if state["status"] == "busy"
        ],
    }


def probe_npu_devices(
    device_ids_to_check: Sequence[str],
    *,
    command: str = "npu-smi",
    timeout: float = 15.0,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    run = runner or subprocess.run
    try:
        result = run(
            [command, "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NpuProbeError("device_probe_failed", f"无法执行 npu-smi: {exc}") from exc
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        raise NpuProbeError(
            "device_probe_failed",
            f"npu-smi info 退出码为 {result.returncode}",
            {"returncode": result.returncode, "stderr": stderr, "stdout": stdout},
        )
    parsed = parse_npu_smi_output(stdout, device_ids_to_check)
    parsed["command"] = [command, "info"]
    parsed["stdout"] = stdout
    parsed["stderr"] = stderr
    return parsed


def probe_authorized_devices(
    *,
    environment: Mapping[str, str] | None = None,
    visible_devices: Sequence[Any] | None = None,
    required_count: int | None = None,
    command: str = "npu-smi",
    timeout: float = 15.0,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    pool_devices, visibility_env = parse_pool_environment(
        env.get("GOAL_PLUS_DEVICE_POOLS")
    )
    if visible_devices is None:
        raw_visible = env.get(visibility_env)
        visible = (
            _visibility_device_ids(raw_visible, visibility_env)
            if raw_visible
            else list(pool_devices)
        )
    else:
        visible = _device_ids(list(visible_devices), "visible_devices")
        raw_visible = env.get(visibility_env)
        if raw_visible:
            actual_visible = _visibility_device_ids(raw_visible, visibility_env)
            if set(actual_visible) != set(visible):
                raise NpuProbeError(
                    "device_visibility_mismatch",
                    "Provider 可见设备与 ASCEND_RT_VISIBLE_DEVICES 不一致",
                    {
                        "provider_visible_devices": visible,
                        "environment_visible_devices": actual_visible,
                    },
                )
    if not set(visible).issubset(set(pool_devices)):
        raise NpuProbeError(
            "device_visibility_mismatch",
            "Provider 可见设备超出 GOAL_PLUS_DEVICE_POOLS 授权范围",
            {"pool_devices": pool_devices, "visible_devices": visible},
        )
    effective_devices = [
        device for device in pool_devices if device in set(visible)
    ]
    device_probe = probe_npu_devices(
        effective_devices,
        command=command,
        timeout=timeout,
        runner=runner,
    )
    if required_count is not None:
        required = _positive_int(required_count, "required_count")
        if len(device_probe["available_devices"]) < required:
            raise NpuProbeError(
                "insufficient_available_devices",
                "未占用设备数量不足以满足本次部署需求",
                {
                    "required_devices": required,
                    "available_devices": device_probe["available_devices"],
                    "busy_devices": device_probe["busy_devices"],
                },
            )
        selected_devices = device_probe["available_devices"][:required]
    else:
        selected_devices = list(device_probe["available_devices"])
    return {
        "pool_id": "ascend-npu",
        "visibility_env": visibility_env,
        "authorized_devices": pool_devices,
        "visible_devices": visible,
        "effective_devices": effective_devices,
        "selected_devices": selected_devices,
        "required_devices": required_count,
        "probe": device_probe,
    }


def _result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "failure": None,
        "resources": {},
    }


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visible-devices",
        help="Provider 可见设备 ID，逗号分隔；不提供时读取 ASCEND_RT_VISIBLE_DEVICES",
    )
    parser.add_argument("--required-count", type=int)
    parser.add_argument("--npu-smi", default="npu-smi")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = _result()
    try:
        visible = (
            [item.strip() for item in args.visible_devices.split(",")]
            if args.visible_devices
            else None
        )
        result["resources"] = probe_authorized_devices(
            visible_devices=visible,
            required_count=args.required_count,
            command=args.npu_smi,
            timeout=args.timeout,
        )
        result["status"] = "pass"
    except (NpuProbeError, OSError, ValueError, TypeError) as exc:
        error = exc if isinstance(exc, NpuProbeError) else NpuProbeError(
            "resource_input_invalid",
            str(exc),
        )
        result["failure"] = {
            "stage": "npu_resource_probe",
            "type": error.failure_type,
            "message": error.message,
            "artifacts": [],
        }
        if error.details is not None:
            result["failure"]["details"] = error.details
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
