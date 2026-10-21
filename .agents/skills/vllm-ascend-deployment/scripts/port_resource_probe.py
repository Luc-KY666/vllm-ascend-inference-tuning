"""Fail-closed wildcard TCP bind probe for Provider-supplied ports."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import socket
from collections.abc import Callable, Sequence
from typing import Any


SCHEMA_VERSION = "vllm-ascend-port-resource-probe-v1"


class PortProbeError(ValueError):
    """A fail-closed Provider-port probe error."""

    def __init__(self, failure_type: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message
        self.details = details


def _port(value: Any, field: str = "port") -> int:
    if isinstance(value, str):
        normalized = value.strip()
        if not re.fullmatch(r"[1-9][0-9]*", normalized):
            raise PortProbeError("resource_input_invalid", f"{field} 必须是正整数")
        value = int(normalized)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PortProbeError("resource_input_invalid", f"{field} 必须是正整数")
    if value > 65535:
        raise PortProbeError(
            "resource_input_invalid",
            f"{field} 必须位于 1..65535",
        )
    return value


def probe_port(
    port: Any,
    *,
    socket_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    normalized_port = _port(port)
    factory = socket_factory or socket.socket
    sock = None
    try:
        sock = factory(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("0.0.0.0", normalized_port))
    except OSError as exc:
        if exc.errno in {errno.EADDRINUSE, errno.EADDRNOTAVAIL}:
            return {
                "port": normalized_port,
                "status": "unavailable",
                "failure_type": "port_unavailable",
                "message": str(exc),
            }
        return {
            "port": normalized_port,
            "status": "failed",
            "failure_type": "port_probe_failed",
            "message": str(exc),
        }
    finally:
        if sock is not None:
            sock.close()
    return {"port": normalized_port, "status": "available"}


def ordered_provider_ports(
    candidates: Sequence[Any],
    *,
    preferred: Any | None = None,
) -> list[int]:
    ports = [_port(value, "port_candidates item") for value in candidates]
    if len(set(ports)) != len(ports):
        raise PortProbeError(
            "resource_input_invalid",
            "port_candidates 不能包含重复端口",
        )
    if preferred is None:
        return ports
    preferred_port = _port(preferred, "preferred_port")
    if preferred_port not in ports:
        raise PortProbeError(
            "preferred_port_not_provider_candidate",
            "preferred_port 不在 Provider 提供的候选端口中",
            {"preferred_port": preferred_port, "port_candidates": ports},
        )
    return [preferred_port] + [port for port in ports if port != preferred_port]


def select_provider_ports(
    candidates: Sequence[Any],
    required_count: int = 1,
    *,
    preferred: Any | None = None,
    checker: Callable[[int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if isinstance(required_count, bool) or not isinstance(required_count, int):
        raise PortProbeError("resource_input_invalid", "required_count 必须是正整数")
    if required_count <= 0:
        raise PortProbeError("resource_input_invalid", "required_count 必须是正整数")
    ordered = ordered_provider_ports(candidates, preferred=preferred)
    if not ordered:
        raise PortProbeError("port_unavailable", "Provider 没有提供候选端口")
    check = checker or probe_port
    checks: list[dict[str, Any]] = []
    selected: list[int] = []
    for candidate in ordered:
        try:
            result = check(candidate)
        except OSError as exc:
            result = {
                "port": candidate,
                "status": "failed",
                "failure_type": "port_probe_failed",
                "message": str(exc),
            }
        checks.append(result)
        if result.get("status") == "available":
            selected.append(candidate)
            if len(selected) >= required_count:
                break
    if len(selected) < required_count:
        failure_type = (
            "port_probe_failed"
            if any(
                item.get("failure_type") == "port_probe_failed"
                for item in checks
            )
            else "port_unavailable"
        )
        raise PortProbeError(
            failure_type,
            f"Provider 候选端口中只有 {len(selected)} 个可用，至少需要 {required_count} 个",
            {"checks": checks, "required_count": required_count},
        )
    return {
        "preferred_port": None if preferred is None else _port(preferred),
        "candidates": ordered,
        "checks": checks,
        "selected": selected,
        "required_count": required_count,
    }


def _result() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "failure": None,
        "resources": {},
    }


def _write_json(path: str, value: dict[str, Any]) -> None:
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
    parser.add_argument("--ports", required=True, help="Provider 候选端口，逗号分隔")
    parser.add_argument("--required-count", type=int, default=1)
    parser.add_argument(
        "--preferred-port",
        help="可选的 Provider 外部偏好端口；不提供时保持候选顺序",
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = _result()
    try:
        candidates = [value.strip() for value in args.ports.split(",")]
        result["resources"] = {
            "ports": select_provider_ports(
                candidates,
                args.required_count,
                preferred=args.preferred_port,
            )
        }
        result["status"] = "pass"
    except (PortProbeError, OSError, ValueError, TypeError) as exc:
        error = exc if isinstance(exc, PortProbeError) else PortProbeError(
            "resource_input_invalid",
            str(exc),
        )
        result["failure"] = {
            "stage": "port_resource_probe",
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
