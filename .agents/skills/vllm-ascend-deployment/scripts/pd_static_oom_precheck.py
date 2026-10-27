"""Run one static OOM precheck for both single-host PD roles."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

from static_oom_precheck import precheck


def _role_payload(payload: Mapping[str, Any], role: str) -> dict[str, Any]:
    normalized = payload.get("normalized") if isinstance(payload.get("normalized"), Mapping) else payload
    model = dict(normalized.get("model", {}))
    global_runtime = dict(normalized.get("runtime", {}))
    pd = normalized.get("pd", {})
    role_input = pd.get(role, {}) if isinstance(pd, Mapping) else {}
    role_runtime = dict(global_runtime)
    role_runtime.update(role_input.get("runtime", {}) if isinstance(role_input, Mapping) else {})
    explicit_tp = role_input.get("tp")
    role_runtime["tp"] = explicit_tp
    role_runtime["dp"] = 1
    role_runtime["allowed_tp"] = role_input.get("allowed_tp")
    role_runtime["official_min_tp"] = 1
    hardware_all = payload.get("hardware", {})
    hardware = hardware_all.get(role, {}) if isinstance(hardware_all, Mapping) else {}
    if not isinstance(hardware, Mapping):
        hardware = {}
    devices = role_input.get("devices", []) if isinstance(role_input, Mapping) else []
    return {
        "model": model,
        "runtime": role_runtime,
        "hardware": {
            "device_count": hardware.get("device_count", len(devices)),
            "card_memory_bytes": hardware.get("card_memory_bytes"),
        },
        "recommendation_mode": explicit_tp is None,
    }


def precheck_pd(payload: Mapping[str, Any]) -> dict[str, Any]:
    results = {role: precheck(_role_payload(payload, role)) for role in ("prefill", "decode")}
    effective: dict[str, dict[str, Any]] = {}
    for role, result in results.items():
        role_input = payload.get("normalized", payload).get("pd", {}).get(role, {})
        if role_input.get("tp") is None:
            recommendation = result.get("recommendation", {})
            recommended_tp = recommendation.get("recommended_tp")
            effective_status = "pass" if recommendation.get("status") == "available" else recommendation.get("status", "unknown")
            effective[role] = {
                "status": effective_status,
                "tp": recommended_tp,
                "source": "static_oom_minimum",
            }
        else:
            effective[role] = {
                "status": result.get("status", "unknown"),
                "tp": role_input.get("tp"),
                "source": "explicit",
            }
    statuses = {item["status"] for item in effective.values()}
    if "reject" in statuses:
        aggregate_status = "reject"
    elif statuses == {"pass"}:
        aggregate_status = "pass"
    else:
        aggregate_status = "unknown"
    return {
        "schema_version": "vllm-ascend-static-oom-pd-v1",
        "status": aggregate_status,
        "prefill": results["prefill"],
        "decode": results["decode"],
        "effective": effective,
        "aggregate": {
            "status": aggregate_status,
            "roles": {role: item["status"] for role, item in effective.items()},
        },
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, Mapping):
            raise ValueError("stdin 必须是 JSON object")
        result = precheck_pd(payload)
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        result = {
            "schema_version": "vllm-ascend-static-oom-pd-v1",
            "status": "unknown",
            "aggregate": {"status": "unknown"},
            "diagnostics": [f"输入无效: {exc}"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"pass", "unknown"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
