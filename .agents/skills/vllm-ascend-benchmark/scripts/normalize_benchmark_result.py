"""把 vLLM benchmark 或流式摘要规范化为统一的独立测量报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from streaming_metrics import summarize_payload
except ImportError:  # pragma: no cover - 直接从包外导入时的兼容路径
    summarize_payload = None


SCHEMA_VERSION = "vllm-ascend-benchmark-v1"
METRIC_NAMES = ("TTFT", "TPOT", "E2EL")
PERCENTILES = ("P50", "P95")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _identity(payload: Mapping[str, Any]) -> str | None:
    service = payload.get("service")
    if isinstance(service, Mapping):
        value = _first(service, "served_model_name", "model")
        if isinstance(value, str):
            return value
    model_config = payload.get("model_config")
    if isinstance(model_config, Mapping):
        value = _first(model_config, "served_model_name", "model_name", "model")
        if isinstance(value, str):
            return value
    for key in ("served_model_name", "model_id", "model", "model_name"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _integer(payload: Mapping[str, Any], *keys: str) -> int | None:
    value = _first(payload, *keys)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = payload.get("counts")
    source = nested if isinstance(nested, Mapping) else payload
    requested = _integer(source, "requested", "requested_count", "num_prompts", "request_count", "total_requests")
    completed = _integer(source, "completed", "completed_count", "successful", "success_count")
    if completed is None:
        completed = _integer(payload, "completed", "completed_count", "successful", "success_count")
    if completed is None:
        completed = _integer(payload, "num_prompts")
    failed = _integer(source, "failed", "failed_count", "failed_requests", "errors")
    if failed is None:
        failed = _integer(payload, "failed", "failed_count", "failed_requests", "errors")
    return {
        "requested": requested,
        "completed": completed,
        "failed": failed,
        "requested_source": "explicit" if requested is not None else None,
        "completed_source": "explicit" if completed is not None else None,
        "failed_source": "explicit" if failed is not None else None,
    }


def _metric_value(value: Any) -> float | None:
    if _finite(value) and float(value) >= 0:
        return float(value)
    return None


def _extract_metrics(payload: Mapping[str, Any]) -> dict[str, dict[str, float | None]] | None:
    candidates: list[Mapping[str, Any]] = []
    for key in ("metrics", "performance"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    candidates.append(payload)
    result: dict[str, dict[str, float | None]] = {}

    for metric in METRIC_NAMES:
        lower = metric.lower()
        values: dict[str, float | None] = {}
        for candidate in candidates:
            nested = candidate.get(metric) or candidate.get(lower) or candidate.get(f"{lower}_ms")
            if isinstance(nested, Mapping):
                for percentile in PERCENTILES:
                    if values.get(percentile) is None:
                        values[percentile] = _metric_value(
                            _first(nested, percentile, percentile.lower(), percentile.replace("P", "p"))
                        )
            for percentile in PERCENTILES:
                key_options = (
                    f"{percentile.lower()}_{lower}_ms",
                    f"{lower}_{percentile.lower()}_ms",
                    f"{percentile}_{metric}_ms",
                    f"{metric}_{percentile}",
                )
                if values.get(percentile) is None:
                    values[percentile] = _metric_value(_first(candidate, *key_options))
        result[metric] = values

    if not all(result[metric].get(percentile) is not None for metric in METRIC_NAMES for percentile in PERCENTILES):
        return None
    return result


def _base_report() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "invalid",
        "attempt": {},
        "service": {
            "served_model_name": None,
            "service_identity_valid": None,
        },
        "counts": {
            "requested": None,
            "completed": None,
            "failed": None,
        },
        "metrics": {
            metric: {"unit": "ms", "P50": None, "P95": None}
            for metric in METRIC_NAMES
        },
        "throughput": None,
        "source": {
            "format": None,
            "raw_result": None,
            "raw_result_sha256": None,
        },
        "cleanup": {"status": "unknown", "artifact_exported": False},
        "errors": [],
    }


def normalize(paths: list[str], expected_model: str, expected_requests: int) -> dict[str, Any]:
    report = _base_report()
    if len(paths) != 1:
        report["errors"].append(
            {
                "stage": "input",
                "type": "raw_result_ambiguous" if len(paths) > 1 else "raw_result_not_found",
                "message": "原始结果必须恰好有一个候选文件",
                "candidates": paths,
            }
        )
        return report
    path = Path(paths[0])
    report["source"]["raw_result"] = str(path)
    if not path.is_file():
        report["errors"].append(
            {"stage": "input", "type": "raw_result_not_found", "message": f"文件不存在: {path}"}
        )
        return report
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report["errors"].append(
            {"stage": "input", "type": "raw_result_invalid", "message": f"{type(exc).__name__}: {exc}"}
        )
        return report
    if not isinstance(payload, Mapping):
        report["errors"].append(
            {"stage": "input", "type": "raw_result_invalid", "message": "原始结果必须是 JSON object"}
        )
        return report

    report["source"]["raw_result_sha256"] = _sha256(path)
    raw_payload = payload.get("raw_result")
    if not isinstance(raw_payload, Mapping):
        raw_payload = payload
    identity = _identity(payload)
    if identity is None:
        identity = _identity(raw_payload)
    raw_identity = _identity(raw_payload)
    report["service"]["served_model_name"] = identity
    if identity != expected_model:
        report["errors"].append(
            {
                "stage": "identity",
                "type": "model_mismatch",
                "message": f"原始结果模型 {identity!r} 与期望模型 {expected_model!r} 不一致",
            }
        )
    if raw_identity is not None and raw_identity != expected_model:
        report["errors"].append(
            {
                "stage": "identity",
                "type": "raw_model_mismatch",
                "message": f"raw benchmark 模型 {raw_identity!r} 与期望模型 {expected_model!r} 不一致",
            }
        )

    if isinstance(raw_payload.get("requests"), list) and any(
        isinstance(item, Mapping) and "chunks" in item for item in raw_payload["requests"]
    ):
        if summarize_payload is None:
            report["errors"].append(
                {"stage": "metrics", "type": "streaming_helper_unavailable", "message": "无法导入流式解析 helper"}
            )
            return report
        try:
            stream_summary = summarize_payload(
                raw_payload,
                int(raw_payload.get("minimum_decode_samples", 1)),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            report["errors"].append(
                {"stage": "metrics", "type": "stream_summary_invalid", "message": f"{type(exc).__name__}: {exc}"}
            )
            return report
        payload_for_metrics: Mapping[str, Any] = stream_summary
        report["source"]["format"] = "stream_summary"
    else:
        payload_for_metrics = raw_payload
        report["source"]["format"] = "vllm_bench_serve"

    counts = _counts(payload_for_metrics)
    report["counts"].update(
        {
            "requested": counts["requested"],
            "completed": counts["completed"],
            "failed": counts["failed"],
        }
    )
    if counts["requested"] is not None and counts["requested"] != expected_requests:
        report["errors"].append(
            {
                "stage": "counts",
                "type": "request_count_mismatch",
                "message": f"请求数 {counts['requested']} 与期望值 {expected_requests} 不一致",
            }
        )
    elif counts["requested"] is None:
        report["errors"].append(
            {
                "stage": "counts",
                "type": "request_count_missing",
                "message": "缺少本次 benchmark 的请求数",
            }
        )
    if counts["completed"] is None:
        report["errors"].append(
            {"stage": "counts", "type": "completed_count_missing", "message": "缺少 completed 计数"}
        )
    elif counts["completed"] != expected_requests:
        report["errors"].append(
            {
                "stage": "counts",
                "type": "completed_count_mismatch",
                "message": f"完成数 {counts['completed']} 与期望值 {expected_requests} 不一致",
            }
        )
    if counts["failed"] is not None and counts["failed"] > 0:
        report["errors"].append(
            {
                "stage": "counts",
                "type": "benchmark_incomplete",
                "message": f"存在失败请求: {counts['failed']}",
            }
        )

    metrics = _extract_metrics(payload_for_metrics)
    if metrics is None:
        report["errors"].append(
            {
                "stage": "metrics",
                "type": "metrics_missing_or_invalid",
                "message": "缺少 TTFT、TPOT 或 E2EL 的 P50/P95",
            }
        )
    else:
        report["metrics"] = {
            metric: {"unit": "ms", **metrics[metric]}
            for metric in METRIC_NAMES
        }
    throughput = payload_for_metrics.get("throughput")
    if isinstance(throughput, Mapping):
        report["throughput"] = dict(throughput)

    service_valid = payload.get("service_identity_valid")
    if service_valid is None:
        service = payload.get("service")
        if isinstance(service, Mapping):
            service_valid = service.get("service_identity_valid")
    report["service"]["service_identity_valid"] = service_valid
    if service_valid is not True:
        report["errors"].append(
            {
                "stage": "identity",
                "type": "service_identity_invalid",
                "message": "缺少 service_identity_valid=true 或服务身份验收失败",
            }
        )

    if not report["errors"]:
        report["status"] = "success"
        report["cleanup"] = {"status": "passed", "artifact_exported": True}
    return report


def _write_json(path: str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", help="一个或多个候选原始 JSON 路径")
    parser.add_argument("--model", required=True, help="冻结的 served model name")
    parser.add_argument("--expected-requests", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = normalize(args.input, args.model, args.expected_requests)
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "success" else 3


if __name__ == "__main__":
    raise SystemExit(main())
