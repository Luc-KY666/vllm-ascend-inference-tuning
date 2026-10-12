"""对规范化静态输入执行容量预检和最小 TP 推荐。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "vllm-ascend-static-oom-v3"
GIB = 1024**3
DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "fp8": 1,
    "float8e4m3fn": 1,
    "float8e5m2": 1,
    "fp8e4m3fn": 1,
    "fp8e5m2": 1,
    "int8": 1,
}
NO_QUANTIZATION_VALUES = {"", "none", "null", "unquantized", "false"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _config_view(model: Mapping[str, Any]) -> Mapping[str, Any]:
    config = model.get("config")
    if not isinstance(config, Mapping):
        return {}
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping) or not text_config:
        return config
    merged = dict(config)
    merged.update(text_config)
    return merged


def _dtype_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("torch.", "").replace("-", "").replace("_", "")
    return DTYPE_BYTES.get(normalized)


def _select_dtype(
    runtime: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    value = runtime.get("dtype")
    if value is None or value.strip().lower() == "auto":
        value = config.get("torch_dtype")
    if value is None:
        return None, "缺少 dtype 或 model.config.torch_dtype"
    return value, None


def _architecture(model: Mapping[str, Any]) -> tuple[dict[str, int] | None, str | None]:
    config = _config_view(model)
    required = {
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        return None, f"模型配置缺少必要架构字段: {', '.join(missing)}"

    hidden_size = required["hidden_size"]
    layers = required["num_hidden_layers"]
    attention_heads = required["num_attention_heads"]
    key_value_heads = config.get("num_key_value_heads", attention_heads)
    if key_value_heads is None:
        key_value_heads = attention_heads
    if key_value_heads > attention_heads:
        return None, "num_key_value_heads 不能大于 num_attention_heads"

    head_dim = config.get("head_dim")
    if head_dim is None:
        if hidden_size % attention_heads:
            return None, "hidden_size 不能整除 num_attention_heads，且缺少 head_dim"
        head_dim = hidden_size // attention_heads
    elif hidden_size != attention_heads * head_dim:
        return None, "hidden_size 必须等于 num_attention_heads * head_dim"

    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": key_value_heads,
        "head_dim": head_dim,
    }, None


def _quantization_reason(
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> str | None:
    runtime_quantization = runtime.get("quantization")
    if (
        runtime_quantization is not None
        and runtime_quantization.strip().lower() not in NO_QUANTIZATION_VALUES
    ):
        return "存在量化配置 runtime.quantization，但没有明确权重大小"

    config_quantization = config.get("quantization_config")
    if config_quantization not in (None, {}, False, ""):
        return "存在量化配置 model.config.quantization_config，但没有明确权重大小"
    return None


def _explicit_weight(model: Mapping[str, Any]) -> tuple[int | None, str]:
    weight_bytes = model.get("weight_bytes")
    if weight_bytes is not None:
        return weight_bytes, "explicit_weight_bytes"

    weight_gib = model.get("weight_gib")
    if weight_gib is not None:
        return math.ceil(weight_gib * GIB), "explicit_weight_gib"

    return None, "none"


def _weight_requirements(
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[int | None, str, str | None]:
    explicit_weight, method = _explicit_weight(model)
    if explicit_weight is not None:
        return explicit_weight, method, None

    config = _config_view(model)
    quantization_reason = _quantization_reason(config, runtime)
    if quantization_reason:
        return None, "config_derived", quantization_reason

    if config.get("num_experts") is not None:
        return None, "config_derived", "检测到 num_experts，需提供明确权重大小"

    architecture, architecture_reason = _architecture(model)
    if architecture is None:
        return None, "config_derived", architecture_reason

    vocab_size = config.get("vocab_size")
    intermediate_size = config.get("intermediate_size")
    if vocab_size is None:
        return None, "config_derived", "模型配置缺少 vocab_size"
    if intermediate_size is None:
        return None, "config_derived", "模型配置缺少 intermediate_size"

    dtype_value, dtype_reason = _select_dtype(runtime, config)
    if dtype_reason:
        return None, "config_derived", dtype_reason
    dtype_size = _dtype_bytes(dtype_value)
    if dtype_size is None:
        return None, "config_derived", "dtype 不受支持或无法可靠确定"

    attention = (
        architecture["hidden_size"]
        * architecture["num_attention_heads"]
        * architecture["head_dim"]
        + 2
        * architecture["hidden_size"]
        * architecture["num_key_value_heads"]
        * architecture["head_dim"]
        + architecture["num_attention_heads"]
        * architecture["head_dim"]
        * architecture["hidden_size"]
    )
    mlp = 3 * architecture["hidden_size"] * intermediate_size
    norm = 2 * architecture["hidden_size"]
    embedding = vocab_size * architecture["hidden_size"]
    lm_head = 0 if config.get("tie_word_embeddings", False) else embedding
    total_params = embedding + lm_head + architecture["num_hidden_layers"] * (
        attention + mlp + norm
    )
    return total_params * dtype_size, "config_derived", None


def _kv_parameters(
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, int] | None, str | None]:
    architecture, architecture_reason = _architecture(model)
    if architecture is None:
        return None, architecture_reason

    dtype_value, dtype_reason = _select_dtype(runtime, _config_view(model))
    if dtype_reason:
        return None, dtype_reason
    dtype_size = _dtype_bytes(dtype_value)
    if dtype_size is None:
        return None, "KV Cache 的 dtype 不受支持或无法可靠确定"
    return {**architecture, "dtype_bytes": dtype_size}, None


def _evaluate_tp(
    *,
    tp: int,
    dp: int | None,
    runtime: Mapping[str, Any],
    card_memory_bytes: int | None,
    device_count: int | None,
    weight_bytes: int | None,
    weight_method: str,
    weight_reason: str | None,
    kv: Mapping[str, int] | None,
    kv_reason: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tp": tp,
        "dp": dp,
        "status": "unknown",
        "risk_level": "unknown",
        "message": None,
        "weight_method": weight_method,
    }
    if dp is None:
        result["message"] = "缺少 dp"
        return result
    if device_count is None:
        result["message"] = "缺少 device_count"
        return result
    if tp * dp > device_count:
        result.update(
            status="reject",
            risk_level="resource_exceeded",
            message="tp * dp 超过输入的设备数量约束",
            device_count=device_count,
        )
        return result
    if card_memory_bytes is None:
        result["message"] = "缺少 card_memory_bytes"
        return result
    if weight_bytes is None:
        result["message"] = weight_reason or "无法确定模型权重需求"
        return result
    if kv is None:
        result["message"] = kv_reason or "无法确定 KV Cache 需求"
        return result

    heads = kv["num_attention_heads"]
    if tp > heads or heads % tp:
        result.update(
            risk_level="incompatible_configuration",
            message="tp 必须为不大于 num_attention_heads 且可整除 attention heads 的正整数",
        )
        return result

    utilization = runtime.get("gpu_memory_utilization")
    max_seqs = runtime.get("max_num_seqs")
    min_context = runtime.get("min_required_context_len")
    block_size = runtime.get("block_size")
    if any(value is None for value in (utilization, max_seqs, min_context, block_size)):
        result["message"] = (
            "缺少 gpu_memory_utilization、max_num_seqs、"
            "min_required_context_len 或 block_size"
        )
        return result

    budget = math.floor(card_memory_bytes * utilization)
    weight_per_card = (weight_bytes + tp - 1) // tp
    kv_heads_per_rank = max(1, math.ceil(kv["num_key_value_heads"] / tp))
    kv_bytes_per_token = (
        2
        * kv["num_hidden_layers"]
        * kv_heads_per_rank
        * kv["head_dim"]
        * kv["dtype_bytes"]
    )
    context_tokens = math.ceil(min_context / block_size) * block_size
    minimum_kv_bytes = max_seqs * context_tokens * kv_bytes_per_token
    result.update(
        {
            "card_memory_bytes": card_memory_bytes,
            "memory_budget_bytes": budget,
            "weight_bytes_total": weight_bytes,
            "weight_bytes_per_card": weight_per_card,
            "kv_bytes_per_token_per_card": kv_bytes_per_token,
            "minimum_context_tokens_per_sequence": context_tokens,
            "minimum_kv_bytes_per_card": minimum_kv_bytes,
            "gpu_memory_utilization": utilization,
            "min_required_context_len": min_context,
        }
    )

    if weight_per_card >= budget:
        result.update(
            status="reject",
            risk_level="capacity_exceeded",
            message="单卡模型权重需求已经达到或超过显存预算",
        )
    elif weight_per_card + minimum_kv_bytes > budget:
        result.update(
            status="reject",
            risk_level="capacity_exceeded",
            message="模型权重与最低上下文 KV Cache 需求超过显存预算",
        )
    else:
        result.update(status="pass", risk_level="none", message=None)
    return result


def _unknown_result(message: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "unknown",
        "risk_level": "unknown",
        "message": message,
        "current": None,
        "recommendation": {
            "status": "unknown",
            "recommended_tp": None,
            "candidates": [],
            "message": message,
        },
        "diagnostics": [message],
    }


def precheck(payload: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(payload.get("model"))
    hardware = _mapping(payload.get("hardware"))
    runtime = _mapping(payload.get("runtime"))

    card_memory_bytes = hardware.get("card_memory_bytes")
    device_count = hardware.get("device_count")
    weight_bytes, weight_method, weight_reason = _weight_requirements(model, runtime)
    kv, kv_reason = _kv_parameters(model, runtime)

    dp = runtime.get("dp")
    current_tp = runtime.get("tp")
    if current_tp is None:
        current: dict[str, Any] = {
            "status": "missing_parameter",
            "risk_level": "unknown",
            "message": "未提供 tp，当前配置未执行容量判定",
            "tp": None,
            "dp": dp,
        }
    else:
        current = _evaluate_tp(
            tp=current_tp,
            dp=dp,
            runtime=runtime,
            card_memory_bytes=card_memory_bytes,
            device_count=device_count,
            weight_bytes=weight_bytes,
            weight_method=weight_method,
            weight_reason=weight_reason,
            kv=kv,
            kv_reason=kv_reason,
        )

    recommendation: dict[str, Any] = {
        "status": "unknown",
        "recommended_tp": None,
        "candidates": [],
        "message": None,
    }
    allowed = runtime.get("allowed_tp")
    official_min_tp = runtime.get("official_min_tp")
    recommendation_reason = next(
        (
            reason
            for reason in (
                "缺少 dp" if dp is None else None,
                "缺少 allowed_tp" if allowed is None else None,
                "缺少 official_min_tp" if official_min_tp is None else None,
                "缺少 card_memory_bytes" if card_memory_bytes is None else None,
                "缺少 device_count" if device_count is None else None,
                weight_reason,
                kv_reason,
            )
            if reason
        ),
        None,
    )
    if recommendation_reason:
        recommendation["message"] = recommendation_reason
    else:
        saw_unknown = False
        saw_eligible = False
        for candidate_tp in sorted(set(allowed)):
            if candidate_tp < official_min_tp:
                recommendation["candidates"].append(
                    {
                        "tp": candidate_tp,
                        "dp": dp,
                        "status": "skipped",
                        "risk_level": "policy",
                        "message": "低于 official_min_tp",
                    }
                )
                continue
            saw_eligible = True
            candidate = _evaluate_tp(
                tp=candidate_tp,
                dp=dp,
                runtime=runtime,
                card_memory_bytes=card_memory_bytes,
                device_count=device_count,
                weight_bytes=weight_bytes,
                weight_method=weight_method,
                weight_reason=weight_reason,
                kv=kv,
                kv_reason=kv_reason,
            )
            recommendation["candidates"].append(candidate)
            if candidate["status"] == "pass":
                recommendation.update(
                    status="available",
                    recommended_tp=candidate_tp,
                )
                break
            if candidate["status"] == "unknown":
                saw_unknown = True
        if recommendation["recommended_tp"] is None:
            if saw_unknown:
                recommendation.update(
                    status="unknown",
                    message="允许的 TP 候选中至少有一个无法完成兼容性或容量判定",
                )
            elif not saw_eligible:
                recommendation.update(
                    status="unavailable",
                    message="allowed_tp 中没有达到 official_min_tp 的候选",
                )
            else:
                recommendation.update(
                    status="unavailable",
                    message="允许的 TP 候选中没有满足设备或静态容量约束的值",
                )

    current_status = current.get("status")
    if current_status == "missing_parameter":
        status = "unknown"
        message = "未提供当前 tp；请使用 recommendation.recommended_tp 生成候选配置后重新预检"
    elif current_status in {"pass", "reject", "unknown"}:
        status = current_status
        message = current.get("message")
    else:
        status = "unknown"
        message = current.get("message") or "输入无法完成静态预检"

    diagnostics: list[str] = []
    for reason in (weight_reason, kv_reason, recommendation_reason):
        if reason and reason not in diagnostics:
            diagnostics.append(reason)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "risk_level": current.get("risk_level", "unknown"),
        "message": message,
        "current": current,
        "recommendation": recommendation,
        "inputs": {
            "device_count": device_count,
            "card_memory_bytes": card_memory_bytes,
            "allowed_tp": allowed,
            "official_min_tp": official_min_tp,
            "weight_method": weight_method,
        },
        "diagnostics": diagnostics,
    }


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str, value: Any) -> None:
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
    parser.add_argument("--input", required=True, help="JSON 输入文件，使用 - 表示 stdin")
    parser.add_argument("--output", help="可选的 JSON 输出文件")
    args = parser.parse_args(argv)
    try:
        payload = _read_json(args.input)
        if not isinstance(payload, Mapping):
            raise ValueError("输入必须是 JSON object")
        result = precheck(payload)
    except (OSError, ValueError, TypeError, OverflowError, AttributeError) as exc:
        result = _unknown_result(f"输入无效: {type(exc).__name__}: {exc}")
        if args.output:
            _write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"pass", "unknown"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
