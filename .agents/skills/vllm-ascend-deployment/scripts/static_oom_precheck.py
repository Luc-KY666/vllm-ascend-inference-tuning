"""纯计算的模型容量和最小 TP 预检工具。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "vllm-ascend-static-oom-v1"
DEFAULT_ALLOWED_TP = (1, 2, 4, 8, 16)
DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8": 1,
    "fp8": 1,
    "int8": 1,
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _config_view(model: Mapping[str, Any]) -> Mapping[str, Any]:
    config = _mapping(model.get("config")) or _mapping(model.get("model_config"))
    if not config:
        config = model
    text_config = _mapping(config.get("text_config"))
    if text_config:
        merged = dict(config)
        merged.update(text_config)
        return merged
    return config


def _dtype_bytes(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    normalized = value.lower().replace("torch.", "").replace("-", "").replace("_", "")
    if normalized in {"float8e4m3fn", "float8e5m2", "fp8e4m3fn", "fp8e5m2"}:
        return 1
    return DTYPE_BYTES.get(normalized)


def _memory_bytes(hardware: Mapping[str, Any]) -> int | None:
    value = _first(hardware, "card_memory_bytes", "memory_bytes")
    if _positive_number(value):
        return int(value)
    value = _first(hardware, "card_memory_gib", "memory_gib")
    if _positive_number(value):
        return int(float(value) * (1024**3))
    return None


def _parameter_count(model: Mapping[str, Any], runtime: Mapping[str, Any]) -> tuple[int | None, str, str | None]:
    explicit = _first(model, "weight_bytes")
    if _positive_number(explicit):
        return int(explicit), "explicit_weight_bytes", None
    explicit_gib = _first(model, "weight_gib")
    if _positive_number(explicit_gib):
        return int(float(explicit_gib) * (1024**3)), "explicit_weight_gib", None

    config = _config_view(model)
    vocab = _first(config, "vocab_size")
    hidden = _first(config, "hidden_size", "d_model")
    layers = _first(config, "num_hidden_layers", "num_layers", "n_layer")
    heads = _first(config, "num_attention_heads", "n_head")
    kv_heads = _first(config, "num_key_value_heads", "num_kv_heads")
    head_dim = _first(config, "head_dim")
    intermediate = _first(config, "intermediate_size", "ffn_hidden_size")
    dtype = _first(runtime, "dtype")
    if not isinstance(dtype, str) or dtype.lower() == "auto":
        dtype = _first(config, "dtype", "torch_dtype")
    dtype_size = _dtype_bytes(dtype)

    required = {
        "vocab_size": vocab,
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "intermediate_size": intermediate,
        "dtype": dtype_size,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        return None, "config_derived", f"无法从模型配置可靠确定: {', '.join(missing)}"
    if not all(_positive_int(value) for value in (vocab, hidden, layers, heads, intermediate)):
        return None, "config_derived", "模型参数量字段必须为正整数"
    if not _positive_int(kv_heads):
        kv_heads = heads

    if not _positive_int(kv_heads):
        return None, "config_derived", "num_key_value_heads 无法确定"

    experts = _first(config, "num_experts", "n_routed_experts", "num_local_experts")
    moe_intermediate = _first(
        config,
        "moe_intermediate_size",
        "moe_ffn_hidden_size",
        "expert_intermediate_size",
    )
    shared_intermediate = _first(config, "shared_expert_intermediate_size")
    if experts is not None or moe_intermediate is not None:
        if not (_positive_int(experts) and _positive_int(moe_intermediate)):
            return None, "config_derived", "检测到 MoE 字段但无法确定 expert 参数"
        mlp = int(experts) * 3 * int(hidden) * int(moe_intermediate)
        if _positive_int(shared_intermediate):
            mlp += 3 * int(hidden) * int(shared_intermediate)
        router = int(hidden) * int(experts)
    else:
        mlp = 3 * int(hidden) * int(intermediate)
        router = 0

    if head_dim is None:
        if int(hidden) % int(heads) != 0:
            return None, "config_derived", "hidden_size 不能整除 num_attention_heads，且缺少 head_dim"
        head_dim = int(hidden) // int(heads)
    if not _positive_int(head_dim):
        return None, "config_derived", "head_dim 必须为正整数"

    attention = (
        int(hidden) * int(heads) * int(head_dim)
        + 2 * int(hidden) * int(kv_heads) * int(head_dim)
        + int(heads) * int(head_dim) * int(hidden)
    )
    norm = 2 * int(hidden)
    embeddings = int(vocab) * int(hidden)
    tie_embeddings = bool(_first(config, "tie_word_embeddings") or False)
    lm_head = 0 if tie_embeddings else embeddings
    total_params = embeddings + lm_head + int(layers) * (attention + mlp + router + norm)
    return total_params * int(dtype_size), "config_derived", None


def _kv_parameters(model: Mapping[str, Any], runtime: Mapping[str, Any]) -> tuple[dict[str, int] | None, str | None]:
    config = _config_view(model)
    layers = _first(config, "num_hidden_layers", "num_layers", "n_layer")
    heads = _first(config, "num_attention_heads", "n_head")
    kv_heads = _first(config, "num_key_value_heads", "num_kv_heads") or heads
    head_dim = _first(config, "head_dim")
    dtype = _first(runtime, "kv_dtype", "dtype")
    if not isinstance(dtype, str) or dtype.lower() == "auto":
        dtype = _first(config, "kv_dtype", "torch_dtype", "dtype")
    dtype_size = _dtype_bytes(dtype)
    if not (_positive_int(layers) and _positive_int(kv_heads) and _positive_int(dtype_size)):
        return None, "KV Cache 的层数、KV heads 或 dtype 无法可靠确定"
    if head_dim is None and _positive_int(heads) and _positive_int(_first(config, "hidden_size", "d_model")):
        hidden = int(_first(config, "hidden_size", "d_model"))
        head_count = int(heads)
        if hidden % head_count == 0:
            head_dim = hidden // head_count
    if not _positive_int(head_dim):
        return None, "KV Cache 的 head_dim 无法可靠确定"
    return {
        "num_hidden_layers": int(layers),
        "num_key_value_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "dtype_bytes": int(dtype_size),
    }, None


def _allowed_tp(runtime: Mapping[str, Any]) -> list[int]:
    values = runtime.get("allowed_tp", DEFAULT_ALLOWED_TP)
    if not isinstance(values, list):
        values = list(DEFAULT_ALLOWED_TP)
    normalized = sorted({int(value) for value in values if _positive_int(value)})
    return normalized or list(DEFAULT_ALLOWED_TP)


def _evaluate_tp(
    *,
    tp: int,
    dp: int,
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
    card_memory_bytes: int | None,
    weight_bytes: int | None,
    kv: Mapping[str, int] | None,
    weight_method: str,
    weight_reason: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tp": tp,
        "dp": dp,
        "status": "unknown",
        "risk_level": "unknown",
        "message": None,
        "weight_method": weight_method,
    }
    if card_memory_bytes is None:
        result["message"] = "缺少 card_memory_bytes 或 card_memory_gib"
        return result
    if weight_bytes is None:
        result["message"] = weight_reason or "无法确定模型权重需求"
        return result
    if kv is None:
        result["message"] = "无法确定 KV Cache 需求"
        return result

    utilization = runtime.get("gpu_memory_utilization", 0.9)
    if not _positive_number(utilization) or float(utilization) > 1:
        result["message"] = "gpu_memory_utilization 必须位于 (0, 1]"
        return result
    max_seqs = runtime.get("max_num_seqs")
    min_context = runtime.get("min_required_context_len", 4096)
    block_size = runtime.get("block_size", 16)
    if not (_positive_int(max_seqs) and _positive_int(min_context) and _positive_int(block_size)):
        result["message"] = "max_num_seqs、min_required_context_len 和 block_size 必须为正整数"
        return result

    budget = math.floor(card_memory_bytes * float(utilization))
    weight_per_card = math.ceil(weight_bytes / tp)
    kv_heads_per_rank = max(1, math.ceil(kv["num_key_value_heads"] / tp))
    kv_bytes_per_token = (
        2
        * kv["num_hidden_layers"]
        * kv_heads_per_rank
        * kv["head_dim"]
        * kv["dtype_bytes"]
    )
    context_tokens = math.ceil(min_context / block_size) * block_size
    minimum_kv_bytes = int(max_seqs) * context_tokens * kv_bytes_per_token
    result.update(
        {
            "card_memory_bytes": card_memory_bytes,
            "memory_budget_bytes": budget,
            "weight_bytes_total": weight_bytes,
            "weight_bytes_per_card": weight_per_card,
            "kv_bytes_per_token_per_card": kv_bytes_per_token,
            "minimum_context_tokens_per_sequence": context_tokens,
            "minimum_kv_bytes_per_card": minimum_kv_bytes,
            "gpu_memory_utilization": float(utilization),
            "min_required_context_len": int(min_context),
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


def precheck(payload: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(payload.get("model"))
    hardware = _mapping(payload.get("hardware"))
    runtime = _mapping(payload.get("runtime"))
    card_memory_bytes = _memory_bytes(hardware)
    device_count = hardware.get("device_count")
    if not _positive_int(device_count):
        device_count = None
    dp = runtime.get("dp", 1)
    current_tp = runtime.get("tp")
    official_min_tp = runtime.get("official_min_tp", 1)
    weight_bytes, weight_method, weight_reason = _parameter_count(model, runtime)
    kv, kv_reason = _kv_parameters(model, runtime)
    allowed = _allowed_tp(runtime)

    current: dict[str, Any]
    if current_tp is None:
        current = {
            "status": "missing_parameter",
            "risk_level": "unknown",
            "message": "未提供 tp，当前配置未执行容量判定",
            "tp": None,
            "dp": dp,
        }
    elif not (_positive_int(current_tp) and _positive_int(dp)):
        current = {
            "status": "unknown",
            "risk_level": "unknown",
            "message": "tp 和 dp 必须为正整数",
            "tp": current_tp,
            "dp": dp,
        }
    elif device_count is not None and int(current_tp) * int(dp) > int(device_count):
        current = {
            "status": "reject",
            "risk_level": "resource_exceeded",
            "message": "tp * dp 超过输入的设备数量约束",
            "tp": int(current_tp),
            "dp": int(dp),
            "device_count": int(device_count),
        }
    else:
        current = _evaluate_tp(
            tp=int(current_tp),
            dp=int(dp),
            model=model,
            runtime=runtime,
            card_memory_bytes=card_memory_bytes,
            weight_bytes=weight_bytes,
            kv=kv,
            weight_method=weight_method,
            weight_reason=weight_reason or kv_reason,
        )

    recommendation: dict[str, Any] = {
        "status": "unknown",
        "recommended_tp": None,
        "candidates": [],
        "message": None,
    }
    if not _positive_int(dp):
        recommendation.update(status="unknown", message="dp 必须为正整数")
    elif card_memory_bytes is None or weight_bytes is None or kv is None or device_count is None:
        recommendation.update(
            status="unknown",
            message=weight_reason or kv_reason or "缺少卡容量或设备数量，无法推荐最小 TP",
        )
    else:
        for candidate_tp in allowed:
            if candidate_tp < int(official_min_tp) or candidate_tp * int(dp) > int(device_count):
                continue
            candidate = _evaluate_tp(
                tp=candidate_tp,
                dp=int(dp),
                model=model,
                runtime=runtime,
                card_memory_bytes=card_memory_bytes,
                weight_bytes=weight_bytes,
                kv=kv,
                weight_method=weight_method,
                weight_reason=weight_reason or kv_reason,
            )
            recommendation["candidates"].append(candidate)
            if candidate["status"] == "pass":
                recommendation.update(
                    status="available",
                    recommended_tp=candidate_tp,
                    message=None,
                )
                break
        if recommendation["recommended_tp"] is None and recommendation["status"] != "available":
            recommendation.update(
                status="unavailable",
                message="允许的 TP 候选中没有满足设备或静态容量约束的值",
            )

    top_status = current.get("status")
    if top_status == "missing_parameter":
        status = "unknown"
        message = "未提供当前 tp；请使用 recommendation.recommended_tp 生成候选配置后重新预检"
    elif top_status in {"pass", "reject", "unknown"}:
        status = top_status
        message = current.get("message")
    else:
        status = "unknown"
        message = current.get("message") or "输入无法完成静态预检"

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
    }


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "unknown",
            "risk_level": "unknown",
            "message": f"输入无效: {type(exc).__name__}: {exc}",
            "current": None,
            "recommendation": {
                "status": "unknown",
                "recommended_tp": None,
                "candidates": [],
                "message": "输入解析失败",
            },
        }
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
