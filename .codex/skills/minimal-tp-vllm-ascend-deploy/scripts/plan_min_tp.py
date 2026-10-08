#!/usr/bin/env python3
"""Compute the smallest allowed single-node TP for vLLM-Ascend."""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any

GIB = 1024**3
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
ALLOWED_TP = (1, 2, 4, 8, 16)


def _first(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def _config_view(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _weight_bytes(args: argparse.Namespace) -> int:
    if args.weight_gib is not None:
        return int((Decimal(str(args.weight_gib)) * GIB).to_integral_value(rounding=ROUND_FLOOR))
    if args.weight_dir is None:
        raise ValueError("provide --weight-gib or --weight-dir")
    total = sum(
        path.stat().st_size
        for path in Path(args.weight_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
    )
    if total <= 0:
        raise ValueError("no model weight files found in --weight-dir")
    return total


def _load_arch(args: argparse.Namespace) -> tuple[dict[str, int] | None, str]:
    if args.config is None:
        return None, "missing_config"
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    view = _config_view(config)
    layers = _first(view, "num_hidden_layers", "num_layers")
    kv_heads = _first(view, "num_key_value_heads", "num_kv_heads")
    attention_heads = _first(view, "num_attention_heads")
    kv_heads = kv_heads or attention_heads
    head_dim = _first(view, "head_dim")
    if head_dim is None and view.get("hidden_size") and attention_heads:
        head_dim = int(view["hidden_size"]) // int(attention_heads)
    dtype = str(_first(view, "torch_dtype", "dtype") or "float16").lower()
    kv_dtype_bytes = 1 if dtype in {"float8_e4m3fn", "float8_e5m2", "int8", "uint8"} else 2
    if None in (layers, kv_heads, head_dim):
        return None, "missing_architecture_fields"
    return {
        "num_layers": int(layers),
        "num_kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "kv_dtype_bytes": kv_dtype_bytes,
    }, "config"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--weight-dir", type=Path)
    parser.add_argument("--weight-gib", type=Decimal)
    parser.add_argument("--vram-gib", type=Decimal, required=True)
    parser.add_argument("--available-devices", type=int, required=True)
    parser.add_argument("--min-context", type=int, default=4096)
    parser.add_argument("--official-context", type=int)
    parser.add_argument("--official-min-tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=Decimal, default=Decimal("0.9"))
    parser.add_argument("--runtime-headroom-factor", type=Decimal, default=Decimal("0.9"))
    args = parser.parse_args()

    if args.available_devices < 1 or args.available_devices > max(ALLOWED_TP):
        raise SystemExit("--available-devices must be between 1 and 16")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")
    if not 0 < args.runtime_headroom_factor <= 1:
        raise SystemExit("--runtime-headroom-factor must be in (0, 1]")
    if args.official_min_tp < 1:
        raise SystemExit("--official-min-tp must be positive")

    weight_bytes = _weight_bytes(args)
    arch, context_source = _load_arch(args)
    if arch is None and args.official_context is None:
        raise SystemExit("config architecture is incomplete; provide --official-context")
    if arch is None and args.official_context < args.min_context:
        raise SystemExit("official context is below --min-context")

    vram_bytes = int(args.vram_gib * GIB)
    planning_utilization = args.gpu_memory_utilization * args.runtime_headroom_factor
    base_usable_per_card = int((Decimal(vram_bytes) * args.gpu_memory_utilization).to_integral_value(rounding=ROUND_FLOOR))
    safe_usable_per_card = int((Decimal(vram_bytes) * planning_utilization).to_integral_value(rounding=ROUND_FLOOR))
    min_cards_by_weight = math.ceil(weight_bytes / safe_usable_per_card)
    raw_required = max(min_cards_by_weight, args.official_min_tp, 1)
    candidates = [tp for tp in ALLOWED_TP if tp >= raw_required and tp <= args.available_devices]
    if not candidates:
        raise SystemExit("required TP exceeds available devices or allowed TP values")

    result: dict[str, Any] = {
        "weight_bytes": weight_bytes,
        "vram_bytes_per_device": vram_bytes,
        "gpu_memory_utilization": str(args.gpu_memory_utilization),
        "runtime_headroom_factor": str(args.runtime_headroom_factor),
        "planning_memory_utilization": str(planning_utilization),
        "available_devices": args.available_devices,
        "min_context": args.min_context,
        "official_min_tp": args.official_min_tp,
        "min_cards_by_weight": min_cards_by_weight,
        "raw_required_cards": raw_required,
        "allowed_tp": list(ALLOWED_TP),
        "context_source": context_source if arch else "official_fallback",
    }
    for tp in candidates:
        usable_total = tp * base_usable_per_card
        safe_total = tp * safe_usable_per_card
        kv_available = safe_total - weight_bytes
        if arch:
            kv_per_token = 2 * arch["num_layers"] * arch["num_kv_heads"] * arch["head_dim"] * arch["kv_dtype_bytes"]
            if kv_per_token < 1024:
                raise SystemExit("kv_bytes_per_token is below the safety floor of 1024")
            theoretical = kv_available // kv_per_token if kv_available > 0 else 0
        else:
            kv_per_token = None
            theoretical = args.official_context
        if theoretical > 1048576:
            raise SystemExit("theoretical_max_model_len exceeds the safety ceiling of 1048576")
        if kv_available > 0 and theoretical >= args.min_context:
            result.update(
                {
                    "tp": tp,
                    "model_total_usable_vram_bytes": usable_total,
                    "model_total_safe_vram_bytes": safe_total,
                    "available_kv_bytes_total": kv_available,
                    "kv_bytes_per_token": kv_per_token,
                    "theoretical_max_model_len": theoretical,
                }
            )
            if arch:
                result["architecture"] = arch
            print(json.dumps(result, ensure_ascii=True, indent=2))
            return 0

    raise SystemExit("no allowed TP satisfies KV/context requirements")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"input error: {exc}") from exc
