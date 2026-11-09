#!/usr/bin/env python3
"""Shared manifest and feature helpers for the Ascend skills."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
EVENT_FIELDS = (
    "schema_version", "model_id", "source_model", "reduced_layers",
    "hardware", "cann_version", "torch_npu_version", "vllm_version",
    "world_size", "rank", "phase", "batch_size", "input_tokens",
    "output_tokens", "op_name", "api_symbol", "kind", "input_shapes",
    "output_shapes", "input_dtypes", "output_dtypes", "input_formats",
    "output_formats", "message_bytes", "stream", "duration_us", "count",
    "allocation_bytes", "copy_bytes", "kernel_variant", "source_trace",
    "source_file", "matching_confidence", "sample_group", "sample_id",
    "workload_id", "phase_source", "prefill_chunk", "kv_prefill",
    "n_decode", "kv_decode", "collective", "topology", "dtype", "format",
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def stable_code(value: Any) -> float:
    """Encode a categorical dtype/format as a reproducible numeric feature."""
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return float(int.from_bytes(digest[:4], "big") % 1_000_003 + 1)


def parse_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_shape_field(value: Any) -> list[list[int]]:
    '''Parse Ascend fields such as ``"""40960,128;1;1"""``.'''
    if value is None:
        return []
    text = str(value).strip().replace('"""', "").replace('"', "")
    if not text or text.upper() in {"N/A", "NONE", "NULL"}:
        return []
    result: list[list[int]] = []
    for item in re.split(r"[;|]", text):
        dims: list[int] = []
        for token in item.strip().strip("[]()").split(","):
            token = token.strip()
            if token.lstrip("-+").isdigit():
                dims.append(int(token))
            elif _NUMBER_RE.fullmatch(token):
                dims.append(int(float(token)))
        if dims:
            result.append(dims)
    return result


def parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip().replace('"""', "").replace('"', "")
    if not text or text.upper() in {"N/A", "NONE", "NULL"}:
        return []
    return [item.strip() for item in re.split(r"[;|]", text) if item.strip()]


def shape_numel(shapes: Iterable[Iterable[int]]) -> int:
    total = 0
    for shape in shapes:
        dims = list(shape)
        if not dims or any(dim < 0 for dim in dims):
            continue
        product = 1
        for dim in dims:
            product *= dim
        total += product
    return total


def normalize_name(name: Any) -> str:
    text = re.sub(r"_[0-9a-f]{8,}(?=_|$)", "", str(name or "").strip(), flags=re.I)
    return re.sub(r"\s+", "_", text) or "unknown"


def event_key(event: dict[str, Any]) -> str:
    dtypes = decode_json_field(event.get("input_dtypes", ""))
    dtype_key = json_text(dtypes) if isinstance(dtypes, (list, dict)) else str(dtypes)
    fields = (
        event.get("kind", "unknown"), normalize_name(event.get("op_name")),
        event.get("api_symbol", ""), event.get("hardware", ""),
        event.get("world_size", 1), event.get("phase", "unknown"),
        dtype_key,
    )
    if str(event.get("kind", "")).lower() == "communication":
        fields += (event.get("collective", ""), event.get("topology", ""))
    return "|".join(str(item) for item in fields)


def model_file_name(group_key: str) -> str:
    return f"model-{hashlib.sha256(group_key.encode()).hexdigest()[:16]}.joblib"


def write_event_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for row in rows:
            output = {}
            for field in EVENT_FIELDS:
                value = row.get(field, "")
                if field in {"input_shapes", "output_shapes", "input_dtypes",
                             "output_dtypes", "input_formats", "output_formats"}:
                    value = value if isinstance(value, str) else json_text(value)
                output[field] = value
            writer.writerow(output)


def read_event_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def decode_json_field(value: Any) -> Any:
    if isinstance(value, (list, dict, int, float)):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return parse_list_field(text)


def event_features(event: dict[str, Any]) -> dict[str, float]:
    input_shapes = decode_json_field(event.get("input_shapes"))
    output_shapes = decode_json_field(event.get("output_shapes"))
    input_shapes = input_shapes if isinstance(input_shapes, list) else []
    output_shapes = output_shapes if isinstance(output_shapes, list) else []
    features: dict[str, float] = {
        "world_size": parse_number(event.get("world_size"), 1),
        "rank": parse_number(event.get("rank")),
        "batch_size": parse_number(event.get("batch_size"), 1),
        "input_tokens": parse_number(event.get("input_tokens")),
        "output_tokens": parse_number(event.get("output_tokens")),
        "message_bytes": parse_number(event.get("message_bytes")),
        "allocation_bytes": parse_number(event.get("allocation_bytes")),
        "copy_bytes": parse_number(event.get("copy_bytes")),
        "stream": parse_number(event.get("stream")),
        "input_numel": float(shape_numel(input_shapes)),
        "output_numel": float(shape_numel(output_shapes)),
        "prefill_chunk": parse_number(event.get("prefill_chunk")),
        "kv_prefill": parse_number(event.get("kv_prefill")),
        "n_decode": parse_number(event.get("n_decode")),
        "kv_decode": parse_number(event.get("kv_decode")),
    }
    input_dtypes = decode_json_field(event.get("input_dtypes"))
    output_dtypes = decode_json_field(event.get("output_dtypes"))
    input_formats = decode_json_field(event.get("input_formats"))
    output_formats = decode_json_field(event.get("output_formats"))
    features.update({
        "input_dtype_code": stable_code("|".join(map(str, input_dtypes)) if isinstance(input_dtypes, list) else input_dtypes),
        "output_dtype_code": stable_code("|".join(map(str, output_dtypes)) if isinstance(output_dtypes, list) else output_dtypes),
        "input_format_code": stable_code("|".join(map(str, input_formats)) if isinstance(input_formats, list) else input_formats),
        "output_format_code": stable_code("|".join(map(str, output_formats)) if isinstance(output_formats, list) else output_formats),
        "kernel_variant_code": stable_code(event.get("kernel_variant")),
        "collective_code": stable_code(event.get("collective")),
        "topology_code": stable_code(event.get("topology")),
        "dtype_code": stable_code(event.get("dtype")),
        "format_code": stable_code(event.get("format")),
        "phase_code": stable_code(event.get("phase")),
    })
    input_dims = [d for shape in input_shapes if isinstance(shape, list) for d in shape]
    output_dims = [d for shape in output_shapes if isinstance(shape, list) for d in shape]
    for index, value in enumerate(input_dims[:8]):
        features[f"input_dim_{index}"] = float(value)
    for index, value in enumerate(output_dims[:8]):
        features[f"output_dim_{index}"] = float(value)
    if input_shapes and len(input_shapes[0]) >= 2:
        features["matmul_m"] = float(input_shapes[0][-2])
        features["matmul_k"] = float(input_shapes[0][-1])
    if len(input_shapes) >= 2 and input_shapes[1] and len(input_shapes[1]) >= 2:
        features["matmul_n"] = float(input_shapes[1][-1])
    elif output_shapes and output_shapes[0] and len(output_shapes[0]) >= 2:
        features["matmul_n"] = float(output_shapes[0][-1])
    return features


def event_target(event: dict[str, Any]) -> float:
    return parse_number(event.get("duration_us"))
