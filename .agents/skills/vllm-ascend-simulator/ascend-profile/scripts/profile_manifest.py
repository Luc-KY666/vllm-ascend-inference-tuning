#!/usr/bin/env python3
"""Convert an Ascend profiler directory into the Ascend manifest contract."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

COMMON = Path(__file__).resolve().parents[2] / "ascend-common"
sys.path.insert(0, str(COMMON))
from ascend_contract import (  # noqa: E402
    EVENT_FIELDS,
    SCHEMA_VERSION,
    json_text,
    normalize_name,
    parse_list_field,
    parse_number,
    parse_shape_field,
    write_event_csv,
)


COMMUNICATION_MARKERS = (
    "HCCL", "ALLREDUCE", "ALLGATHER", "REDUCESCATTER", "BROADCAST",
    "SEND", "RECV", "COMMUNICATION", "ALLTOALL",
)
MEMORY_MARKERS = (
    "MEMCPY", "MEMSET", "MALLOC", "FREE", "RESHAPEANDCACHE", "CACHE",
)
METADATA_MARKERS = (
    "GETWORKSPACESIZE", "EXECUTOR", "BINARYGETFUNCTION", "MODELEXECUTE",
    "DEVICESYNCHRONIZE", "EVENTCREATE", "EVENTDESTROY", "EVENTRECORD",
    "EVENTSYNCHRONIZE", "STREAMSYNCHRONIZE", "STREAMWAITEVENT",
    "STREAMSYNCTASKFINISH", "KERNELLAUNCH", "LAUNCHKERNEL", "ACLRTGETRES",
    "ACLRTPOINTERGETATTRIBUTES", "ACLRTQUERYEVENTSTATUS", "ACLRTGETSTREAMATTRIBUTE",
    "LAUNCH",
)


def find_profiler_output(root: Path) -> Path:
    if root.is_file():
        root = root.parent
    if (root / "kernel_details.csv").is_file():
        return root
    candidates = sorted(root.rglob("kernel_details.csv"))
    if not candidates:
        raise FileNotFoundError(f"kernel_details.csv not found under {root}")
    return candidates[0].parent


def classify(name: str, op_type: str = "") -> str:
    upper = f"{name} {op_type}".upper()
    if any(marker in upper for marker in METADATA_MARKERS):
        return "metadata"
    if any(marker in upper for marker in COMMUNICATION_MARKERS):
        return "communication"
    if any(marker in upper for marker in MEMORY_MARKERS):
        return "memory"
    return "compute"


def _base(args: argparse.Namespace, kind: str, name: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "source_model": args.source_model,
        "reduced_layers": args.reduced_layers,
        "hardware": args.hardware,
        "cann_version": args.cann_version,
        "torch_npu_version": args.torch_npu_version,
        "vllm_version": args.vllm_version,
        "world_size": args.world_size,
        "rank": args.rank,
        "phase": args.phase,
        "batch_size": args.batch_size,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "op_name": normalize_name(name),
        "api_symbol": "",
        "kind": kind,
        "input_shapes": [],
        "output_shapes": [],
        "input_dtypes": [],
        "output_dtypes": [],
        "input_formats": [],
        "output_formats": [],
        "message_bytes": 0,
        "allocation_bytes": 0,
        "copy_bytes": 0,
        "kernel_variant": "",
        "stream": "",
        "duration_us": 0.0,
        "count": 1,
        "source_trace": str(args.profile_root),
        "source_file": "",
        "matching_confidence": "observed",
        "sample_group": args.sample_group,
        "sample_id": args.sample_id,
        "workload_id": args.workload_id,
        "phase_source": args.phase_source,
        "prefill_chunk": args.prefill_chunk,
        "kv_prefill": args.kv_prefill,
        "n_decode": args.n_decode,
        "kv_decode": args.kv_decode,
        "collective": args.collective,
        "topology": args.topology,
        "dtype": args.dtype,
        "format": args.format,
    }


def _bytes_from_row(raw: dict[str, Any], *names: str) -> int:
    for name in names:
        value = raw.get(name)
        parsed = parse_number(value, -1)
        if parsed >= 0:
            return int(parsed)
    return 0


def read_kernel_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            name = raw.get("Name", "").strip()
            if not name:
                continue
            kind = classify(name, raw.get("Type", ""))
            row = _base(args, kind, name)
            row.update({
                "input_shapes": parse_shape_field(raw.get("Input Shapes")),
                "output_shapes": parse_shape_field(raw.get("Output Shapes")),
                "input_dtypes": parse_list_field(raw.get("Input Data Types")),
                "output_dtypes": parse_list_field(raw.get("Output Data Types")),
                "input_formats": parse_list_field(raw.get("Input Formats")),
                "output_formats": parse_list_field(raw.get("Output Formats")),
                "message_bytes": _bytes_from_row(raw, "Message Bytes", "Message Size", "Bytes"),
                "allocation_bytes": _bytes_from_row(raw, "Allocation Bytes", "Alloc Bytes"),
                "copy_bytes": _bytes_from_row(raw, "Copy Bytes", "Memcpy Bytes"),
                "kernel_variant": raw.get("Type", "").strip(),
                "stream": raw.get("Stream ID", ""),
                "duration_us": parse_number(raw.get("Duration(us)")),
                "source_file": str(path),
                "matching_confidence": "trace_csv_matched",
            })
            row["dtype"] = row["input_dtypes"][0] if row["input_dtypes"] else args.dtype
            row["format"] = row["input_formats"][0] if row["input_formats"] else args.format
            rows.append(row)
    return rows


def _is_host_side_operator(name: str) -> bool:
    """Return True for CPU-side dispatch operators that must not become
    compute/memory/communication latency events.  These are the aten::/atb::
    and similar host operator wrappers; the real device work lives in
    kernel_details.csv."""
    text = name.strip()
    if not text:
        return True
    upper = text.upper()
    host_prefixes = ("ATEN::", "ACC::", "ATB::", "TORCH::", "ACONNTORCH::", "LAUNCH(", "LAUNCHKERNEL")
    return any(upper.startswith(prefix) for prefix in host_prefixes)


def read_api_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Read host-side runtime API statistics.

    ``api_statistic.csv`` is a CPU-side view of calls made by the host.  It is
    kept as the ``api_calls`` auxiliary artifact, but only memory-management
    and communication API rows are merged into the latency events.  Compute
    API rows (e.g. ``aclnnMatmul`` host dispatch) are explicitly excluded so
    CPU-side operators never pollute the measured kernel latency."""
    rows = []
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            name = raw.get("API Name", "").strip()
            if not name:
                continue
            kind = classify(name)
            row = _base(args, kind, name)
            row.update({
                "api_symbol": name,
                "duration_us": parse_number(raw.get("Time(us)")),
                "count": int(parse_number(raw.get("Count"), 1)),
                "source_file": str(path),
                "matching_confidence": "api_statistic",
                "cpu_side": True,
            })
            rows.append(row)
    return rows


def read_operator_rows(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Read device-side memory/communication operators from operator_details.

    Host dispatch operators (aten::*, atb::*, ...) are skipped here; only
    non-host memory copy/set and collective operators with a positive device
    duration are retained."""
    rows = []
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for raw in csv.DictReader(stream):
            name = raw.get("Name", "").strip()
            if not name:
                continue
            if _is_host_side_operator(name):
                continue
            kind = classify(name)
            if kind not in {"memory", "communication"}:
                continue
            duration = parse_number(raw.get("Device Total Duration(us)"))
            if duration <= 0:
                continue
            row = _base(args, kind, name)
            row.update({
                "duration_us": duration,
                "source_file": str(path),
                "matching_confidence": "operator_details_device",
            })
            rows.append(row)
    return rows


def load_diagnostics(profile_output: Path) -> dict[str, Any]:
    candidates = list(profile_output.rglob("kernel_analysis.json"))
    if not candidates:
        return {}
    try:
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"warning": "kernel_analysis.json could not be decoded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--source-model", default="")
    parser.add_argument("--reduced-layers", type=int, default=0)
    parser.add_argument("--hardware", default="Ascend")
    parser.add_argument("--cann-version", default="unknown")
    parser.add_argument("--torch-npu-version", default="unknown")
    parser.add_argument("--vllm-version", default="unknown")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--phase", default="unknown")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=0)
    parser.add_argument("--sample-group", default="profile")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--workload-id", default="")
    parser.add_argument("--phase-source", default="declared")
    parser.add_argument("--prefill-chunk", type=int, default=0)
    parser.add_argument("--kv-prefill", type=int, default=0)
    parser.add_argument("--n-decode", type=int, default=0)
    parser.add_argument("--kv-decode", type=int, default=0)
    parser.add_argument("--collective", default="")
    parser.add_argument("--topology", default="")
    parser.add_argument("--dtype", default="")
    parser.add_argument("--format", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.profile_root = args.profile_root.resolve()
    profile_output = find_profiler_output(args.profile_root)
    args.output.mkdir(parents=True, exist_ok=True)
    kernel_path = profile_output / "kernel_details.csv"
    rows = read_kernel_rows(kernel_path, args)
    api_rows = read_api_rows(profile_output / "api_statistic.csv", args)
    operator_rows = read_operator_rows(profile_output / "operator_details.csv", args)

    # Only real device work enters the latency events: device kernels plus
    # device-duration memory/communication operators.  CPU-side operators and
    # host runtime APIs are deliberately excluded from compute latency.
    main_api_rows = [row for row in api_rows if row["kind"] in {"memory", "communication"}]

    by_kind = {kind: [] for kind in ("compute", "communication", "memory", "metadata")}
    for row in rows + operator_rows + main_api_rows:
        by_kind.setdefault(row["kind"], []).append(row)
    for kind in ("compute", "communication", "memory"):
        write_event_csv(args.output / "events" / f"{kind}.csv", by_kind[kind])
    write_event_csv(args.output / "events" / "metadata.csv", by_kind["metadata"])
    write_event_csv(args.output / "events" / "api_calls.csv", api_rows)
    write_event_csv(args.output / "events" / "kernels.csv", rows)

    diagnostics = load_diagnostics(profile_output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "ascend_profile",
        "model_id": args.model_id,
        "source_model": args.source_model,
        "reduced_layers": args.reduced_layers,
        "hardware_name": args.hardware,
        "world_size": args.world_size,
        "rank": args.rank,
        "phase": args.phase,
        "batch_size": args.batch_size,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "model": {
            "model_id": args.model_id,
            "source_model": args.source_model,
            "reduced_layers": args.reduced_layers,
        },
        "hardware": {
            "name": args.hardware,
            "cann_version": args.cann_version,
            "torch_npu_version": args.torch_npu_version,
            "vllm_version": args.vllm_version,
            "world_size": args.world_size,
            "rank": args.rank,
        },
        "workload": {
            "phase": args.phase,
            "batch_size": args.batch_size,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "sample_group": args.sample_group,
            "sample_id": args.sample_id,
            "workload_id": args.workload_id,
            "phase_source": args.phase_source,
            "prefill_chunk": args.prefill_chunk,
            "kv_prefill": args.kv_prefill,
            "n_decode": args.n_decode,
            "kv_decode": args.kv_decode,
            "collective": args.collective,
            "topology": args.topology,
            "dtype": args.dtype,
            "format": args.format,
        },
        "profile_root": str(args.profile_root),
        "profiler_output": str(profile_output),
        "events": {
            kind: {
                "path": str(args.output / "events" / f"{kind}.csv"),
                "count": len(by_kind[kind]),
                "duration_us": sum(float(row["duration_us"]) for row in by_kind[kind]),
            }
            for kind in ("compute", "communication", "memory", "metadata")
        },
        "auxiliary_events": {
            kind: {"path": str(args.output / "events" / f"{kind}.csv"), "count": len(rows)}
            for kind, rows in (("api_calls", api_rows), ("kernels", rows))
        },
        "diagnostics": diagnostics,
        "coverage": {
            "kernel_details_present": kernel_path.is_file(),
            "api_statistics_present": (profile_output / "api_statistic.csv").is_file(),
            "operator_details_present": (profile_output / "operator_details.csv").is_file(),
            "device_kernel_rows": len(rows),
            "memory_communication_operator_rows": len(operator_rows),
            "host_api_rows_total": len(api_rows),
            "host_api_rows_merged": len(main_api_rows),
            "cpu_side_operators_excluded": True,
            "warnings": diagnostics.get("diagnostics", {}).get("warnings", []),
        },
    }
    (args.output / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (args.output / "coverage.json").write_text(
        json.dumps(manifest["coverage"], indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(args.output / "profile_manifest.json"),
                      "profile_output": str(profile_output),
                      "event_counts": manifest["events"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
