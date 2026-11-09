#!/usr/bin/env python3
"""Measure real HCCL collectives on the visible Ascend devices.

Run with ``torchrun --nproc_per_node=2``.  Each rank writes a raw JSON file;
rank zero merges those files into the standard communication event CSV and
manifest after all ranks have reached the final barrier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

COMMON = Path(__file__).resolve().parents[2] / "ascend-common"
sys.path.insert(0, str(COMMON))
from ascend_contract import SCHEMA_VERSION, write_event_csv  # noqa: E402


COLLECTIVES = ("all_reduce", "reduce_scatter", "all_gather", "broadcast", "send_recv")


def parse_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--message-bytes", type=parse_ints, default=parse_ints("4096,65536,1048576,16777216"))
    parser.add_argument("--collectives", default=",".join(COLLECTIVES))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--model-id", default="")
    parser.add_argument("--source-model", default="")
    parser.add_argument("--hardware", default="Ascend")
    parser.add_argument("--cann-version", default="unknown")
    parser.add_argument("--torch-npu-version", default="unknown")
    parser.add_argument("--topology", default="npu0-npu1")
    parser.add_argument("--sample-group", default="hccl-tp2-fast")
    return parser.parse_args()


def _dtype(torch: Any, name: str) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _dtype_name(name: str) -> str:
    return {"bfloat16": "DT_BF16", "float16": "DT_FLOAT16", "float32": "DT_FLOAT"}[name]


def _timed(torch: Any, operation: Callable[[], None], stream: Any) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record(stream)
    operation()
    end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end)) * 1000.0


def _run_collective(dist: Any, torch: Any, name: str, tensor: Any, count: int, rank: int, world: int) -> None:
    if name == "all_reduce":
        dist.all_reduce(tensor, async_op=False)
    elif name == "reduce_scatter":
        output = torch.empty(count, dtype=tensor.dtype, device=tensor.device)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(output, tensor, async_op=False)
        else:
            chunks = list(tensor.chunk(world))
            dist.reduce_scatter(output, chunks, async_op=False)
    elif name == "all_gather":
        outputs = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(outputs, tensor, async_op=False)
    elif name == "broadcast":
        dist.broadcast(tensor, src=0, async_op=False)
    elif name == "send_recv":
        if world != 2:
            raise RuntimeError("send_recv benchmark currently requires world_size=2")
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)
    else:
        raise ValueError(f"unsupported collective: {name}")


def _event(args: argparse.Namespace, rank: int, world: int, name: str, size: int, duration: float, repeat: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": args.model_id,
        "source_model": args.source_model,
        "reduced_layers": 2,
        "hardware": args.hardware,
        "cann_version": args.cann_version,
        "torch_npu_version": args.torch_npu_version,
        "vllm_version": "",
        "world_size": world,
        "rank": rank,
        "phase": "communication",
        "batch_size": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "op_name": name,
        "api_symbol": f"torch.distributed.{name}",
        "kind": "communication",
        "input_shapes": [[size // 2 * world if name == "reduce_scatter" else size // 2]],
        "output_shapes": [[size // 2]],
        "input_dtypes": [_dtype_name(args.dtype)],
        "output_dtypes": [_dtype_name(args.dtype)],
        "input_formats": ["ND"],
        "output_formats": ["ND"],
        "message_bytes": size,
        "stream": "0",
        "duration_us": duration,
        "count": 1,
        "allocation_bytes": size,
        "copy_bytes": 0,
        "kernel_variant": "HCCL",
        "source_trace": str(args.output.resolve()),
        "source_file": str(args.output.resolve() / f"rank-{rank}.json"),
        "matching_confidence": "hccl_event_timed",
        "sample_group": args.sample_group,
        "sample_id": f"{name}-bytes{size}-rank{rank}-repeat{repeat}",
        "workload_id": f"{name}-bytes{size}",
        "phase_source": "hccl_benchmark",
        "prefill_chunk": 0,
        "kv_prefill": 0,
        "n_decode": 0,
        "kv_decode": 0,
        "collective": name,
        "topology": args.topology,
        "dtype": _dtype_name(args.dtype),
        "format": "ND",
    }


def _write_failure(args: argparse.Namespace, rank: int, world: int, exc: Exception, stage: str) -> None:
    failure = {
        "rank": rank, "world_size": world, "status": "failed", "stage": stage,
        "error": f"{type(exc).__name__}: {exc}",
        "environment": {key: os.environ.get(key, "") for key in
                        ("MASTER_ADDR", "MASTER_PORT", "HCCL_SOCKET_IFNAME", "HCCL_IF_IP")},
    }
    (args.output / f"rank-{rank}-failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
    if rank == 0:
        (args.output / "coverage.json").write_text(json.dumps({"status": "failed", "failure": failure}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(failure, indent=2), file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("--warmup must be >= 0 and --repeats must be positive")
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise SystemExit("torch and torch_npu are required for HCCL profiling") from exc
    import torch_npu  # noqa: F401  # registers torch.npu

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world != 2:
        raise SystemExit("profile_hccl.py requires exactly two torchrun ranks")
    torch.npu.set_device(local_rank)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        dist.init_process_group(backend="hccl", init_method="env://")
    except Exception as exc:
        _write_failure(args, rank, world, exc, "init_process_group")
        return 2
    stream = torch.npu.current_stream()
    rows: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, Any]] = {}
    dtype = _dtype(torch, args.dtype)
    names = [name.strip() for name in args.collectives.split(",") if name.strip()]
    group_healthy = True

    try:
        for name in names:
            if name not in COLLECTIVES:
                coverage[name] = {"status": "unsupported_name"}
                continue
            coverage[name] = {"status": "ok", "message_bytes": [], "samples": 0}
            for requested_size in args.message_bytes:
                count = max(1, math.ceil(requested_size / torch.tensor([], dtype=dtype).element_size()))
                actual_size = count * torch.tensor([], dtype=dtype).element_size()
                tensor_count = count * world if name == "reduce_scatter" else count
                tensor = torch.ones(tensor_count, dtype=dtype, device=f"npu:{local_rank}")
                try:
                    dist.barrier()
                except Exception as exc:
                    _write_failure(args, rank, world, exc, f"barrier_before_{name}_{requested_size}")
                    group_healthy = False
                    try:
                        dist.destroy_process_group()
                    except Exception:
                        pass
                    return 2
                for _ in range(args.warmup):
                    _run_collective(dist, torch, name, tensor, count, rank, world)
                torch.npu.synchronize()
                for repeat in range(args.repeats):
                    dist.barrier()
                    duration = _timed(
                        torch,
                        lambda: _run_collective(dist, torch, name, tensor, count, rank, world),
                        stream,
                    )
                    rows.append(_event(args, rank, world, name, actual_size, duration, repeat))
                coverage[name]["message_bytes"].append(actual_size)
                coverage[name]["samples"] += args.repeats
    except Exception as exc:
        _write_failure(args, rank, world, exc, "collective_measurement")
        group_healthy = False
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        return 2
    finally:
        rank_file = args.output / f"rank-{rank}.json"
        rank_file.write_text(json.dumps({"rank": rank, "events": rows, "coverage": coverage}, indent=2) + "\n", encoding="utf-8")
        if group_healthy:
            dist.barrier()
        if rank == 0 and group_healthy:
            all_rows: list[dict[str, Any]] = []
            all_coverage: dict[str, Any] = {}
            for other_rank in range(world):
                data = json.loads((args.output / f"rank-{other_rank}.json").read_text(encoding="utf-8"))
                all_rows.extend(data.get("events", []))
                for key, value in data.get("coverage", {}).items():
                    all_coverage.setdefault(key, {"status": value.get("status"), "message_bytes": [], "samples": 0})
                    all_coverage[key]["message_bytes"] = sorted(set(all_coverage[key]["message_bytes"] + value.get("message_bytes", [])))
                    all_coverage[key]["samples"] += value.get("samples", 0)
            event_path = args.output / "events" / "communication.csv"
            write_event_csv(event_path, all_rows)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "manifest_type": "ascend_hccl_profile",
                "model_id": args.model_id,
                "source_model": args.source_model,
                "reduced_layers": 2,
                "hardware_name": args.hardware,
                "world_size": world,
                "rank": 0,
                "phase": "communication",
                "model": {"model_id": args.model_id, "source_model": args.source_model, "reduced_layers": 2},
                "hardware": {"name": args.hardware, "cann_version": args.cann_version,
                             "torch_npu_version": args.torch_npu_version,
                             "world_size": world, "rank": 0},
                "workload": {"phase": "communication", "sample_group": args.sample_group, "topology": args.topology},
                "events": {"communication": {"path": str(event_path), "count": len(all_rows), "duration_us": sum(row["duration_us"] for row in all_rows)}},
                "coverage": {"status": "complete" if all(value.get("status") == "ok" for value in all_coverage.values()) else "incomplete", "collectives": all_coverage},
            }
            (args.output / "profile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            (args.output / "coverage.json").write_text(json.dumps(manifest["coverage"], indent=2) + "\n", encoding="utf-8")
        if group_healthy:
            dist.barrier()
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
