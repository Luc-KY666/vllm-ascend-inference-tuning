#!/usr/bin/env python3
"""Build a reproducible profile workload plan from observed manifests.

The output is an execution plan. ``run_profile_sweep.py`` executes its
workloads on an Ascend host; this command never invents measured durations.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
from typing import Any


FAST_ATTENTION = [
    (8, 8, 0, 0), (32, 32, 0, 0), (128, 128, 0, 0),
    (0, 128, 1, 128), (0, 512, 1, 512), (0, 2048, 1, 2048),
    (32, 512, 1, 512),
]
DEFAULT_COLLECTIVES = ("all_reduce", "reduce_scatter", "all_gather", "broadcast", "send_recv")


def ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _read_events(manifest_path: Path, manifest: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    root = manifest_path.parent
    path = Path(manifest.get("events", {}).get(kind, {}).get("path", root / "events" / f"{kind}.csv"))
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _inventory(paths: list[Path]) -> list[dict[str, Any]]:
    descriptors: dict[tuple[str, ...], dict[str, Any]] = {}
    for manifest_path in paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for kind in ("compute", "communication", "memory"):
            for row in _read_events(manifest_path, manifest, kind):
                fields = ("kind", "op_name", "api_symbol", "hardware", "world_size", "phase", "input_dtypes", "collective", "topology")
                key = tuple(str(row.get(field, "")) for field in fields)
                descriptors[key] = {field: row.get(field, "") for field in fields}
    return [descriptors[key] for key in sorted(descriptors)]


def _attention(value: str) -> list[tuple[int, int, int, int]]:
    if value == "fast":
        return FAST_ATTENTION
    parsed = json.loads(value)
    return [tuple(int(item) for item in point) for point in parsed]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--hardware", default="Ascend")
    parser.add_argument("--cann-version", default="unknown")
    parser.add_argument("--torch-npu-version", default="unknown")
    parser.add_argument("--vllm-version", default="unknown")
    parser.add_argument(
        "--profile-runner",
        type=Path,
        default=Path(os.environ.get("ASCEND_PROFILE_RUNNER", "")),
        help="bash script that performs one collection run",
    )
    parser.add_argument("--runner-env", help="JSON object of extra env for --profile-runner")
    parser.add_argument("--tp", type=int, default=2, choices=(1, 2))
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--phases", default="prefill,decode")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--input-tokens", default="8,32,128")
    parser.add_argument("--output-tokens", default="1")
    parser.add_argument("--world-sizes", default="1,2")
    parser.add_argument("--message-bytes", default="4096,65536,1048576,16777216")
    parser.add_argument("--attention-shots", default="fast")
    parser.add_argument("--collectives", default=",".join(DEFAULT_COLLECTIVES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sampling-budget", choices=("fast", "standard", "exhaustive"), default="fast")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model:
        raise SystemExit("--model is required (path to the source model checkpoint)")
    manifests = [path.resolve() for path in args.manifest]
    inventory = _inventory(manifests)
    batches = ints(args.batch_sizes)
    input_tokens = ints(args.input_tokens)
    output_tokens = ints(args.output_tokens)
    world_sizes = [value for value in ints(args.world_sizes) if value <= args.tp]
    phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    attention = _attention(args.attention_shots)
    rows: list[dict[str, Any]] = []
    counter = 0

    for phase, batch, tokens, output_tokens_value, world_size, repeat in itertools.product(
        phases, batches, input_tokens, output_tokens, world_sizes, range(args.repeats)
    ):
        counter += 1
        workload_id = f"{phase}-b{batch}-i{tokens}-o{output_tokens_value}-tp{world_size}-r{repeat}"
        if phase == "prefill":
            profile_args = {
                "profile_group": "dense", "fixed_shot_size": batch * tokens,
                "phase": "prefill", "batch_size": batch,
                "input_tokens": tokens, "output_tokens": output_tokens_value,
            }
        else:
            history = max(tokens, 1)
            profile_args = {
                "profile_group": "attention",
                "attention_shot": {"prefill_chunk": 0, "kv_prefill": 0, "n_decode": batch, "kv_decode": history},
                "phase": "decode", "batch_size": batch,
                "input_tokens": history, "output_tokens": output_tokens_value,
            }
        rows.append({"sample_id": f"sample-{counter:05d}", "workload_id": workload_id,
                     "kind": "compute", "world_size": world_size, "repeat": repeat,
                     "profile_args": profile_args})

    for point, repeat in itertools.product(attention, range(args.repeats)):
        prefill_chunk, kv_prefill, n_decode, kv_decode = point
        phase = "mixed" if prefill_chunk and n_decode else ("prefill" if prefill_chunk else "decode")
        counter += 1
        workload_id = f"attention-p{prefill_chunk}-kp{kv_prefill}-n{n_decode}-kd{kv_decode}-r{repeat}"
        rows.append({"sample_id": f"sample-{counter:05d}", "workload_id": workload_id,
                     "kind": "attention", "world_size": args.tp, "repeat": repeat,
                     "profile_args": {
                         "profile_group": "attention", "phase": phase,
                         "batch_size": max(1, n_decode), "input_tokens": kv_prefill or kv_decode,
                         "output_tokens": 1,
                         "attention_shot": {"prefill_chunk": prefill_chunk, "kv_prefill": kv_prefill,
                                            "n_decode": n_decode, "kv_decode": kv_decode},
                     }})

    rows.append({"sample_id": f"sample-{counter + 1:05d}", "workload_id": "hccl-tp2-fast",
                 "kind": "communication", "world_size": 2, "repeat": 1,
                 "profile_args": {"collectives": [item.strip() for item in args.collectives.split(",") if item.strip()],
                                  "message_bytes": ints(args.message_bytes), "warmup": 3, "repeats": 5}})
    payload = {
        "schema_version": 2, "plan_type": "ascend_profile_sweep",
        "model": {"model_id": args.model_id, "source_model": str(Path(args.model).resolve()), "reduced_layers": args.layers},
        "hardware": {"name": args.hardware, "devices": args.devices.split(","), "max_tp": args.tp},
        "software": {"cann_version": args.cann_version, "torch_npu_version": args.torch_npu_version,
                     "vllm_version": args.vllm_version},
        "runner": {"script": str(args.profile_runner.resolve()) if args.profile_runner else "",
                    "env": json.loads(args.runner_env) if args.runner_env else {}},
        "source_manifests": [str(path) for path in manifests],
        "operator_inventory": inventory, "sampling_budget": args.sampling_budget,
        "workloads": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"workloads": len(rows), "operators": len(inventory), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
