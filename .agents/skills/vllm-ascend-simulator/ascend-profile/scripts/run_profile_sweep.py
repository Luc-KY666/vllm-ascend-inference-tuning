#!/usr/bin/env python3
"""Execute an ``ascend_profile_sweep`` plan and preserve every raw result."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROFILE_MODEL = Path(__file__).resolve().with_name("profile_model.py")
PROFILE_HCCL = Path(__file__).resolve().with_name("profile_hccl.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--torchrun", default=shutil.which("torchrun") or "torchrun")
    parser.add_argument("--max-workloads", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args()


def _profile_command(args: argparse.Namespace, plan: dict[str, Any], workload: dict[str, Any], output: Path) -> list[str]:
    profile = workload["profile_args"]
    world_size = int(workload.get("world_size", 1))
    devices = ",".join(plan["hardware"]["devices"][:world_size])
    command = [args.python, str(PROFILE_MODEL), "--model", plan["model"]["source_model"],
               "--output", str(output), "--layers", str(plan["model"]["reduced_layers"]),
               "--tp", str(world_size), "--devices", devices, "--hardware", plan["hardware"]["name"],
               "--cann-version", plan.get("software", {}).get("cann_version", "unknown"),
               "--torch-npu-version", plan.get("software", {}).get("torch_npu_version", "unknown"),
               "--vllm-version", plan.get("software", {}).get("vllm_version", "unknown"),
               "--profile-group", profile["profile_group"], "--phase", profile["phase"],
               "--batch-size", str(profile["batch_size"]), "--input-tokens", str(profile["input_tokens"]),
               "--output-tokens", str(profile["output_tokens"]), "--iterations", str(args.iterations),
               "--sample-id", workload["sample_id"], "--workload-id", workload["workload_id"],
               "--phase-source", "shot", "--topology", f"tp{world_size}"]
    runner = plan.get("runner") or {}
    if runner.get("script"):
        command.extend(["--profile-runner", str(runner["script"])])
    if runner.get("env"):
        command.extend(["--runner-env", json.dumps(runner["env"])])
    if "fixed_shot_size" in profile:
        command.extend(["--fixed-shot-size", str(profile["fixed_shot_size"])])
        command.extend(["--max-batched-tokens", str(max(8, int(profile["fixed_shot_size"])))])
    if "attention_shot" in profile:
        command.extend(["--attention-shot", json.dumps(profile["attention_shot"])])
        shot = profile["attention_shot"]
        command.extend(["--max-batched-tokens", str(max(8, int(shot["prefill_chunk"]) + int(shot["n_decode"])))])
        command.extend(["--max-num-seqs", str(max(1, int(shot["n_decode"]) + (1 if shot["prefill_chunk"] else 0)))])
    return command


def _hccl_command(args: argparse.Namespace, plan: dict[str, Any], workload: dict[str, Any], output: Path, port: int) -> list[str]:
    profile = workload["profile_args"]
    return [args.torchrun, "--master_addr=127.0.0.1", f"--master_port={port}", "--nproc_per_node=2", str(PROFILE_HCCL),
            "--output", str(output), "--model-id", plan.get("model", {}).get("model_id", ""),
            "--source-model", plan["model"]["source_model"], "--hardware", plan["hardware"]["name"],
            "--cann-version", plan.get("software", {}).get("cann_version", "unknown"),
            "--torch-npu-version", plan.get("software", {}).get("torch_npu_version", "unknown"),
            "--collectives", ",".join(profile["collectives"]),
            "--message-bytes", ",".join(str(value) for value in profile["message_bytes"]),
            "--warmup", str(profile["warmup"]), "--repeats", str(profile["repeats"])]


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    workloads = plan.get("workloads", [])
    if args.max_workloads is not None:
        workloads = workloads[:args.max_workloads]
    results = []
    for index, workload in enumerate(workloads):
        work_dir = args.output / f"{index:04d}-{workload['workload_id']}"
        work_dir.mkdir(parents=True, exist_ok=True)
        if workload.get("kind") == "communication":
            command = _hccl_command(args, plan, workload, work_dir, 29500 + index)
        else:
            command = _profile_command(args, plan, workload, work_dir)
        (work_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
        if args.dry_run:
            results.append({"workload_id": workload["workload_id"], "status": "planned", "command": command})
            continue
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(plan["hardware"]["devices"][:int(workload.get("world_size", 1))])
        completed = subprocess.run(command, env=env, cwd=Path(__file__).resolve().parents[3])
        status = "complete" if completed.returncode == 0 else "failed"
        results.append({"workload_id": workload["workload_id"], "status": status,
                        "returncode": completed.returncode, "output": str(work_dir)})
        if completed.returncode and not args.allow_failures:
            raise SystemExit(f"workload failed: {workload['workload_id']} rc={completed.returncode}")
    index = {"schema_version": 2, "plan": str(args.plan.resolve()), "results": results,
             "manifests": [str(path) for path in args.output.glob("*/profile_manifest.json")]}
    (args.output / "collection_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
