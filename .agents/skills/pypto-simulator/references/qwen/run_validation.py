#!/usr/bin/env python3
"""Run adjacent native-A, sleep-simulator, native-B Qwen validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WORKLOAD = ROOT / "pypto-emulate" / "scripts" / "run_validation_workload.py"
SLEEP_LAUNCHER = ROOT / "pypto-emulate" / "scripts" / "run_sleep_simulator.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--memory-profile",
        type=Path,
        help="strict-ready measured memory profile for virtual weight/KV materialization",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="reuse completed cases and archive failed attempts")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--prompt", default="Huawei is")
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--sim-runtime-lib-dir",
        type=Path,
        help="optional build/lib containing a custom a2a3sim runtime",
    )
    parser.add_argument("--ring-dep-pool", type=int, default=262144)
    parser.add_argument("--ring-task-window", type=int, default=262144)
    parser.add_argument("--ring-heap", type=int, default=1073741824)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--device-wait-seconds", type=float, default=3600.0)
    parser.add_argument("--run-timeout-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--ptoas-root",
        type=Path,
        default=Path("/root/revati/.venvs/pypto-simulator"),
    )
    parser.add_argument(
        "--pypto-lib",
        type=Path,
        default=Path("/root/revati/pypto-serving/pypto-lib"),
    )
    args = parser.parse_args()
    positive = (
        args.requests,
        args.max_new_tokens,
        args.max_model_len,
        args.max_num_seqs,
        args.max_num_batched_tokens,
        args.ring_dep_pool,
        args.ring_task_window,
        args.ring_heap,
    )
    if args.device < 0 or args.warmup_requests < 0 or any(value <= 0 for value in positive):
        parser.error("device must be non-negative and workload sizes positive")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    values = [float(row["wall_ms"]) for row in rows if row["phase"] == "measure"]
    if not values:
        raise ValueError("no measured requests")
    return {
        "requests": len(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def wait_for_device(device: int, timeout_s: float, poll_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "proc-mem", "-i", str(device)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and "No process in device." in result.stdout:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"NPU {device} remained occupied: {result.stdout}{result.stderr}")
        print(f"[pypto-validation] waiting for NPU {device}", flush=True)
        time.sleep(poll_s)


def serving_args(args: argparse.Namespace, platform_name: str, device: int) -> list[str]:
    prompt_count = args.warmup_requests + args.requests
    command = [
        "--model",
        str(args.model.resolve()),
        "--platform",
        platform_name,
        "--device",
        str(device),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--ring-dep-pool",
        str(args.ring_dep_pool),
        "--ring-task-window",
        str(args.ring_task_window),
        "--ring-heap",
        str(args.ring_heap),
        "--generate-config",
        json.dumps(
            {
                "max_new_tokens": args.max_new_tokens,
                "temperature": 0,
                "top_p": 1,
                "ignore_eos": True,
            },
            separators=(",", ":"),
        ),
    ]
    for _ in range(prompt_count):
        command.extend(("--prompt", args.prompt))
    return command


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def validate_rows(
    rows: list[dict[str, Any]], args: argparse.Namespace, *, expected_token: int | None
) -> None:
    expected_rows = args.warmup_requests + args.requests
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} request rows, got {len(rows)}")
    for row in rows:
        if row["token_count"] != args.max_new_tokens or row["finish_reason"] != "length":
            raise RuntimeError(f"invalid request result: {row}")
        if expected_token is not None and any(token != expected_token for token in row["token_ids"]):
            raise RuntimeError(f"simulator returned unexpected control token: {row}")


def run_case(
    label: str,
    kind: str,
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
    expected_token: int,
) -> dict[str, Any]:
    run_dir = root / label
    if (run_dir / "summary.json").is_file():
        print(f"[pypto-validation] reuse completed {label}", flush=True)
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if run_dir.exists():
        suffix = 1
        while (root / f"{label}.failed-{suffix}").exists():
            suffix += 1
        run_dir.rename(root / f"{label}.failed-{suffix}")
    run_dir.mkdir()
    requests_path = run_dir / "requests.jsonl"
    workload = [
        sys.executable,
        str(WORKLOAD),
        "--output",
        str(requests_path),
        "--warmup-requests",
        str(args.warmup_requests),
        "--requests",
        str(args.requests),
        "--",
    ]
    telemetry = run_dir / "sleep_telemetry.jsonl"
    if kind == "native":
        wait_for_device(args.device, args.device_wait_seconds, args.poll_seconds)
        command = workload + serving_args(args, "a2a3", args.device)
    else:
        command = [
            sys.executable,
            str(SLEEP_LAUNCHER),
            "--plan",
            str(args.plan.resolve()),
            "--telemetry",
            str(telemetry),
        ]
        if args.memory_profile is not None:
            command.extend(("--memory-profile", str(args.memory_profile.resolve())))
        if args.sim_runtime_lib_dir is not None:
            command.extend(("--runtime-lib-dir", str(args.sim_runtime_lib_dir.resolve())))
        command.extend([
            "--",
            *workload,
            *serving_args(args, "a2a3sim", 0),
        ])

    started = datetime.now(timezone.utc)
    print(f"[pypto-validation] start {label}", flush=True)
    with (run_dir / "run.log").open("x", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.run_timeout_seconds,
                check=False,
            )
            returncode = result.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = 124
            timed_out = True
    record = {
        "schema_version": 1,
        "label": label,
        "kind": kind,
        "started_at": started.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "timed_out": timed_out,
        "command": command,
    }
    (run_dir / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if returncode != 0:
        raise RuntimeError(f"{label} failed with return code {returncode}; see {run_dir / 'run.log'}")
    rows = load_jsonl(requests_path)
    validate_rows(rows, args, expected_token=expected_token if kind == "simulator" else None)
    telemetry_records = load_jsonl(telemetry) if kind == "simulator" else []
    if kind == "simulator" and not telemetry_records:
        raise RuntimeError("simulator produced no interception telemetry")
    summary = summarize(rows)
    summary["unique_modeled_stubs_executed"] = len(telemetry_records)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[pypto-validation] complete {label}: mean={summary['mean_ms']:.3f} ms", flush=True)
    return summary


def _validation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "plan": str(args.plan.resolve()),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "memory_profile": (
            None if args.memory_profile is None else str(args.memory_profile.resolve())
        ),
        "memory_profile_sha256": (
            None if args.memory_profile is None else hashlib.sha256(args.memory_profile.read_bytes()).hexdigest()
        ),
        "device": args.device,
        "prompt": args.prompt,
        "warmup_requests": args.warmup_requests,
        "requests": args.requests,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "sim_runtime_lib_dir": (
            None if args.sim_runtime_lib_dir is None else str(args.sim_runtime_lib_dir.resolve())
        ),
        "ring_dep_pool": args.ring_dep_pool,
        "ring_task_window": args.ring_task_window,
        "ring_heap": args.ring_heap,
    }


def main() -> int:
    args = parse_args()
    root = args.output.resolve()
    if root.exists() and not args.resume:
        print(f"error: refusing existing result root: {root}", file=sys.stderr)
        return 2
    if args.resume and not root.is_dir():
        print(f"error: resume root does not exist: {root}", file=sys.stderr)
        return 2
    if not args.model.is_dir() or not args.plan.is_file():
        print("error: model directory or simulation plan is missing", file=sys.stderr)
        return 2
    if args.memory_profile is not None and not args.memory_profile.is_file():
        print("error: memory profile is missing", file=sys.stderr)
        return 2
    if args.memory_profile is not None:
        try:
            memory_manifest = json.loads(args.memory_profile.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"error: invalid memory profile: {error}", file=sys.stderr)
            return 2
        if memory_manifest.get("schema_version") != 1 or not memory_manifest.get("strict_ready", False):
            print("error: acceptance requires a schema-v1 strict-ready memory profile", file=sys.stderr)
            return 2
    root.mkdir(parents=True, exist_ok=args.resume)
    config_path = root / "validation_config.json"
    expected_config = _validation_config(args)
    if config_path.is_file():
        observed_config = json.loads(config_path.read_text(encoding="utf-8"))
        if observed_config != expected_config:
            print("error: resume configuration does not match validation_config.json", file=sys.stderr)
            return 2
    else:
        config_path.write_text(
            json.dumps(expected_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.resume and (root / "failure.json").exists():
        suffix = 1
        while (root / f"failure.previous-{suffix}.json").exists():
            suffix += 1
        (root / "failure.json").rename(root / f"failure.previous-{suffix}.json")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    expected_token = int(plan["sampled_token_id"])

    env = os.environ.copy()
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    env["PTOAS_ROOT"] = str(args.ptoas_root.resolve())
    env["PYPTO_CACHE"] = "1"
    env.setdefault("PYPTO_CACHE_DIR", "/root/revati/pypto-serving/artifacts/pypto_cache")
    previous = env.get("PYTHONPATH")
    pypto_lib = str(args.pypto_lib.resolve())
    env["PYTHONPATH"] = pypto_lib if not previous else f"{pypto_lib}{os.pathsep}{previous}"

    try:
        runs = {
            "native-a": run_case("native-a", "native", args, root, env, expected_token),
            "sim-sleep": run_case("sim-sleep", "simulator", args, root, env, expected_token),
            "native-b": run_case("native-b", "native", args, root, env, expected_token),
        }
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        (root / "failure.json").write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}"}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"error: {error}", file=sys.stderr)
        return 3

    metric_errors: dict[str, float] = {}
    native_drift: dict[str, float] = {}
    for metric in ("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"):
        midpoint = (float(runs["native-a"][metric]) + float(runs["native-b"][metric])) / 2.0
        metric_errors[metric] = (float(runs["sim-sleep"][metric]) - midpoint) / midpoint * 100.0
        native_drift[metric] = (
            float(runs["native-b"][metric]) - float(runs["native-a"][metric])
        ) / float(runs["native-a"][metric]) * 100.0
    report = {
        "schema_version": 1,
        "workload": {
            "model": str(args.model.resolve()),
            "plan": str(args.plan.resolve()),
            "device": args.device,
            "prompt": args.prompt,
            "warmup_requests": args.warmup_requests,
            "requests": args.requests,
            "max_new_tokens": args.max_new_tokens,
        },
        "runs": runs,
        "signed_error_pct_against_native_midpoint": metric_errors,
        "native_a_to_b_drift_pct": native_drift,
        "all_primary_metrics_within_5pct": all(
            abs(metric_errors[name]) <= 5.0 for name in ("mean_ms", "p50_ms", "p95_ms")
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
