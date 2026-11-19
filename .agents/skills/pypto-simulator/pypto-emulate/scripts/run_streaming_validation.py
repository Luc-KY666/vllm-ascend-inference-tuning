#!/usr/bin/env python3
"""Run a same-NPU native/simulator/native streaming-latency bracket."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WORKLOAD = HERE / "run_streaming_benchmark.py"
SIMULATOR = HERE / "run_native_delay_simulator.py"
METRICS = ("ttft_ms", "tpot_ms", "e2el_ms")
STATS = ("mean", "p50", "p95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--memory-profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--idle-stability-seconds", type=float, default=60.0)
    parser.add_argument("--monitor-seconds", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--ring-dep-pool", type=int, default=262144)
    parser.add_argument("--ring-task-window", type=int, default=262144)
    parser.add_argument("--ring-heap", type=int, default=1073741824)
    parser.add_argument(
        "--ptoas-root", type=Path, default=Path("/root/revati/.venvs/pypto-simulator")
    )
    parser.add_argument(
        "--pypto-serving", type=Path, default=Path("/root/revati/pypto-serving")
    )
    args = parser.parse_args()
    positive = (
        args.poll_seconds,
        args.idle_stability_seconds,
        args.monitor_seconds,
        args.max_model_len,
        args.max_num_seqs,
        args.max_num_batched_tokens,
        args.ring_dep_pool,
        args.ring_task_window,
        args.ring_heap,
    )
    if args.device < 0 or any(value <= 0 for value in positive):
        parser.error("device must be non-negative and sizes/intervals must be positive")
    return args


def npu_processes(device: int) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["npu-smi", "info", "-t", "proc-mem", "-i", str(device)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"npu-smi proc-mem failed for NPU {device}: {result.stderr.strip()}")
    pattern = re.compile(
        r"Process id:(\d+)\s+Process name:(.*?)\s+Process memory\(MB\):(\d+)"
    )
    return [
        {"pid": int(pid), "name": name.strip(), "memory_mb": int(memory_mb)}
        for pid, name, memory_mb in pattern.findall(result.stdout)
    ]


def wait_for_device(device: int, poll_seconds: float, stability_seconds: float) -> None:
    idle_since = None
    while True:
        processes = npu_processes(device)
        now = time.monotonic()
        if not processes:
            if idle_since is None:
                idle_since = now
            idle_for = now - idle_since
            if idle_for >= stability_seconds:
                return
            print(
                f"[stream-validation] NPU {device} idle for {idle_for:.1f}s; "
                f"requiring {stability_seconds:.1f}s stable idle",
                flush=True,
            )
            time.sleep(min(poll_seconds, stability_seconds - idle_for))
            continue
        idle_since = None
        print(
            f"[stream-validation] NPU {device} occupied by "
            f"{[(item['pid'], item['name']) for item in processes]}; waiting",
            flush=True,
        )
        time.sleep(poll_seconds)


def serving_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        str(args.ptoas_root / "bin" / "python"),
        str(WORKLOAD),
        "--manifest",
        str(args.manifest.resolve()),
        "--output",
        str(output),
        "--",
        "--model",
        str(args.model.resolve()),
        "--platform",
        "a2a3",
        "--device",
        str(args.device),
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
        '{"max_new_tokens":2,"temperature":0,"top_p":1,"ignore_eos":true,"stream":true}',
        "--prompt",
        "benchmark",
    ]


def monitored_run(
    command: list[str], env: dict[str, str], device: int, log_path: Path, interval: float
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    seen_pids: set[int] = set()
    contaminated = False
    sample_count = 0
    peak_processes = 0
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while process.poll() is None:
            processes = npu_processes(device)
            sample_count += 1
            peak_processes = max(peak_processes, len(processes))
            seen_pids.update(int(item["pid"]) for item in processes)
            if len(processes) > 1:
                contaminated = True
            time.sleep(interval)
        returncode = process.wait()
    return {
        "started_at": started.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "monitor_sample_count": sample_count,
        "monitor_interval_seconds": interval,
        "observed_npu_pids": sorted(seen_pids),
        "peak_concurrent_npu_processes": peak_processes,
        "contaminated_by_multiple_npu_pids": contaminated,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    measured = [row for row in rows if row["phase"] == "measure"]
    if not measured:
        raise ValueError("benchmark has no measured requests")
    summary: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in measured]
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"benchmark has invalid {metric} values")
        summary[metric] = {
            "mean": statistics.mean(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }
    return summary


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row["phase"] == "measure"]
    case_ids = sorted({str(row["case"]) for row in measured})
    return {
        "request_count": len(measured),
        "overall": metric_summary(measured),
        "cases": {
            case_id: metric_summary([row for row in measured if row["case"] == case_id])
            for case_id in case_ids
        },
    }


def compare_summaries(
    native_a: dict[str, Any], simulator: dict[str, Any], native_b: dict[str, Any]
) -> dict[str, Any]:
    def compare_scope(a: dict[str, Any], sim: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for metric in METRICS:
            result[metric] = {}
            for stat in STATS:
                av = float(a[metric][stat])
                sv = float(sim[metric][stat])
                bv = float(b[metric][stat])
                midpoint = (av + bv) / 2.0
                result[metric][stat] = {
                    "native_a_ms": av,
                    "simulator_ms": sv,
                    "native_b_ms": bv,
                    "native_midpoint_ms": midpoint,
                    "simulator_error_pct": (sv - midpoint) / midpoint * 100.0,
                    "native_drift_pct": (bv - av) / av * 100.0,
                }
        return result

    return {
        "overall": compare_scope(native_a["overall"], simulator["overall"], native_b["overall"]),
        "cases": {
            case_id: compare_scope(
                native_a["cases"][case_id],
                simulator["cases"][case_id],
                native_b["cases"][case_id],
            )
            for case_id in native_a["cases"]
        },
    }


def all_within_threshold(comparison: dict[str, Any], threshold: float = 5.0) -> bool:
    scopes = [comparison["overall"], *comparison["cases"].values()]
    return all(
        abs(float(scope[metric][stat]["simulator_error_pct"])) <= threshold
        for scope in scopes
        for metric in METRICS
        for stat in STATS
    )


def validate_rows(
    rows: list[dict[str, Any]], manifest: dict[str, Any], sampled_token: int | None
) -> None:
    expected = sum(int(case["requests"]) for case in manifest["cases"]) * (
        int(manifest["warmup_rounds"]) + int(manifest["measure_rounds"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(rows)}")
    for row in rows:
        if row["token_count"] != row["output_tokens"] or row["finish_reason"] != "length":
            raise RuntimeError(f"invalid request completion: {row['request_id']}")
        if len(row["token_event_ms"]) != row["output_tokens"]:
            raise RuntimeError(f"incomplete token timestamps: {row['request_id']}")
        if sampled_token is not None and any(
            int(token) != sampled_token for token in row["token_ids"]
        ):
            raise RuntimeError(f"wrong simulator control token: {row['request_id']}")


def run_case(
    label: str,
    simulator: bool,
    args: argparse.Namespace,
    root: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
    sampled_token: int,
) -> dict[str, Any]:
    attempt = 0
    while True:
        wait_for_device(
            args.device, args.poll_seconds, args.idle_stability_seconds
        )
        attempt += 1
        run_dir = root / f"{label}.attempt-{attempt}"
        run_dir.mkdir()
        requests = run_dir / "requests.jsonl"
        child = serving_command(args, requests)
        if simulator:
            command = [
                str(args.ptoas_root / "bin" / "python"),
                str(SIMULATOR),
                "--plan",
                str(args.plan.resolve()),
                "--audit",
                str(run_dir / "audit.jsonl"),
                "--memory-profile",
                str(args.memory_profile.resolve()),
                "--memory-audit",
                str(run_dir / "memory-audit.jsonl"),
                "--",
                *child,
            ]
        else:
            command = child
        print(f"[stream-validation] start {label} attempt {attempt}", flush=True)
        record = monitored_run(
            command, env, args.device, run_dir / "run.log", args.monitor_seconds
        )
        record["command"] = command
        (run_dir / "run.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if record["contaminated_by_multiple_npu_pids"]:
            print(
                f"[stream-validation] discard contaminated {label} attempt {attempt}; "
                "waiting for an uncontaminated retry",
                flush=True,
            )
            continue
        if record["returncode"] != 0:
            raise RuntimeError(
                f"{label} failed with return code {record['returncode']}; see {run_dir / 'run.log'}"
            )
        rows = load_rows(requests)
        validate_rows(rows, manifest, sampled_token if simulator else None)
        summary = summarize(rows)
        summary["run"] = record
        summary["attempt"] = attempt
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        final_dir = root / label
        run_dir.rename(final_dir)
        print(f"[stream-validation] complete {label}", flush=True)
        return summary


def main() -> int:
    args = parse_args()
    required = (args.model, args.plan, args.memory_profile, args.manifest)
    ptoas_python = args.ptoas_root / "bin" / "python"
    if (
        not args.model.is_dir()
        or any(not path.is_file() for path in required[1:])
        or not ptoas_python.is_file()
    ):
        print(
            "error: model, plan, memory profile, manifest, or PTOAS Python is missing",
            file=sys.stderr,
        )
        return 2
    if args.output.exists():
        print(f"error: refusing existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        sampled_token = int(plan["sampled_token_id"])
    except (OSError, ValueError, KeyError) as error:
        print(f"error: invalid input artifact: {error}", file=sys.stderr)
        return 2
    root = args.output.resolve()
    root.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "plan": str(args.plan.resolve()),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        "memory_profile": str(args.memory_profile.resolve()),
        "memory_profile_sha256": hashlib.sha256(args.memory_profile.read_bytes()).hexdigest(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "device": args.device,
        "idle_stability_seconds": args.idle_stability_seconds,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    (root / "validation_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    env["PTOAS_ROOT"] = str(args.ptoas_root.resolve())
    env["PYPTO_CACHE"] = "1"
    env.setdefault("PYPTO_CACHE_DIR", str((args.pypto_serving / "artifacts/pypto_cache").resolve()))
    roots = [str(args.pypto_serving.resolve()), str((args.pypto_serving / "pypto-lib").resolve())]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)

    try:
        summaries = {
            "native-a": run_case(
                "native-a", False, args, root, env, manifest, sampled_token
            ),
            "simulator": run_case(
                "simulator", True, args, root, env, manifest, sampled_token
            ),
            "native-b": run_case(
                "native-b", False, args, root, env, manifest, sampled_token
            ),
        }
        comparison = compare_summaries(
            summaries["native-a"], summaries["simulator"], summaries["native-b"]
        )
        report = {
            "schema_version": 1,
            "acceptance_threshold_pct": 5.0,
            "accepted": all_within_threshold(comparison),
            "config": config,
            "summaries": summaries,
            "comparison": comparison,
        }
        (root / "bracket_summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["accepted"] else 4
    except (OSError, ValueError, RuntimeError) as error:
        (root / "failure.json").write_text(
            json.dumps({"error": f"{type(error).__name__}: {error}"}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
