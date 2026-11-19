#!/usr/bin/env python3
"""Run a native a2a3 command with strict device-side delay child replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HOOK = Path(__file__).with_name("install_native_delay_hook.py").resolve()
VMEM_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "native_delay"
    / "virtual_memory_interposer.cpp"
)
_VMEM_MAGIC = 0x5059544F564D454D
_VMEM_VERSION = 1
_VMEM_RECORD = struct.Struct("<QQQQQQQqqq32s")
PROFILE_HOOK = (
    Path(__file__).resolve().parents[2]
    / "pypto-profile"
    / "scripts"
    / "install_serving_profile_hook.py"
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--memory-profile",
        type=Path,
        required=True,
        help="strict-ready exact memory profile for virtual weight/KV allocation",
    )
    parser.add_argument(
        "--memory-audit",
        type=Path,
        required=True,
        help="write one record per NPU virtual-address reservation",
    )
    parser.add_argument("--graph-timing", type=Path, help="diagnostic-only per-L3 wall timing")
    parser.add_argument("--profile-output", type=Path, help="diagnostic-only native-delay DFX root")
    parser.add_argument("--profile-program", action="append", default=[])
    parser.add_argument("--profile-level", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--npu-memory-telemetry",
        type=Path,
        help="diagnostic-only npu-smi process-memory time series; requires --npu-memory-device",
    )
    parser.add_argument("--npu-memory-device", type=int)
    parser.add_argument("--zero-duration", action="store_true", help="diagnostic reachability gate")
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an a2a3 command is required after --")
    if not any(
        command[index : index + 2] == ["--platform", "a2a3"]
        for index in range(max(0, len(command) - 1))
    ):
        parser.error("the child command must explicitly contain --platform a2a3")
    if bool(args.profile_output) != bool(args.profile_program):
        parser.error("--profile-output and at least one --profile-program are required together")
    if bool(args.npu_memory_telemetry) != (args.npu_memory_device is not None):
        parser.error("--npu-memory-telemetry and --npu-memory-device are required together")
    if args.npu_memory_device is not None and args.npu_memory_device < 0:
        parser.error("--npu-memory-device must be non-negative")
    return args, command


def _npu_process_memory(device: int) -> list[dict[str, object]]:
    result = subprocess.run(
        ["npu-smi", "info", "-t", "proc-mem", "-i", str(device)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"npu-smi proc-mem failed for device {device}")
    pattern = re.compile(
        r"Process id:(\d+)\s+Process name:(.*?)\s+Process memory\(MB\):(\d+)"
    )
    return [
        {"pid": int(pid), "name": name.strip(), "memory_mb": int(memory_mb)}
        for pid, name, memory_mb in pattern.findall(result.stdout)
    ]


def _run_with_memory_telemetry(
    command: list[str], env: dict[str, str], device: int, output: Path
) -> subprocess.CompletedProcess:
    existing = _npu_process_memory(device)
    if existing:
        raise RuntimeError(f"NPU {device} is already occupied: {existing}")
    process = subprocess.Popen(command, env=env)
    samples: list[dict[str, object]] = []
    monitor_error: str | None = None
    while process.poll() is None:
        try:
            processes = _npu_process_memory(device)
            samples.append(
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "total_process_memory_mb": sum(int(item["memory_mb"]) for item in processes),
                    "processes": processes,
                }
            )
        except RuntimeError as error:
            monitor_error = str(error)
        time.sleep(0.1)
    returncode = process.wait()
    per_pid: dict[int, dict[str, object]] = {}
    for sample in samples:
        for item in sample["processes"]:
            pid = int(item["pid"])
            entry = per_pid.setdefault(
                pid,
                {"pid": pid, "names": set(), "peak_memory_mb": 0},
            )
            entry["names"].add(str(item["name"]))
            entry["peak_memory_mb"] = max(
                int(entry["peak_memory_mb"]), int(item["memory_mb"])
            )
    process_peaks = [
        {
            "pid": pid,
            "names": sorted(entry["names"]),
            "peak_memory_mb": int(entry["peak_memory_mb"]),
        }
        for pid, entry in sorted(per_pid.items())
    ]
    contaminated = len(process_peaks) > 1
    peak = max((int(sample["total_process_memory_mb"]) for sample in samples), default=0)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "device": device,
                "sample_interval_ms": 100,
                "sample_count": len(samples),
                "peak_process_memory_mb": peak,
                "process_peaks": process_peaks,
                "contaminated_by_multiple_npu_pids": contaminated,
                "monitor_error": monitor_error,
                "samples": samples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if (monitor_error is not None or contaminated) and returncode == 0:
        returncode = 6
    return subprocess.CompletedProcess(command, returncode)


def _build_virtual_memory_interposer(output: Path) -> None:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        raise RuntimeError("C++ compiler is required for native virtual memory")
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        str(VMEM_SOURCE),
        "-ldl",
        "-pthread",
        "-o",
        str(output),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "virtual-memory interposer build failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def _memory_expectations(profile: dict) -> dict[str, int]:
    if profile.get("schema_version") != 1 or profile.get("strict_ready") is not True:
        raise RuntimeError("memory profile must be schema-v1 and strict-ready")
    footprint = profile.get("footprint_model")
    if not isinstance(footprint, dict) or footprint.get("strict_ready") is not True:
        raise RuntimeError("memory profile must contain a strict exact footprint model")
    expected: dict[str, int] = {}
    for model in profile.get("models", []):
        operation = str(model.get("operation", ""))
        if operation not in {"weight_materialize", "kv_materialize"}:
            continue
        points = list(model.get("points", []))
        if model.get("strict_ready") is not True or len(points) != 1:
            raise RuntimeError(f"memory profile operation {operation} is not exact")
        expected[operation] = int(points[0]["bytes"])
    if set(expected) != {"weight_materialize", "kv_materialize"}:
        raise RuntimeError("memory profile is missing exact weight/KV points")
    return expected


def _validate_memory_audit(path: Path, expected: dict[str, int]) -> dict[str, object]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid native virtual-memory audit: {error}") from error
    if not records:
        raise RuntimeError("native virtual-memory audit is empty")
    totals: dict[str, int] = {}
    counts: dict[str, int] = {}
    for record in records:
        if record.get("event") != "native_delay_virtual_allocation":
            raise RuntimeError("native virtual-memory audit has an unknown event")
        operation = str(record.get("operation", ""))
        logical_bytes = int(record.get("logical_bytes", 0))
        address = int(record.get("virtual_address", 0))
        if logical_bytes <= 0 or address <= 0:
            raise RuntimeError("native virtual-memory audit has an invalid allocation")
        if int(record.get("physical_backing_bytes", -1)) != 0:
            raise RuntimeError("native virtual-memory allocation committed physical backing")
        if int(record.get("reserved_address_bytes", 0)) < logical_bytes:
            raise RuntimeError("native virtual-memory reservation is smaller than its logical extent")
        totals[operation] = totals.get(operation, 0) + logical_bytes
        counts[operation] = counts.get(operation, 0) + 1
    for operation, byte_count in expected.items():
        if totals.get(operation) != byte_count:
            raise RuntimeError(
                f"native virtual-memory {operation} bytes mismatch: "
                f"{totals.get(operation, 0)} != {byte_count}"
            )
    if totals.get("kv_scratch", 0) <= 0:
        raise RuntimeError("native virtual-memory audit has no KV scratch allocation")
    return {
        "allocation_counts_by_operation": dict(sorted(counts.items())),
        "logical_bytes_by_operation": dict(sorted(totals.items())),
        "physical_backing_bytes": 0,
    }


def _initial_control_record() -> bytes:
    return _VMEM_RECORD.pack(
        _VMEM_MAGIC,
        _VMEM_VERSION,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        b"".ljust(32, b"\0"),
    )


def _validate_audit(
    path: Path,
    plan: dict,
    *,
    expected_plan_sha256: str | None = None,
    expected_zero_duration: bool | None = None,
) -> dict[str, int]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid native delay audit: {error}") from error
    expected: dict[str, set[int]] = {}
    for entry in plan.get("kernel_models", []):
        expected.setdefault(str(entry.get("program", "default")), set()).add(int(entry["func_id"]))
    observed: dict[str, set[int]] = {}
    for record in records:
        if record.get("event") != "native_delay_callable_replaced":
            continue
        program = str(record["program"])
        if expected_plan_sha256 is not None and record.get("plan_sha256") != expected_plan_sha256:
            raise RuntimeError(f"native delay audit has wrong plan SHA256 for {program}")
        if (
            expected_zero_duration is not None
            and record.get("zero_duration") is not expected_zero_duration
        ):
            raise RuntimeError(f"native delay audit has wrong zero-duration mode for {program}")
        replacements = record.get("replacements", [])
        non_modeled = [int(item["func_id"]) for item in replacements if item.get("modeled") is not True]
        if non_modeled:
            raise RuntimeError(
                f"native delay audit registered unmodeled replacements for {program}: {non_modeled}"
            )
        omitted = {int(func_id) for func_id in record.get("unregistered_unmodeled_func_ids", [])}
        modeled = {int(item["func_id"]) for item in replacements}
        if omitted & modeled:
            raise RuntimeError(
                f"native delay audit overlaps modeled and unregistered IDs for {program}: "
                f"{sorted(omitted & modeled)}"
            )
        observed.setdefault(program, set()).update(modeled)
    if observed != expected:
        missing = {
            program: sorted(funcs - observed.get(program, set()))
            for program, funcs in expected.items()
            if funcs - observed.get(program, set())
        }
        extra = {
            program: sorted(funcs - expected.get(program, set()))
            for program, funcs in observed.items()
            if funcs - expected.get(program, set())
        }
        raise RuntimeError(f"native delay audit mismatch: missing={missing}, extra={extra}")
    return {program: len(funcs) for program, funcs in observed.items()}


def main() -> int:
    args, command = parse_args()
    plan_path = args.plan.resolve()
    audit_path = args.audit.resolve()
    memory_profile_path = args.memory_profile.resolve()
    memory_audit_path = args.memory_audit.resolve()
    memory_telemetry_path = (
        args.npu_memory_telemetry.resolve() if args.npu_memory_telemetry else None
    )
    graph_timing_path = args.graph_timing.resolve() if args.graph_timing else None
    profile_output = args.profile_output.resolve() if args.profile_output else None
    if not plan_path.is_file():
        print(f"error: missing plan: {plan_path}", file=sys.stderr)
        return 2
    if not memory_profile_path.is_file():
        print(f"error: missing memory profile: {memory_profile_path}", file=sys.stderr)
        return 2
    outputs = [
        audit_path,
        audit_path.with_suffix(audit_path.suffix + ".run.json"),
        memory_audit_path,
    ]
    if graph_timing_path is not None:
        outputs.append(graph_timing_path)
    if memory_telemetry_path is not None:
        outputs.append(memory_telemetry_path)
    if profile_output is not None:
        outputs.append(profile_output)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        print(f"error: refusing existing output: {existing}", file=sys.stderr)
        return 2
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        memory_profile = json.loads(memory_profile_path.read_text(encoding="utf-8"))
        memory_expectations = _memory_expectations(memory_profile)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: invalid plan or memory profile: {error}", file=sys.stderr)
        return 2
    if (
        plan.get("schema_version") != 1
        or plan.get("execution_mode") != "sleep"
        or plan.get("backend") != "a2a3"
        or not plan.get("coverage", {}).get("runtime_stub_ready", False)
    ):
        print("error: plan must be schema-v1 strict-ready a2a3 sleep plan", file=sys.stderr)
        return 2

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    memory_audit_path.parent.mkdir(parents=True, exist_ok=True)
    if memory_telemetry_path is not None:
        memory_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    if graph_timing_path is not None:
        graph_timing_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix="pypto-native-delay-bootstrap-") as temporary:
        bootstrap = Path(temporary)
        control_path = bootstrap / "virtual-memory-control.bin"
        interposer_path = bootstrap / "libpypto_native_virtual_memory.so"
        control_path.write_bytes(_initial_control_record())
        try:
            _build_virtual_memory_interposer(interposer_path)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 3
        (bootstrap / "sitecustomize.py").write_text(
            "import importlib.util\n"
            f"_spec = importlib.util.spec_from_file_location('pypto_native_delay_hook', {str(HOOK)!r})\n"
            "_module = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_module)\n"
            "_module.install()\n"
            + (
                "import importlib.util as _profile_importlib_util\n"
                f"_profile_spec = _profile_importlib_util.spec_from_file_location('pypto_native_delay_profile_hook', {str(PROFILE_HOOK)!r})\n"
                "_profile_module = _profile_importlib_util.module_from_spec(_profile_spec)\n"
                "_profile_spec.loader.exec_module(_profile_module)\n"
                "_profile_module.install()\n"
                if profile_output is not None
                else ""
            ),
            encoding="ascii",
        )
        env = os.environ.copy()
        env["PYPTO_NATIVE_DELAY_ENABLE"] = "1"
        env["PYPTO_NATIVE_DELAY_PLAN"] = str(plan_path)
        env["PYPTO_NATIVE_DELAY_AUDIT"] = str(audit_path)
        env["PYPTO_NATIVE_DELAY_ZERO"] = "1" if args.zero_duration else "0"
        env["PYPTO_NATIVE_DELAY_MEMORY_PROFILE"] = str(memory_profile_path)
        env["PYPTO_NATIVE_DELAY_MEMORY_AUDIT"] = str(memory_audit_path)
        env["PYPTO_NATIVE_DELAY_VMEM_CONTROL"] = str(control_path)
        previous_preload = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = (
            str(interposer_path)
            if not previous_preload
            else f"{interposer_path} {previous_preload}"
        )
        if graph_timing_path is not None:
            env["PYPTO_NATIVE_DELAY_GRAPH_TIMING"] = str(graph_timing_path)
        if profile_output is not None:
            env["PYPTO_PROFILE_ARTIFACT_ROOT"] = str(profile_output)
            env["PYPTO_PROFILE_SWIMLANE_LEVEL"] = str(args.profile_level)
            env["PYPTO_PROFILE_PROGRAMS"] = ",".join(args.profile_program)
        previous = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(bootstrap) if not previous else f"{bootstrap}{os.pathsep}{previous}"
        try:
            result = (
                subprocess.run(command, env=env, check=False)
                if memory_telemetry_path is None
                else _run_with_memory_telemetry(
                    command,
                    env,
                    int(args.npu_memory_device),
                    memory_telemetry_path,
                )
            )
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            result = subprocess.CompletedProcess(command, 6)

    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    record = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "returncode": result.returncode,
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "memory_profile": str(memory_profile_path),
        "memory_profile_sha256": hashlib.sha256(memory_profile_path.read_bytes()).hexdigest(),
        "memory_audit": str(memory_audit_path),
        "npu_memory_telemetry": (
            None if memory_telemetry_path is None else str(memory_telemetry_path)
        ),
        "zero_duration": args.zero_duration,
        "graph_timing": None if graph_timing_path is None else str(graph_timing_path),
        "profile_output": None if profile_output is None else str(profile_output),
        "profile_programs": args.profile_program,
    }
    if result.returncode == 0:
        try:
            record["replaced_func_ids_by_program"] = _validate_audit(
                audit_path,
                plan,
                expected_plan_sha256=plan_sha256,
                expected_zero_duration=args.zero_duration,
            )
            record["virtual_memory"] = _validate_memory_audit(
                memory_audit_path,
                memory_expectations,
            )
            if graph_timing_path is not None:
                graph_records = [
                    json.loads(line)
                    for line in graph_timing_path.read_text(encoding="ascii").splitlines()
                ]
                if not graph_records or any(
                    item.get("event") != "native_delay_graph_wall" for item in graph_records
                ):
                    raise RuntimeError("native delay graph timing is empty or invalid")
                record["graph_timing_record_count"] = len(graph_records)
        except (OSError, ValueError, RuntimeError) as error:
            record["audit_error"] = str(error)
            result = subprocess.CompletedProcess(command, 4)
            print(f"error: {error}", file=sys.stderr)
    audit_path.with_suffix(audit_path.suffix + ".run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
