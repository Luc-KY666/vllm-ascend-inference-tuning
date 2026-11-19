#!/usr/bin/env python3
"""Run a Python command with isolated PyPTO timewarp simulation enabled."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HOOK = Path(__file__).with_name("install_timewarp_hook.py").resolve()
TIMEKEEPER = Path(__file__).with_name("timekeeper.py").resolve()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--memory-profile", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--runtime-lib-dir", type=Path)
    parser.add_argument("--simpler-log-level", choices=("debug", "info", "timing", "warn", "error", "null"))
    parser.add_argument("--sim-aicpu-threads", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument(
        "--allow-deadline-overrun",
        action="store_true",
        help="diagnostic-only; acceptance rejects host control-plane wall-time overruns",
    )
    parser.add_argument("--allow-non-strict-memory-profile", action="store_true")
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if args.sim_aicpu_threads not in (0, *range(2, 17)):
        parser.error("--sim-aicpu-threads must be 0 or in [2, 16]")
    if args.sim_aicpu_threads > 4 and args.runtime_lib_dir is None:
        parser.error("--sim-aicpu-threads above 4 requires --runtime-lib-dir")
    return args, command


def _wait_for_server(path: Path, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"timekeeper exited during startup with {process.returncode}")
        if path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(0.2)
                    client.connect(str(path))
                return
            except OSError:
                pass
        time.sleep(0.02)
    raise TimeoutError(f"timekeeper socket did not become ready: {path}")


def _load_client(socket_path: Path, timeout_s: float):
    spec = importlib.util.spec_from_file_location("pypto_timewarp_launcher_clock", TIMEKEEPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load timekeeper client")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TimewarpClient(str(socket_path), timeout_s)


def _validate_output(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.exists():
        raise ValueError(f"refusing existing {label}: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_runtime_lib_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    relative = Path("a2a3/sim/tensormap_and_ringbuffer/libhost_runtime.so")
    for candidate in (resolved, resolved / "lib"):
        if (candidate / relative).is_file():
            return candidate
    raise ValueError(
        "custom simulator runtime must contain "
        f"{relative} directly or under lib/: {resolved}"
    )


def main() -> int:
    args, command = parse_args()
    plan = args.plan.resolve()
    if not plan.is_file():
        print(f"error: plan does not exist: {plan}", file=sys.stderr)
        return 2
    try:
        plan_data = json.loads(plan.read_text(encoding="utf-8"))
        if plan_data.get("execution_mode") != "timewarp" or plan_data.get("backend") != "a2a3sim":
            raise ValueError("plan is not an a2a3sim timewarp plan")
        memory_profile = args.memory_profile.resolve() if args.memory_profile else None
        if memory_profile is not None and not memory_profile.is_file():
            raise ValueError(f"memory profile does not exist: {memory_profile}")
        if args.allow_non_strict_memory_profile and memory_profile is None:
            raise ValueError("--allow-non-strict-memory-profile requires --memory-profile")
        runtime_lib_dir = _resolve_runtime_lib_dir(args.runtime_lib_dir)
        telemetry = _validate_output(args.telemetry, "telemetry file")
        trace = _validate_output(args.trace, "trace file")
        snapshot = _validate_output(args.snapshot, "snapshot file")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="pypto-timewarp-") as temporary:
        root = Path(temporary)
        socket_path = root / "timekeeper.sock"
        clock_path = root / "timekeeper.clock"
        bootstrap = root / "bootstrap"
        bootstrap.mkdir()
        (bootstrap / "sitecustomize.py").write_text(
            "import importlib.util\n"
            f"_spec = importlib.util.spec_from_file_location('pypto_timewarp_hook', {str(HOOK)!r})\n"
            "_module = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_module)\n"
            "_module.install()\n",
            encoding="ascii",
        )
        server_command = [
            sys.executable,
            str(TIMEKEEPER),
            "--socket",
            str(socket_path),
            "--clock",
            str(clock_path),
        ]
        if trace is not None:
            server_command.extend(("--trace", str(trace)))
        server = subprocess.Popen(
            server_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        returncode = 2
        try:
            _wait_for_server(socket_path, server, min(args.timeout_s, 10.0))
            env = os.environ.copy()
            env.update(
                {
                    "PYPTO_TIMEWARP_PLAN": str(plan),
                    "PYPTO_TIMEWARP_SOCKET": str(socket_path),
                    "PYPTO_TIMEWARP_CLOCK": str(clock_path),
                    "PYPTO_TIMEWARP_TIMEOUT_S": str(args.timeout_s),
                    "PYPTO_TIMEWARP_STRICT": "0" if args.no_strict else "1",
                    "PYPTO_TIMEWARP_AICPU_THREADS": str(args.sim_aicpu_threads),
                }
            )
            if memory_profile is not None:
                env["PYPTO_TIMEWARP_MEMORY_PROFILE"] = str(memory_profile)
            if args.allow_non_strict_memory_profile:
                env["PYPTO_TIMEWARP_MEMORY_ALLOW_NON_STRICT"] = "1"
            if telemetry is not None:
                env["PYPTO_TIMEWARP_TELEMETRY"] = str(telemetry)
            if runtime_lib_dir is not None:
                env["PYPTO_TIMEWARP_RUNTIME_LIB_DIR"] = str(runtime_lib_dir)
            if args.simpler_log_level is not None:
                env["PYPTO_TIMEWARP_SIMPLER_LOG_LEVEL"] = args.simpler_log_level.upper()
            previous = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(bootstrap) if not previous else f"{bootstrap}{os.pathsep}{previous}"
            returncode = subprocess.run(command, env=env, check=False).returncode
            state = _load_client(socket_path, args.timeout_s).snapshot()
            if snapshot is not None:
                snapshot.write_text(
                    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            if returncode == 0 and (
                state.get("pending_steps") or int(state.get("output_pending", 0)) != 0
            ):
                raise RuntimeError(
                    "timewarp ended with unresolved steps or observer outputs: "
                    f"steps={sorted(state.get('pending_steps', {}))}, "
                    f"output_pending={state.get('output_pending')}"
                )
            if (
                returncode == 0
                and int(state.get("deadline_overrun_count", 0)) > 0
                and not args.allow_deadline_overrun
            ):
                raise RuntimeError(
                    "timewarp host control plane exceeded modeled deadlines: "
                    f"count={state['deadline_overrun_count']}, "
                    f"total_ns={state['deadline_overrun_total_ns']}, "
                    f"max_ns={state['deadline_overrun_max_ns']}"
                )
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            returncode = 3
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)

    if telemetry is not None and returncode == 0:
        try:
            records = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()]
        except (OSError, ValueError) as error:
            print(f"error: invalid timewarp telemetry: {error}", file=sys.stderr)
            return 4
        if not records:
            print("error: no timewarp stub executed", file=sys.stderr)
            return 4
        unmodeled = [record for record in records if record.get("event") == "unmodeled_stub_execution"]
        if unmodeled:
            print(f"error: unmodeled stubs executed: {unmodeled}", file=sys.stderr)
            return 5
        print(f"[pypto-timewarp] executed {len(records)} unique modeled stubs", file=sys.stderr)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
