#!/usr/bin/env python3
"""Run a Python command with the PyPTO sleep hook installed in every process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = Path(__file__).with_name("install_sleep_hook.py").resolve()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--memory-profile",
        type=Path,
        help="strict-ready measured memory profile for virtual weight/KV materialization",
    )
    parser.add_argument("--telemetry", type=Path, help="write and validate first-execution stub telemetry")
    parser.add_argument("--graph-timing", type=Path, help="diagnostic-only per-graph wall-time JSONL")
    parser.add_argument(
        "--sim-aicpu-threads",
        type=int,
        default=0,
        help=(
            "diagnostic-only simulator AICPU topology override; "
            "0 preserves the pinned program configuration and is required for acceptance"
        ),
    )
    parser.add_argument(
        "--runtime-lib-dir",
        type=Path,
        help="optional build/lib containing a custom a2a3sim runtime",
    )
    parser.add_argument(
        "--simpler-log-level",
        choices=("debug", "info", "timing", "warn", "error", "null"),
        help="optional simpler/C++ diagnostic log threshold",
    )
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument(
        "--allow-non-strict-memory-profile",
        action="store_true",
        help="diagnostic-only; acceptance must use a strict-ready memory profile",
    )
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.sim_aicpu_threads not in (0, *range(2, 17)):
        parser.error("--sim-aicpu-threads must be 0 or in [2, 16]")
    if args.sim_aicpu_threads > 4 and args.runtime_lib_dir is None:
        parser.error("--sim-aicpu-threads above 4 requires --runtime-lib-dir")
    return args, command


def main() -> int:
    args, command = parse_args()
    plan = args.plan.resolve()
    if not plan.is_file():
        print(f"error: plan does not exist: {plan}", file=sys.stderr)
        return 2
    telemetry = args.telemetry.resolve() if args.telemetry else None
    graph_timing = args.graph_timing.resolve() if args.graph_timing else None
    memory_profile = args.memory_profile.resolve() if args.memory_profile else None
    if memory_profile is not None and not memory_profile.is_file():
        print(f"error: memory profile does not exist: {memory_profile}", file=sys.stderr)
        return 2
    if args.allow_non_strict_memory_profile and memory_profile is None:
        print("error: --allow-non-strict-memory-profile requires --memory-profile", file=sys.stderr)
        return 2
    for output_path, label in ((telemetry, "telemetry"), (graph_timing, "graph timing")):
        if output_path is not None:
            if output_path.exists():
                print(f"error: refusing existing {label} file: {output_path}", file=sys.stderr)
                return 2
            output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pypto-sleep-bootstrap-") as temporary:
        bootstrap = Path(temporary)
        (bootstrap / "sitecustomize.py").write_text(
            "import importlib.util\n"
            f"_spec = importlib.util.spec_from_file_location('pypto_sleep_hook', {str(HOOK)!r})\n"
            "_module = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_module)\n"
            "_module.install()\n",
            encoding="ascii",
        )
        env = os.environ.copy()
        env["PYPTO_SLEEP_PLAN"] = str(plan)
        env["PYPTO_SLEEP_STRICT"] = "0" if args.no_strict else "1"
        env["PYPTO_SLEEP_AICPU_THREADS"] = str(args.sim_aicpu_threads)
        if memory_profile is not None:
            env["PYPTO_SLEEP_MEMORY_PROFILE"] = str(memory_profile)
        if args.allow_non_strict_memory_profile:
            env["PYPTO_SLEEP_MEMORY_ALLOW_NON_STRICT"] = "1"
        if args.runtime_lib_dir is not None:
            env["PYPTO_SLEEP_RUNTIME_LIB_DIR"] = str(args.runtime_lib_dir.resolve())
        if args.simpler_log_level is not None:
            env["PYPTO_SLEEP_SIMPLER_LOG_LEVEL"] = args.simpler_log_level.upper()
        if telemetry is not None:
            env["PYPTO_SLEEP_TELEMETRY"] = str(telemetry)
        if graph_timing is not None:
            env["PYPTO_SLEEP_GRAPH_TIMING"] = str(graph_timing)
        previous = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(bootstrap) if not previous else f"{bootstrap}{os.pathsep}{previous}"
        returncode = subprocess.run(command, env=env, check=False).returncode

    if telemetry is not None and returncode == 0:
        try:
            records = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()]
        except (OSError, ValueError) as error:
            print(f"error: invalid sleep telemetry: {error}", file=sys.stderr)
            return 4
        if not records:
            print("error: no sleep stub executed", file=sys.stderr)
            return 4
        unmodeled = [record for record in records if record.get("event") == "unmodeled_stub_execution"]
        if unmodeled:
            print(f"error: unmodeled stubs executed: {unmodeled}", file=sys.stderr)
            return 5
        print(f"[pypto-sleep] executed {len(records)} unique modeled stubs", file=sys.stderr)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
