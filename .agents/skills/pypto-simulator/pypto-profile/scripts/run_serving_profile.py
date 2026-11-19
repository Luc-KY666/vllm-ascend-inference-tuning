#!/usr/bin/env python3
"""Run pypto-serving with native per-task DFX collection enabled."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HOOK = Path(__file__).with_name("install_serving_profile_hook.py").resolve()


def _capture_dispatch(value: str) -> tuple[str, int]:
    program, separator, ordinal = value.partition("=")
    try:
        parsed = int(ordinal)
    except ValueError as error:
        raise argparse.ArgumentTypeError("capture dispatch must be PROGRAM=POSITIVE_ORDINAL") from error
    if not separator or not program.strip() or parsed < 1:
        raise argparse.ArgumentTypeError("capture dispatch must be PROGRAM=POSITIVE_ORDINAL")
    return program.strip(), parsed


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument(
        "--npu-device",
        type=int,
        help="monitor this NPU for run-time multi-PID contamination",
    )
    parser.add_argument("--level", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--program",
        action="append",
        default=[],
        help="profile only this L3 program; repeat for more programs",
    )
    parser.add_argument(
        "--capture-dispatch",
        action="append",
        default=[],
        type=_capture_dispatch,
        metavar="PROGRAM=ORDINAL",
        help="profile only one 1-based invocation of a program; repeat for more programs",
    )
    parser.add_argument(
        "--bypass-program",
        action="append",
        default=[],
        help="skip this L3 program during a decode-only diagnostic run",
    )
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    capture_names = [name for name, _ in args.capture_dispatch]
    if len(capture_names) != len(set(capture_names)):
        parser.error("--capture-dispatch program names must be unique")
    if args.program and args.capture_dispatch:
        parser.error("use either --program or --capture-dispatch, not both")
    return args, command


def _npu_processes(device: int) -> list[int]:
    result = subprocess.run(
        ["npu-smi", "info", "-t", "proc-mem", "-i", str(device)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"npu-smi failed for device {device}")
    pattern = re.compile(r"Process id:(\d+)\s+Process name:.*?Process memory\(MB\):\d+")
    return [int(pid) for pid in pattern.findall(result.stdout)]


def main() -> int:
    args, command = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.allow_existing:
        print(f"error: refusing non-empty output directory: {output}")
        return 2
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    observed_pids: set[int] = set()
    monitor_errors: list[str] = []
    contaminated = False
    with tempfile.TemporaryDirectory(prefix="pypto-profile-bootstrap-") as temporary:
        bootstrap = Path(temporary)
        (bootstrap / "sitecustomize.py").write_text(
            "import importlib.util\n"
            f"_spec = importlib.util.spec_from_file_location('pypto_profile_hook', {str(HOOK)!r})\n"
            "_module = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_module)\n"
            "_module.install()\n",
            encoding="ascii",
        )
        env = os.environ.copy()
        env["PYPTO_PROFILE_ARTIFACT_ROOT"] = str(output)
        env["PYPTO_PROFILE_SWIMLANE_LEVEL"] = str(args.level)
        env["PYPTO_PROFILE_PROGRAMS"] = ",".join(args.program)
        env["PYPTO_PROFILE_CAPTURE_DISPATCHES"] = json.dumps(dict(args.capture_dispatch), sort_keys=True)
        env["PYPTO_PROFILE_BYPASS_PROGRAMS"] = ",".join(args.bypass_program)
        # Large outlined graphs can exceed the normal 45 s watchdog during
        # dep-gen. Preserve explicit operator choices, but make profiling usable.
        env.setdefault("SIMPLER_OP_EXECUTE_TIMEOUT_US", "300000000")
        env.setdefault("SIMPLER_STREAM_SYNC_TIMEOUT_MS", "310000")
        previous = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(bootstrap) if not previous else f"{bootstrap}{os.pathsep}{previous}"
        process = subprocess.Popen(command, env=env)
        while True:
            if args.npu_device is not None:
                try:
                    pids = _npu_processes(args.npu_device)
                    observed_pids.update(pids)
                    if len(pids) > 1:
                        contaminated = True
                except (OSError, RuntimeError) as error:
                    monitor_errors.append(str(error))
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(1.0)

    child_returncode = returncode
    if contaminated or monitor_errors:
        returncode = returncode if returncode else 3

    run_record = {
        "schema_version": 1,
        "collector": "pypto-serving-native-dfx",
        "started_at": started.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "child_returncode": child_returncode,
        "cwd": str(Path.cwd()),
        "command": command,
        "host": platform.node(),
        "selected_programs": sorted(args.program),
        "capture_dispatches": dict(args.capture_dispatch),
        "bypassed_programs": sorted(args.bypass_program),
        "swimlane_level": args.level,
        "timeouts": {
            "SIMPLER_OP_EXECUTE_TIMEOUT_US": env["SIMPLER_OP_EXECUTE_TIMEOUT_US"],
            "SIMPLER_STREAM_SYNC_TIMEOUT_MS": env["SIMPLER_STREAM_SYNC_TIMEOUT_MS"],
        },
        "npu_monitor": {
            "device": args.npu_device,
            "observed_pids": sorted(observed_pids),
            "contaminated": contaminated,
            "errors": monitor_errors,
        },
    }
    (output / "collection_run.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
