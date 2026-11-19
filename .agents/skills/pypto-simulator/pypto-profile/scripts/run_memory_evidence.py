#!/usr/bin/env python3
"""Run one native command with exact model memory evidence collection enabled."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HOOK = Path(__file__).with_name("install_memory_evidence_hook.py").resolve()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a native pypto-serving command is required after --")
    if not args.workload_id.strip():
        parser.error("--workload-id must be non-empty")
    return args, command


def main() -> int:
    args, command = parse_args()
    output = args.output.resolve()
    if output.exists():
        print(f"error: refusing existing output: {output}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix="pypto-memory-evidence-") as temporary:
        bootstrap = Path(temporary)
        (bootstrap / "sitecustomize.py").write_text(
            "import importlib.util\n"
            f"_spec = importlib.util.spec_from_file_location('pypto_memory_evidence', {str(HOOK)!r})\n"
            "_module = importlib.util.module_from_spec(_spec)\n"
            "_spec.loader.exec_module(_module)\n"
            "_module.install()\n",
            encoding="ascii",
        )
        env = os.environ.copy()
        env["PYPTO_MEMORY_EVIDENCE_OUTPUT"] = str(output)
        env["PYPTO_MEMORY_EVIDENCE_WORKLOAD_ID"] = args.workload_id.strip()
        previous = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(bootstrap) if not previous else f"{bootstrap}{os.pathsep}{previous}"
        result = subprocess.run(command, env=env, check=False)

    run_record = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "workload_id": args.workload_id.strip(),
        "output": str(output),
        "command": command,
        "returncode": result.returncode,
    }
    output.with_suffix(output.suffix + ".run.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.returncode == 0:
        try:
            records = [json.loads(line) for line in output.read_text(encoding="ascii").splitlines()]
        except (OSError, ValueError) as error:
            print(f"error: invalid memory evidence: {error}", file=sys.stderr)
            return 4
        expected = {"weight_materialize", "kv_sizing", "kv_materialize"}
        observed = {str(record.get("event")) for record in records}
        if not expected <= observed:
            print(f"error: incomplete memory evidence: missing {sorted(expected - observed)}", file=sys.stderr)
            return 4
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
