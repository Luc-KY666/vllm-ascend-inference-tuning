#!/usr/bin/env python3
"""Normalize an exporter-produced PyPTO task profile without device access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

COMMON = Path(__file__).resolve().parents[2] / "pypto-common"
sys.path.insert(0, str(COMMON))

from pypto_contract import (
    ContractError,
    normalize_profile_manifest,
    read_json,
    should_intercept,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = normalize_profile_manifest(read_json(args.input))
    except (OSError, ValueError, ContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    measured = sum(bool(task["duration_samples_us"]) for task in manifest["tasks"])
    coverage = {
        "schema_version": manifest["schema_version"],
        "task_count": len(manifest["tasks"]),
        "measured_task_count": measured,
        "unmeasured_task_count": len(manifest["tasks"]) - measured,
        "intercepted_task_count": sum(
            should_intercept(task) for task in manifest["tasks"]
        ),
        "npu_accessed": False,
    }
    write_json(args.output / "profile_manifest.json", manifest)
    write_json(args.output / "coverage.json", coverage)
    print(args.output / "profile_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
