#!/usr/bin/env python3
"""Aggregate profile, latency-model, and emulator coverage into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--emulator-manifest", type=Path)
    parser.add_argument("--coverage-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profile = load(args.profile_manifest)
    model = load(args.model_manifest)
    emulator = load(args.emulator_manifest)
    generated_coverage = load(args.coverage_report)
    profile_events = profile.get("events", {})
    model_coverage = model.get("coverage", {})
    unsupported = emulator.get("unsupported", generated_coverage.get("unsupported", []))
    unmodeled = emulator.get("unmodeled_acl_nn_symbols", generated_coverage.get("unmodeled_acl_nn_symbols", []))
    low_confidence = emulator.get(
        "low_confidence_model_symbols",
        generated_coverage.get("low_confidence_model_symbols", []),
    )
    report = {
        "schema_version": 1,
        "manifest_type": "ascend_coverage",
        "profile": {
            "manifest": str(args.profile_manifest.resolve()),
            "events": {
                kind: {"count": data.get("count", 0), "duration_us": data.get("duration_us", 0)}
                for kind, data in profile_events.items()
                if isinstance(data, dict)
            },
            "warnings": profile.get("coverage", {}).get("warnings", []),
        },
        "latency_model": {
            "manifest": str(args.model_manifest.resolve()),
            "coverage": model_coverage,
            "threshold_mape_pct": model.get("default_mape_threshold_pct", 15.0),
        },
        "emulator": {
            "manifest": str(args.emulator_manifest.resolve()) if args.emulator_manifest else "",
            "unsupported": unsupported,
            "unmodeled_acl_nn_symbols": unmodeled,
            "low_confidence_model_symbols": low_confidence,
        },
        "strict_ready": not unsupported and not unmodeled and not low_confidence,
        "notes": [
            "low-confidence models require an independent workload sweep before strict emulation",
            "strict_ready does not replace an Ascend TP2/full-request integration test",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "strict_ready": report["strict_ready"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
