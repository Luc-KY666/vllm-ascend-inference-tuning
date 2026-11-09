#!/usr/bin/env python3
"""Validate model bundle coverage and the configured MAPE threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--max-mape", type=float)
    parser.add_argument("--require-complete", action="store_true",
                        help="Reject incomplete or low-confidence model bundles")
    args = parser.parse_args()
    manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    threshold = args.max_mape if args.max_mape is not None else manifest.get("default_mape_threshold_pct", 15.0)
    failures = []
    for entry in manifest.get("models", []):
        mape = entry.get("metrics", {}).get("mape_pct")
        if not entry.get("low_confidence") and mape is not None and mape > threshold:
            failures.append({"group_key": entry["group_key"], "mape_pct": mape})
        if args.require_complete and entry.get("low_confidence"):
            failures.append({"group_key": entry["group_key"], "reason": "low_confidence"})
    coverage = manifest.get("coverage", {})
    if args.require_complete and manifest.get("status") != "complete":
        failures.append({"reason": "model_status", "status": manifest.get("status", "unknown")})
    for group_key in coverage.get("unresolved_operator_groups", []):
        failures.append({"group_key": group_key, "reason": "unresolved_operator_group"})
    result = {"passed": not failures, "threshold_mape_pct": threshold,
              "failures": failures, "coverage": coverage,
              "status": manifest.get("status", "unknown"),
              "strict_loadable": bool(manifest.get("strict_loadable", False))}
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
