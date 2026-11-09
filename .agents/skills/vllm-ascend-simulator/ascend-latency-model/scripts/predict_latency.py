#!/usr/bin/env python3
"""Predict one event latency from a Ascend model bundle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

COMMON = Path(__file__).resolve().parents[2] / "ascend-common"
sys.path.insert(0, str(COMMON))
from ascend_contract import event_features, event_key  # noqa: E402


def load_bundle(path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest, path.parent


def predict(manifest: dict[str, Any], root: Path, event: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    import joblib

    if strict and not manifest.get("strict_loadable", False):
        raise RuntimeError("latency model is incomplete or contains low-confidence groups")

    group_key = event_key(event)
    entries = {entry["group_key"]: entry for entry in manifest.get("models", [])}
    entry = entries.get(group_key)
    if entry is None:
        raise KeyError(f"no latency model for group: {group_key}")
    if strict and entry.get("low_confidence"):
        raise RuntimeError(f"latency model group is low-confidence: {group_key}")
    payload = joblib.load(root / entry["model_file"])
    features = event_features(event)
    vector = [[features.get(name, 0.0) for name in payload["feature_names"]]]
    if payload["strategy"] == "median":
        value = float(payload["median_us"])
    else:
        value = float(payload["estimator"].predict(vector)[0])
        if payload.get("target_transform") == "log":
            value = math.exp(value)
    return {"group_key": group_key, "duration_us": max(0.0, value),
            "strategy": payload["strategy"], "low_confidence": entry["low_confidence"]}


def predict_lookup(lookup: dict[str, Any], event: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    if strict and lookup.get("status") != "complete":
        raise RuntimeError("latency lookup is incomplete")
    group_key = event_key(event)
    entry = lookup.get("groups", {}).get(group_key)
    if entry is None:
        raise KeyError(f"no latency lookup for group: {group_key}")
    if strict and entry.get("low_confidence"):
        raise RuntimeError(f"latency lookup group is low-confidence: {group_key}")
    return {"group_key": group_key, "duration_us": max(0.0, float(entry["duration_us"])),
            "strategy": "observed_median", "low_confidence": bool(entry.get("low_confidence", True))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--event", type=Path, help="JSON event object")
    parser.add_argument("--event-json", help="Inline JSON event object")
    parser.add_argument("--lookup", type=Path, help="Compact JSON lookup emitted by fit_latency_models.py")
    parser.add_argument("--strict", action="store_true", help="Reject incomplete or low-confidence predictions")
    args = parser.parse_args()
    if bool(args.event) == bool(args.event_json):
        parser.error("provide exactly one of --event or --event-json")
    event = json.loads(args.event.read_text(encoding="utf-8")) if args.event else json.loads(args.event_json)
    try:
        if args.lookup:
            lookup = json.loads(args.lookup.read_text(encoding="utf-8"))
            result = predict_lookup(lookup, event, strict=args.strict)
        else:
            manifest, root = load_bundle(args.model_manifest.resolve())
            result = predict(manifest, root, event, strict=args.strict)
    except (KeyError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
