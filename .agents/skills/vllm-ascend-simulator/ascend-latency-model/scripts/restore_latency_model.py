#!/usr/bin/env python3
"""Restore the latency model bundle from disk and answer predictions without refitting.

This is the "fast restore" counterpart of fit_latency_models.py: it loads the
per-operator joblib artifacts (random forest or median fallback) recorded in
model_manifest.json and re-validates them against the source collection CSVs.
No scikit-learn fitting happens here — only ``joblib.load`` + ``predict`` — so
a saved bundle can be re-imported in seconds.

Usage:
  python restore_latency_model.py --model-manifest /tmp/ascend-model/model_manifest.json \
      [--output /tmp/ascend-model/restored] [--event-json '{"kind": ...}']

Prints one JSON summary; writes restored_metrics.csv and a compact
restored_manifest.json into --output when given.
"""

from __future__ import annotations

import argparse
import csv
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


def read_collection_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def normalize_group_key(key: str) -> str:
    """Canonicalize numeric group-key fields so `1`, `1.0`, and `"1"` match.

    Collection CSVs round-trip integer fields through the CSV writer as floats
    (world_size=1 -> "1.0"), while the manifest records the original integer.
    World size is the 5th `|`-separated field of the group key."""
    parts = key.split("|")
    if len(parts) > 4:
        try:
            number = float(parts[4])
            if number == int(number):
                parts[4] = str(int(number))
        except (TypeError, ValueError):
            pass
    return "|".join(parts)


def predict_rows(payload: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    import numpy as np

    names = payload["feature_names"]
    matrix = np.asarray(
        [[float(row.get(name, 0.0) or 0.0) for name in names] for row in rows], dtype=float
    )
    if payload["strategy"] == "median":
        values = [float(payload["median_us"]) for _ in rows]
    else:
        values = payload["estimator"].predict(matrix).tolist()
        if payload.get("target_transform") == "log":
            values = [math.exp(v) for v in values]
    actuals = [float(row["duration_us"]) for row in rows]
    return actuals, [max(0.0, v) for v in values]


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    if not actual:
        return {"samples": 0, "mae_us": None, "rmse_us": None, "mape_pct": None}
    errs = [abs(a - p) for a, p in zip(actual, predicted)]
    nonzero = [abs(a - p) / abs(a) for a, p in zip(actual, predicted) if a]
    return {
        "samples": len(actual),
        "mae_us": sum(errs) / len(errs),
        "rmse_us": math.sqrt(sum(e * e for e in errs) / len(errs)),
        "mape_pct": (sum(nonzero) / len(nonzero) * 100) if nonzero else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Directory for restored_metrics.csv / restored_manifest.json")
    parser.add_argument("--event-json", help="Optional single event JSON to predict right after restore")
    args = parser.parse_args()

    manifest, root = load_bundle(args.model_manifest.resolve())
    entries = manifest.get("models", [])
    if not entries:
        raise SystemExit("model manifest has no models")

    # Index source rows by canonicalized group key so we can validate without refitting.
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for rel in manifest.get("collections", {}).get("files", []):
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            for row in read_collection_csv(path):
                rows_by_group.setdefault(normalize_group_key(event_key(row)), []).append(row)

    summary: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda e: e["group_key"]):
        import joblib

        payload = joblib.load(root / entry["model_file"])
        rows = rows_by_group.get(normalize_group_key(entry["group_key"]), [])
        actual, predicted = predict_rows(payload, rows) if rows else ([], [])
        m = metrics(actual, predicted)
        m.update({
            "group_key": entry["group_key"], "strategy": payload["strategy"],
            "low_confidence": entry.get("low_confidence", True),
        })
        summary.append(m)

    out: dict[str, Any] = {"schema_version": manifest.get("schema_version", 1), "groups": summary}

    if args.event_json:
        event = json.loads(args.event_json)
        entry = next((e for e in entries if normalize_group_key(e["group_key"]) == normalize_group_key(event_key(event))), None)
        if entry is None:
            print(json.dumps({"error": f"no model for {event_key(event)}"}), file=sys.stderr)
            return 1
        import joblib

        payload = joblib.load(root / entry["model_file"])
        names = payload["feature_names"]
        vector = [[float(event_features(event).get(n, 0.0) or 0.0) for n in names]]
        if payload["strategy"] == "median":
            value = float(payload["median_us"])
        else:
            value = float(payload["estimator"].predict(vector)[0])
            if payload.get("target_transform") == "log":
                value = math.exp(value)
        out["single_prediction"] = {
            "group_key": entry["group_key"], "duration_us": max(0.0, value),
            "strategy": payload["strategy"],
        }

    print(json.dumps(out, indent=2))

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "restored_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["group_key", "strategy", "samples", "mae_us", "rmse_us", "mape_pct", "low_confidence"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
        with (args.output / "restored_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(out, stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
