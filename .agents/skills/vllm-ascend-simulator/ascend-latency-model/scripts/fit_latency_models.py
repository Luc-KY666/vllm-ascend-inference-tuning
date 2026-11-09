#!/usr/bin/env python3
"""Fit grouped per-operator latency models from Ascend profile manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

COMMON = Path(__file__).resolve().parents[2] / "ascend-common"
sys.path.insert(0, str(COMMON))
from ascend_contract import (  # noqa: E402
    SCHEMA_VERSION,
    event_features,
    event_key,
    event_target,
    model_file_name,
    read_event_csv,
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    rows: list[dict[str, Any]] = []
    for kind in ("compute", "communication", "memory"):
        event = data.get("events", {}).get(kind, {})
        csv_path = Path(event.get("path", root / "events" / f"{kind}.csv"))
        if not csv_path.is_absolute():
            csv_path = root / csv_path
        if not csv_path.is_file():
            continue
        for row in read_event_csv(csv_path):
            for key, value in data.get("hardware", {}).items():
                row.setdefault(key, value)
            for key, value in data.get("model", {}).items():
                row.setdefault(key, value)
            for key, value in data.get("workload", {}).items():
                row.setdefault(key, value)
            row.setdefault("kind", kind)
            row["manifest"] = str(path)
            rows.append(row)
    return rows


def sanitize(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return safe[:80] or "group"


def write_collection_csv(
    collections_dir: Path, group_key: str, group_rows: list[dict[str, Any]]
) -> Path | None:
    """Write one per-operator collection table.

    Each row records the operator latency under a specific input shape,
    together with the numeric feature vector used by the random forest.
    The table is the source of truth consumed by ``fit_latency_models.py``;
    it keeps shape-to-latency measurements reviewable outside the joblib
    artifacts."""
    if not group_rows:
        return None
    kind = str(group_rows[0].get("kind", "unknown"))
    op_name = str(group_rows[0].get("op_name", "unknown"))
    digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:12]
    path = collections_dir / f"{kind}__{sanitize(op_name)}__{digest}.csv"
    collections_dir.mkdir(parents=True, exist_ok=True)
    feature_names = sorted({name for row in group_rows for name in event_features(row)})
    base_fields = [
        "op_name", "api_symbol", "kind", "hardware", "world_size", "rank",
        "phase", "collective", "topology", "dtype", "format",
        "input_shapes", "output_shapes", "input_dtypes", "output_dtypes",
    ]
    provenance_fields = ["duration_us", "count", "workload_id", "sample_id", "sample_group"]
    fields = base_fields + feature_names + provenance_fields
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in group_rows:
            out = {field: row.get(field, "") for field in fields}
            out.update({name: event_features(row).get(name, "") for name in feature_names})
            writer.writerow(out)
    return path


def metric_values(actual: list[float], predicted: list[float]) -> dict[str, float | int | None]:
    if not actual:
        return {"samples": 0, "mae_us": None, "rmse_us": None, "mape_pct": None,
                "p50_abs_error_us": None, "p95_abs_error_us": None}
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    nonzero = [abs(a - p) / abs(a) for a, p in zip(actual, predicted) if a]
    ordered = sorted(errors)
    percentile = lambda values, fraction: values[min(len(values) - 1, int(len(values) * fraction))]
    return {
        "samples": len(actual),
        "mae_us": sum(errors) / len(errors),
        "rmse_us": math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)),
        "mape_pct": (sum(nonzero) / len(nonzero) * 100) if nonzero else None,
        "p50_abs_error_us": percentile(ordered, 0.50),
        "p95_abs_error_us": percentile(ordered, 0.95),
    }


def split_rows(rows: list[dict[str, Any]], seed: int, folds: int) -> tuple[list[tuple[list[int], list[int]]], str, int]:
    """Build grouped k-fold splits, never mixing repeats from one workload."""
    from sklearn.model_selection import GroupKFold

    groups = [str(row.get("workload_id") or row.get("sample_id") or row.get("sample_group") or event_key(row)) for row in rows]
    independent = len(set(groups))
    if independent >= 2 and len(rows) >= 4:
        n_splits = min(max(2, folds), independent)
        splitter = GroupKFold(n_splits=n_splits)
        splits = [(list(train), list(test)) for train, test in splitter.split(rows, groups=groups)]
        return splits, "grouped_kfold", independent
    order = list(range(len(rows)))
    cut = max(1, int(len(order) * 0.75))
    return [(order[:cut], order[cut:] or order[:1])], "row_fallback", independent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--required-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--max-mape", type=float, default=15.0)
    parser.add_argument("--cv-folds", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for latency model fitting") from exc
    import joblib
    import numpy as np

    rows: list[dict[str, Any]] = []
    for path in args.manifest:
        rows.extend(load_manifest(path.resolve()))
    if not rows:
        raise SystemExit("no profile event rows found")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(event_key(row), []).append(row)
    required_rows: list[dict[str, Any]] = []
    for path in args.required_manifest or args.manifest:
        required_rows.extend(load_manifest(path.resolve()))
    required_keys = sorted({event_key(row) for row in required_rows})
    unresolved = sorted(set(required_keys) - set(grouped))

    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "models"
    model_dir.mkdir(exist_ok=True)
    collections_dir = args.output / "collections"
    collections_dir.mkdir(exist_ok=True)
    model_entries = []
    metric_rows = []
    collection_paths: list[str] = []
    feature_ranges: dict[str, dict[str, float]] = {}

    for group_key, group_rows in sorted(grouped.items()):
        collection_path = write_collection_csv(collections_dir, group_key, group_rows)
        if collection_path is not None:
            collection_paths.append(str(collection_path.relative_to(args.output)))

        feature_dicts = [event_features(row) for row in group_rows]
        feature_names = sorted({name for features in feature_dicts for name in features})
        matrix = np.asarray([[features.get(name, 0.0) for name in feature_names] for features in feature_dicts], dtype=float)
        targets = np.asarray([event_target(row) for row in group_rows], dtype=float)
        path = model_dir / model_file_name(group_key)
        splits, split_kind, independent_groups = split_rows(group_rows, args.seed, args.cv_folds)
        for name in feature_names:
            values = matrix[:, feature_names.index(name)]
            feature_ranges.setdefault(name, {"min": float(values.min()), "max": float(values.max())})
            feature_ranges[name]["min"] = min(feature_ranges[name]["min"], float(values.min()))
            feature_ranges[name]["max"] = max(feature_ranges[name]["max"], float(values.max()))

        low_confidence = len(group_rows) < args.min_samples or independent_groups < 2 or split_kind != "grouped_kfold"
        if low_confidence:
            strategy = "median"
            median = float(np.median(targets)) if len(targets) else 0.0
            actual = targets[splits[0][1]].tolist()
            predicted = [median for _ in actual]
            payload = {"strategy": strategy, "group_key": group_key, "median_us": median, "feature_names": feature_names,
                       "feature_ranges": {name: {"min": float(matrix[:, i].min()), "max": float(matrix[:, i].max())} for i, name in enumerate(feature_names)}}
        else:
            strategy = "random_forest"
            # Groups with a wide duration range (communication, mixed-size GEMMs)
            # are fitted on log scale so relative error dominates instead of
            # being swamped by the largest samples.
            use_log = bool(float(targets.min()) > 0 and float(targets.max()) / float(targets.min()) > 5.0)
            train_targets = np.log(targets) if use_log else targets
            actual_parts: list[float] = []
            predicted_parts: list[float] = []
            for train_indices, test_indices in splits:
                estimator = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                                  min_samples_split=args.min_samples_split, random_state=args.seed, n_jobs=-1)
                estimator.fit(matrix[train_indices], train_targets[train_indices])
                prediction = estimator.predict(matrix[test_indices])
                if use_log:
                    prediction = np.exp(prediction)
                actual_parts.extend(targets[test_indices].tolist())
                predicted_parts.extend(prediction.tolist())
            actual = actual_parts
            predicted = predicted_parts
            estimator = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                              min_samples_split=args.min_samples_split, random_state=args.seed, n_jobs=-1)
            estimator.fit(matrix, train_targets)
            payload = {"strategy": strategy, "group_key": group_key, "estimator": estimator,
                       "target_transform": "log" if use_log else "identity",
                       "feature_names": feature_names,
                       "feature_ranges": {name: {"min": float(matrix[:, i].min()), "max": float(matrix[:, i].max())} for i, name in enumerate(feature_names)}}
        joblib.dump(payload, path)
        metrics = metric_values(actual, predicted)
        metrics.update({"group_key": group_key, "strategy": strategy, "split": split_kind,
                        "independent_groups": independent_groups, "low_confidence": low_confidence})
        metric_rows.append(metrics)
        model_entries.append({
            "group_key": group_key, "model_file": str(path.relative_to(args.output)),
            "strategy": strategy, "samples": len(group_rows), "independent_groups": independent_groups,
            "low_confidence": low_confidence, "feature_names": feature_names,
            "target_transform": payload.get("target_transform", "identity"),
            "feature_ranges": payload["feature_ranges"],
            "observed_median_us": float(np.median(targets)) if len(targets) else 0.0,
            "metrics": metrics,
        })

    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["group_key", "strategy", "split", "samples", "independent_groups", "mae_us", "rmse_us",
                  "mape_pct", "p50_abs_error_us", "p95_abs_error_us", "low_confidence"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in metric_rows)

    high_error = [entry["group_key"] for entry in model_entries
                  if not entry["low_confidence"] and entry["metrics"].get("mape_pct") is not None
                  and entry["metrics"]["mape_pct"] > args.max_mape]
    kind_counts = Counter(str(row.get("kind", "unknown")) for row in rows)
    coverage = {
        "input_rows": len(rows), "groups": len(grouped), "kind_counts": dict(kind_counts),
        "low_confidence_groups": sum(bool(entry["low_confidence"]) for entry in model_entries),
        "unresolved_operator_groups": unresolved, "high_error_groups": high_error,
        "feature_ranges": feature_ranges,
    }
    complete = not unresolved and not high_error and coverage["low_confidence_groups"] == 0
    manifest = {
        "schema_version": SCHEMA_VERSION, "manifest_type": "ascend_latency_model",
        "backend": "scikit-learn.RandomForestRegressor",
        "hyperparameters": {"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                             "min_samples_split": args.min_samples_split, "random_state": args.seed,
                             "cv_folds": args.cv_folds},
        "min_samples": args.min_samples, "default_mape_threshold_pct": args.max_mape,
        "status": "complete" if complete else "incomplete", "strict_loadable": complete,
        "models": model_entries, "coverage": coverage,
        "collections": {"directory": "collections", "files": collection_paths},
    }
    (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "features.json").write_text(json.dumps({entry["group_key"]: entry["feature_names"] for entry in model_entries}, indent=2) + "\n", encoding="utf-8")
    lookup = {"schema_version": SCHEMA_VERSION, "lookup_type": "observed_median", "status": manifest["status"],
              "groups": {entry["group_key"]: {"duration_us": entry["observed_median_us"], "low_confidence": entry["low_confidence"]}
                         for entry in model_entries}}
    (args.output / "latency_lookup.json").write_text(json.dumps(lookup, separators=(",", ":")) + "\n", encoding="utf-8")
    (args.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output / "model_manifest.json"), "status": manifest["status"],
                      "groups": len(grouped), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
