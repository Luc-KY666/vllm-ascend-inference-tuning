from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "export_memory_profile.py"
SPEC = importlib.util.spec_from_file_location("export_memory_profile", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _log(path: Path, *, pages: int, weight_start: str, weight_end: str, kv_start: str, kv_end: str) -> Path:
    kv_gb = pages * MODULE.QWEN3_14B_KV_PAGE_BYTES / 1_000_000_000
    path.write_text(
        f"{weight_start} INFO | [init_kv_cache] uploading static tensors ...\n"
        f"{weight_end} INFO | [init_kv_cache] profile warmup (scratch 1 page) ...\n"
        f"{kv_start} INFO | KV cache sizing (vLLM-style): total=65.45 GB, utilization=0.90, "
        "peak_non_kv=39.80 GB, simpler_committed=39.20 GB, kv_budget=19.11 GB\n"
        f"{kv_start} INFO | [init_kv_cache] num_pages={pages}, allocating ...\n"
        f"{kv_end} INFO | [init_kv_cache] allocated {pages} pages: {kv_gb:.2f} GB KV cache\n"
        f"{kv_end} INFO |   weights (estimated):       27.98 GB\n",
        encoding="ascii",
    )
    return path


def _evidence(path: Path, *, workload_id: str, pages: int, reserved_bytes: int) -> Path:
    page_bytes = MODULE.QWEN3_14B_KV_PAGE_BYTES
    total_bytes = 65_452_113_920
    simpler_committed_bytes = 39_155_604_908
    driver_visible_non_kv_bytes = simpler_committed_bytes + reserved_bytes
    records = [
        {
            "schema_version": 1,
            "event": "weight_materialize",
            "workload_id": workload_id,
            "bytes": 29_540_610_048,
        },
        {
            "schema_version": 1,
            "event": "kv_sizing",
            "workload_id": workload_id,
            "runtime_reserved_bytes": reserved_bytes,
            "free_bytes": total_bytes - driver_visible_non_kv_bytes,
            "total_bytes": total_bytes,
            "driver_visible_non_kv_bytes": driver_visible_non_kv_bytes,
            "simpler_committed_bytes": simpler_committed_bytes,
        },
        {
            "schema_version": 1,
            "event": "kv_materialize",
            "workload_id": workload_id,
            "page_count": pages,
            "page_bytes": page_bytes,
            "bytes": pages * page_bytes,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="ascii")
    return path


def test_parse_log_exports_exact_kv_and_rounded_weight(tmp_path: Path) -> None:
    path = _log(
        tmp_path / "run.log",
        pages=911,
        weight_start="01:00:00.000",
        weight_end="01:00:38.500",
        kv_start="01:01:00.000",
        kv_end="01:01:32.000",
    )
    records = {record["operation"]: record for record in MODULE.parse_log(path)}
    assert records["weight_materialize"]["duration_us"] == 38_500_000
    assert records["weight_materialize"]["bytes_evidence"] == "rounded_serving_log"
    assert records["kv_materialize"]["duration_us"] == 32_000_000
    assert records["kv_materialize"]["bytes"] == 911 * MODULE.QWEN3_14B_KV_PAGE_BYTES
    assert records["kv_materialize"]["bytes_evidence"] == "exact_page_count"
    assert records["runtime_reserved_memory"]["bytes"] == 600_000_000


def test_manifest_requires_exact_bytes_for_strict_mode(tmp_path: Path) -> None:
    logs = [
        _log(
            tmp_path / f"run-{index}.log",
            pages=910,
            weight_start="01:00:00.000",
            weight_end=f"01:00:{38 + index:02d}.000",
            kv_start="01:01:00.000",
            kv_end=f"01:01:{31 + index:02d}.000",
        )
        for index in range(3)
    ]
    manifest = MODULE.build_manifest(logs, min_workloads=3)
    models = {model["operation"]: model for model in manifest["models"]}
    assert models["kv_materialize"]["strict_ready"] is True
    assert models["kv_materialize"]["min_bytes"] == models["kv_materialize"]["max_bytes"]
    assert models["weight_materialize"]["strict_ready"] is False
    assert manifest["strict_ready"] is False

    exact_manifest = MODULE.build_manifest(logs, min_workloads=3, exact_weight_bytes=29_540_610_048)
    exact_models = {model["operation"]: model for model in exact_manifest["models"]}
    assert exact_models["weight_materialize"]["strict_ready"] is True
    assert exact_models["weight_materialize"]["points"][0]["bytes"] == 29_540_610_048
    assert exact_manifest["footprint_model"]["strict_ready"] is False
    assert exact_manifest["strict_ready"] is False

    evidence = [
        _evidence(
            tmp_path / f"evidence-{index}.jsonl",
            workload_id=f"native-memory-{index}",
            pages=910,
            reserved_bytes=500_000_000 + index,
        )
        for index in range(3)
    ]
    strict_manifest = MODULE.build_manifest(logs, min_workloads=3, evidence_paths=evidence)
    strict_models = {model["operation"]: model for model in strict_manifest["models"]}
    assert strict_models["weight_materialize"]["strict_ready"] is True
    assert strict_models["kv_materialize"]["strict_ready"] is True
    assert strict_manifest["footprint_model"]["strict_ready"] is True
    assert strict_manifest["footprint_model"]["native_total_bytes"] == 65_452_113_920
    assert (
        strict_manifest["footprint_model"]["native_simpler_committed_bytes"]
        == 39_155_604_908
    )
    assert strict_manifest["strict_ready"] is True


def test_parse_log_rejects_changed_kv_page_count(tmp_path: Path) -> None:
    path = _log(
        tmp_path / "run.log",
        pages=911,
        weight_start="01:00:00.000",
        weight_end="01:00:38.000",
        kv_start="01:01:00.000",
        kv_end="01:01:32.000",
    )
    text = path.read_text(encoding="ascii").replace("allocated 911 pages", "allocated 910 pages")
    path.write_text(text, encoding="ascii")
    try:
        MODULE.parse_log(path)
    except ValueError as error:
        assert "page count changed" in str(error)
    else:
        raise AssertionError("expected a page-count mismatch")
