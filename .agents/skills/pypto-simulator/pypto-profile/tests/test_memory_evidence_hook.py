from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "scripts" / "install_memory_evidence_hook.py"
SPEC = importlib.util.spec_from_file_location("install_memory_evidence_hook", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_record_writes_workload_identity_and_exact_integer_fields(tmp_path: Path) -> None:
    output = tmp_path / "evidence.jsonl"
    with mock.patch.dict(
        os.environ,
        {
            "PYPTO_MEMORY_EVIDENCE_OUTPUT": str(output),
            "PYPTO_MEMORY_EVIDENCE_WORKLOAD_ID": "native-memory-C",
        },
        clear=True,
    ):
        MODULE._record({"event": "kv_sizing", "total_bytes": 65_450_000_000})
    record = json.loads(output.read_text(encoding="ascii"))
    assert record == {
        "schema_version": 1,
        "workload_id": "native-memory-C",
        "event": "kv_sizing",
        "total_bytes": 65_450_000_000,
    }


def test_native_guard_rejects_simulator() -> None:
    MODULE._require_native(SimpleNamespace(_compiled=SimpleNamespace(platform="a2a3")))
    try:
        MODULE._require_native(SimpleNamespace(_compiled=SimpleNamespace(platform="a2a3sim")))
    except MODULE.MemoryEvidenceError as error:
        assert "requires native a2a3" in str(error)
    else:
        raise AssertionError("expected simulator platform rejection")
