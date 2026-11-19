from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "pypto-emulate" / "scripts" / "install_native_delay_hook.py"
LAUNCHER = ROOT / "pypto-emulate" / "scripts" / "run_native_delay_simulator.py"
VMEM_SOURCE = ROOT / "pypto-emulate" / "native_delay" / "virtual_memory_interposer.cpp"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_device_stub_uses_full_sys_counter_duration_and_control_output() -> None:
    hook = _load(HOOK, "native_delay_hook_test")
    entry = {
        "duration_us": 10.0,
        "control_outputs": [
            {
                "strategy": "constant_token",
                "arg_index": 1,
                "token_id": 2813,
                "padding_value": 0,
            }
        ],
    }

    source = hook._stub_source(entry, zero_duration=False)

    assert "get_sys_cnt() - started_at < 500ULL" in source
    assert "reinterpret_cast<__gm__ Tensor *>(args[1])" in source
    assert "= 2813;" in source
    assert "pipe_barrier(PIPE_ALL)" in source
    assert "sleep" not in source


def test_device_stub_selects_block_rank_and_aiv_lane_duration() -> None:
    hook = _load(HOOK, "native_delay_block_hook_test")
    entry = {
        "duration_us": 10.0,
        "block_duration_us_by_subslot": {
            "AIV0": [2.0, 4.0],
            "AIV1": [3.0, 5.0],
        },
        "control_outputs": [],
    }

    source = hook._stub_source(entry, zero_duration=False)

    assert "get_block_idx(args)" in source
    assert "get_sub_block_id(args) == 0" in source
    assert "if (block_idx == 0) target_ticks = 100ULL" in source
    assert "if (block_idx == 1) target_ticks = 200ULL" in source
    assert "if (block_idx == 0) target_ticks = 150ULL" in source
    assert "if (block_idx == 1) target_ticks = 250ULL" in source
    assert "get_sys_cnt() - started_at < target_ticks" in source


def test_zero_duration_removes_only_wait() -> None:
    hook = _load(HOOK, "native_delay_zero_hook_test")
    entry = {
        "duration_us": 10.0,
        "control_outputs": [
            {
                "strategy": "constant_token",
                "arg_index": 1,
                "token_id": 7,
                "padding_value": 0,
            }
        ],
    }

    source = hook._stub_source(entry, zero_duration=True)

    assert "get_sys_cnt" not in source
    assert "= 7;" in source


def test_virtual_memory_interposer_reserves_va_without_physical_allocation() -> None:
    source = VMEM_SOURCE.read_text(encoding="ascii")

    assert "aclrtReserveMemAddress" in source
    assert "aclrtReleaseMemAddress" in source
    assert "aclrtMallocPhysical" not in source
    assert 'extern "C" int rtMalloc' in source
    assert 'extern "C" int rtFree' in source


def test_virtual_memory_controller_requires_consumed_one_shot_token(tmp_path: Path) -> None:
    hook = _load(HOOK, "native_delay_virtual_memory_test")
    control = tmp_path / "control.bin"
    audit = tmp_path / "memory.jsonl"
    control.write_bytes(
        hook._VMEM_RECORD.pack(
            hook._VMEM_MAGIC,
            hook._VMEM_VERSION,
            hook._VMEM_IDLE,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            b"".ljust(32, b"\0"),
        )
    )
    controller = hook._VirtualMemoryController(control, audit)

    def consume() -> str:
        record = controller._decode(control.read_bytes())
        assert record["state"] == hook._VMEM_PENDING
        assert record["expected_bytes"] == 4096
        record.update(
            state=hook._VMEM_CONSUMED,
            virtual_address=0x400000,
            reserved_bytes=2 * 1024 * 1024,
            consumer_pid=os.getpid() + 1,
            error_code=0,
        )
        control.write_bytes(controller._encode(record))
        return "allocated"

    assert controller.allocate("weight_materialize", 4096, consume) == "allocated"
    event = json.loads(audit.read_text(encoding="ascii"))
    assert event["logical_bytes"] == 4096
    assert event["physical_backing_bytes"] == 0
    assert event["virtual_address"] == 0x400000
    assert controller._decode(control.read_bytes())["state"] == hook._VMEM_IDLE

    with pytest.raises(hook.NativeDelayHookError, match="token was not consumed"):
        controller.allocate("kv_materialize", 8192, lambda: "real allocation")


def test_native_virtual_memory_preserves_exact_logical_kv_capacity() -> None:
    hook = _load(HOOK, "native_delay_exact_memory_test")
    total = 65_452_113_920
    non_kv = 39_577_563_136
    baseline = 39_155_604_908
    kv_bytes = 19_314_769_920
    profile = hook._MemoryProfile(
        {
            "schema_version": 1,
            "strict_ready": True,
            "footprint_model": {
                "strict_ready": True,
                "native_total_bytes": total,
                "driver_visible_non_kv_bytes": non_kv,
            },
            "models": [
                {
                    "operation": "weight_materialize",
                    "strict_ready": True,
                    "points": [{"bytes": 29_540_610_048, "duration_us": 1.0}],
                },
                {
                    "operation": "kv_materialize",
                    "strict_ready": True,
                    "points": [{"bytes": kv_bytes, "duration_us": 1.0}],
                },
            ],
        }
    )

    free_before, logical_total = hook._exact_memory_info(
        profile,
        actual_free=total - 10_000_000_000,
        actual_total=total,
        committed=baseline,
        baseline=baseline,
    )
    kv_budget = int(logical_total * 0.90 - (logical_total - free_before))
    assert kv_budget // 20_971_520 == 921

    free_after, _ = hook._exact_memory_info(
        profile,
        actual_free=total - 10_000_000_000,
        actual_total=total,
        committed=baseline + kv_bytes,
        baseline=baseline,
    )
    assert free_after == total - non_kv - kv_bytes


def test_memory_audit_requires_exact_weight_and_kv_totals(tmp_path: Path) -> None:
    launcher = _load(LAUNCHER, "native_delay_memory_audit_test")
    audit = tmp_path / "memory.jsonl"
    records = [
        {
            "event": "native_delay_virtual_allocation",
            "operation": operation,
            "logical_bytes": byte_count,
            "virtual_address": address,
            "reserved_address_bytes": byte_count + 4096,
            "physical_backing_bytes": 0,
        }
        for operation, byte_count, address in (
            ("weight_materialize", 100, 0x1000),
            ("kv_scratch", 20, 0x2000),
            ("kv_materialize", 200, 0x3000),
        )
    ]
    audit.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="ascii")

    summary = launcher._validate_memory_audit(
        audit,
        {"weight_materialize": 100, "kv_materialize": 200},
    )

    assert summary["physical_backing_bytes"] == 0
    assert summary["logical_bytes_by_operation"] == {
        "kv_materialize": 200,
        "kv_scratch": 20,
        "weight_materialize": 100,
    }

    records[2]["physical_backing_bytes"] = 1
    audit.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="ascii")
    with pytest.raises(RuntimeError, match="committed physical backing"):
        launcher._validate_memory_audit(
            audit,
            {"weight_materialize": 100, "kv_materialize": 200},
        )


def test_npu_process_memory_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load(LAUNCHER, "native_delay_memory_parser_test")
    output = """
    Process id:123 Process name:python3           Process memory(MB):4567
    Process id:890 Process name:VLLM::Worker_TP   Process memory(MB):12000
    """

    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: launcher.subprocess.CompletedProcess(
            args[0], 0, stdout=output, stderr=""
        ),
    )

    assert launcher._npu_process_memory(2) == [
        {"pid": 123, "name": "python3", "memory_mb": 4567},
        {"pid": 890, "name": "VLLM::Worker_TP", "memory_mb": 12000},
    ]


def test_audit_accepts_unregistered_unmodeled_func_ids(tmp_path: Path) -> None:
    launcher = _load(LAUNCHER, "native_delay_launcher_test")
    audit = tmp_path / "audit.jsonl"
    records = [
        {
            "event": "native_delay_callable_replaced",
            "program": "decode_fwd",
            "plan_sha256": "plan-sha",
            "zero_duration": True,
            "replacements": [{"func_id": 3, "modeled": True}],
            "unregistered_unmodeled_func_ids": [9],
        }
    ]
    audit.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    plan = {"kernel_models": [{"program": "decode_fwd", "func_id": 3}]}

    assert launcher._validate_audit(
        audit,
        plan,
        expected_plan_sha256="plan-sha",
        expected_zero_duration=True,
    ) == {"decode_fwd": 1}
