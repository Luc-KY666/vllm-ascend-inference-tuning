from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pypto-emulate" / "scripts" / "analyze_native_delay_profile.py"


def _load():
    spec = importlib.util.spec_from_file_location("native_delay_analysis", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_uses_validation_percentile_and_reports_signed_error() -> None:
    module = _load()

    summary = module._summary([(10.0, 11.0), (20.0, 22.0), (40.0, 44.0)])

    assert summary["sample_count"] == 3
    assert summary["mean_absolute_error_us"] == 7 / 3
    assert summary["p95_absolute_error_us"] == 4.0
    assert summary["max_absolute_error_us"] == 4.0
    assert summary["mean_absolute_percent_error"] == 10.0
    assert summary["mean_signed_error_us"] == 7 / 3


def test_capture_specs_reject_duplicates() -> None:
    module = _load()

    try:
        module._capture_specs(["decode=/tmp/a", "decode=/tmp/b"])
    except ValueError as error:
        assert "duplicate capture program" in str(error)
    else:
        raise AssertionError("expected duplicate program rejection")
