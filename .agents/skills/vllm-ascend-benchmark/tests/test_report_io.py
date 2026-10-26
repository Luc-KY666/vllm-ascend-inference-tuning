from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_report_io", ROOT / "scripts" / "report_io.py")
assert SPEC and SPEC.loader
report_io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_io)


class ReportIoTests(unittest.TestCase):
    def test_atomic_write_report_writes_parseable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "benchmark_report.json"
            report_io.atomic_write_report(target, {"status": "running"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"status": "running"})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_failed_replace_removes_temporary_file_and_preserves_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "benchmark_report.json"
            target.write_text('{"status":"old"}\n', encoding="utf-8")
            with mock.patch.object(report_io.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    report_io.atomic_write_report(target, {"status": "new"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"status": "old"})
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_sha256_and_tail_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text("0123456789", encoding="utf-8")
            self.assertEqual(report_io.tail_text(path, 4), "6789")
            self.assertEqual(
                report_io.sha256_file(path),
                "84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882",
            )


if __name__ == "__main__":
    unittest.main()
