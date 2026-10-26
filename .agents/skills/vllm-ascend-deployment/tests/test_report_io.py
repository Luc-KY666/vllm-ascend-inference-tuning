from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_io", ROOT / "scripts" / "report_io.py")
assert SPEC and SPEC.loader
report_io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_io)


class ReportIoTests(unittest.TestCase):
    def test_atomic_write_report_writes_parseable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "deployment_report.json"
            report_io.atomic_write_report(target, {"status": "running"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"status": "running"})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_tail_text_supports_stream_and_path(self) -> None:
        self.assertEqual(report_io.tail_text(StringIO("abcdef"), 3), "def")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            path.write_text("abcdef", encoding="utf-8")
            self.assertEqual(report_io.tail_text(path, 4), "cdef")

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            path.write_bytes(b"abc")
            self.assertEqual(
                report_io.sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_failed_replace_preserves_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "deployment_report.json"
            target.write_text('{"status": "running"}\n', encoding="utf-8")
            with patch.object(report_io.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    report_io.atomic_write_report(target, {"status": "failed"})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["status"], "running")
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
