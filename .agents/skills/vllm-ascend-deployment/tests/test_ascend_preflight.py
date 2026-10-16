from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import ascend_preflight  # noqa: E402


class FakeNpu:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 2


class FakeTorch:
    npu = FakeNpu()


class AscendPreflightTests(unittest.TestCase):
    def importer(self, name: str) -> object:
        return {
            "vllm": SimpleNamespace(__file__="/env/vllm.py", __version__="1"),
            "torch": FakeTorch(),
            "torch_npu": SimpleNamespace(__file__="/env/torch_npu.py"),
            "vllm_ascend": SimpleNamespace(
                __file__="/env/vllm_ascend.py",
                __version__="module-version-must-not-be-read",
            ),
        }[name]

    def runner(self, argv: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    def test_dependency_probe_requires_ascend_runtime(self) -> None:
        result = ascend_preflight._dependency_check(importer=self.importer)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["ascend_runtime"]["device_count"], 2)

    def test_module_version_comes_from_distribution_metadata(self) -> None:
        record, _ = ascend_preflight._module_record(
            "vllm_ascend",
            importer=self.importer,
            version_getter=lambda name: {
                "vllm-ascend": "metadata-version",
            }[name],
        )
        self.assertEqual(record["version"], "metadata-version")

    def test_vllm_help_probe_does_not_start_a_server(self) -> None:
        result = ascend_preflight._vllm_serve_check(
            command="vllm",
            timeout=5,
            which=lambda value: "/env/bin/vllm",
            runner=self.runner,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["command"], ["/env/bin/vllm", "serve", "--help=all"])

    def test_runtime_combines_atb_dependencies_and_vllm(self) -> None:
        expected = {
            key: ascend_preflight.os.environ.get(key)
            for key in ascend_preflight.PROTECTED_ENV
        }
        code = ascend_preflight.check_runtime(
            json.dumps(expected),
            5,
            importer=self.importer,
            loader=lambda name: SimpleNamespace(_name=name),
            which=lambda value: "/env/bin/vllm",
            runner=self.runner,
        )
        self.assertEqual(code, 0)

    def test_main_does_not_require_provider_tool_paths(self) -> None:
        original = ascend_preflight.run_runtime_checks
        try:
            ascend_preflight.run_runtime_checks = lambda *args, **kwargs: {
                "schema_version": ascend_preflight.SCHEMA_VERSION,
                "stage": "ascend_vllm_preflight",
                "status": "pass",
            }
            with self.subTest("no provider paths"):
                self.assertEqual(ascend_preflight.main([]), 0)
        finally:
            ascend_preflight.run_runtime_checks = original

    def test_missing_vllm_executable_is_a_failure(self) -> None:
        result = ascend_preflight._vllm_serve_check(
            command="vllm",
            timeout=5,
            which=lambda value: None,
            runner=self.runner,
        )
        self.assertEqual(result["failure_type"], "vllm_executable_missing")


if __name__ == "__main__":
    unittest.main()
