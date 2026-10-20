from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import probe_environment  # noqa: E402


class FakeNpu:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 2


class FakeTorch:
    npu = FakeNpu()


class ProbeEnvironmentTests(unittest.TestCase):
    def importer(self, name: str) -> object:
        return {
            "vllm": SimpleNamespace(__file__="/env/vllm.py"),
            "torch": FakeTorch(),
            "torch_npu": SimpleNamespace(__file__="/env/torch_npu.py"),
            "vllm_ascend": SimpleNamespace(__file__="/env/vllm_ascend.py"),
        }[name]

    def version_getter(self, name: str) -> str:
        return {"vllm": "1", "torch": "2", "torch-npu": "3", "vllm-ascend": "4"}[name]

    def runner(self, argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[:2] == ["/env/bin/vllm", "serve"]:
            return SimpleNamespace(returncode=0, stdout="serve help", stderr="")
        if argv[:3] == ["/env/bin/vllm", "bench", "serve"]:
            return SimpleNamespace(returncode=0, stdout="bench help", stderr="")
        return SimpleNamespace(returncode=0, stdout="probe ok", stderr="")

    def which(self, name: str) -> str | None:
        return {
            "vllm": "/env/bin/vllm",
            "npu-smi": "/env/bin/npu-smi",
            "ss": "/env/bin/ss",
            "netstat": None,
            "lsof": None,
        }.get(name)

    def loader(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(_name=name)

    def expected_env(self) -> dict[str, str | None]:
        return {
            key: probe_environment.os.environ.get(key)
            for key in probe_environment.PROTECTED_ENV
        }

    def test_port_probe_selects_first_verified_tool(self) -> None:
        result = probe_environment._port_probe_tools(
            timeout=5,
            which=self.which,
            runner=self.runner,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["name"], "ss")
        self.assertEqual(result["argv"], ["ss", "-H", "-ltn"])
        self.assertNotIn("selected", result)
        self.assertEqual(result["candidates"][1]["status"], "missing")
        self.assertNotIn("argv", result["candidates"][0])
        self.assertNotIn("stdout_tail", result["candidates"][0])

    def test_port_probe_uses_python_fallback_when_external_tools_missing(self) -> None:
        result = probe_environment._port_probe_tools(
            timeout=5,
            which=lambda name: None,
            runner=self.runner,
        )
        self.assertEqual(result["name"], "python_socket")
        self.assertEqual(result["mode"], "socket_proc")

    def test_cli_checks_write_complete_help_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            serve_path = Path(directory) / "serve.txt"
            bench_path = Path(directory) / "bench.txt"
            serve = probe_environment._cli_check(
                command="vllm",
                subcommand=["serve"],
                timeout=5,
                which=self.which,
                runner=self.runner,
                help_artifact=str(serve_path),
            )
            bench = probe_environment._cli_check(
                command="vllm",
                subcommand=["bench", "serve"],
                timeout=5,
                which=self.which,
                runner=self.runner,
                help_artifact=str(bench_path),
            )
            self.assertEqual(serve["status"], "pass")
            self.assertEqual(bench["status"], "pass")
            self.assertEqual(serve_path.read_text(encoding="utf-8"), "serve help")
            self.assertEqual(bench_path.read_text(encoding="utf-8"), "bench help")

    def test_cli_failure_keeps_bounded_diagnostic_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def failing_runner(argv: list[str], **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(
                    returncode=1,
                    stdout="stdout " * 1000,
                    stderr="stderr " * 1000,
                )

            result = probe_environment._cli_check(
                command="vllm",
                subcommand=["serve"],
                timeout=5,
                which=self.which,
                runner=failing_runner,
                help_artifact=str(Path(directory) / "serve.txt"),
            )

        self.assertEqual(result["status"], "failed")
        self.assertLessEqual(len(result["diagnostic_tail"]), 2000)
        self.assertNotIn("stdout", result)
        self.assertNotIn("stderr", result)
        self.assertNotIn("stdout_tail", result)
        self.assertNotIn("stderr_tail", result)

    def test_run_probe_combines_cann_dependencies_and_both_cli_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[list[str]] = []

            def recording_runner(argv: list[str], **kwargs: object) -> SimpleNamespace:
                calls.append(argv)
                return self.runner(argv, **kwargs)

            result = probe_environment.run_probe(
                json.dumps(self.expected_env()),
                provider_path="managed_deployment",
                execution_context="current",
                timeout=5,
                cli_timeout=77,
                serve_help_artifact=str(root / "vllm-serve-help-all.txt"),
                bench_help_artifact=str(root / "bench-serve-help-all.txt"),
                importer=self.importer,
                version_getter=self.version_getter,
                loader=self.loader,
                which=self.which,
                runner=recording_runner,
            )
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["cann"]["status"], "pass")
            self.assertEqual(result["sandbox"], {"status": "not_detected"})
            self.assertEqual(result["port_probe_tool"]["name"], "ss")
            self.assertNotIn("selected", result["port_probe_tool"])
            self.assertEqual(result["vllm_serve"]["argv"][-1], "--help=all")
            self.assertEqual(result["vllm_bench_serve"]["argv"][-1], "--help=all")
            self.assertEqual(result["vllm_serve"]["timeout_seconds"], 77)
            self.assertEqual(result["vllm_bench_serve"]["timeout_seconds"], 77)
            self.assertNotIn('"stdout"', json.dumps(result))
            self.assertNotIn('"stderr"', json.dumps(result))
            self.assertNotIn('"selected"', json.dumps(result))
            self.assertEqual(
                calls,
                [
                    ["/env/bin/npu-smi", "info"],
                    ["/env/bin/vllm", "serve", "--help=all"],
                    ["/env/bin/vllm", "bench", "serve", "--help=all"],
                    probe_environment.PORT_PROBE_CANDIDATES[0][1],
                    probe_environment.PYTHON_PORT_PROBE[1],
                ],
            )

    def test_protected_environment_change_fails_probe(self) -> None:
        expected = self.expected_env()
        expected["TMPDIR"] = "/different/tmp"
        with tempfile.TemporaryDirectory() as directory:
            result = probe_environment.run_probe(
                json.dumps(expected),
                provider_path="predeployed",
                execution_context="current",
                timeout=5,
                serve_help_artifact=str(Path(directory) / "serve.txt"),
                bench_help_artifact=str(Path(directory) / "bench.txt"),
                importer=self.importer,
                version_getter=self.version_getter,
                loader=self.loader,
                which=self.which,
                runner=self.runner,
            )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["type"], "protected_environment_changed")

    def test_main_writes_snapshot_history_without_circular_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_run_probe = probe_environment.run_probe
            probe_environment.run_probe = lambda *args, **kwargs: {
                "schema_version": probe_environment.SCHEMA_VERSION,
                "status": "pass",
                "provider_path": "predeployed",
                "execution_context": "current",
                "shell_fingerprint": {},
                "sandbox": {"status": "not_detected"},
                "artifacts": {
                    "serve_help": str(root / "serve.txt"),
                    "bench_help": str(root / "bench.txt"),
                    "sandbox_detection": None,
                },
            }
            try:
                rc = probe_environment.main(
                    [
                        "--attempt-dir",
                        str(root),
                        "--provider-path",
                        "predeployed",
                        "--execution-context",
                        "current",
                        "--probe-phase",
                        "initial",
                    ]
                )
            finally:
                probe_environment.run_probe = original_run_probe
            self.assertEqual(rc, 0)
            report = json.loads((root / "environment_probe.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["probe_history"]), 1)
            self.assertEqual(set(report["probe_history"][0]), {
                "execution_context",
                "status",
                "shell_fingerprint",
                "sandbox_status",
                "failure",
            })
            self.assertNotIn("stdout_tail", json.dumps(report))
            self.assertEqual(
                report["artifacts"]["sandbox_detection"],
                str(root / "sandbox-detection.json"),
            )

    def test_sandbox_marks_missing_device_after_npu_failure_as_detected(self) -> None:
        result = probe_environment._sandbox_record(
            npu_path="/env/bin/npu-smi",
            npu_result={
                "status": "failed",
                "stdout": "Failed to open the character device when interacting with the OS kernel",
                "stderr": "",
                "argv": ["/env/bin/npu-smi", "info"],
                "returncode": 187,
            },
            port_tools={"status": "pass"},
        )
        self.assertEqual(result["status"], "detected")


if __name__ == "__main__":
    unittest.main()
