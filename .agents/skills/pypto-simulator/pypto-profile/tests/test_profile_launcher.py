from __future__ import annotations

import argparse
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "pypto-profile" / "scripts" / "run_serving_profile.py"
HOOK = ROOT / "pypto-profile" / "scripts" / "install_serving_profile_hook.py"


class ProfileLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("run_serving_profile", LAUNCHER)
        assert spec and spec.loader
        cls.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.launcher)

    def test_parses_positive_dispatch_ordinal(self) -> None:
        self.assertEqual(
            self.launcher._capture_dispatch("prefill_fwd=1"),
            ("prefill_fwd", 1),
        )

    def test_rejects_invalid_dispatch_ordinal(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            self.launcher._capture_dispatch("decode_fwd=0")
    def test_profile_hook_compiles_executable_child_artifacts(self) -> None:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
        from pypto_serving.model.common.compiler.compiler import KernelCompiler
        from pypto_serving.model.common.executor import utils
        from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin

        original_compile = KernelCompiler.compile
        original_build_config = utils.build_pypto_run_config
        original_configure_rings = L3DispatchMixin._configure_l3_rings
        original_run_l3 = L3DispatchMixin._run_l3
        try:
            with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
                os.environ,
                {
                    "PYPTO_PROFILE_ARTIFACT_ROOT": temporary,
                    "PYPTO_PROFILE_SWIMLANE_LEVEL": "4",
                    "PYPTO_PROFILE_PROGRAMS": "prefill_fwd",
                    "PYPTO_PROFILE_CAPTURE_DISPATCHES": "{}",
                    "PYPTO_PROFILE_BYPASS_PROGRAMS": "",
                },
            ):
                spec = importlib.util.spec_from_file_location("test_profile_hook", HOOK)
                assert spec and spec.loader
                hook = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hook)
                hook.install()

                config = utils.build_pypto_run_config(platform="a2a3", device_ids=[0])
                captured = {}

                class FakeJit:
                    def compile(self, *, config, **kwargs):
                        captured["config"] = config
                        captured["kwargs"] = kwargs
                        return object.__new__(DistributedCompiledProgram)

                result = KernelCompiler(run_config=config).compile(
                    "prefill_fwd", FakeJit(), marker="value"
                )
                self.assertFalse(captured["config"].codegen_only)
                self.assertEqual(captured["kwargs"], {"marker": "value"})
                self.assertEqual(result.name, "prefill_fwd")
                self.assertTrue(
                    str(captured["config"].save_kernels_dir).endswith("build/prefill_fwd")
                )
        finally:
            KernelCompiler.compile = original_compile
            utils.build_pypto_run_config = original_build_config
            L3DispatchMixin._configure_l3_rings = original_configure_rings
            L3DispatchMixin._run_l3 = original_run_l3


if __name__ == "__main__":
    unittest.main()
