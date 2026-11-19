#!/usr/bin/env python3
"""Install level-4 PyPTO DFX collection into pypto-serving L3 dispatches."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    root_raw = os.environ.get("PYPTO_PROFILE_ARTIFACT_ROOT")
    if not root_raw:
        raise RuntimeError("PYPTO_PROFILE_ARTIFACT_ROOT is required")
    root = Path(root_raw).resolve()
    root.mkdir(parents=True, exist_ok=True)
    level = int(os.environ.get("PYPTO_PROFILE_SWIMLANE_LEVEL", "4"))
    if level not in range(1, 5):
        raise RuntimeError("PYPTO_PROFILE_SWIMLANE_LEVEL must be in [1, 4]")

    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.common.compiler.l3_callable import L3Callable
    from pypto_serving.model.common.executor import utils
    from pypto_serving.model.common.runner.l3_dispatch import L3DispatchMixin

    from pypto.runtime import RunConfig

    selected = {
        name.strip()
        for name in os.environ.get("PYPTO_PROFILE_PROGRAMS", "").split(",")
        if name.strip()
    }
    try:
        raw_dispatches = json.loads(os.environ.get("PYPTO_PROFILE_CAPTURE_DISPATCHES", "{}"))
        capture_dispatches = {str(name): int(ordinal) for name, ordinal in raw_dispatches.items()}
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("PYPTO_PROFILE_CAPTURE_DISPATCHES must be a JSON object") from error
    if any(not name or ordinal < 1 for name, ordinal in capture_dispatches.items()):
        raise RuntimeError("capture dispatch ordinals must be positive")
    if selected and capture_dispatches:
        raise RuntimeError("program selection and dispatch selection are mutually exclusive")
    bypassed = {
        name.strip()
        for name in os.environ.get("PYPTO_PROFILE_BYPASS_PROGRAMS", "").split(",")
        if name.strip()
    }
    if selected & bypassed:
        raise RuntimeError(f"programs cannot be selected and bypassed: {sorted(selected & bypassed)}")
    (root / "capture_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "swimlane_level": level,
                "selected_programs": sorted(selected),
                "capture_dispatches": capture_dispatches,
                "bypassed_programs": sorted(bypassed),
                "npu_measured_surface": "selected programs only",
                "warning": "Bypassed programs are not hardware measurements and must not enter latency models.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )

    original_build_config = utils.build_pypto_run_config
    original_configure_rings = L3DispatchMixin._configure_l3_rings
    original_run_l3 = L3DispatchMixin._run_l3
    dispatch_counts: dict[str, int] = {}

    def compile_executable(self, name, jit_fn, **compile_kwargs):
        """Emit executable child artifacts needed by ordinal bootstrap dispatches."""
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

        configs = {**self._extra_configs, "codegen_only": False}
        if self._run_config.save_kernels_dir is not None and "save_kernels_dir" not in configs:
            configs["save_kernels_dir"] = str(Path(self._run_config.save_kernels_dir) / name)
        run_config = dataclasses.replace(self._run_config, **configs)
        compiled = jit_fn.compile(config=run_config, **compile_kwargs)
        if not isinstance(compiled, DistributedCompiledProgram):
            raise TypeError(
                f"{name} did not compile to DistributedCompiledProgram; "
                f"got {type(compiled).__name__}"
            )
        return L3Callable(
            compiled=compiled,
            name=name,
            aicpu_thread_num=run_config.distributed_config.aicpu_thread_num,
        )

    def build_pypto_run_config(**kwargs):
        kwargs["pypto_build_dir"] = str(root / "build")
        config = original_build_config(**kwargs)
        return dataclasses.replace(config, save_kernels=True, save_kernels_dir=str(root / "build"))

    def configure_l3_rings(self, runtime):
        original_configure_rings(self, runtime)
        base = self._l3_run_config or RunConfig()
        self._pypto_profile_bootstrap_run_config = dataclasses.replace(
            base,
            enable_chip_swimlane=0,
            enable_dep_gen=False,
            save_kernels=True,
            save_kernels_dir=str(root / "build"),
        )
        self._pypto_profile_run_config = dataclasses.replace(
            base,
            enable_chip_swimlane=level,
            enable_dep_gen=False,
            save_kernels=True,
            save_kernels_dir=str(root / "build"),
        )

    def run_l3(self, callable_spec, *args, config=None):
        name = callable_spec.name
        dispatch_counts[name] = dispatch_counts.get(name, 0) + 1
        if name in bypassed:
            print(f"[pypto-profile] bypassing {name}", flush=True)
            return None
        if capture_dispatches:
            if capture_dispatches.get(name) != dispatch_counts[name]:
                bootstrap_config = self._pypto_profile_bootstrap_run_config
                return original_run_l3(
                    self, callable_spec, *args, config=bootstrap_config
                )
        elif selected and name not in selected:
            return original_run_l3(self, callable_spec, *args, config=config)
        profile_config = self._pypto_profile_run_config
        return original_run_l3(self, callable_spec, *args, config=profile_config)

    KernelCompiler.compile = compile_executable
    utils.build_pypto_run_config = build_pypto_run_config
    L3DispatchMixin._configure_l3_rings = configure_l3_rings
    L3DispatchMixin._run_l3 = run_l3
    _INSTALLED = True


if __name__ == "__main__":
    install()
