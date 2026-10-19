"""Probe the current Ascend environment and the vLLM serve entrypoint."""

from __future__ import annotations

import argparse
import ctypes
import importlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import Any


SCHEMA_VERSION = "vllm-ascend-environment-preflight-v1"
PROTECTED_ENV = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "TMPDIR",
    "TMP",
    "TEMP",
    "PYTHONPYCACHEPREFIX",
)
REQUIRED_MODULES = ("vllm", "torch", "torch_npu", "vllm_ascend")
DISTRIBUTION_NAMES = {
    "vllm": "vllm",
    "torch": "torch",
    "torch_npu": "torch-npu",
    "vllm_ascend": "vllm-ascend",
}


def _distribution_version(
    module_name: str,
    *,
    version_getter: Callable[[str], str] = importlib_metadata.version,
) -> str | None:
    """Read the installed distribution version without relying on module attributes."""
    distribution_name = DISTRIBUTION_NAMES.get(module_name, module_name)
    try:
        return version_getter(distribution_name)
    except Exception:
        # Version metadata is diagnostic only; an incomplete editable install must
        # not turn an otherwise usable runtime into a preflight failure.
        return None


def _module_record(
    name: str,
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib_metadata.version,
) -> tuple[dict[str, Any], Any | None]:
    try:
        module = importer(name)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }, None
    return {
        "status": "pass",
        "file": getattr(module, "__file__", None),
        "version": _distribution_version(name, version_getter=version_getter),
    }, module


def _dependency_check(
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib_metadata.version,
) -> dict[str, Any]:
    dependencies: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    for name in REQUIRED_MODULES:
        record, module = _module_record(
            name,
            importer=importer,
            version_getter=version_getter,
        )
        dependencies[name] = record
        if module is not None:
            modules[name] = module

    torch = modules.get("torch")
    npu = getattr(torch, "npu", None)
    runtime: dict[str, Any] = {"status": "failed"}
    try:
        available_fn = getattr(npu, "is_available")
        available = bool(available_fn())
        runtime["available"] = available
        count_fn = getattr(npu, "device_count", None)
        if count_fn is not None:
            runtime["device_count"] = int(count_fn())
        if not available:
            runtime["failure_type"] = "ascend_runtime_unavailable"
            runtime["error"] = "torch.npu.is_available() returned false"
        elif runtime.get("device_count", 1) <= 0:
            runtime["failure_type"] = "ascend_runtime_unavailable"
            runtime["error"] = "torch.npu.device_count() returned zero"
        else:
            runtime["status"] = "pass"
    except Exception as exc:
        runtime.update(
            {
                "failure_type": "ascend_runtime_probe_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return {
        "status": (
            "pass"
            if all(item.get("status") == "pass" for item in dependencies.values())
            and runtime["status"] == "pass"
            else "failed"
        ),
        "modules": dependencies,
        "ascend_runtime": runtime,
    }


def _vllm_serve_check(
    *,
    command: str = "vllm",
    timeout: float,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    help_artifact: str | None = None,
) -> dict[str, Any]:
    executable = which(command)
    if not executable:
        return {
            "status": "failed",
            "failure_type": "vllm_executable_missing",
            "command": [command, "serve", "--help=all"],
            "help_artifact": None,
        }
    argv = [executable, "serve", "--help=all"]
    try:
        result = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed",
            "failure_type": "vllm_serve_probe_failed",
            "command": argv,
            "error": str(exc),
            "help_artifact": None,
        }
    record = {
        "status": "pass" if result.returncode == 0 else "failed",
        "failure_type": None if result.returncode == 0 else "vllm_serve_unavailable",
        "command": argv,
        "returncode": result.returncode,
        "stdout": (result.stdout or "")[-4000:],
        "stderr": (result.stderr or "")[-4000:],
        "help_artifact": None,
    }
    if help_artifact is not None:
        try:
            target = Path(help_artifact)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result.stdout or "", encoding="utf-8")
            record["help_artifact"] = str(target)
        except OSError as exc:
            record["help_artifact_error"] = str(exc)
    return record


def _atb_check(
    *,
    loader: Callable[[str], Any] = ctypes.CDLL,
) -> dict[str, Any]:
    try:
        library = loader("libatb.so")
    except OSError as exc:
        return {
            "status": "failed",
            "failure_type": "atb_loader_failed",
            "error": str(exc),
        }
    return {
        "status": "pass",
        "library": getattr(library, "_name", "libatb.so"),
    }


def run_runtime_checks(
    expected_json: str,
    timeout: float = 30.0,
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib_metadata.version,
    loader: Callable[[str], Any] = ctypes.CDLL,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    serve_help_artifact: str | None = None,
) -> dict[str, Any]:
    expected = json.loads(expected_json)
    protected_keys = tuple(expected) or PROTECTED_ENV
    changed = [key for key in protected_keys if os.environ.get(key) != expected.get(key)]
    if changed:
        return {
            "schema_version": SCHEMA_VERSION,
            "stage": "activation",
            "status": "failed",
            "failure_type": "protected_environment_changed",
            "changed_resource_variables": changed,
        }

    checks: dict[str, Any] = {"atb_loader": _atb_check(loader=loader)}
    if checks["atb_loader"]["status"] != "pass":
        status = "failed"
    else:
        checks["dependencies"] = _dependency_check(
            importer=importer,
            version_getter=version_getter,
        )
        checks["vllm_serve"] = _vllm_serve_check(
            timeout=timeout,
            which=which,
            runner=runner,
            help_artifact=serve_help_artifact,
        )
        status = (
            "pass"
            if all(item.get("status") == "pass" for item in checks.values())
            else "failed"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "ascend_vllm_preflight",
        "status": status,
        "python": sys.executable,
        "checks": checks,
    }


def check_runtime(
    expected_json: str,
    timeout: float = 30.0,
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib_metadata.version,
    loader: Callable[[str], Any] = ctypes.CDLL,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
    serve_help_artifact: str | None = None,
) -> int:
    report = run_runtime_checks(
        expected_json,
        timeout,
        importer=importer,
        version_getter=version_getter,
        loader=loader,
        which=which,
        runner=runner,
        serve_help_artifact=serve_help_artifact,
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report.get("status") == "pass" else 2


def _positive_timeout(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return number


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", default=30, type=_positive_timeout)
    parser.add_argument(
        "--protected-env",
        default="{}",
        help="内联 JSON object 字符串（不是 JSON 文件路径）的受保护环境变量快照",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--serve-help-output",
        help="将 vllm serve --help=all 的完整 stdout 落盘为文件 artifact，供后续阶段复用",
    )
    args = parser.parse_args(argv)

    try:
        protected = json.loads(args.protected_env)
        if not isinstance(protected, dict):
            raise ValueError("--protected-env 必须是 JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    if args.protected_env == "{}":
        protected = {key: os.environ.get(key) for key in PROTECTED_ENV}
    report = run_runtime_checks(
        json.dumps(protected),
        timeout=args.timeout,
        serve_help_artifact=args.serve_help_output,
    )
    if args.output:
        _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
