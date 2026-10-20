"""Probe the shared vLLM-Ascend environment once per execution context."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import Any


SCHEMA_VERSION = "vllm-ascend-environment-probe-v2"
DEFAULT_COMMAND_TIMEOUT = 30.0
DEFAULT_VLLM_CLI_TIMEOUT = 180.0
REQUIRED_MODULES = ("vllm", "torch", "torch_npu", "vllm_ascend")
DISTRIBUTION_NAMES = {
    "vllm": "vllm",
    "torch": "torch",
    "torch_npu": "torch-npu",
    "vllm_ascend": "vllm-ascend",
}
PORT_PROBE_CANDIDATES = (
    ("ss", ["ss", "-H", "-ltn"], "listen_table"),
    ("netstat", ["netstat", "-ltn"], "listen_table"),
    ("lsof", ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], "listen_table"),
)
PYTHON_PORT_PROBE = (
    "python_socket",
    [
        sys.executable,
        "-c",
        (
            "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
            "s.bind(('127.0.0.1', 0)); s.close(); "
            "open('/proc/net/tcp', encoding='ascii').read(); print('socket-proc-ok')"
        ),
    ],
    "socket_proc",
)


def _tail(value: str | None, limit: int = 4000) -> str:
    return (value or "")[-limit:]


def _diagnostic_fields(result: Mapping[str, Any], limit: int = 2000) -> dict[str, Any]:
    """Keep failure context without serializing successful command output."""
    if result.get("status") == "pass":
        return {}
    output = "\n".join(
        value for value in (result.get("stdout"), result.get("stderr")) if value
    )
    fields: dict[str, Any] = {}
    if output:
        fields["diagnostic_tail"] = _tail(output, limit)
    if result.get("error"):
        fields["error"] = result["error"]
    return fields


def _candidate_record(
    name: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name, "status": result.get("status")}
    if result.get("status") != "pass":
        record["returncode"] = result.get("returncode")
        record.update(_diagnostic_fields(result))
    return record


def _distribution_version(
    module_name: str,
    *,
    version_getter: Callable[[str], str] = importlib_metadata.version,
) -> str | None:
    try:
        return version_getter(DISTRIBUTION_NAMES.get(module_name, module_name))
    except Exception:
        return None


def _python_check() -> dict[str, Any]:
    return {
        "status": "pass",
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


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
    modules: dict[str, Any] = {}
    records: dict[str, Any] = {}
    for name in REQUIRED_MODULES:
        record, module = _module_record(
            name,
            importer=importer,
            version_getter=version_getter,
        )
        records[name] = record
        if module is not None:
            modules[name] = module

    runtime: dict[str, Any] = {"status": "failed"}
    torch = modules.get("torch")
    npu = getattr(torch, "npu", None)
    try:
        available = bool(getattr(npu, "is_available")())
        runtime["available"] = available
        count = int(getattr(npu, "device_count")())
        runtime["device_count"] = count
        if not available or count <= 0:
            runtime.update(
                {
                    "failure_type": "ascend_runtime_unavailable",
                    "error": "torch.npu is unavailable or reports zero devices",
                }
            )
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
            if all(item.get("status") == "pass" for item in records.values())
            and runtime["status"] == "pass"
            else "failed"
        ),
        "modules": records,
        "ascend_runtime": runtime,
    }


def _library_check(
    *,
    loader: Callable[[str], Any] = ctypes.CDLL,
) -> dict[str, Any]:
    libraries: dict[str, Any] = {}
    for name in ("libascendcl.so", "libatb.so"):
        try:
            library = loader(name)
        except OSError as exc:
            libraries[name] = {
                "status": "failed",
                "failure_type": "cann_library_load_failed",
                "error": str(exc),
            }
        else:
            libraries[name] = {
                "status": "pass",
                "loaded_name": getattr(library, "_name", name),
            }
    return {
        "status": "pass"
        if all(item["status"] == "pass" for item in libraries.values())
        else "failed",
        "libraries": libraries,
    }


def _run(
    argv: list[str],
    *,
    timeout: float,
    runner: Callable[..., Any],
) -> dict[str, Any]:
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
            "argv": argv,
            "returncode": None,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": "",
            "stderr": "",
        }
    return {
        "status": "pass" if result.returncode == 0 else "failed",
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }


def _port_probe_tools(
    *,
    timeout: float,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for name, argv, mode in PORT_PROBE_CANDIDATES:
        executable = which(name)
        if executable is None:
            candidates.append({"name": name, "status": "missing"})
            continue
        result = _run(argv, timeout=timeout, runner=runner)
        candidate = _candidate_record(name, result)
        candidates.append(candidate)
        if selected is None and result["status"] == "pass":
            selected = {
                "name": name,
                "argv": argv,
                "mode": mode,
                "executable": executable,
                "verified": True,
            }

    fallback_name, fallback_argv, fallback_mode = PYTHON_PORT_PROBE
    fallback = _run(fallback_argv, timeout=timeout, runner=runner)
    candidates.append(_candidate_record(fallback_name, fallback))
    if selected is None and fallback["status"] == "pass":
        selected = {
            "name": fallback_name,
            "argv": fallback_argv,
            "mode": fallback_mode,
            "executable": sys.executable,
            "verified": True,
        }

    result = {
        "status": "pass" if selected is not None else "failed",
        "candidates": candidates,
        "failure_type": None if selected is not None else "port_probe_tool_unavailable",
    }
    if selected is not None:
        result.update(selected)
    return result


def _npu_smi_probe(
    *,
    timeout: float,
    runner: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[str | None, dict[str, Any]]:
    npu_path = which("npu-smi")
    npu_result = (
        _run([npu_path, "info"], timeout=timeout, runner=runner)
        if npu_path
        else {
            "status": "missing",
            "argv": ["npu-smi", "info"],
            "returncode": None,
            "stdout": "",
            "stderr": "npu-smi not found",
        }
    )
    return npu_path, npu_result


def _sandbox_record(
    *,
    npu_path: str | None,
    npu_result: Mapping[str, Any],
    port_tools: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    device_path = Path("/dev/davinci_manager")
    device_exists = device_path.exists()
    device_readable = os.access(device_path, os.R_OK) if device_exists else False
    text = f"{npu_result.get('stdout', '')}\n{npu_result.get('stderr', '')}".lower()
    permission_error = any(token in text for token in ("eperm", "eacces", "permission denied"))
    namespace_error = any(
        token in text for token in ("namespace", "device node", "davinci_manager", "not permitted")
    )
    device_visibility_error = any(
        token in text
        for token in (
            "failed to open the character device",
            "failed to obtain the number of devices",
            "different containers share the same device",
        )
    )
    if (device_exists and not device_readable) or permission_error or namespace_error or device_visibility_error:
        status = "detected"
    elif npu_path is None:
        status = "not_detected"
    elif npu_result.get("status") != "pass":
        status = "unknown"
    else:
        status = "not_detected"
    return {
        "status": status,
        "device_node": {
            "path": str(device_path),
            "exists": device_exists,
            "readable": device_readable,
            "rc": 0 if device_readable else 1,
        },
        # Keep the verified tool in the sandbox artifact so callers do not
        # rediscover it before the resource phase.
        "port_probe": dict(port_tools),
        "ascend_runtime": dict(runtime or {}),
    }


def _sandbox_probe(
    *,
    timeout: float,
    runner: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Compatibility wrapper for callers that need the combined sandbox snapshot."""
    npu_path, npu_result = _npu_smi_probe(timeout=timeout, runner=runner, which=which)
    port_tools = _port_probe_tools(timeout=timeout, which=which, runner=runner)
    return _sandbox_record(
        npu_path=npu_path,
        npu_result=npu_result,
        port_tools=port_tools,
    )


def _http_client_check() -> dict[str, Any]:
    try:
        import http.client  # noqa: F401
        import urllib.request  # noqa: F401
    except Exception as exc:
        return {
            "status": "failed",
            "failure_type": "http_client_missing",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "pass", "implementation": "python-stdlib"}


def _cli_check(
    *,
    command: str,
    subcommand: list[str],
    timeout: float,
    which: Callable[[str], str | None],
    runner: Callable[..., Any],
    help_artifact: str,
) -> dict[str, Any]:
    executable = which(command)
    argv = [executable or command, *subcommand, "--help=all"]
    if executable is None:
        return {
            "status": "failed",
            "failure_type": "vllm_executable_missing",
            "argv": argv,
            "executable": None,
            "help_artifact": None,
        }
    result = _run(argv, timeout=timeout, runner=runner)
    artifact_path = Path(help_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    stdout = result.get("stdout", "")
    stderr = result.get("stderr", "")
    complete_output = stdout + (f"\n{stderr}" if stderr else "")
    artifact_path.write_text(complete_output, encoding="utf-8")
    failure_type = None
    if result["status"] != "pass":
        failure_type = "vllm_serve_unavailable" if subcommand == ["serve"] else "vllm_bench_serve_unavailable"
    record = {
        "status": result["status"],
        "failure_type": failure_type,
        "argv": argv,
        "executable": executable,
        "timeout_seconds": timeout,
        "returncode": result.get("returncode"),
        "help_artifact": str(artifact_path),
    }
    record.update(_diagnostic_fields(result))
    return record


def _shell_fingerprint() -> dict[str, Any]:
    net_ns = None
    try:
        net_ns = os.readlink(f"/proc/{os.getpid()}/ns/net")
    except OSError:
        pass
    path_value = os.environ.get("PATH", "")
    return {
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "network_namespace": net_ns,
        "path_sha256": hashlib.sha256(path_value.encode()).hexdigest(),
    }


def run_probe(
    *,
    provider_path: str,
    execution_context: str,
    timeout: float,
    cli_timeout: float | None = None,
    serve_help_artifact: str,
    bench_help_artifact: str,
    sandbox_details_out: dict[str, Any] | None = None,
    importer: Callable[[str], Any] = importlib.import_module,
    version_getter: Callable[[str], str] = importlib_metadata.version,
    loader: Callable[[str], Any] = ctypes.CDLL,
    which: Callable[[str], str | None] = shutil.which,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    python = _python_check()
    vllm_executable_path = which("vllm")
    vllm_executable = {
        "status": "pass" if vllm_executable_path else "failed",
        "executable": vllm_executable_path,
        "failure_type": None if vllm_executable_path else "vllm_executable_missing",
    }
    dependencies = _dependency_check(importer=importer, version_getter=version_getter)
    npu_path, npu_result = _npu_smi_probe(timeout=timeout, runner=runner, which=which)
    libraries = _library_check(loader=loader)
    npu_smi = {
        "executable": npu_path,
        "status": npu_result.get("status"),
        "argv": npu_result.get("argv"),
        "returncode": npu_result.get("returncode"),
    }
    npu_smi.update(_diagnostic_fields(npu_result))
    cann = {
        "status": "pass"
        if libraries["status"] == "pass"
        and npu_smi.get("status") == "pass"
        else "failed",
        "libraries": libraries["libraries"],
        "npu_smi": npu_smi,
    }
    http_client = _http_client_check()
    effective_cli_timeout = timeout if cli_timeout is None else cli_timeout
    serve = _cli_check(
        command="vllm",
        subcommand=["serve"],
        timeout=effective_cli_timeout,
        which=lambda _name: vllm_executable_path,
        runner=runner,
        help_artifact=serve_help_artifact,
    )
    bench = _cli_check(
        command="vllm",
        subcommand=["bench", "serve"],
        timeout=effective_cli_timeout,
        which=lambda _name: vllm_executable_path,
        runner=runner,
        help_artifact=bench_help_artifact,
    )
    port_tools = _port_probe_tools(timeout=timeout, which=which, runner=runner)
    sandbox = _sandbox_record(
        npu_path=npu_path,
        npu_result=npu_result,
        port_tools=port_tools,
        runtime=dependencies.get("ascend_runtime"),
    )
    if sandbox_details_out is not None:
        sandbox_details_out.update(sandbox)
    failures: list[dict[str, Any]] = []
    for name, check in (
        ("python", python),
        ("vllm_executable", vllm_executable),
        ("cann", cann),
        ("dependencies", dependencies),
        ("http_client", http_client),
        ("vllm_serve", serve),
        ("vllm_bench_serve", bench),
    ):
        if check.get("status") != "pass":
            failures.append({"type": check.get("failure_type") or f"{name}_failed"})
    if sandbox["port_probe"].get("status") != "pass":
        failures.append({"type": "port_probe_tool_unavailable"})
    if sandbox["status"] in {"detected", "unknown"}:
        failures.append({"type": f"sandbox_isolation_{sandbox['status']}"})

    if execution_context == "current" and sandbox["status"] in {"detected", "unknown"}:
        status = "escalation_required"
        failure = {"type": "sandbox_escalation_required"}
    elif execution_context == "elevated" and sandbox["status"] in {"detected", "unknown"}:
        status = "failed"
        failure = {"type": "sandbox_isolation_persisted"}
    elif failures:
        status = "failed"
        failure = failures[0]
    else:
        status = "pass"
        failure = None
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "provider_path": provider_path,
        "execution_context": execution_context,
        "shell_fingerprint": _shell_fingerprint(),
        "python": python,
        "vllm_executable": vllm_executable,
        "sandbox": {"status": sandbox["status"]},
        "cann": cann,
        "dependencies": dependencies,
        "vllm_serve": serve,
        "vllm_bench_serve": bench,
        "http_client": http_client,
        "port_probe_tool": sandbox["port_probe"],
        "artifacts": {
            "serve_help": serve_help_artifact,
            "bench_help": bench_help_artifact,
            "sandbox_detection": None,
        },
        "failure": failure,
        "failures": failures,
    }


def _history_entry(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "execution_context": report.get("execution_context"),
        "status": report.get("status"),
        "shell_fingerprint": report.get("shell_fingerprint"),
        "sandbox_status": report.get("sandbox", {}).get("status"),
        "failure": report.get("failure"),
    }


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _write_sandbox(
    path: str,
    report: Mapping[str, Any],
    approval: str,
    sandbox_record: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": "vllm-ascend-sandbox-detection-v1",
        "status": report.get("sandbox", {}).get("status"),
        "execution_context": report.get("execution_context"),
        "approval": approval,
        "provider_path": report.get("provider_path"),
        "shell_fingerprint": report.get("shell_fingerprint"),
        "sandbox": dict(sandbox_record or {"status": report.get("sandbox", {}).get("status")}),
        "npu_smi": report.get("cann", {}).get("npu_smi"),
        "port_probe_tool": report.get("port_probe_tool"),
        "probe_history": report.get("probe_history", []),
    }
    _write_json(path, payload)


def _positive_timeout(value: str) -> float:
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--provider-path", required=True, choices=("managed_deployment", "predeployed"))
    parser.add_argument("--execution-context", required=True, choices=("current", "elevated"))
    parser.add_argument("--probe-phase", required=True, choices=("initial", "elevated"))
    parser.add_argument("--timeout", default=DEFAULT_COMMAND_TIMEOUT, type=_positive_timeout)
    parser.add_argument("--cli-timeout", default=DEFAULT_VLLM_CLI_TIMEOUT, type=_positive_timeout)
    parser.add_argument("--output")
    parser.add_argument("--sandbox-output")
    parser.add_argument("--serve-help-output")
    parser.add_argument("--bench-help-output")
    parser.add_argument("--approval", default="not_required", choices=("not_required", "requested", "granted", "denied"))
    args = parser.parse_args(argv)

    attempt_dir = Path(args.attempt_dir)
    output = args.output or str(attempt_dir / "environment_probe.json")
    sandbox_output = args.sandbox_output or str(attempt_dir / "sandbox-detection.json")
    serve_help = args.serve_help_output or str(attempt_dir / "vllm-serve-help-all.txt")
    bench_help = args.bench_help_output or str(attempt_dir / "bench-serve-help-all.txt")

    sandbox_details: dict[str, Any] = {}
    report = run_probe(
        provider_path=args.provider_path,
        execution_context=args.execution_context,
        timeout=args.timeout,
        cli_timeout=args.cli_timeout,
        serve_help_artifact=serve_help,
        bench_help_artifact=bench_help,
        sandbox_details_out=sandbox_details,
    )
    current_snapshot = _history_entry(report)
    if args.probe_phase == "elevated" and Path(output).exists():
        previous = json.loads(Path(output).read_text(encoding="utf-8"))
        previous_history = previous.get("probe_history") or [_history_entry(previous)]
        report["probe_history"] = [
            *(
                item
                if set(item) <= {
                    "execution_context",
                    "status",
                    "shell_fingerprint",
                    "sandbox_status",
                    "failure",
                }
                else _history_entry(item)
                for item in previous_history
            ),
            current_snapshot,
        ]
    else:
        report["probe_history"] = [current_snapshot]
    report["artifacts"]["sandbox_detection"] = sandbox_output
    _write_json(output, report)
    _write_sandbox(sandbox_output, report, args.approval, sandbox_details)
    print(
        json.dumps(
            {
                "status": report["status"],
                "execution_context": report["execution_context"],
                "output": output,
                "failure": report.get("failure"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if report["status"] == "pass":
        return 0
    if report["status"] == "escalation_required":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
