#!/usr/bin/env python3
"""Shared monotonic-offset clock and conservative PyPTO step coordinator."""

from __future__ import annotations

import argparse
import json
import mmap
import os
import socket
import socketserver
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CLOCK_STRUCT = struct.Struct("=Qq")
_CLOCK_SIZE = _CLOCK_STRUCT.size


class SharedClock:
    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        self.path = str(path)
        flags = os.O_RDWR | (os.O_CREAT | os.O_TRUNC if create else 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, _CLOCK_SIZE)
            self._map = mmap.mmap(fd, _CLOCK_SIZE)
        finally:
            os.close(fd)
        if create:
            self.write_offset_ns(0)

    def write_offset_ns(self, offset_ns: int) -> None:
        sequence, _ = _CLOCK_STRUCT.unpack_from(self._map)
        if sequence & 1:
            sequence += 1
        _CLOCK_STRUCT.pack_into(self._map, 0, sequence + 1, int(offset_ns))
        _CLOCK_STRUCT.pack_into(self._map, 0, sequence + 2, int(offset_ns))

    def offset_ns(self) -> int:
        while True:
            first, offset_ns = _CLOCK_STRUCT.unpack_from(self._map)
            second, _ = _CLOCK_STRUCT.unpack_from(self._map)
            if first == second and not first & 1:
                return int(offset_ns)

    def now_ns(self) -> int:
        return time.monotonic_ns() + self.offset_ns()

    def advance_to_ns(self, target_ns: int) -> tuple[int, int]:
        before_offset = self.offset_ns()
        before = time.monotonic_ns() + before_offset
        if target_ns > before:
            wall_ns = time.monotonic_ns()
            offset_ns = self.offset_ns()
            current = wall_ns + offset_ns
            if target_ns > current:
                self.write_offset_ns(offset_ns + target_ns - current)
        after_offset = self.offset_ns()
        return time.monotonic_ns() + after_offset, max(0, after_offset - before_offset)

    def close(self) -> None:
        self._map.close()


@dataclass
class StepState:
    frontier_ns: int
    graph_count: int = 0
    sealed: bool = False


class TimeKeeper:
    def __init__(self, clock: SharedClock, trace_path: Path | None = None) -> None:
        self.clock = clock
        self.trace_path = trace_path
        self._condition = threading.Condition()
        self._steps: dict[str, StepState] = {}
        self.device_frontier_ns = clock.now_ns()
        self.graph_events = 0
        self.completed_steps = 0
        self.output_pending = 0
        self.output_generations = 0
        self.delivered_outputs = 0
        self.jumps = 0
        self.jumped_ns = 0
        self.deadline_overrun_count = 0
        self.deadline_overrun_total_ns = 0
        self.deadline_overrun_max_ns = 0
        self._trace("timekeeper_start")

    def _trace(self, event: str, **values: Any) -> None:
        if self.trace_path is None:
            return
        record = {
            "event": event,
            "wall_ns": time.monotonic_ns(),
            "virtual_ns": self.clock.now_ns(),
            **values,
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self.trace_path.open("a", encoding="ascii") as stream:
            stream.write(payload)

    def begin_step(self, key: str) -> int:
        with self._condition:
            if key in self._steps:
                raise RuntimeError(f"duplicate timewarp step: {key}")
            now = self.clock.now_ns()
            self._steps[key] = StepState(frontier_ns=max(now, self.device_frontier_ns))
            self._trace("step_begin", step=key)
            return now

    def graph(self, key: str | None, program: str, duration_ns: int) -> int:
        if duration_ns < 0:
            raise ValueError("graph duration cannot be negative")
        with self._condition:
            now = self.clock.now_ns()
            start = max(now, self.device_frontier_ns)
            finish = start + duration_ns
            self.device_frontier_ns = finish
            if key is not None:
                state = self._steps.get(key)
                if state is None:
                    raise RuntimeError(f"graph references unknown step: {key}")
                if state.sealed:
                    raise RuntimeError(f"graph references sealed step: {key}")
                state.frontier_ns = max(state.frontier_ns, finish)
                state.graph_count += 1
            else:
                _, jumped = self.clock.advance_to_ns(finish)
                if jumped:
                    self.jumps += 1
                    self.jumped_ns += jumped
            self.graph_events += 1
            self._trace(
                "graph",
                step=key,
                program=program,
                duration_ns=duration_ns,
                start_ns=start,
                finish_ns=finish,
            )
            return finish

    def startup_duration(self, operation: str, duration_ns: int) -> int:
        if duration_ns < 0:
            raise ValueError("startup duration cannot be negative")
        with self._condition:
            target = self.clock.now_ns() + duration_ns
            now, jumped = self.clock.advance_to_ns(target)
            if jumped:
                self.jumps += 1
                self.jumped_ns += jumped
            self.device_frontier_ns = max(self.device_frontier_ns, now)
            self._trace(
                "startup_duration",
                operation=operation,
                duration_ns=duration_ns,
                jumped_ns=jumped,
            )
            return now

    def seal_step(self, key: str) -> int:
        with self._condition:
            state = self._steps.get(key)
            if state is None:
                now = self.clock.now_ns()
                state = StepState(frontier_ns=max(now, self.device_frontier_ns))
                self._steps[key] = state
            if state.sealed:
                raise RuntimeError(f"duplicate timewarp step seal: {key}")
            state.sealed = True
            self._trace(
                "step_seal", step=key, deadline_ns=state.frontier_ns, graphs=state.graph_count
            )
            self._condition.notify_all()
            return state.frontier_ns

    def output_expected(self) -> int:
        with self._condition:
            self.output_pending += 1
            self.output_generations += 1
            generation = self.output_generations
            self._trace("output_expected", generation=generation)
            self._condition.notify_all()
            return generation

    def output_delivered(self) -> int:
        with self._condition:
            if self.output_pending <= 0:
                raise RuntimeError("no timewarp output is awaiting delivery")
            self.output_pending -= 1
            self.delivered_outputs += 1
            self._trace("output_delivered", remaining=self.output_pending)
            self._condition.notify_all()
            return self.clock.now_ns()

    def wait_step(self, key: str, timeout_s: float) -> int:
        with self._condition:
            deadline = time.monotonic() + timeout_s
            while (
                key not in self._steps
                or not self._steps[key].sealed
                or self.output_pending > 0
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timewarp step cannot advance: step={key}, "
                        f"output_pending={self.output_pending}"
                    )
                self._condition.wait(remaining)
            state = self._steps.pop(key)
            observed_ns = self.clock.now_ns()
            overrun_ns = (
                max(0, observed_ns - state.frontier_ns) if state.graph_count else 0
            )
            if overrun_ns:
                self.deadline_overrun_count += 1
                self.deadline_overrun_total_ns += overrun_ns
                self.deadline_overrun_max_ns = max(
                    self.deadline_overrun_max_ns, overrun_ns
                )
            now, jumped = self.clock.advance_to_ns(state.frontier_ns)
            if jumped:
                self.jumps += 1
                self.jumped_ns += jumped
            self.completed_steps += 1
            self._trace(
                "step_deliver",
                step=key,
                deadline_ns=state.frontier_ns,
                graphs=state.graph_count,
                jumped_ns=jumped,
                overrun_ns=overrun_ns,
            )
            return now

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "now_ns": self.clock.now_ns(),
                "offset_ns": self.clock.offset_ns(),
                "device_frontier_ns": self.device_frontier_ns,
                "pending_steps": {
                    key: {
                        "frontier_ns": state.frontier_ns,
                        "graph_count": state.graph_count,
                        "sealed": state.sealed,
                    }
                    for key, state in sorted(self._steps.items())
                },
                "graph_events": self.graph_events,
                "completed_steps": self.completed_steps,
                "output_pending": self.output_pending,
                "output_generations": self.output_generations,
                "delivered_outputs": self.delivered_outputs,
                "jumps": self.jumps,
                "jumped_ns": self.jumped_ns,
                "deadline_overrun_count": self.deadline_overrun_count,
                "deadline_overrun_total_ns": self.deadline_overrun_total_ns,
                "deadline_overrun_max_ns": self.deadline_overrun_max_ns,
            }


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while line := self.rfile.readline():
            try:
                request = json.loads(line)
                response = self.server.dispatch(request)  # type: ignore[attr-defined]
                payload = {"ok": True, **response}
            except Exception as error:  # noqa: BLE001 - RPC failures must become responses.
                payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            self.wfile.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
            self.wfile.flush()


class TimeKeeperServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: str, keeper: TimeKeeper) -> None:
        self.keeper = keeper
        super().__init__(path, _Handler)

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "begin_step":
            return {"now_ns": self.keeper.begin_step(str(request["step"]))}
        if command == "graph":
            step = request.get("step")
            return {
                "deadline_ns": self.keeper.graph(
                    None if step is None else str(step),
                    str(request["program"]),
                    int(request["duration_ns"]),
                )
            }
        if command == "startup_duration":
            return {
                "now_ns": self.keeper.startup_duration(
                    str(request["operation"]), int(request["duration_ns"])
                )
            }
        if command == "seal_step":
            return {"deadline_ns": self.keeper.seal_step(str(request["step"]))}
        if command == "output_expected":
            return {"generation": self.keeper.output_expected()}
        if command == "output_delivered":
            return {"now_ns": self.keeper.output_delivered()}
        if command == "wait_step":
            return {
                "now_ns": self.keeper.wait_step(
                    str(request["step"]), float(request.get("timeout_s", 120.0))
                )
            }
        if command == "snapshot":
            return {"snapshot": self.keeper.snapshot()}
        raise ValueError(f"unknown command: {command!r}")


class TimewarpClient:
    def __init__(self, socket_path: str | None = None, timeout_s: float | None = None) -> None:
        self.socket_path = socket_path or os.environ["PYPTO_TIMEWARP_SOCKET"]
        self.timeout_s = float(
            timeout_s if timeout_s is not None else os.environ.get("PYPTO_TIMEWARP_TIMEOUT_S", "120")
        )
        self._local = threading.local()

    def _rpc(self, command: str, **values: Any) -> dict[str, Any]:
        request = json.dumps({"command": command, **values}, separators=(",", ":")).encode() + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout_s)
            client.connect(self.socket_path)
            client.sendall(request)
            stream = client.makefile("rb")
            line = stream.readline()
        if not line:
            raise RuntimeError("timewarp server closed without a response")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "timewarp RPC failed")))
        return response

    @staticmethod
    def step_key(replica: int, step_id: int) -> str:
        return f"{replica}:{step_id}"

    def set_current_step(self, key: str | None) -> None:
        self._local.step = key

    def current_step(self) -> str | None:
        return getattr(self._local, "step", None)

    def begin_step(self, key: str) -> int:
        self.set_current_step(key)
        return int(self._rpc("begin_step", step=key)["now_ns"])

    def graph(self, program: str, duration_ns: int) -> int:
        return int(
            self._rpc(
                "graph", step=self.current_step(), program=program, duration_ns=duration_ns
            )["deadline_ns"]
        )

    def startup_duration(self, operation: str, duration_ns: int) -> int:
        return int(
            self._rpc(
                "startup_duration", operation=operation, duration_ns=duration_ns
            )["now_ns"]
        )

    def seal_step(self, key: str) -> int:
        return int(self._rpc("seal_step", step=key)["deadline_ns"])

    def output_expected(self) -> int:
        return int(self._rpc("output_expected")["generation"])

    def output_delivered(self) -> int:
        return int(self._rpc("output_delivered")["now_ns"])

    def wait_step(self, key: str) -> int:
        return int(
            self._rpc("wait_step", step=key, timeout_s=self.timeout_s)["now_ns"]
        )

    def snapshot(self) -> dict[str, Any]:
        return dict(self._rpc("snapshot")["snapshot"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--clock", required=True)
    parser.add_argument("--trace", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    socket_path = Path(args.socket)
    clock_path = Path(args.clock)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        if args.trace.exists():
            raise FileExistsError(f"refusing existing trace: {args.trace}")
    clock = SharedClock(clock_path, create=True)
    keeper = TimeKeeper(clock, args.trace)
    server = TimeKeeperServer(str(socket_path), keeper)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        clock.close()
        if socket_path.exists():
            socket_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
