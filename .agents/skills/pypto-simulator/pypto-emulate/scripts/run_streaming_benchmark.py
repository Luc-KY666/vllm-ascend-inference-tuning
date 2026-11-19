#!/usr/bin/env python3
"""Run a manifest-driven streaming benchmark through one pypto-serving engine."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import mmap
import os
import random
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

WORDS = (" and", " the", " a", " to", " in", " of", " for", " with", " from", " on", " at", " is")
_TIMEWARP_CLOCK_STRUCT = struct.Struct("=Qq")
_TIMEWARP_CLOCK: mmap.mmap | None = None


def benchmark_now() -> float:
    """Read virtual monotonic time only under the explicit timewarp launcher."""
    global _TIMEWARP_CLOCK
    path = os.environ.get("PYPTO_TIMEWARP_CLOCK")
    if not path:
        return time.perf_counter()
    if _TIMEWARP_CLOCK is None:
        fd = os.open(path, os.O_RDONLY)
        try:
            _TIMEWARP_CLOCK = mmap.mmap(fd, _TIMEWARP_CLOCK_STRUCT.size, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)
    while True:
        first, offset_ns = _TIMEWARP_CLOCK_STRUCT.unpack_from(_TIMEWARP_CLOCK)
        second, _ = _TIMEWARP_CLOCK_STRUCT.unpack_from(_TIMEWARP_CLOCK)
        if first == second and not first & 1:
            return (time.monotonic_ns() + offset_ns) / 1_000_000_000


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("pypto-serving arguments are required after --")
    return args, command


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("benchmark manifest must use schema_version 1")
    if type(manifest.get("seed")) is not int:
        raise ValueError("benchmark seed must be an integer")
    for field in ("warmup_rounds", "measure_rounds"):
        value = manifest.get(field)
        if type(value) is not int or value < (1 if field == "measure_rounds" else 0):
            raise ValueError(f"{field} is invalid")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark manifest requires at least one case")
    ids: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError(f"invalid or duplicate case id: {case_id!r}")
        ids.add(case_id)
        for field in ("requests", "concurrency", "input_tokens", "output_tokens"):
            value = case.get(field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"case {case_id} has invalid {field}")
        if case["concurrency"] > case["requests"]:
            raise ValueError(f"case {case_id} concurrency exceeds requests")
        if case["output_tokens"] < 2:
            raise ValueError(f"case {case_id} needs at least two output tokens for TPOT")
    return manifest


def build_prompt(tokenizer: Any, target: int, seed: int) -> str:
    words = tuple(word for word in WORDS if len(tokenizer.encode(word)) == 1)
    if not words:
        raise ValueError("tokenizer has no known single-token benchmark words")
    rng = random.Random(seed)
    for _ in range(128):
        prompt = "".join(rng.choice(words) for _ in range(target))
        if len(tokenizer.encode(prompt)) == target:
            return prompt
    raise ValueError(f"could not construct an exact {target}-token prompt")


async def consume_request(
    engine: Any,
    prompt: str,
    config: Any,
    request_id: str,
    ready: asyncio.Event,
) -> dict[str, Any]:
    await ready.wait()
    started = benchmark_now()
    token_times: list[float] = []
    token_ids: list[int] = []
    completion_tokens = 0
    finish_reason: str | None = None
    terminal = None
    outputs = engine.add_request(request_id, prompt, config)
    async with contextlib.aclosing(outputs):
        async for output in outputs:
            now = benchmark_now()
            observed = int(output.completion_tokens)
            if observed > completion_tokens:
                token_times.extend([now] * (observed - completion_tokens))
                completion_tokens = observed
            if output.token_ids:
                token_ids = [int(token) for token in output.token_ids]
            if output.finished:
                terminal = now
                finish_reason = engine.normalize_finish_reason(output.finish_reason)
    if terminal is None or not token_times:
        raise RuntimeError(f"request {request_id} produced no complete token stream")
    if completion_tokens != config.max_new_tokens or finish_reason != "length":
        raise RuntimeError(
            f"request {request_id} ended with tokens={completion_tokens}, reason={finish_reason}"
        )
    ttft = token_times[0] - started
    e2el = terminal - started
    tpot = (token_times[-1] - token_times[0]) / (completion_tokens - 1)
    return {
        "request_id": request_id,
        "ttft_ms": ttft * 1000.0,
        "tpot_ms": tpot * 1000.0,
        "e2el_ms": e2el * 1000.0,
        "token_event_ms": [(stamp - started) * 1000.0 for stamp in token_times],
        "token_ids": token_ids,
        "token_count": completion_tokens,
        "finish_reason": finish_reason,
    }


async def run_case_round(
    engine: Any,
    case: dict[str, Any],
    prompts: list[str],
    config: Any,
    phase: str,
    round_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    concurrency = int(case["concurrency"])
    for wave_start in range(0, len(prompts), concurrency):
        wave = prompts[wave_start : wave_start + concurrency]
        ready = asyncio.Event()
        tasks = []
        for offset, prompt in enumerate(wave, start=wave_start):
            request_id = f"bench-{phase}-{case['id']}-r{round_index}-q{offset + 1}"
            tasks.append(
                asyncio.create_task(consume_request(engine, prompt, config, request_id, ready))
            )
        await asyncio.sleep(0)
        ready.set()
        wave_rows = await asyncio.gather(*tasks)
        rows.extend(wave_rows)
    return rows


def row_with_context(
    row: dict[str, Any], case: dict[str, Any], phase: str, round_index: int, request_index: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": phase,
        "case": case["id"],
        "round": round_index,
        "request_index": request_index,
        "input_tokens": case["input_tokens"],
        "output_tokens": case["output_tokens"],
        "concurrency": case["concurrency"],
        **row,
    }


def main() -> int:
    args, command = parse_args()
    output = args.output.resolve()
    if output.exists():
        print(f"error: refusing existing output: {output}", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest.resolve())
    except (OSError, ValueError) as error:
        print(f"error: invalid benchmark manifest: {error}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)

    from pypto_serving.cli import main as cli

    async def generate_manifest(engine: Any, _prompts: list[str], base_config: Any) -> int:
        prompt_sets: dict[str, list[str]] = {}
        for case_index, case in enumerate(manifest["cases"]):
            prompt_sets[case["id"]] = [
                build_prompt(
                    engine.tokenizer,
                    int(case["input_tokens"]),
                    int(manifest["seed"]) + case_index * 1000 + request_index,
                )
                for request_index in range(int(case["requests"]))
            ]
        metadata = {
            "schema_version": 1,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "prompt_sha256": {
                case_id: [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]
                for case_id, prompts in prompt_sets.items()
            },
        }
        output.with_suffix(output.suffix + ".meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        total = 0
        with output.open("x", encoding="utf-8") as stream:
            phases = (
                ("warmup", int(manifest["warmup_rounds"])),
                ("measure", int(manifest["measure_rounds"])),
            )
            for phase, rounds in phases:
                for case in manifest["cases"]:
                    config = replace(
                        base_config,
                        max_new_tokens=int(case["output_tokens"]),
                        stream=True,
                        temperature=0.0,
                        top_p=1.0,
                        ignore_eos=True,
                    )
                    for round_index in range(1, rounds + 1):
                        rows = await run_case_round(
                            engine,
                            case,
                            prompt_sets[case["id"]],
                            config,
                            phase,
                            round_index,
                        )
                        for request_index, row in enumerate(rows, start=1):
                            record = row_with_context(
                                row, case, phase, round_index, request_index
                            )
                            stream.write(json.dumps(record, sort_keys=True) + "\n")
                            total += int(record["token_count"])
                        stream.flush()
                        print(
                            f"[stream-benchmark {phase}] case={case['id']} "
                            f"round={round_index} requests={len(rows)}",
                            flush=True,
                        )
        return total

    cli._generate_prompts = generate_manifest
    sys.argv = ["pypto-serving", *command]
    cli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
