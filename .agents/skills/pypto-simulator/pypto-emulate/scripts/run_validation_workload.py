#!/usr/bin/env python3
"""Run sequential warmup and measured requests through one pypto-serving engine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--requests", type=int, default=5)
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("pypto-serving arguments are required after --")
    if args.warmup_requests < 0 or args.requests < 1:
        parser.error("warmup requests must be non-negative and measured requests positive")
    return args, command


def _result_row(phase: str, index: int, elapsed_s: float, result: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": phase,
        "request_index": index,
        "wall_ms": elapsed_s * 1000.0,
        "token_ids": [int(value) for value in result.token_ids],
        "token_count": len(result.token_ids),
        "finish_reason": str(result.finish_reason),
        "text": str(result.text),
    }


def main() -> int:
    args, command = parse_args()
    output = args.output.resolve()
    if output.exists():
        print(f"error: refusing existing output: {output}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)

    from pypto_serving.cli import main as cli

    expected = args.warmup_requests + args.requests

    async def generate_sequential(engine, prompts, generate_config):
        if len(prompts) != expected:
            raise RuntimeError(f"expected {expected} prompts, got {len(prompts)}")
        total = 0
        with output.open("x", encoding="utf-8") as stream:
            for offset, prompt in enumerate(prompts):
                phase = "warmup" if offset < args.warmup_requests else "measure"
                index = offset + 1 if phase == "warmup" else offset - args.warmup_requests + 1
                started = time.perf_counter()
                results = await engine.generate_batch([prompt], generate_config)
                elapsed = time.perf_counter() - started
                if len(results) != 1:
                    raise RuntimeError(f"expected one result, got {len(results)}")
                row = _result_row(phase, index, elapsed, results[0])
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                total += row["token_count"]
                print(
                    f"[validation {phase} {index}] wall={row['wall_ms']:.3f} ms "
                    f"tokens={row['token_count']} finish={row['finish_reason']}",
                    flush=True,
                )
        return total

    cli._generate_prompts = generate_sequential
    sys.argv = ["pypto-serving", *command]
    cli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
