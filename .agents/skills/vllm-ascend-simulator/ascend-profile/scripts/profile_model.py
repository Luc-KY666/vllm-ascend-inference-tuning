#!/usr/bin/env python3
"""Run a configurable Ascend profile runner and publish a manifest.

The collection step is driven by a ``--profile-runner`` script.  The script is
invoked through bash with a standard set of environment variables (see
SKILL.md); any additional variables are passed via ``--runner-env``.  The
manifest normalization step is runner-agnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MANIFEST_SCRIPT = Path(__file__).resolve().with_name("profile_manifest.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--hardware", default="Ascend")
    parser.add_argument("--cann-version", default="unknown")
    parser.add_argument("--torch-npu-version", default="unknown")
    parser.add_argument("--vllm-version", default="unknown")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--phase", choices=("prefill", "decode", "mixed", "unknown"), default="unknown")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-tokens", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--profile-group", default="dense")
    parser.add_argument(
        "--fixed-shot-size",
        type=int,
        help=(
            "Collect one deterministic workload point instead of the first "
            "profile grid point. For dense this is packed token count; for "
            "per_sequence this is sequence count."
        ),
    )
    parser.add_argument(
        "--attention-shot",
        help="JSON object with prefill_chunk, kv_prefill, n_decode, kv_decode",
    )
    parser.add_argument("--max-batched-tokens", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--profile-preload", type=Path)
    parser.add_argument("--profile-parent", type=Path, default=Path("/tmp/ascend_ascend_profiles"))
    parser.add_argument(
        "--profile-runner",
        type=Path,
        default=Path(os.environ.get("ASCEND_PROFILE_RUNNER", "")),
        help="bash script that performs one collection run (required unless --profile-root is used)",
    )
    parser.add_argument(
        "--runner-env",
        help="JSON object of extra environment variables passed to --profile-runner",
    )
    parser.add_argument("--profile-root", type=Path, help="Use existing profile artifacts; skip collection")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--workload-id", default="")
    parser.add_argument("--phase-source", default="declared")
    parser.add_argument("--topology", default="")
    parser.add_argument("--allow-profile-failure", action="store_true")
    return parser.parse_args()


def latest_raw_profile(root: Path) -> Path:
    candidates = [path for path in root.rglob("kernel_details.csv")]
    if not candidates:
        raise FileNotFoundError(f"no kernel_details.csv under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def run_collection(args: argparse.Namespace) -> Path:
    script = Path(args.profile_runner).resolve()
    if not script.is_file():
        raise FileNotFoundError(
            f"--profile-runner not found: {script} (set ASCEND_PROFILE_RUNNER or pass --profile-runner)"
        )
    env = os.environ.copy()
    env.update({
        "SOURCE_MODEL": str(args.model.resolve()),
        "REDUCED_MODEL": str((args.output / "reduced_model").resolve()),
        "REDUCED_LAYERS": str(args.layers),
        "PROFILE_TP": str(args.tp),
        "HARDWARE": args.hardware,
        "ASCEND_RT_VISIBLE_DEVICES": args.devices,
        "PROFILE_GROUP": args.profile_group,
        "PROFILE_ITERATIONS": str(args.iterations),
        "PROFILE_MAX_TOKENS": str(args.max_batched_tokens),
        "PROFILE_MAX_SEQS": str(args.max_num_seqs),
        "PROFILE_GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
        "PROFILE_PARENT": str(args.profile_parent.resolve()),
        "RESULTS_DIR": str((args.output / "profile_results").resolve()),
    })
    if args.fixed_shot_size is not None:
        if args.fixed_shot_size < 1:
            raise ValueError("--fixed-shot-size must be positive")
        env["PROFILE_FIXED_SHOT_SIZE"] = str(args.fixed_shot_size)
    if args.attention_shot:
        shot = json.loads(args.attention_shot)
        required = ("prefill_chunk", "kv_prefill", "n_decode", "kv_decode")
        if any(key not in shot for key in required):
            raise ValueError(f"--attention-shot requires {required}")
        env["PROFILE_ATTENTION_SHOT"] = json.dumps(
            {key: int(shot[key]) for key in required}
        )
    if args.profile_preload:
        env["PROFILE_PRELOAD"] = str(args.profile_preload.resolve())
    if args.runner_env:
        env.update(json.loads(args.runner_env))
    args.output.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(["bash", str(script)], env=env, cwd=script.parent)
    if completed.returncode and not args.allow_profile_failure:
        raise RuntimeError(f"Ascend profile command failed with rc={completed.returncode}")
    return latest_raw_profile(args.output / "profile_results")


def validate_phase(args: argparse.Namespace) -> None:
    if args.phase in {"decode", "mixed"} and not args.attention_shot:
        raise ValueError("decode/mixed profiling requires --attention-shot with KV history")
    if not args.attention_shot:
        return
    shot = json.loads(args.attention_shot)
    prefill = int(shot["prefill_chunk"]) > 0
    decode = int(shot["n_decode"]) > 0
    actual = "mixed" if prefill and decode else ("prefill" if prefill else "decode")
    if args.phase not in {"unknown", actual}:
        raise ValueError(f"phase {args.phase!r} does not match attention shot phase {actual!r}")


def main() -> int:
    args = parse_args()
    validate_phase(args)
    args.output = args.output.resolve()
    profile_root = args.profile_root.resolve() if args.profile_root else run_collection(args)
    phase_source = args.phase_source
    if phase_source == "declared" and (args.attention_shot or args.fixed_shot_size is not None):
        phase_source = "shot"
    command = [
        sys.executable, str(MANIFEST_SCRIPT),
        "--profile-root", str(profile_root),
        "--output", str(args.output),
        "--model-id", args.model_id,
        "--source-model", str(args.model.resolve()),
        "--reduced-layers", str(args.layers),
        "--hardware", args.hardware,
        "--cann-version", args.cann_version,
        "--torch-npu-version", args.torch_npu_version,
        "--vllm-version", args.vllm_version,
        "--world-size", str(args.tp),
        "--rank", str(args.rank),
        "--phase", args.phase,
        "--batch-size", str(args.batch_size),
        "--input-tokens", str(args.input_tokens),
        "--output-tokens", str(args.output_tokens),
        "--sample-group", f"{args.phase}-b{args.batch_size}-i{args.input_tokens}-o{args.output_tokens}",
        "--sample-id", args.sample_id or f"{args.phase}-b{args.batch_size}-i{args.input_tokens}-o{args.output_tokens}",
        "--workload-id", args.workload_id or f"{args.phase}-b{args.batch_size}-i{args.input_tokens}-o{args.output_tokens}",
        "--phase-source", phase_source,
        "--topology", args.topology or f"tp{args.tp}",
    ]
    if args.attention_shot:
        shot = json.loads(args.attention_shot)
        for key, flag in (("prefill_chunk", "--prefill-chunk"),
                          ("kv_prefill", "--kv-prefill"),
                          ("n_decode", "--n-decode"),
                          ("kv_decode", "--kv-decode")):
            command.extend([flag, str(int(shot[key]))])
    subprocess.run(command, check=True)
    print(f"profile_root: {profile_root}")
    print(f"manifest: {args.output / 'profile_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
