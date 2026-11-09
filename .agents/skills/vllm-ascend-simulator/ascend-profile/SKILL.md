---
name: ascend-profile
description: Collect and normalize reduced-model Ascend/vLLM profiles into standard JSON and CSV manifests.
---

# Ascend Profile

Read `../README.md` first for the pipeline contract and responsibility
boundaries shared by the simulator skills.

Use this skill when a model needs a small, reproducible Ascend profile before
latency modeling or emulation generation. It accepts schema-v1 inputs and
emits schema-v2 manifests with explicit workload and communication metadata.

## Workflow

1. Use `scripts/profile_model.py` with a `--profile-runner` script to run one
   reduced-model Ascend/vLLM profile and normalize it. Use `--profile-root`
   when the trace already exists. The runner receives a standard environment
   contract (`SOURCE_MODEL`, `REDUCED_MODEL`, `REDUCED_LAYERS`, `PROFILE_TP`,
   `ASCEND_RT_VISIBLE_DEVICES`, `PROFILE_GROUP`, `PROFILE_ITERATIONS`,
   `PROFILE_MAX_TOKENS`, `PROFILE_MAX_SEQS`, `PROFILE_PARENT`, `RESULTS_DIR`,
   and optionally `PROFILE_FIXED_SHOT_SIZE` / `PROFILE_ATTENTION_SHOT` /
   `PROFILE_PRELOAD`); pass any runner-specific variables with
   `--runner-env '{"KEY":"value"}'`. An existing runner such as
   `qwen3-8b-emulate/profile_qwen3_reduced_ascend.sh` may be passed the same
   way as any other script.
2. Publish `profile_manifest.json`, `coverage.json`, and category CSV files
   under the requested output directory.
3. For multiple points, use `scripts/run_profile_sweep.py` with a plan created
   by `ascend-latency-model/scripts/build_profile_sweep.py`. Every workload gets
   its own directory and raw artifacts; failed workloads remain in
   `collection_index.json`.
4. For real TP2 communication, run `scripts/profile_hccl.py` with
   `torchrun --master_addr=127.0.0.1 --master_port=29500 --nproc_per_node=2`.
   It measures HCCL directly and never synthesizes communication durations.

Example using a pre-existing trace (no runner needed):

```bash
python .agents/skills/vllm-ascend-simulator/ascend-profile/scripts/profile_model.py \
  --model /path/to/source/model \
  --profile-root /path/to/raw/run.MvQwoU \
  --output /tmp/ascend-profile
```

Example collection on one NPU:

```bash
python .agents/skills/vllm-ascend-simulator/ascend-profile/scripts/profile_model.py \
  --model /path/to/source/model \
  --profile-runner /path/to/profile_runner.sh \
  --layers 2 --tp 1 --devices 0 --output /tmp/ascend-profile
```

Use TP2 and two visible devices for communication coverage. A TP1 profile is
not evidence that HCCL is absent.

`--max-batched-tokens`, `--max-num-seqs`, `--gpu-memory-utilization`,
`--iterations`, and `--profile-preload` are passed through to the runner.
Runners are expected to produce a `kernel_details.csv` somewhere under the
`RESULTS_DIR`/`PROFILE_PARENT` tree; `profile_model.py` locates the newest one
and normalizes it.

For a deterministic batch/token point, pass `--fixed-shot-size`. With the
`dense` group the value is the packed token count; with `per_sequence` it is
the number of one-token sequences. This is useful when strict Ascend decoder
attribution stops a larger sweep at its first failed shot. Dense fixed shots
measure token scaling and should not be interpreted as a concurrent-sequence
HCCL measurement.

For attention, pass `--profile-group attention --attention-shot
'{"prefill_chunk":32,"kv_prefill":512,"n_decode":1,"kv_decode":512}'`.
The four values distinguish pure prefill, pure decode, and mixed shots and are
copied into every event row. A decode shot is a one-token increment with KV
history, not a metadata-only phase label.

## Contract

The generated manifest is schema version 2 (readers remain compatible with
version 1). Each event row preserves input and
output shapes, dtype/format, stream, duration, source file, and matching
confidence, sample/workload IDs, phase source, attention axes, and collective
metadata. `metadata` rows are kept separate from `compute`, `communication`,
and `memory` rows. Decoder warnings are retained in `diagnostics` and
`coverage.json` instead of being silently discarded.

### Device-only latency events

Only real device work enters `compute`, `memory`, and `communication` events:

- `compute` comes exclusively from `kernel_details.csv` device kernels (AI
  Core/Vector/MIX kernels such as `MatMulV2_*`, `FusedInferAttentionScore`,
  `RmsNorm_*`).
- `memory` and `communication` come from device kernels plus
  non-host memory copy/set operators in `operator_details.csv` whose
  `Device Total Duration(us)` is positive, and from memory-management or
  collective runtime APIs in `api_statistic.csv` (for example `aclrtMalloc`,
  `aclrtFree`, `aclrtMemcpyAsync`, `aclrtMemset`, `MemcpyAsync`).
- CPU-side dispatch operators (`aten::*`, `atb::*`, `torch::*`, `launch`, ...)
  and host runtime APIs are never merged into the latency events. The raw
  `api_statistic.csv` rows are preserved only in `events/api_calls.csv` as an
  auxiliary artifact.

The manifest `coverage` block records how many host API rows were excluded
(`cpu_side_operators_excluded` is always true) so consumers can verify that
no CPU-side operator polluted a measurement.

## Prerequisites

- Python 3.10+.
- For collection: CANN, `torch_npu`, vLLM Ascend, an Ascend device, and a
  `--profile-runner` script that emits an Ascend profiler output directory
  (with `kernel_details.csv`). The runner may use any model or serving stack.
- For normalization of an existing trace: only Python standard-library modules.
