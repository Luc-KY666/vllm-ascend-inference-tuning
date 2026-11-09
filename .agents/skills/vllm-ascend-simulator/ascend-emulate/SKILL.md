---
name: ascend-emulate
description: Generate and build transparent ACLNN, HCCL, memory, and virtual-time emulation modules from profile/model manifests.
---

# Ascend Transparent Emulation

Read `../README.md` first for the pipeline contract and responsibility
boundaries shared by the simulator skills.

Use this skill after `ascend-profile` and `ascend-latency-model`. It generates
a CANN-version-specific `LD_PRELOAD` library from the observed API inventory;
do not reuse a generated library across CANN versions without regenerating and
revalidating it.

## Generate, build, and check coverage

```bash
python .agents/skills/vllm-ascend-simulator/ascend-emulate/scripts/generate_emulator.py \
  --profile-manifest /tmp/ascend-profile/profile_manifest.json \
  --model-manifest /tmp/ascend-model/model_manifest.json \
  --output /tmp/ascend-emulator

bash .agents/skills/vllm-ascend-simulator/ascend-emulate/scripts/build_emulator.sh \
  /tmp/ascend-emulator/generated

python .agents/skills/vllm-ascend-simulator/ascend-emulate/scripts/coverage_report.py \
  --profile-manifest /tmp/ascend-profile/profile_manifest.json \
  --model-manifest /tmp/ascend-model/model_manifest.json \
  --emulator-manifest /tmp/ascend-emulator/generated/emulator_manifest.json \
  --output /tmp/ascend-emulator/coverage.json
```

The build script compiles a release library. Treat `unsupported`, missing
headers, ambiguous prototypes, and low-confidence/missing models as coverage
failures in strict mode. An internal execute symbol may use the common ACLNN
execute ABI even when its workspace prototype is unavailable; that workspace
entry remains an explicit passthrough note.

## Run

```bash
ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_EMULATOR_TIME_MODE=sleep \
  bash .agents/skills/vllm-ascend-simulator/ascend-emulate/scripts/run_emulated_model.sh \
  /tmp/ascend-emulator/generated /path/to/model
```

Important variables:

- `ASCEND_EMULATOR_EAGER=1` is eager mode; `0` enables the vLLM graph path.
- `ASCEND_EMULATOR_TIME_MODE=jump` accumulates predicted work without waiting.
- `ASCEND_EMULATOR_TIME_MODE=sleep` advances per-stream virtual clocks and
  waits only at stream/device/event synchronization points.
- `ASCEND_EMULATOR_STRICT=1` and
  `ASCEND_EMULATOR_ALLOW_PASSTHROUGH=0` reject incomplete compute coverage.
- `ASCEND_EMULATOR_VIRTUAL_MEMORY=1` virtualizes large physical allocations,
  suppresses device kernels, and emulates device copies. Device-to-host copies
  synthesize zero data so upper-layer control flow is deterministic.
- `ASCEND_EMULATOR_VIRTUAL_MIN_BYTES` defaults to 4 MiB. Smaller allocations
  stay real for runtime metadata.
- `ASCEND_EMULATOR_SHAPE_LOOKUP=1` uses shape-aware latency tables. Disable it
  only as a diagnostic median-latency baseline.
- `ASCEND_EMULATOR_LOG_OVERHEAD=1` enables expensive wrapper diagnostics;
  never use diagnostic runs as performance results.

## Runtime model

ACLNN workspace wrappers do not call the real tiling path while emulating.
They read only the tensor dimensions needed to distinguish observed latency
rows, predict once, and encode the duration in a tagged fake executor. Execute
wrappers decode that duration and submit it to virtual time. Tensor shapes use
a small thread-local cache; a cache miss falls back to `aclGetViewShape`.

The generated latency table exact-matches observed feature rows and uses a
two-nearest inverse-distance interpolation on a miss. Symbols without a shape
table use the observed median. Predictions are made per exported symbol, so
the hot execute path has no global operator-name dispatch.

Graph mode uses the CANN runtime interfaces `aclmdlRICaptureBegin`,
`aclmdlRICaptureEnd`, `aclmdlRIExecuteAsync`, and `aclmdlRIExecute`. Work seen
during capture is stored per graph and submitted again on every replay. Do not
substitute unrelated graph API names based on log wording.

Virtual memory reserves real virtual addresses but does not commit large HBM
backing. Kernel launches are suppressed, device copies are emulated, and
device-to-host reads return deterministic zeros. The library is for control
flow and timing only; it does not promise numerically correct tensors or model
text.

## Validation

For every target model/CANN/vLLM combination:

1. Confirm coverage reports `strict_ready=true` and no unsupported observed
   API.
2. Run eager and graph modes with identical server and request parameters.
3. Use at least three warmups, a fixed request seed, and a native run adjacent
   to each simulated run. Report average, p50, p95, min/max, and request-order
   drift; do not reuse historical baselines.
4. Test both `jump` and `sleep`. `jump` wall time measures interceptor/control
   overhead, not predicted device time.
5. In graph diagnostics, require nonzero capture and replay counters. A graph
   request returning HTTP 200 is not proof that replay timing was modeled.
6. Verify HBM remains near runtime baseline and logs contain no real kernel
   touching virtual memory, invalid token IDs, or silent passthrough.

The 5% target is a measured acceptance threshold, not a property of the
design. If a mode misses it, preserve the failing evidence and isolate host
overhead, latency-model error, and control-flow divergence separately.

## Prerequisites

- CANN headers and libraries matching the target `torch_npu`.
- A C++17 compiler, `dl`, and `pthread`.
- An Ascend/vLLM environment for integration validation.
