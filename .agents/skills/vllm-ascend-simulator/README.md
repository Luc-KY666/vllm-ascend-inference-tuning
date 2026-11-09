# vLLM Ascend Simulator Skills

This directory contains the profiling, latency-modeling, and runtime-emulation
skills used to simulate vLLM workloads on Ascend. The skills form one pipeline
but remain independently discoverable by agents.

## Directory layout

```text
vllm-ascend-simulator/
├── README.md
├── ascend-common/          Shared manifest and feature helpers; not a skill
├── ascend-profile/         Collect and normalize real Ascend measurements
├── ascend-latency-model/   Fit and validate per-operator latency models
└── ascend-emulate/         Generate and run the LD_PRELOAD simulator
```

Codex and Pi discover the three nested `SKILL.md` files recursively from
`.agents/skills/`. `README.md` is an overview, not an additional skill.

## Responsibility boundaries

### `ascend-profile`

Collects reduced-model or workload-sweep profiles from real Ascend hardware
and normalizes them into schema-versioned JSON/CSV manifests. It separates
device compute, communication, and memory work from host-side API metadata.
It does not train models or synthesize missing device measurements.

Primary output: `profile_manifest.json`, event CSV files, coverage, and
diagnostics.

### `ascend-latency-model`

Consumes one or more profile manifests, writes auditable per-operator
collection tables, and fits or restores compute, communication, and memory
latency predictors. It owns model validation, confidence status, coverage
ranges, and compact runtime lookup data. It does not collect hardware traces
or implement runtime interception.

Primary output: `model_manifest.json`, `collections/`, `models/`, validation
metrics, and `latency_lookup.json`.

### `ascend-emulate`

Consumes the profile and model manifests, inspects the matching CANN headers,
and generates a release-built shared library that intercepts ACLNN, HCCL,
memory, synchronization, and graph replay interfaces. It owns virtual memory,
shape-aware runtime lookup, virtual time, strict coverage checks, and
end-to-end vLLM validation. It does not promise numerically correct tensors or
model text.

Primary output: `generated/libascend_emulator.so`,
`emulator_manifest.json`, and `coverage_report.json`.

### `ascend-common`

Contains the shared event schema, feature extraction, and key-normalization
helpers used by the profile and model scripts. It has no `SKILL.md` because it
is an implementation dependency rather than a user-selectable workflow.

## Recommended workflow

```bash
# 1. Collect and normalize real device measurements.
python .agents/skills/vllm-ascend-simulator/ascend-profile/scripts/profile_model.py \
  --model /path/to/model \
  --profile-runner /path/to/profile_runner.sh \
  --layers 2 --tp 1 --devices 0 \
  --output /tmp/ascend-profile

# 2. Fit and validate latency models.
python .agents/skills/vllm-ascend-simulator/ascend-latency-model/scripts/fit_latency_models.py \
  --manifest /tmp/ascend-profile/profile_manifest.json \
  --output /tmp/ascend-model

# 3. Generate and build the runtime simulator.
python .agents/skills/vllm-ascend-simulator/ascend-emulate/scripts/generate_emulator.py \
  --profile-manifest /tmp/ascend-profile/profile_manifest.json \
  --model-manifest /tmp/ascend-model/model_manifest.json \
  --output /tmp/ascend-emulator

bash .agents/skills/vllm-ascend-simulator/ascend-emulate/scripts/build_emulator.sh \
  /tmp/ascend-emulator/generated

# 4. Check strict coverage before running vLLM.
python .agents/skills/vllm-ascend-simulator/ascend-emulate/scripts/coverage_report.py \
  --profile-manifest /tmp/ascend-profile/profile_manifest.json \
  --model-manifest /tmp/ascend-model/model_manifest.json \
  --emulator-manifest /tmp/ascend-emulator/generated/emulator_manifest.json \
  --output /tmp/ascend-emulator/coverage.json
```

Do not skip directly from an incomplete profile to runtime emulation. A
successful HTTP response is not proof of timing coverage: require strict
coverage, compare eager and graph modes with adjacent native baselines, and
confirm graph capture/replay counters when graph mode is enabled.

Current benchmark evidence and unresolved limitations are recorded in the
repository-level `ASCEND_BENCHMARK.md` and `ASCEND_HANDOVER.md`.
