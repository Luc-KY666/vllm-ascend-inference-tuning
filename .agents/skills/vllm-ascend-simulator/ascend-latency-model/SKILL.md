---
name: ascend-latency-model
description: Build batch/shape sweeps and fit per-operator compute, communication, and memory latency models.
---

# Ascend Latency Model

Read `../README.md` first for the pipeline contract and responsibility
boundaries shared by the simulator skills.

Use this skill after `ascend-profile` has produced one or more profile
manifests. For every operator group the fitter first writes a per-operator
collection CSV (`output/collections/<kind>__<op_name>__<hash>.csv`) recording
the measured latency under each observed input shape, then trains a
scikit-learn random forest directly from that collection table. The default
backend is scikit-learn random forest; low-sample groups are retained as
explicit low-confidence median models.

## Commands

Create a deterministic sweep plan. Pass multiple `--manifest` values to union
the observed operator inventory. The output is executable and contains no
synthetic durations:

```bash
python .agents/skills/vllm-ascend-simulator/ascend-latency-model/scripts/build_profile_sweep.py \
  --manifest /tmp/ascend-profile/profile_manifest.json \
  --model /path/to/source/model \
  --profile-runner /path/to/profile_runner.sh \
  --output /tmp/ascend-sweep.json
```

Execute it on Ascend with the profile skill:

```bash
python .agents/skills/vllm-ascend-simulator/ascend-profile/scripts/run_profile_sweep.py \
  --plan /tmp/ascend-sweep.json \
  --output /tmp/ascend-collection \
  --allow-failures
```

Fit models from one or more manifests:

```bash
python .agents/skills/vllm-ascend-simulator/ascend-latency-model/scripts/fit_latency_models.py \
  --manifest /tmp/ascend-profile/profile_manifest.json \
  --output /tmp/ascend-model
```

Predict and validate:

```bash
python .agents/skills/vllm-ascend-simulator/ascend-latency-model/scripts/predict_latency.py \
  --model-manifest /tmp/ascend-model/model_manifest.json \
  --event-json '{"kind":"compute","op_name":"MatMulV2","world_size":1,"duration_us":0}'

# Compact median lookup for a runtime without Python/joblib:
python .agents/skills/vllm-ascend-simulator/ascend-latency-model/scripts/predict_latency.py \
  --model-manifest /tmp/ascend-model/model_manifest.json \
  --lookup /tmp/ascend-model/latency_lookup.json \
  --event-json '{"kind":"compute","op_name":"MatMulV2","world_size":1,"duration_us":0}'

python .agents/skills/vllm-ascend-simulator/ascend-latency-model/scripts/validate_latency_model.py \
  --model-manifest /tmp/ascend-model/model_manifest.json
```

Fast restore without refitting (loads the saved joblib bundle only):

```bash
python .agents/skills/vllm-ascend-simulator/ascend-latency-model/scripts/restore_latency_model.py \
  --model-manifest /tmp/ascend-model/model_manifest.json \
  --output /tmp/ascend-model/restored \
  --event-json '{"kind":"compute","op_name":"MatMulV2","world_size":1,"duration_us":0}'
```

`restore_latency_model.py` uses only `joblib.load` and `predict`, validates the
saved estimators against `collections/*.csv`, and writes
`restored_metrics.csv` / `restored_manifest.json`. The sweep plan includes
prefill, KV-history decode, explicit attention shots, and a real TP2 HCCL
workload; it never treats synthetic durations as hardware measurements.

## Model contract

Models are grouped by event kind, normalized operator/API name, hardware,
world size, phase, and dtype. Communication groups additionally include the
collective and topology. Numeric features include shape dimensions,
numel, M/K/N when available, batch/tokens, message/allocation/copy bytes,
stream, dtype/format codes, rank, attention axes, and collective codes. Each
group has a `joblib` artifact plus JSON metadata, validation metrics, and
coverage ranges.

Every group also has a collection CSV under `collections/`. The fitter reads
only the three device latency event kinds produced by `ascend-profile`
(`compute`, `memory`, `communication`); metadata and CPU-side API rows are
ignored. Training flows manifest rows → per-operator collection table →
random forest, so model quality is auditable shape-by-shape. Groups whose
measured duration spans more than ~5x (for example communication across
message sizes) are fitted on log scale to keep relative error meaningful;
the payload records `target_transform` and `predict_latency.py` inverts it
automatically.

The fitter also writes `latency_lookup.json`. It is a compact, median-based
exact-group lookup for C++/shell consumers; RF predictions remain available
through the Python API. Models trained from one workload are marked
`low_confidence` because their validation split is not independent.

The model manifest is `status=incomplete` when groups lack independent
workloads, exceed the MAPE threshold, or are unresolved. Use
`validate_latency_model.py --require-complete` and the matching predictor
`--strict` flag to reject incomplete or low-confidence bundles. Strict
emulation must reject low-confidence or missing groups unless the user
explicitly enables passthrough.

## Dependencies

`fit_latency_models.py` requires `scikit-learn`, `numpy`, and `joblib`. Profile
manifest parsing and sweep generation use only the Python standard library.
