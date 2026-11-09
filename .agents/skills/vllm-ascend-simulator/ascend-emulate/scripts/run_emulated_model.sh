#!/usr/bin/env bash
set -euo pipefail

# Launch vLLM with a generated Ascend emulator preloaded.  Compute is emulated,
# large (>= ASCEND_EMULATOR_VIRTUAL_MIN_BYTES) device allocations are reserved virtual
# addresses, and device copies are emulated, so a
# request can run end-to-end without real weights, real KV-cache HBM, or
# meaningful output tokens.
GENERATED=${1:?usage: run_emulated_model.sh /path/to/generated model [vllm args...]}
MODEL=${2:?usage: run_emulated_model.sh /path/to/generated model [vllm args...]}
shift 2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INTERPOSER=${INTERPOSER:-$GENERATED/libascend_emulator.so}
PORT=${VLLM_PORT:-30000}
DEVICE=${ASCEND_RT_VISIBLE_DEVICES:-0}
SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL")}
# Eager vs graph mode. ASCEND_EMULATOR_EAGER=0 (or EAGER=0) drops --enforce-eager so
# vLLM can attempt NPU graph capture; the default stays eager because the
# emulator's per-operator virtual timing works at the eager launch layer.
if [[ "${ASCEND_EMULATOR_EAGER:-${EAGER:-1}}" == "1" ]]; then
    EAGER_FLAG="--enforce-eager"
else
    EAGER_FLAG=""
fi

if [[ ! -r "$INTERPOSER" ]]; then
    "$SCRIPT_DIR/build_emulator.sh" "$GENERATED" "$INTERPOSER"
fi

export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export LD_PRELOAD="$INTERPOSER"
export ASCEND_EMULATOR_INTERCEPT_DLSYM=${ASCEND_EMULATOR_INTERCEPT_DLSYM:-1}
export ASCEND_EMULATOR_EMULATE=${ASCEND_EMULATOR_EMULATE:-1}
export ASCEND_EMULATOR_STRICT=${ASCEND_EMULATOR_STRICT:-1}
export ASCEND_EMULATOR_VIRTUAL_MEMORY=${ASCEND_EMULATOR_VIRTUAL_MEMORY:-1}
export ASCEND_EMULATOR_ALLOW_PASSTHROUGH=${ASCEND_EMULATOR_ALLOW_PASSTHROUGH:-0}
# Allocations of at least this many bytes are reserved virtual addresses
# (compute buffers); smaller buffers stay real (metadata/sampler buffers).
export ASCEND_EMULATOR_VIRTUAL_MIN_BYTES=${ASCEND_EMULATOR_VIRTUAL_MIN_BYTES:-4194304}
# Outside virtual-memory mode, copies at least this large are emulated so
# weight loading never touches HBM. Virtual-memory mode emulates every device
# copy and synthesizes zero-filled device-to-host results.
export ASCEND_EMULATOR_MEMCPY_SKIP_MIN_BYTES=${ASCEND_EMULATOR_MEMCPY_SKIP_MIN_BYTES:-65536}
EMULATOR_ROOT=$(cd "$GENERATED/.." && pwd)
if [[ -z "${ASCEND_EMULATOR_PROFILE_MANIFEST:-}" && -r "$EMULATOR_ROOT/profile_manifest.json" ]]; then
    export ASCEND_EMULATOR_PROFILE_MANIFEST="$EMULATOR_ROOT/profile_manifest.json"
fi
if [[ -z "${ASCEND_EMULATOR_LATENCY_MODEL:-}" && -r "$EMULATOR_ROOT/model_manifest.json" ]]; then
    export ASCEND_EMULATOR_LATENCY_MODEL="$EMULATOR_ROOT/model_manifest.json"
fi

exec vllm serve "$MODEL" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "${TP_SIZE:-1}" \
    --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN:-128}" \
    --max-num-seqs "${MAX_NUM_SEQS:-1}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.50}" \
    ${EAGER_FLAG:-} "$@"
