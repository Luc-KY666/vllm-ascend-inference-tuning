#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENERATED=${1:?usage: build_emulator.sh /path/to/generated [output.so]}
OUTPUT=${2:-$GENERATED/libascend_emulator.so}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.1}

test -r "$GENERATED/CMakeLists.txt" || {
    printf 'generated CMake project not found: %s\n' "$GENERATED" >&2
    exit 1
}

g++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -fPIC -shared \
    "$GENERATED/compute_interposer.cpp" \
    "$GENERATED/communication_interposer.cpp" \
    "$GENERATED/memory_interposer.cpp" \
    "$GENERATED/device_interposer.cpp" \
    "$GENERATED/latency_table.cpp" \
    "$GENERATED/virtual_time.cpp" \
    -I"$ASCEND_HOME_PATH/aarch64-linux/include" \
    -ldl -pthread -o "$OUTPUT"
printf 'emulator: %s\n' "$OUTPUT"
