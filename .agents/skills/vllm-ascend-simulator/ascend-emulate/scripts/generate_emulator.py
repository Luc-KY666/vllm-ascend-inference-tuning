#!/usr/bin/env python3
"""Generate a version-specific Ascend ACLNN/HCCL/allocator emulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

COMMON = Path(__file__).resolve().parents[2] / "ascend-common"
sys.path.insert(0, str(COMMON))
from ascend_contract import decode_json_field, shape_numel  # noqa: E402


EXECUTE_SUFFIX = "GetWorkspaceSize"
KNOWN_HCCL = (
    "HcclAllReduce", "HcclBroadcast", "HcclReduceScatter", "HcclAllGather",
    "HcclSend", "HcclRecv",
)


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def api_stem(name: str) -> str:
    """Return the ACLNN API name encoded as the first token of a fused kernel
    name (``aclnnMatmul_MatMulCommon_MatMulV2`` -> ``aclnnMatmul``)."""
    if name.startswith("aclnn"):
        return name.split("_", 1)[0]
    return name


def hccl_alias(name: str) -> list[str]:
    """Map a profiled collective name to the corresponding HCCL ABI symbol
    (``all_reduce`` -> ``HcclAllReduce``).  This is purely name-derived and
    carries no model-specific knowledge."""
    tokens = [token for token in str(name or "").split("_") if token]
    if tokens == ["send", "recv"]:
        return ["HcclSend", "HcclRecv"]
    if not tokens:
        return []
    return ["Hccl" + "".join(token.capitalize() for token in tokens)]


def match_api_symbols(op_name: str, api_symbols: list[str]) -> list[str]:
    """Generic kernel-name to API-symbol alias discovery.

    Returns every observed API symbol whose normalized identifier contains
    the operator name or is contained by it.  This links ``RmsNorm`` to
    ``aclnnRmsNorm``, ``FusedInferAttentionScore`` to
    ``aclnnInnerFusedInferAttentionScore``, and ``all_reduce`` to
    ``torch.distributed.all_reduce`` without hardcoding any operator.
    """
    op_norm = _normalize_identifier(op_name)
    if not op_norm:
        return []
    matches = []
    for symbol in api_symbols:
        sym_norm = _normalize_identifier(symbol)
        if sym_norm and (op_norm in sym_norm or sym_norm in op_norm):
            matches.append(symbol)
    return matches


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def event_symbols(profile_manifest: Path) -> tuple[list[str], list[str]]:
    manifest = load_json(profile_manifest)
    symbols: set[str] = set()
    api_path = manifest.get("events", {}).get("api_calls", {}).get("path")
    candidates = [Path(api_path)] if api_path else []
    candidates.extend(profile_manifest.parent.glob("events/*.csv"))
    for path in candidates:
        if not path.is_absolute():
            path = profile_manifest.parent / path
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                symbol = row.get("api_symbol", "").strip()
                if symbol:
                    symbols.add(symbol)
                op_name = row.get("op_name", "").strip()
                if op_name.startswith("aclnn"):
                    symbols.add(api_stem(op_name))
    aclnn = sorted(symbol for symbol in symbols if symbol.startswith("aclnn"))
    hccl = sorted(symbol for symbol in symbols if symbol in KNOWN_HCCL or symbol.upper().startswith("HCCL"))
    return aclnn, hccl


def find_header(symbol: str, include_root: Path) -> tuple[str | None, str | None]:
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\s*\(")
    for path in include_root.rglob("*.h"):
        if "aclnnop" not in path.parts and "hccl" not in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            relative = path.relative_to(include_root).as_posix()
            return relative, text
    return None, None


def prototype(symbol: str, text: str) -> tuple[str, list[str]] | None:
    match = re.search(r"\b" + re.escape(symbol) + r"\s*\((.*?)\)\s*;", text, re.S)
    if not match:
        return None
    body = re.sub(r"/\*.*?\*/|//[^\n]*", "", match.group(1), flags=re.S)
    params: list[str] = []
    current: list[str] = []
    depth = 0
    for char in body:
        if char in "(<[":
            depth += 1
        elif char in ")>]":
            depth -= 1
        if char == "," and depth == 0:
            params.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current and "".join(current).strip():
        params.append("".join(current).strip())
    cleaned = []
    names = []
    for index, param in enumerate(params):
        param = re.sub(r"\bACLNN_API\b", "", param).strip()
        name_match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^]]*\])?$", param)
        if name_match:
            name = name_match.group(1)
            type_text = param[:name_match.start(1)].strip()
        else:
            name = f"arg{index}"
            type_text = param
        if not type_text:
            return None
        cleaned.append(type_text)
        names.append(name)
    return cleaned, names


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _entry_names(entry: dict[str, Any], api_symbols: list[str]) -> set[str]:
    """All latency-lookup names a model group can satisfy."""
    parts = str(entry.get("group_key", "")).split("|")
    names: set[str] = set()
    op_name = parts[1] if len(parts) >= 2 else ""
    api = parts[2] if len(parts) >= 3 else ""
    if api:
        names.add(api)
    if op_name:
        names.add(op_name)
        names.add(api_stem(op_name))
        names.update(hccl_alias(op_name))
        names.update(match_api_symbols(op_name, api_symbols))
    return {name for name in names if name}


def model_defaults(model_manifest: Path, api_symbols: list[str]) -> dict[str, float]:
    data = load_json(model_manifest)
    defaults: dict[str, float] = {}
    for entry in data.get("models", []):
        value = float(entry.get("observed_median_us", 0.0))
        # Keep both API and device-kernel names.  API names drive wrappers;
        # kernel names make the lookup artifact useful for diagnostics.
        for name in sorted(_entry_names(entry, api_symbols)):
            defaults.setdefault(name, value)
    return defaults


def model_status(model_manifest: Path, api_symbols: list[str]) -> tuple[set[str], set[str]]:
    data = load_json(model_manifest)
    known: set[str] = set()
    low_confidence: set[str] = set()
    for entry in data.get("models", []):
        names = _entry_names(entry, api_symbols)
        known.update(names)
        if entry.get("low_confidence"):
            low_confidence.update(names)
    return known, low_confidence


RF_FEATURES = ["input_numel", "output_numel", "matmul_m", "matmul_k", "matmul_n"] + \
              [f"input_dim_{i}" for i in range(8)] + [f"output_dim_{i}" for i in range(8)]
RF_FEATURE_COUNT = len(RF_FEATURES)  # 21


def _sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return safe[:80] or "group"


def rf_feature_vector(row: dict[str, Any]) -> list[float]:
    """Recompute the shape-only feature vector the C++ runtime can extract
    from aclTensor args at the *GetWorkspaceSize call. Must stay in lockstep
    with the C++ ascend_features_* emitters in compute_source()."""
    ins = decode_json_field(row.get("input_shapes"))
    outs = decode_json_field(row.get("output_shapes"))
    ins = ins if isinstance(ins, list) else []
    outs = outs if isinstance(outs, list) else []
    f = [0.0] * RF_FEATURE_COUNT
    f[0] = float(shape_numel(ins))
    f[1] = float(shape_numel(outs))
    if ins and isinstance(ins[0], list) and len(ins[0]) >= 2:
        f[2] = float(ins[0][-2])
        f[3] = float(ins[0][-1])
    if len(ins) >= 2 and ins[1] and isinstance(ins[1], list) and len(ins[1]) >= 2:
        f[4] = float(ins[1][-1])
    elif outs and outs[0] and isinstance(outs[0], list) and len(outs[0]) >= 2:
        f[4] = float(outs[0][-1])
    indims = [d for sh in ins if isinstance(sh, list) for d in sh]
    outdims = [d for sh in outs if isinstance(sh, list) for d in sh]
    for i, v in enumerate(indims[:8]):
        f[5 + i] = float(v)
    for i, v in enumerate(outdims[:8]):
        f[13 + i] = float(v)
    return f


def _select_lut_features(vectors: list[list[int]]) -> list[int]:
    varying = [i for i in range(RF_FEATURE_COUNT) if len({row[i] for row in vectors}) > 1]
    costs = {
        index: (6 if index == 0 else 3 if index == 1 else
                1 if index in {2, 3} or 13 <= index <= 20 else 2)
        for index in varying
    }
    complete: list[tuple[int, int, tuple[int, ...]]] = []
    for size in range(1, len(varying) + 1):
        for subset in itertools.combinations(varying, size):
            if len({tuple(row[i] for i in subset) for row in vectors}) == len(vectors):
                complete.append((sum(costs[i] for i in subset), size, subset))
    if complete:
        return list(min(complete)[2])
    selected: list[int] = []
    distinct = 1
    # Direct M/K/N dimensions are cheapest to obtain. Raw dimensions come
    # next; numel requires reading every tensor descriptor.
    priority = {index: rank for rank, index in enumerate(
                [2, 3, 4, *range(5, 13), *range(13, 21), 0, 1])}
    while distinct < len(vectors):
        choices = []
        for index in varying:
            if index in selected:
                continue
            count = len({tuple(row[i] for i in [*selected, index]) for row in vectors})
            choices.append((count, -priority[index], index))
        if not choices:
            break
        count, _rank, index = max(choices)
        if count <= distinct:
            break
        selected.append(index)
        distinct = count
    return selected


def latency_table_source(
    model_manifest: Path, aclnn_symbols: list[str]
) -> tuple[str, set[str], dict[str, set[int]]]:
    """Export a per-API shape -> latency lookup table with nearest-neighbour
    interpolation, replacing the random-forest tree walk.

    Each distinct (shape-feature -> duration) point observed in the collection
    tables becomes one row. At runtime the execute wrapper reads the concrete
    tensor shapes, then ``ascend_latency_predict`` does an exact match on the
    varying dimensions and, on a miss, an inverse-distance-weighted average of
    the two nearest rows. This is a flat scan over a handful of rows instead
    of walking hundreds of decision trees, keeping the interceptor overhead
    near zero. Returns -1 for symbols without a table so callers fall back to
    the median lookup."""
    data = load_json(model_manifest)
    root = model_manifest.parent
    execute_symbols = [s for s in aclnn_symbols if not s.endswith(EXECUTE_SUFFIX)]
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for entry in data.get("models", []):
        group_key = str(entry.get("group_key", ""))
        parts = group_key.split("|")
        kind = parts[0] if parts else "unknown"
        op_name = parts[1] if len(parts) >= 2 else "group"
        digest = hashlib.sha256(group_key.encode()).hexdigest()[:12]
        csv_path = root / "collections" / f"{kind}__{_sanitize_filename(op_name)}__{digest}.csv"
        if not csv_path.is_file():
            continue
        symbols = [s for s in execute_symbols if s in _entry_names(entry, execute_symbols)]
        if not symbols:
            continue
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for s in symbols:
            rows_by_symbol.setdefault(s, []).extend(rows)

    has_table: set[str] = set()
    feature_masks: dict[str, set[int]] = {}
    chunks = [
        "// Shape -> latency lookup tables with nearest-neighbour interpolation.",
        "#include <cmath>",
    ]
    for symbol in sorted(rows_by_symbol):
        by_feat: dict[tuple[int, ...], list[float]] = {}
        for row in rows_by_symbol[symbol]:
            feat = tuple(int(v) for v in rf_feature_vector(row))
            by_feat.setdefault(feat, []).append(float(row.get("duration_us", 0)))
        vecs = [list(k) for k in by_feat.keys()]
        durs = [sorted(v)[len(v) // 2] for v in by_feat.values()]
        if not vecs:
            continue
        has_table.add(symbol)
        sid = safe_id(symbol)
        predictor = f"ascend_latency_predict_{sid}"
        if len(vecs) == 1:
            feature_masks[symbol] = set()
            chunks.append(f'extern "C" float {predictor}(const float*) {{ return {durs[0]!r}f; }}')
            continue
        vary = _select_lut_features(vecs)
        feature_masks[symbol] = set(vary)
        if not vary:
            chunks.append(f'extern "C" float {predictor}(const float*) {{ return {durs[0]!r}f; }}')
            continue
        ranges = []
        for i in vary:
            lo = min(v[i] for v in vecs)
            hi = max(v[i] for v in vecs)
            ranges.append(max(1.0, float(hi - lo)))
        k = len(vary)
        chunks.append(f"static const int kLutVary_{sid}[] = {{" + ", ".join(map(str, vary)) + "};")
        chunks.append(f"static const float kLutRange_{sid}[] = {{" + ", ".join(f"{r!r}f" for r in ranges) + "};")
        rows_lines = ["  {" + ", ".join(str(v[i]) for i in vary) + "}," for v in vecs]
        chunks.append(f"static const long long kLutFeat_{sid}[][{k}] = {{")
        chunks.extend(rows_lines)
        chunks.append("};")
        chunks.append(f"static const float kLutDur_{sid}[] = {{" + ", ".join(f"{d!r}f" for d in durs) + "};")
        chunks.append(f'extern "C" float {predictor}(const float* f) {{')
        chunks.append(f"  const int n = {len(vecs)}; const int k = {k};")
        chunks.append("  int best = 0; int best2 = -1; double bd = 1e300; double bd2 = 1e300;")
        chunks.append("  for (int r = 0; r < n; r++) {")
        chunks.append("    double d2 = 0.0;")
        chunks.append(f"    for (int j = 0; j < k; j++) {{ double diff = ((double)f[kLutVary_{sid}[j]] - (double)kLutFeat_{sid}[r][j]) / (double)kLutRange_{sid}[j]; d2 += diff * diff; }}")
        chunks.append("    if (d2 < bd) { bd2 = bd; best2 = best; bd = d2; best = r; } else if (d2 < bd2) { bd2 = d2; best2 = r; }")
        chunks.append("  }")
        chunks.append("  if (bd < 1e-9) return kLutDur_" + sid + "[best];")
        chunks.append("  if (best2 < 0) return kLutDur_" + sid + "[best];")
        chunks.append("  double w1 = 1.0 / (bd + 1e-9), w2 = 1.0 / (bd2 + 1e-9);")
        chunks.append("  return (float)((kLutDur_" + sid + "[best] * w1 + kLutDur_" + sid + "[best2] * w2) / (w1 + w2));")
        chunks.append("}")

    return "\n".join(chunks) + "\n", has_table, feature_masks


def compute_source(
    symbols: list[str], include_root: Path, defaults: dict[str, float],
    model_known: set[str], model_low_confidence: set[str], table_symbols: set[str],
    feature_masks: dict[str, set[int]],
) -> tuple[str, list[dict[str, str]]]:
    headers: dict[str, str] = {}
    declarations: list[tuple[str, list[str], list[str], str]] = []
    unsupported: list[dict[str, str]] = []
    # Every ACLNN execute entry point shares this calling convention.  It is
    # safe to wrap a header-less execute symbol (for example the internal
    # ``aclnnInnerFusedInferAttentionScore``) with this uniform signature;
    # workspace calls keep their real prototypes when a header is available.
    uniform_execute_types = ["void*", "unsigned long long", "aclOpExecutor*", "aclrtStream"]
    uniform_execute_names = ["workspace", "workspaceSize", "executor", "stream"]
    for symbol in symbols:
        header, text = find_header(symbol, include_root)
        if not header or not text:
            if symbol.startswith("aclnn") and not symbol.endswith(EXECUTE_SUFFIX):
                declarations.append((symbol, list(uniform_execute_types), list(uniform_execute_names), ""))
                continue
            unsupported.append({"symbol": symbol, "reason": "header_not_found"})
            continue
        parsed = prototype(symbol, text)
        if not parsed:
            unsupported.append({"symbol": symbol, "reason": "prototype_not_parsed", "header": header})
            continue
        types, names = parsed
        headers[header] = header
        declarations.append((symbol, types, names, header))

    includes = "\n".join(f'#include <{header}>' for header in sorted(headers))
    lines = [
        "#include <acl/acl.h>", "#include <aclnn/acl_meta.h>",
        "#include <dlfcn.h>", "#include <atomic>", "#include <cstdint>", "#include <cstdio>", "#include <cstdlib>",
        "#include <cstring>", "#include <mutex>", "#include <string>",
        "#include <unordered_map>", includes,
        "extern \"C\" void ascend_time_jump_us(unsigned long long);",
        "extern \"C\" void ascend_op_latency(unsigned long long, void*);",
        "extern \"C\" void ascend_overhead_note(const char*, long long);",
        "extern \"C\" long long ascend_now_ns(void);",
        "extern \"C\" void ascend_collective_barrier(const char*);",
        "extern \"C\" void* ascend_hccl_wrapper_for_symbol(const char*);",
        "extern \"C\" void ascend_hccl_set_real_symbol(const char*, void*);",
        "extern \"C\" void* ascend_memory_wrapper_for_symbol(const char*);",
        "extern \"C\" void ascend_memory_set_real_symbol(const char*, void*);",
        "extern \"C\" void* ascend_device_wrapper_for_symbol(const char*);",
        "extern \"C\" void* ascend_real_dlsym(const char*);",
        "using DlsymFn = void* (*)(void*, const char*);",
        "static DlsymFn real_dlsym() { static auto fn = reinterpret_cast<DlsymFn>(dlvsym(RTLD_NEXT, \"dlsym\", \"GLIBC_2.34\")); return fn; }",
        "static bool env_flag(const char* name, bool def) { const char* v = std::getenv(name); return (!v || !*v) ? def : !(std::strcmp(v, \"0\") == 0 || std::strcmp(v, \"false\") == 0 || std::strcmp(v, \"off\") == 0); }",
        "static bool emulate() { static const bool value = env_flag(\"ASCEND_EMULATOR_EMULATE\", true); return value; }",
        "static bool strict() { static const bool value = env_flag(\"ASCEND_EMULATOR_STRICT\", true); return value; }",
        "static bool allow_passthrough() { static const bool value = env_flag(\"ASCEND_EMULATOR_ALLOW_PASSTHROUGH\", false); return value; }",
        "static bool shape_lookup() { static const bool value = env_flag(\"ASCEND_EMULATOR_SHAPE_LOOKUP\", true); return value; }",
        "static long long fixed_op_us() { static const long long value = [] { const char* v = std::getenv(\"ASCEND_EMULATOR_FIXED_OP_US\"); return (v && *v) ? std::atoll(v) : -1LL; }(); return value; }",
        "static std::mutex g_symbol_mutex;",
        "static std::unordered_map<std::string, void*> g_real_symbols;",
        "#define ASCEND_EMULATOR_FEATURE_COUNT 21",
        "struct AscendTensorShape { uint64_t ndim; int64_t dim[8]; };",
        "struct AscendTensorShapeSlot { const aclTensor* tensor; AscendTensorShape shape; };",
        "static constexpr uintptr_t ASCEND_TENSOR_CACHE_SIZE = 256;",
        "static thread_local AscendTensorShapeSlot g_tensor_shapes[ASCEND_TENSOR_CACHE_SIZE]{};",
        "static void* g_acl_create_tensor = nullptr;",
        "static void* g_acl_init_tensor = nullptr;",
        "static void* g_acl_destroy_tensor = nullptr;",
        "static std::atomic<unsigned long long> g_tensor_created{0};",
        "static std::atomic<unsigned long long> g_tensor_destroyed{0};",
        "static std::atomic<unsigned long long> g_tensor_cached{0};",
        "static aclOpExecutor* make_executor(float duration_us) { uintptr_t payload = static_cast<uintptr_t>(duration_us * 1024.0f + 0.5f); if (payload == 0) payload = 1; return reinterpret_cast<aclOpExecutor*>((payload << 1) | uintptr_t{1}); }",
        "static bool is_emulated_executor(const aclOpExecutor* executor) { return (reinterpret_cast<uintptr_t>(executor) & uintptr_t{1}) != 0; }",
        "static float executor_duration(const aclOpExecutor* executor) { return static_cast<float>(reinterpret_cast<uintptr_t>(executor) >> 1) / 1024.0f; }",
        "using AclGetViewShapeFn = aclnnStatus (*)(const aclTensor*, int64_t**, uint64_t*);",
        "static AclGetViewShapeFn real_acl_get_view_shape() { static AclGetViewShapeFn fn = nullptr; if (!fn) fn = reinterpret_cast<AclGetViewShapeFn>(ascend_real_dlsym(\"aclGetViewShape\")); return fn; }",
        "static bool ascend_read_shape(const aclTensor* t, AscendTensorShape& s) {",
        "  s.ndim = 0;",
        "  for (int i = 0; i < 8; i++) s.dim[i] = 0;",
        "  if (!t) return false;",
        "  uintptr_t raw = reinterpret_cast<uintptr_t>(t);",
        "  auto& slot = g_tensor_shapes[((raw >> 4) ^ (raw >> 17)) & (ASCEND_TENSOR_CACHE_SIZE - 1)];",
        "  if (slot.tensor == t) { s = slot.shape; return true; }",
        "  auto gvs = real_acl_get_view_shape();",
        "  int64_t* dims = nullptr; uint64_t nd = 0;",
        "  if (!gvs || gvs(t, &dims, &nd) != 0 || !dims) return false;",
        "  if (nd > 8) nd = 8;",
        "  s.ndim = nd;",
        "  for (uint64_t i = 0; i < nd; i++) s.dim[i] = dims[i];",
        "  return true;",
        "}",
        "static AscendTensorShapeSlot& shape_slot(const aclTensor* tensor) { uintptr_t raw = reinterpret_cast<uintptr_t>(tensor); return g_tensor_shapes[((raw >> 4) ^ (raw >> 17)) & (ASCEND_TENSOR_CACHE_SIZE - 1)]; }",
        "static void remember_shape(const aclTensor* tensor, const int64_t* dims, uint64_t ndim) { if (!tensor || !dims) return; auto& slot = shape_slot(tensor); if (!slot.tensor) g_tensor_cached.fetch_add(1, std::memory_order_relaxed); slot.shape = {}; slot.shape.ndim = ndim > 8 ? 8 : ndim; for (uint64_t i = 0; i < slot.shape.ndim; ++i) slot.shape.dim[i] = dims[i]; slot.tensor = tensor; }",
        "static void forget_shape(const aclTensor* tensor) { if (!tensor) return; auto& slot = shape_slot(tensor); if (slot.tensor == tensor) { slot.tensor = nullptr; g_tensor_cached.fetch_sub(1, std::memory_order_relaxed); } }",
        "extern \"C\" aclTensor* aclCreateTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride, int64_t offset, aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum, void* tensorData) { using Fn = aclTensor* (*)(const int64_t*, uint64_t, aclDataType, const int64_t*, int64_t, aclFormat, const int64_t*, uint64_t, void*); auto fn = reinterpret_cast<Fn>(g_acl_create_tensor ? g_acl_create_tensor : ascend_real_dlsym(\"aclCreateTensor\")); aclTensor* tensor = fn ? fn(viewDims, viewDimsNum, dataType, stride, offset, format, storageDims, storageDimsNum, tensorData) : nullptr; if (tensor) g_tensor_created.fetch_add(1, std::memory_order_relaxed); if (shape_lookup()) remember_shape(tensor, viewDims, viewDimsNum); return tensor; }",
        "extern \"C\" aclnnStatus aclInitTensor(aclTensor* tensor, const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t* stride, int64_t offset, aclFormat format, const int64_t* storageDims, uint64_t storageDimsNum, void* tensorData) { using Fn = aclnnStatus (*)(aclTensor*, const int64_t*, uint64_t, aclDataType, const int64_t*, int64_t, aclFormat, const int64_t*, uint64_t, void*); auto fn = reinterpret_cast<Fn>(g_acl_init_tensor ? g_acl_init_tensor : ascend_real_dlsym(\"aclInitTensor\")); aclnnStatus rc = fn ? fn(tensor, viewDims, viewDimsNum, dataType, stride, offset, format, storageDims, storageDimsNum, tensorData) : static_cast<aclnnStatus>(-1); if (rc == 0 && shape_lookup()) remember_shape(tensor, viewDims, viewDimsNum); return rc; }",
        "extern \"C\" aclnnStatus aclDestroyTensor(const aclTensor* tensor) { if (shape_lookup()) forget_shape(tensor); g_tensor_destroyed.fetch_add(1, std::memory_order_relaxed); using Fn = aclnnStatus (*)(const aclTensor*); auto fn = reinterpret_cast<Fn>(g_acl_destroy_tensor ? g_acl_destroy_tensor : ascend_real_dlsym(\"aclDestroyTensor\")); return fn ? fn(tensor) : 0; }",
        "extern \"C\" void ascend_tensor_registry_stats(unsigned long long* live, unsigned long long* created, unsigned long long* destroyed) { if (live) *live = g_tensor_cached.load(std::memory_order_relaxed); if (created) *created = g_tensor_created.load(std::memory_order_relaxed); if (destroyed) *destroyed = g_tensor_destroyed.load(std::memory_order_relaxed); }",
    ]
    for symbol in sorted(table_symbols):
        lines.append(f'extern "C" float ascend_latency_predict_{safe_id(symbol)}(const float*);')
    generated_wrappers: set[str] = set()
    get_symbols = {symbol for symbol, _types, _names, _header in declarations if symbol.endswith(EXECUTE_SUFFIX)}
    for symbol, types, names, _header in declarations:
        sid = safe_id(symbol)
        if symbol.endswith(EXECUTE_SUFFIX):
            execute = symbol.removesuffix(EXECUTE_SUFFIX)
            tensor_params = [(t, n) for t, n in zip(types, names) if "aclTensor*" in t]
            output_tensors = [
                (t, n) for t, n in tensor_params
                if not t.strip().startswith("const")
                or n.lower() in {"out", "output", "result", "y"}
                or n.lower().startswith(("out", "output", "result"))
            ]
            input_tensors = [(t, n) for t, n in tensor_params if (t, n) not in output_tensors]
            tensor_args = [name for _t, name in input_tensors + output_tensors]
            feature_fn = "ascend_features_" + safe_id(execute)
            feature_mask = feature_masks.get(execute, set())
            input_limit = 0
            output_limit = 0
            if 0 in feature_mask or any(5 <= index <= 12 for index in feature_mask):
                input_limit = len(input_tensors)
            elif 4 in feature_mask:
                input_limit = min(2, len(input_tensors))
            elif 2 in feature_mask or 3 in feature_mask:
                input_limit = min(1, len(input_tensors))
            if 1 in feature_mask or any(13 <= index <= 20 for index in feature_mask):
                output_limit = len(output_tensors)
            elif 4 in feature_mask and len(input_tensors) < 2:
                output_limit = min(1, len(output_tensors))
            if tensor_args and feature_mask:
                ## Shape-feature extractor, mirroring rf_feature_vector().
                lines.append("static void " + feature_fn + "(" + ", ".join(f"{t} {n}" for t, n in input_tensors + output_tensors) + ", float* f) {")
                for name in tensor_args:
                    lines.append(f"  (void){name};")
                lines.append("  for (int i = 0; i < ASCEND_EMULATOR_FEATURE_COUNT; i++) f[i] = 0.0f;")
                lines.append("  AscendTensorShape ascend_in[8]{}; uint64_t ascend_nin = 0;")
                lines.append("  AscendTensorShape ascend_out[8]{}; uint64_t ascend_nout = 0;")
                lines.append("  AscendTensorShape ascend_sh{};")
                for typ, name in input_tensors[:input_limit]:
                    lines.append(f"  if (ascend_read_shape({name}, ascend_sh)) ascend_in[ascend_nin++] = ascend_sh;")
                for typ, name in output_tensors[:output_limit]:
                    lines.append(f"  if (ascend_read_shape({name}, ascend_sh)) ascend_out[ascend_nout++] = ascend_sh;")
                lines.append("  for (uint64_t t = 0; t < ascend_nin; t++) { double p = 1.0; bool bad = false; for (uint64_t i = 0; i < ascend_in[t].ndim; i++) { if (ascend_in[t].dim[i] < 0) { bad = true; break; } p *= static_cast<double>(ascend_in[t].dim[i]); } if (!bad) f[0] += p; }")
                lines.append("  for (uint64_t t = 0; t < ascend_nout; t++) { double p = 1.0; bool bad = false; for (uint64_t i = 0; i < ascend_out[t].ndim; i++) { if (ascend_out[t].dim[i] < 0) { bad = true; break; } p *= static_cast<double>(ascend_out[t].dim[i]); } if (!bad) f[1] += p; }")
                lines.append("  if (ascend_nin >= 1 && ascend_in[0].ndim >= 2) { f[2] = static_cast<float>(ascend_in[0].dim[ascend_in[0].ndim - 2]); f[3] = static_cast<float>(ascend_in[0].dim[ascend_in[0].ndim - 1]); }")
                lines.append("  if (ascend_nin >= 2 && ascend_in[1].ndim >= 2) f[4] = static_cast<float>(ascend_in[1].dim[ascend_in[1].ndim - 1]);")
                lines.append("  else if (ascend_nout >= 1 && ascend_out[0].ndim >= 2) f[4] = static_cast<float>(ascend_out[0].dim[ascend_out[0].ndim - 1]);")
                lines.append("  uint64_t ascend_idx = 0; for (uint64_t t = 0; t < ascend_nin && ascend_idx < 8; t++) for (uint64_t i = 0; i < ascend_in[t].ndim && ascend_idx < 8; i++) f[5 + ascend_idx++] = static_cast<float>(ascend_in[t].dim[i]);")
                lines.append("  ascend_idx = 0; for (uint64_t t = 0; t < ascend_nout && ascend_idx < 8; t++) for (uint64_t i = 0; i < ascend_out[t].ndim && ascend_idx < 8; i++) f[13 + ascend_idx++] = static_cast<float>(ascend_out[t].dim[i]);")
                lines.append("}")
            lines.append(f"using {sid}Fn = aclnnStatus (*) (" + ", ".join(types) + ");")
            lines.append(f"static void* g_{sid} = nullptr;")
            lines.append(f"static aclnnStatus wrap_{sid}(" + ", ".join(f"{typ} {name}" for typ, name in zip(types, names)) + ") {")
            lines.append(f"  auto fn = reinterpret_cast<{sid}Fn>(g_{sid}); if (!fn) return static_cast<aclnnStatus>(-1);")
            executor_name = next((name for typ, name in zip(types, names) if "aclOpExecutor" in typ and "**" in typ.replace(" ", "")), None)
            workspace_name = next((name for name in names if name and "workspace" in name.lower()), None)
            if executor_name:
                # Emulation: skip the real GetWorkspaceSize. It launches an
                # internal tiling kernel and then waits for it; in virtual
                # memory that kernel is suppressed, so the wait spins for
                # ~100s of us per op. We only need the tensor shapes and an
                # executor token, so synthesize both and return immediately.
                lines.append("  if (emulate()) {")
                fallback = max(0.0, defaults.get(execute, 1.0))
                if tensor_args and feature_mask:
                    lines.append(f"    float predicted = {fallback!r}f;")
                    lines.append("    if (shape_lookup()) {")
                    lines.append("      float feat[ASCEND_EMULATOR_FEATURE_COUNT];")
                    lines.append("      long long _shape_t0 = ascend_now_ns();")
                    lines.append(f"      {feature_fn}(" + ", ".join(tensor_args) + ", feat);")
                    lines.append("      ascend_overhead_note(\"compute_shape_extract\", ascend_now_ns() - _shape_t0);")
                    lines.append(f"      float shape_prediction = ascend_latency_predict_{safe_id(execute)}(feat);")
                    lines.append("      if (shape_prediction >= 0.0f) predicted = shape_prediction;")
                    lines.append("    }")
                    lines.append("    aclOpExecutor* fake = make_executor(predicted);")
                else:
                    lines.append(f"    aclOpExecutor* fake = make_executor({fallback!r}f);")
                if workspace_name:
                    lines.append(f"    if ({workspace_name}) *{workspace_name} = 0;")
                lines.append(f"    *{executor_name} = fake;")
                lines.append("    return 0; }")
                lines.append("  long long _wst0 = ascend_now_ns();")
                lines.append(f"  aclnnStatus rc = fn(" + ", ".join(names) + ");")
                lines.append("  ascend_overhead_note(\"compute_workspace_real\", ascend_now_ns() - _wst0);")
                lines.append("  return rc; }")
            else:
                lines.append(f"  return fn(" + ", ".join(names) + "); }")
            generated_wrappers.add(symbol)
            if execute not in generated_wrappers:
                exec_sid = safe_id(execute)
                fallback = max(0.0, defaults.get(execute, 1.0))
                strict_failure = execute not in model_known or execute in model_low_confidence
                prediction = f"{fallback!r}f"
                if execute in table_symbols:
                    prediction = (
                        f"is_emulated_executor(executor) ? "
                        f"executor_duration(executor) : {prediction}"
                    )
                lines.extend([
                    f"using {exec_sid}Fn = aclnnStatus (*)(void*, unsigned long long, aclOpExecutor*, aclrtStream);",
                    f"static void* g_{exec_sid} = nullptr;",
                    f"static aclnnStatus wrap_{exec_sid}(void* workspace, unsigned long long workspaceSize, aclOpExecutor* executor, aclrtStream stream) {{",
                    f"  auto fn = reinterpret_cast<{exec_sid}Fn>(g_{exec_sid}); if (!fn) return static_cast<aclnnStatus>(-1);",
                    f"  if (emulate()) {{ long long _t0 = ascend_now_ns(); if (strict() && !allow_passthrough() && {str(strict_failure).lower()}) return static_cast<aclnnStatus>(-1); long long fixed = fixed_op_us(); float duration = fixed >= 0 ? static_cast<float>(fixed) : {prediction}; ascend_op_latency(static_cast<unsigned long long>(duration), stream); ascend_overhead_note(\"compute_execute\", ascend_now_ns() - _t0); return 0; }}",
                    "  return fn(workspace, workspaceSize, executor, stream); }",
                ])
                generated_wrappers.add(execute)
            continue
        if symbol in generated_wrappers:
            continue
        lines.append(f"using {sid}Fn = aclnnStatus (*) (" + ", ".join(types) + ");")
        lines.append(f"static void* g_{sid} = nullptr;")
        lines.append(f"static aclnnStatus wrap_{sid}(" + ", ".join(f"{typ} {name}" for typ, name in zip(types, names)) + ") {")
        lines.append(f"  auto fn = reinterpret_cast<{sid}Fn>(g_{sid}); if (!fn) return static_cast<aclnnStatus>(-1);")
        if len(types) == 4 and "aclOpExecutor" in " ".join(types):
            exec_name = next((name for typ, name in zip(types, names) if "aclOpExecutor*" in typ and "aclOpExecutor**" not in typ), "executor")
            fallback = max(0.0, defaults.get(symbol, 1.0))
            strict_failure = symbol not in model_known or symbol in model_low_confidence
            prediction = f"{fallback!r}f"
            if symbol in table_symbols:
                prediction = f"is_emulated_executor({exec_name}) ? executor_duration({exec_name}) : {prediction}"
            lines.append(f"  if (emulate()) {{ long long _t0 = ascend_now_ns(); if (strict() && !allow_passthrough() && {str(strict_failure).lower()}) return static_cast<aclnnStatus>(-1); long long fixed = fixed_op_us(); float duration = fixed >= 0 ? static_cast<float>(fixed) : {prediction}; ascend_op_latency(static_cast<unsigned long long>(duration), {names[3]}); ascend_overhead_note(\"compute_execute\", ascend_now_ns() - _t0); return 0; }}")
        lines.append("  return fn(" + ", ".join(names) + "); }")
        generated_wrappers.add(symbol)
    lines.extend([
        "static aclnnStatus wrap_aclDestroyAclOpExecutor(aclOpExecutor* executor) {",
        "  if (is_emulated_executor(executor)) return 0;",
        "  auto fn = reinterpret_cast<aclnnStatus (*)(aclOpExecutor*)>(ascend_real_dlsym(\"aclDestroyAclOpExecutor\"));",
        "  return fn ? fn(executor) : 0; }",
        "extern \"C\" void* ascend_real_dlsym(const char* name) { { std::lock_guard<std::mutex> lock(g_symbol_mutex); auto it = g_real_symbols.find(name); if (it != g_real_symbols.end()) return it->second; } auto real = real_dlsym(); void* sym = real ? real(RTLD_NEXT, name) : nullptr; if (sym) { std::lock_guard<std::mutex> lock(g_symbol_mutex); g_real_symbols[name] = sym; } return sym; }",
        "extern \"C\" void* dlsym(void* handle, const char* name) {",
        "  long long _dt0 = ascend_now_ns();",
        "  auto real = real_dlsym(); if (!real) return nullptr; void* actual = real(handle, name);",
        "  if (std::strcmp(name, \"aclCreateTensor\") == 0) { g_acl_create_tensor = actual; return reinterpret_cast<void*>(&aclCreateTensor); }",
        "  if (std::strcmp(name, \"aclInitTensor\") == 0) { g_acl_init_tensor = actual; return reinterpret_cast<void*>(&aclInitTensor); }",
        "  if (std::strcmp(name, \"aclDestroyTensor\") == 0) { g_acl_destroy_tensor = actual; return reinterpret_cast<void*>(&aclDestroyTensor); }",
    ])
    for symbol, _types, _names, _header in declarations:
        sid = safe_id(symbol)
        lines.extend([
            f"  if (std::strcmp(name, \"{symbol}\") == 0) {{ g_{sid} = actual; return reinterpret_cast<void*>(&wrap_{sid}); }}",
        ])
    lines.extend([
        "  if (std::strcmp(name, \"aclDestroyAclOpExecutor\") == 0) return reinterpret_cast<void*>(&wrap_aclDestroyAclOpExecutor);",
        "  if (void* wrapper = ascend_hccl_wrapper_for_symbol(name)) { ascend_hccl_set_real_symbol(name, actual); ascend_overhead_note(\"dlsym_dispatch\", ascend_now_ns() - _dt0); return wrapper; }",
        "  if (void* wrapper = ascend_memory_wrapper_for_symbol(name)) { ascend_memory_set_real_symbol(name, actual); ascend_overhead_note(\"dlsym_dispatch\", ascend_now_ns() - _dt0); return wrapper; }",
        "  if (void* wrapper = ascend_device_wrapper_for_symbol(name)) { ascend_overhead_note(\"dlsym_dispatch\", ascend_now_ns() - _dt0); return wrapper; }",
        "  ascend_overhead_note(\"dlsym_dispatch\", ascend_now_ns() - _dt0);",
        "  return actual; }",
    ])
    return "\n".join(lines) + "\n", unsupported


def communication_source(
    defaults: dict[str, float], model_known: set[str], model_low_confidence: set[str]
) -> str:
    source = r'''#include <hccl/hccl.h>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

extern "C" void ascend_time_jump_us(unsigned long long);
extern "C" void ascend_op_latency(unsigned long long, void*);
extern "C" void ascend_collective_barrier(const char*);
extern "C" void* ascend_real_dlsym(const char*);

static bool env_flag(const char* name, bool def) {
  const char* v = std::getenv(name);
  return (!v || !*v) ? def : !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}
static bool emulate() { static const bool value = env_flag("ASCEND_EMULATOR_EMULATE", true); return value; }
static bool strict() { static const bool value = env_flag("ASCEND_EMULATOR_STRICT", true); return value; }
static bool allow_passthrough() { static const bool value = env_flag("ASCEND_EMULATOR_ALLOW_PASSTHROUGH", false); return value; }
static long long fixed_comm_us() { static const long long value = [] { const char* v = std::getenv("ASCEND_EMULATOR_FIXED_COMM_US"); return (v && *v) ? std::atoll(v) : -1LL; }(); return value; }
static unsigned long long predict(const char* name, unsigned long long bytes) {
  const char* v = std::getenv("ASCEND_EMULATOR_FIXED_COMM_US");
  if (v) return std::strtoull(v, nullptr, 10);
  return bytes ? (bytes / 1024 + 1) : 1;
}
static unsigned long long dtype_size(HcclDataType type) {
  switch (type) { case HCCL_DATA_TYPE_INT8: case HCCL_DATA_TYPE_UINT8: return 1;
    case HCCL_DATA_TYPE_INT16: case HCCL_DATA_TYPE_UINT16: case HCCL_DATA_TYPE_FP16: return 2;
    case HCCL_DATA_TYPE_INT32: case HCCL_DATA_TYPE_UINT32: case HCCL_DATA_TYPE_FP32: return 4;
    case HCCL_DATA_TYPE_INT64: case HCCL_DATA_TYPE_UINT64: case HCCL_DATA_TYPE_FP64: return 8;
    default: return 2; }
}
#define REAL(name) reinterpret_cast<decltype(&name)>(real_symbol(#name))
static void* real_symbol(const char* name) { return ascend_real_dlsym(name); }

extern "C" HcclResult HcclAllReduce(void* send, void* recv, uint64_t count, HcclDataType type, HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  if (emulate()) { ascend_collective_barrier("HcclAllReduce"); ascend_time_jump_us(predict(count * dtype_size(type))); return HCCL_SUCCESS; }
  auto fn = REAL(HcclAllReduce); return fn ? fn(send, recv, count, type, op, comm, stream) : HCCL_E_NOT_FOUND;
}
extern "C" HcclResult HcclBroadcast(void* buf, uint64_t count, HcclDataType type, uint32_t root, HcclComm comm, aclrtStream stream) {
  if (emulate()) { ascend_collective_barrier("HcclBroadcast"); ascend_time_jump_us(predict(count * dtype_size(type))); return HCCL_SUCCESS; }
  auto fn = REAL(HcclBroadcast); return fn ? fn(buf, count, type, root, comm, stream) : HCCL_E_NOT_FOUND;
}
extern "C" HcclResult HcclReduceScatter(void* send, void* recv, uint64_t count, HcclDataType type, HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  if (emulate()) { ascend_collective_barrier("HcclReduceScatter"); ascend_time_jump_us(predict(count * dtype_size(type))); return HCCL_SUCCESS; }
  auto fn = REAL(HcclReduceScatter); return fn ? fn(send, recv, count, type, op, comm, stream) : HCCL_E_NOT_FOUND;
}
extern "C" HcclResult HcclAllGather(void* send, void* recv, uint64_t count, HcclDataType type, HcclComm comm, aclrtStream stream) {
  if (emulate()) { ascend_collective_barrier("HcclAllGather"); ascend_time_jump_us(predict(count * dtype_size(type))); return HCCL_SUCCESS; }
  auto fn = REAL(HcclAllGather); return fn ? fn(send, recv, count, type, comm, stream) : HCCL_E_NOT_FOUND;
}
extern "C" HcclResult HcclSend(void* buf, uint64_t count, HcclDataType type, uint32_t rank, HcclComm comm, aclrtStream stream) {
  if (emulate()) { ascend_collective_barrier("HcclSend"); ascend_time_jump_us(predict(count * dtype_size(type))); return HCCL_SUCCESS; }
  auto fn = REAL(HcclSend); return fn ? fn(buf, count, type, rank, comm, stream) : HCCL_E_NOT_FOUND;
}
extern "C" HcclResult HcclRecv(void* buf, uint64_t count, HcclDataType type, uint32_t rank, HcclComm comm, aclrtStream stream) {
  if (emulate()) { ascend_collective_barrier("HcclRecv"); ascend_time_jump_us(predict(count * dtype_size(type))); return HCCL_SUCCESS; }
  auto fn = REAL(HcclRecv); return fn ? fn(buf, count, type, rank, comm, stream) : HCCL_E_NOT_FOUND;
}
extern "C" void* ascend_hccl_wrapper_for_symbol(const char* name) {
  if (!name) return nullptr;
  if (!std::strcmp(name, "HcclAllReduce")) return reinterpret_cast<void*>(&HcclAllReduce);
  if (!std::strcmp(name, "HcclBroadcast")) return reinterpret_cast<void*>(&HcclBroadcast);
  if (!std::strcmp(name, "HcclReduceScatter")) return reinterpret_cast<void*>(&HcclReduceScatter);
  if (!std::strcmp(name, "HcclAllGather")) return reinterpret_cast<void*>(&HcclAllGather);
  if (!std::strcmp(name, "HcclSend")) return reinterpret_cast<void*>(&HcclSend);
  if (!std::strcmp(name, "HcclRecv")) return reinterpret_cast<void*>(&HcclRecv);
  return nullptr;
}
extern "C" void ascend_hccl_set_real_symbol(const char*, void*) {}
'''
    model_lines = [
        "static bool strict_model_failure(const char* name) {",
        "  bool is_strict = strict();",
        "  bool is_passthrough = allow_passthrough();",
        "  if (!is_strict || is_passthrough) return false;",
        "  bool known = false; bool low = false;",
    ]
    for name in sorted(model_known):
        if name.startswith("Hccl") or name.upper().startswith("HCCL"):
            model_lines.append(f"  if (std::strcmp(name, {json.dumps(name)}) == 0) known = true;")
    for name in sorted(model_low_confidence):
        if name.startswith("Hccl") or name.upper().startswith("HCCL"):
            model_lines.append(f"  if (std::strcmp(name, {json.dumps(name)}) == 0) low = true;")
    model_lines.extend(["  return !known || low;", "}"])
    prediction_lines = [
        "static unsigned long long modeled_duration(const char* name, unsigned long long bytes) {",
        "  long long fixed = fixed_comm_us();",
        "  if (fixed >= 0) return static_cast<unsigned long long>(fixed);",
    ]
    for name, value in sorted(defaults.items()):
        if name.startswith("Hccl") or name.upper().startswith("HCCL"):
            prediction_lines.append(
                f"  if (std::strcmp(name, {json.dumps(name)}) == 0) return {max(0.0, value):.0f}ULL;"
            )
    prediction_lines.extend(["  return bytes ? (bytes / 1024 + 1) : 1;", "}"])
    source = source.replace(
        "static unsigned long long predict(const char* name, unsigned long long bytes) {\n  const char* v = std::getenv(\"ASCEND_EMULATOR_FIXED_COMM_US\");\n  if (v) return std::strtoull(v, nullptr, 10);\n  return bytes ? (bytes / 1024 + 1) : 1;\n}",
        "\n".join(prediction_lines),
    )
    source = source.replace("#define REAL(name)", "\n".join(model_lines) + "\n#define REAL(name)")
    for name in KNOWN_HCCL:
        source = source.replace(
            f'if (emulate()) {{ ascend_collective_barrier("{name}"); ascend_time_jump_us(predict(count * dtype_size(type)));',
            f'if (emulate()) {{ if (strict_model_failure("{name}")) return HCCL_E_NOT_FOUND; ascend_collective_barrier("{name}"); ascend_op_latency(modeled_duration("{name}", count * dtype_size(type)), stream);',
        )
    return source


def rt_skip_source() -> str:
    # ATB-launched device kernels (e.g. reshape_and_cache) reach the runtime
    # through rtKernelLaunch*. In virtual-memory mode their output buffers are
    # reserved addresses, so the real kernel would fault; emulating the op means
    # no real device kernel runs. ACLNN ops never reach these entry points
    # (they are intercepted at the execute level), so this only affects ATB.
    return r'''
static bool log_rt_skips() { static const bool value = env_flag("ASCEND_EMULATOR_LOG_RT_SKIPS", false); return value; }
static void note_rt_skip(const char* name) { if (log_rt_skips()) std::fprintf(stderr, "[ascend-rt-skip] %s\n", name); }
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, aclrtLaunchKernelCfg* cfg, void* hostArgs, size_t argsSize, aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory()) { note_rt_skip("aclrtLaunchKernelWithHostArgs"); ascend_op_latency(1, stream); ascend_overhead_note("rt_launch_skip", ascend_now_ns() - _t0); return 0; }
  using Fn = aclError (*)(aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*, void*, size_t, aclrtPlaceHolderInfo*, size_t);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtLaunchKernelWithHostArgs")); ascend_overhead_note("rt_launch_real", ascend_now_ns() - _t0); return fn ? fn(funcHandle, blockDim, stream, cfg, hostArgs, argsSize, placeHolderArray, placeHolderNum) : -1;
}
extern "C" int32_t rtKernelLaunch(const void* stubFunc, uint32_t blockDim, void* args, uint32_t argsSize, void* smDesc, void* stm) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory()) { note_rt_skip("rtKernelLaunch"); ascend_op_latency(1, stm); ascend_overhead_note("rt_launch_skip", ascend_now_ns() - _t0); return 0; }
  using Fn = int32_t (*)(const void*, uint32_t, void*, uint32_t, void*, void*);
  auto fn = reinterpret_cast<Fn>(real_lookup("rtKernelLaunch")); ascend_overhead_note("rt_launch_real", ascend_now_ns() - _t0); return fn ? fn(stubFunc, blockDim, args, argsSize, smDesc, stm) : -1;
}
extern "C" int32_t rtKernelLaunchWithFlag(const void* stubFunc, uint32_t blockDim, void* argsInfo, void* smDesc, void* stm, uint32_t flags) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory()) { note_rt_skip("rtKernelLaunchWithFlag"); ascend_op_latency(1, stm); ascend_overhead_note("rt_launch_skip", ascend_now_ns() - _t0); return 0; }
  using Fn = int32_t (*)(const void*, uint32_t, void*, void*, void*, uint32_t);
  auto fn = reinterpret_cast<Fn>(real_lookup("rtKernelLaunchWithFlag")); ascend_overhead_note("rt_launch_real", ascend_now_ns() - _t0); return fn ? fn(stubFunc, blockDim, argsInfo, smDesc, stm, flags) : -1;
}
extern "C" int32_t rtKernelLaunchWithFlagV2(const void* stubFunc, uint32_t blockDim, void* argsInfo, void* smDesc, void* stm, uint32_t flags, void* cfgInfo) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory()) { note_rt_skip("rtKernelLaunchWithFlagV2"); ascend_op_latency(1, stm); ascend_overhead_note("rt_launch_skip", ascend_now_ns() - _t0); return 0; }
  using Fn = int32_t (*)(const void*, uint32_t, void*, void*, void*, uint32_t, void*);
  auto fn = reinterpret_cast<Fn>(real_lookup("rtKernelLaunchWithFlagV2")); ascend_overhead_note("rt_launch_real", ascend_now_ns() - _t0); return fn ? fn(stubFunc, blockDim, argsInfo, smDesc, stm, flags, cfgInfo) : -1;
}
extern "C" int32_t rtKernelLaunchWithHandle(void* hdl, uint64_t tilingKey, uint32_t blockDim, void* argsInfo, void* smDesc, void* stm, const void* kernelInfo) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory()) { note_rt_skip("rtKernelLaunchWithHandle"); ascend_op_latency(1, stm); ascend_overhead_note("rt_launch_skip", ascend_now_ns() - _t0); return 0; }
  using Fn = int32_t (*)(void*, uint64_t, uint32_t, void*, void*, void*, const void*);
  auto fn = reinterpret_cast<Fn>(real_lookup("rtKernelLaunchWithHandle")); ascend_overhead_note("rt_launch_real", ascend_now_ns() - _t0); return fn ? fn(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, kernelInfo) : -1;
}
extern "C" int32_t rtKernelLaunchWithHandleV2(void* hdl, uint64_t tilingKey, uint32_t blockDim, void* argsInfo, void* smDesc, void* stm, const void* cfgInfo) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory()) { note_rt_skip("rtKernelLaunchWithHandleV2"); ascend_op_latency(1, stm); ascend_overhead_note("rt_launch_skip", ascend_now_ns() - _t0); return 0; }
  using Fn = int32_t (*)(void*, uint64_t, uint32_t, void*, void*, void*, const void*);
  auto fn = reinterpret_cast<Fn>(real_lookup("rtKernelLaunchWithHandleV2")); ascend_overhead_note("rt_launch_real", ascend_now_ns() - _t0); return fn ? fn(hdl, tilingKey, blockDim, argsInfo, smDesc, stm, cfgInfo) : -1;
}
'''


def memory_source(defaults: dict[str, float]) -> str:
    source = r'''#include <acl/acl_rt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <unordered_set>

extern "C" void ascend_time_jump_us(unsigned long long);
extern "C" void ascend_op_latency(unsigned long long, void*);
extern "C" void ascend_host_blocking(unsigned long long);
extern "C" void ascend_overhead_note(const char*, long long);
extern "C" long long ascend_now_ns(void);
extern "C" void* ascend_real_dlsym(const char*);
static std::mutex g_mutex;
static std::map<uintptr_t, size_t> g_virtual;
static std::unordered_set<void*> g_fake_handles;
extern "C" void ascend_memory_registry_stats(unsigned long long* ranges, unsigned long long* handles) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (ranges) *ranges = g_virtual.size();
  if (handles) *handles = g_fake_handles.size();
}
static bool is_virtual_ptr(const void* ptr) {
  if (!ptr) return false;
  auto cur = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_virtual.lower_bound(cur);
  if (it == g_virtual.begin()) return false;
  --it;
  return cur < it->first + it->second;
}
static bool env_flag(const char* name, bool def) {
  const char* v = std::getenv(name);
  return (!v || !*v) ? def : !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}
static bool log_mem_skips() { static const bool value = env_flag("ASCEND_EMULATOR_LOG_MEM_SKIPS", false); return value; }
static bool log_allocs() { static const bool value = env_flag("ASCEND_EMULATOR_LOG_ALLOCS", false); return value; }
static void note_skip(const char* fn, size_t bytes) { if (log_mem_skips()) std::fprintf(stderr, "[ascend-mem-skip] %s bytes=%zu\n", fn, bytes); }
static bool emulate() { static const bool value = env_flag("ASCEND_EMULATOR_EMULATE", true); return value; }
static bool virtual_memory() { static const bool value = env_flag("ASCEND_EMULATOR_VIRTUAL_MEMORY", true); return value; }
static void* real_lookup(const char* name) { return ascend_real_dlsym(name); }
static size_t memcpy_skip_min() {
  static const size_t cached = [] { const char* v = std::getenv("ASCEND_EMULATOR_MEMCPY_SKIP_MIN_BYTES"); if (v && *v) { long long n = std::atoll(v); if (n > 0) return static_cast<size_t>(n); } return static_cast<size_t>(65536); }();
  return cached;
}
static size_t virtual_min() {
  static const size_t cached = [] { const char* v = std::getenv("ASCEND_EMULATOR_VIRTUAL_MIN_BYTES"); if (v && *v) { long long n = std::atoll(v); if (n > 0) return static_cast<size_t>(n); } return static_cast<size_t>(4194304); }();
  return cached;
}
static void note_alloc(const char* kind, size_t size, void* addr) { if (log_allocs()) std::fprintf(stderr, "[ascend-alloc] %s size=%zu addr=%p\n", kind, size, addr); }
static long long fixed_mem_us() { static const long long value = [] { const char* v = std::getenv("ASCEND_EMULATOR_FIXED_MEM_US"); return (v && *v) ? std::atoll(v) : -1LL; }(); return value; }
static unsigned long long duration(const char* name, size_t bytes) {
  const char* fixed = std::getenv("ASCEND_EMULATOR_FIXED_MEM_US");
  if (fixed) return std::strtoull(fixed, nullptr, 10);
  (void)name; return bytes ? (bytes / 1024 + 1) : 1;
}
static bool emulate_copy(size_t count, aclrtMemcpyKind kind, const void* dst, const void* src) {
  if (!emulate()) return false;
  if (count >= memcpy_skip_min()) return true;
  if (virtual_memory() && kind != ACL_MEMCPY_HOST_TO_HOST) return true;
  return virtual_memory() && (is_virtual_ptr(dst) || is_virtual_ptr(src));
}
static void synthesize_device_read(void* dst, size_t destMax, size_t count, aclrtMemcpyKind kind) {
  if (kind == ACL_MEMCPY_DEVICE_TO_HOST && dst) std::memset(dst, 0, std::min(destMax, count));
}

extern "C" aclError aclrtMalloc(void** ptr, size_t size, aclrtMemMallocPolicy policy) {
  long long _t0 = ascend_now_ns();
  using Fn = aclError (*)(void**, size_t, aclrtMemMallocPolicy);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMalloc")); aclError rc = fn ? fn(ptr, size, policy) : ACL_ERROR_API_NOT_SUPPORT;
  if (rc == ACL_SUCCESS) note_alloc("real", size, ptr ? *ptr : nullptr);
  ascend_overhead_note("mem_malloc", ascend_now_ns() - _t0);
  return rc;
}
extern "C" aclError aclrtMallocAlign32(void** ptr, size_t size, aclrtMemMallocPolicy policy) {
  long long _t0 = ascend_now_ns();
  using Fn = aclError (*)(void**, size_t, aclrtMemMallocPolicy);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMallocAlign32")); aclError rc = fn ? fn(ptr, size, policy) : ACL_ERROR_API_NOT_SUPPORT;
  if (rc == ACL_SUCCESS) note_alloc("real-align32", size, ptr ? *ptr : nullptr);
  ascend_overhead_note("mem_malloc_align32", ascend_now_ns() - _t0);
  return rc;
}
extern "C" aclError aclrtMallocCached(void** ptr, size_t size, aclrtMemMallocPolicy policy) {
  long long _t0 = ascend_now_ns();
  using Fn = aclError (*)(void**, size_t, aclrtMemMallocPolicy);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMallocCached")); aclError rc = fn ? fn(ptr, size, policy) : ACL_ERROR_API_NOT_SUPPORT;
  if (rc == ACL_SUCCESS) note_alloc("real-cached", size, ptr ? *ptr : nullptr);
  ascend_overhead_note("mem_malloc_cached", ascend_now_ns() - _t0);
  return rc;
}
extern "C" aclError aclrtFree(void* ptr) {
  long long _t0 = ascend_now_ns();
  using Fn = aclError (*)(void*); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtFree"));
  ascend_overhead_note("mem_free", ascend_now_ns() - _t0);
  return fn ? fn(ptr) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtMallocPhysical(void** handle, size_t size, const aclrtPhysicalMemProp* prop, uint64_t flags) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory() && size >= virtual_min()) {
    void* fake = std::malloc(1);
    if (!fake) return ACL_ERROR_API_NOT_SUPPORT;
    { std::lock_guard<std::mutex> lock(g_mutex); g_fake_handles.insert(fake); }
    *handle = fake; note_alloc("physical-virtual", size, nullptr);
    ascend_time_jump_us(1);
    ascend_overhead_note("mem_alloc_virtual", ascend_now_ns() - _t0);
    return ACL_SUCCESS;
  }
  using Fn = aclError (*)(void**, size_t, const aclrtPhysicalMemProp*, uint64_t);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMallocPhysical"));
  aclError rc = fn ? fn(handle, size, prop, flags) : ACL_ERROR_API_NOT_SUPPORT;
  if (rc == ACL_SUCCESS) note_alloc("physical-real", size, handle ? *handle : nullptr);
  ascend_overhead_note("mem_alloc_real", ascend_now_ns() - _t0);
  return rc;
}
static bool is_fake_handle(void* handle) { std::lock_guard<std::mutex> lock(g_mutex); return g_fake_handles.count(handle) != 0; }
extern "C" aclError aclrtMapMem(void* virPtr, size_t size, size_t offset, void* handle, uint64_t flags) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory() && is_fake_handle(handle)) { { std::lock_guard<std::mutex> lock(g_mutex); g_virtual[reinterpret_cast<uintptr_t>(virPtr)] = size; } note_alloc("map-virtual", size, virPtr); ascend_time_jump_us(1); ascend_overhead_note("mem_map_virtual", ascend_now_ns() - _t0); return ACL_SUCCESS; }
  using Fn = aclError (*)(void*, size_t, size_t, void*, uint64_t);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMapMem"));
  ascend_overhead_note("mem_map_real", ascend_now_ns() - _t0);
  return fn ? fn(virPtr, size, offset, handle, flags) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtUnmapMem(void* virPtr) {
  long long _t0 = ascend_now_ns();
  if (emulate() && virtual_memory()) {
    bool known = false; { std::lock_guard<std::mutex> lock(g_mutex); known = g_virtual.erase(reinterpret_cast<uintptr_t>(virPtr)) != 0; }
    if (known) { note_alloc("unmap-virtual", 0, virPtr); ascend_overhead_note("mem_unmap_virtual", ascend_now_ns() - _t0); return ACL_SUCCESS; }
  }
  using Fn = aclError (*)(void*); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtUnmapMem"));
  ascend_overhead_note("mem_unmap_real", ascend_now_ns() - _t0);
  return fn ? fn(virPtr) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtFreePhysical(void* handle) {
  { std::lock_guard<std::mutex> lock(g_mutex); if (g_fake_handles.erase(handle)) { std::free(handle); return ACL_SUCCESS; } }
  using Fn = aclError (*)(void*); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtFreePhysical"));
  return fn ? fn(handle) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtReserveMemAddress(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags) {
  long long _t0 = ascend_now_ns();
  using Fn = aclError (*)(void**, size_t, size_t, void*, uint64_t);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtReserveMemAddress"));
  aclError rc = fn ? fn(virPtr, size, alignment, expectPtr, flags) : ACL_ERROR_API_NOT_SUPPORT;
  if (rc == ACL_SUCCESS) note_alloc("reserve", size, virPtr ? *virPtr : nullptr);
  ascend_overhead_note("mem_reserve", ascend_now_ns() - _t0);
  return rc;
}
extern "C" aclError aclrtReleaseMemAddress(void* virPtr) {
  long long _t0 = ascend_now_ns();
  using Fn = aclError (*)(void*); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtReleaseMemAddress"));
  ascend_overhead_note("mem_release", ascend_now_ns() - _t0);
  return fn ? fn(virPtr) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind) {
  long long _t0 = ascend_now_ns();
  if (emulate_copy(count, kind, dst, src)) { synthesize_device_read(dst, destMax, count, kind); note_skip("aclrtMemcpy", count); ascend_host_blocking(duration("aclrtMemcpy", count)); ascend_overhead_note("mem_memcpy_skip", ascend_now_ns() - _t0); return ACL_SUCCESS; }
  using Fn = aclError (*)(void*, size_t, const void*, size_t, aclrtMemcpyKind); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMemcpy"));
  ascend_overhead_note("mem_memcpy_check", ascend_now_ns() - _t0);
  return fn ? fn(dst, destMax, src, count, kind) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtMemset(void* ptr, size_t maxCount, int32_t value, size_t count) {
  if (emulate() && virtual_memory() && is_virtual_ptr(ptr)) { note_skip("aclrtMemset", count); ascend_host_blocking(duration("aclrtMemset", count)); return ACL_SUCCESS; }
  using Fn = aclError (*)(void*, size_t, int32_t, size_t); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMemset"));
  return fn ? fn(ptr, maxCount, value, count) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtMemcpyAsync(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream) {
  long long _t0 = ascend_now_ns();
  if (emulate_copy(count, kind, dst, src)) { synthesize_device_read(dst, destMax, count, kind); note_skip("aclrtMemcpyAsync", count); ascend_op_latency(duration("aclrtMemcpyAsync", count), stream); ascend_overhead_note("mem_memcpyasync_skip", ascend_now_ns() - _t0); return ACL_SUCCESS; }
  using Fn = aclError (*)(void*, size_t, const void*, size_t, aclrtMemcpyKind, aclrtStream);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMemcpyAsync"));
  ascend_overhead_note("mem_memcpyasync_check", ascend_now_ns() - _t0);
  return fn ? fn(dst, destMax, src, count, kind, stream) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtMemcpyAsyncWithCondition(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream) {
  if (emulate_copy(count, kind, dst, src)) { synthesize_device_read(dst, destMax, count, kind); note_skip("aclrtMemcpyAsyncWithCondition", count); ascend_op_latency(duration("aclrtMemcpyAsyncWithCondition", count), stream); return ACL_SUCCESS; }
  using Fn = aclError (*)(void*, size_t, const void*, size_t, aclrtMemcpyKind, aclrtStream);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMemcpyAsyncWithCondition"));
  return fn ? fn(dst, destMax, src, count, kind, stream) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtMemsetAsync(void* ptr, size_t maxCount, int32_t value, size_t count, aclrtStream stream) {
  if (emulate() && virtual_memory() && is_virtual_ptr(ptr)) { note_skip("aclrtMemsetAsync", count); ascend_op_latency(duration("aclrtMemsetAsync", count), stream); return ACL_SUCCESS; }
  using Fn = aclError (*)(void*, size_t, int32_t, size_t, aclrtStream);
  auto fn = reinterpret_cast<Fn>(real_lookup("aclrtMemsetAsync"));
  return fn ? fn(ptr, maxCount, value, count, stream) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" void* ascend_memory_wrapper_for_symbol(const char* name) {
  if (!name) return nullptr;
  if (!std::strcmp(name, "aclrtMalloc")) return reinterpret_cast<void*>(&aclrtMalloc);
  if (!std::strcmp(name, "aclrtMallocAlign32")) return reinterpret_cast<void*>(&aclrtMallocAlign32);
  if (!std::strcmp(name, "aclrtMallocCached")) return reinterpret_cast<void*>(&aclrtMallocCached);
  if (!std::strcmp(name, "aclrtFree")) return reinterpret_cast<void*>(&aclrtFree);
  if (!std::strcmp(name, "aclrtMallocPhysical")) return reinterpret_cast<void*>(&aclrtMallocPhysical);
  if (!std::strcmp(name, "aclrtMapMem")) return reinterpret_cast<void*>(&aclrtMapMem);
  if (!std::strcmp(name, "aclrtUnmapMem")) return reinterpret_cast<void*>(&aclrtUnmapMem);
  if (!std::strcmp(name, "aclrtFreePhysical")) return reinterpret_cast<void*>(&aclrtFreePhysical);
  if (!std::strcmp(name, "aclrtReserveMemAddress")) return reinterpret_cast<void*>(&aclrtReserveMemAddress);
  if (!std::strcmp(name, "aclrtReleaseMemAddress")) return reinterpret_cast<void*>(&aclrtReleaseMemAddress);
  if (!std::strcmp(name, "aclrtMemcpy")) return reinterpret_cast<void*>(&aclrtMemcpy);
  if (!std::strcmp(name, "aclrtMemset")) return reinterpret_cast<void*>(&aclrtMemset);
  if (!std::strcmp(name, "aclrtMemcpyAsync")) return reinterpret_cast<void*>(&aclrtMemcpyAsync);
  if (!std::strcmp(name, "aclrtMemcpyAsyncWithCondition")) return reinterpret_cast<void*>(&aclrtMemcpyAsyncWithCondition);
  if (!std::strcmp(name, "aclrtMemsetAsync")) return reinterpret_cast<void*>(&aclrtMemsetAsync);
  return nullptr;
}
extern "C" void ascend_memory_set_real_symbol(const char*, void*) {}
'''
    duration_lines = [
        "static unsigned long long duration(const char* name, size_t bytes) {",
        "  long long fixed = fixed_mem_us();",
        "  if (fixed >= 0) return static_cast<unsigned long long>(fixed);",
    ]
    for name, value in sorted(defaults.items()):
        if name.startswith("aclrt") or "Memcpy" in name or "Memset" in name:
            duration_lines.append(
                f"  if (std::strcmp(name, {json.dumps(name)}) == 0) return {max(0.0, value):.0f}ULL;"
            )
    duration_lines.extend(["  return bytes ? (bytes / 1024 + 1) : 1;", "}"])
    source = source.replace(
        "static unsigned long long duration(const char* name, size_t bytes) {\n  const char* fixed = std::getenv(\"ASCEND_EMULATOR_FIXED_MEM_US\");\n  if (fixed) return std::strtoull(fixed, nullptr, 10);\n  (void)name; return bytes ? (bytes / 1024 + 1) : 1;\n}",
        "\n".join(duration_lines),
    )
    return source + rt_skip_source()


def device_interposer_source() -> str:
    # In sleep mode the host must not block inside each operator wrapper
    # (that serializes the asynchronous native timeline). Instead every
    # operator advances a per-stream virtual clock, and the host blocks here,
    # at the real synchronization points vLLM itself uses.
    return r'''#include <acl/acl_rt.h>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>

extern "C" void ascend_device_sync(void*);
extern "C" void ascend_device_sync_all(void);
extern "C" void ascend_device_record_event(void*, void*);
extern "C" void ascend_device_wait_event(void*, void*);
extern "C" void ascend_device_sync_event(void*);
extern "C" void ascend_graph_capture_begin(void*);
extern "C" void ascend_graph_capture_end(void*, void*);
extern "C" void ascend_graph_replay(void*, void*);
extern "C" void ascend_graph_destroy(void*);
extern "C" void* ascend_real_dlsym(const char*);
extern "C" void ascend_overhead_note(const char*, long long);
extern "C" long long ascend_now_ns(void);

static bool env_flag(const char* name, bool def) {
  const char* v = std::getenv(name);
  return (!v || !*v) ? def : !(std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}
static bool emulate() { static const bool value = env_flag("ASCEND_EMULATOR_EMULATE", true); return value; }
static void* real_lookup(const char* name) { return ascend_real_dlsym(name); }

extern "C" aclError aclrtSynchronizeStream(aclrtStream stream) {
  long long _t0 = ascend_now_ns();
  if (emulate()) { ascend_device_sync(stream); ascend_overhead_note("dev_sync_stream", ascend_now_ns() - _t0); return 0; }
  using Fn = aclError (*)(aclrtStream); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtSynchronizeStream")); ascend_overhead_note("dev_sync_stream_passthru", ascend_now_ns() - _t0); return fn ? fn(stream) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout) {
  long long _t0 = ascend_now_ns();
  if (emulate()) { ascend_device_sync(stream); ascend_overhead_note("dev_sync_stream", ascend_now_ns() - _t0); return 0; }
  using Fn = aclError (*)(aclrtStream, int32_t); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtSynchronizeStreamWithTimeout")); ascend_overhead_note("dev_sync_stream_passthru", ascend_now_ns() - _t0); return fn ? fn(stream, timeout) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtSynchronizeDevice(void) {
  long long _t0 = ascend_now_ns();
  if (emulate()) { ascend_device_sync_all(); ascend_overhead_note("dev_sync_device", ascend_now_ns() - _t0); return 0; }
  using Fn = aclError (*)(void); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtSynchronizeDevice")); ascend_overhead_note("dev_sync_device_passthru", ascend_now_ns() - _t0); return fn ? fn() : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtSynchronizeDeviceWithTimeout(int32_t timeout) {
  if (emulate()) { (void)timeout; ascend_device_sync_all(); return ACL_SUCCESS; }
  using Fn = aclError (*)(int32_t); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtSynchronizeDeviceWithTimeout"));
  return fn ? fn(timeout) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  long long _t0 = ascend_now_ns();
  if (emulate()) { ascend_device_record_event(event, stream); ascend_overhead_note("dev_record_event", ascend_now_ns() - _t0); return 0; }
  using Fn = aclError (*)(aclrtEvent, aclrtStream); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtRecordEvent")); ascend_overhead_note("dev_record_event_passthru", ascend_now_ns() - _t0); return fn ? fn(event, stream) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
  long long _t0 = ascend_now_ns();
  if (emulate()) { ascend_device_wait_event(stream, event); ascend_overhead_note("dev_wait_event", ascend_now_ns() - _t0); return 0; }
  using Fn = aclError (*)(aclrtStream, aclrtEvent); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtStreamWaitEvent")); ascend_overhead_note("dev_wait_event_passthru", ascend_now_ns() - _t0); return fn ? fn(stream, event) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtStreamWaitEventWithTimeout(aclrtStream stream, aclrtEvent event, int32_t timeout) {
  if (emulate()) { (void)timeout; ascend_device_wait_event(stream, event); return ACL_SUCCESS; }
  using Fn = aclError (*)(aclrtStream, aclrtEvent, int32_t); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtStreamWaitEventWithTimeout"));
  return fn ? fn(stream, event, timeout) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtSynchronizeEvent(aclrtEvent event) {
  long long _t0 = ascend_now_ns();
  if (emulate()) { ascend_device_sync_event(event); ascend_overhead_note("dev_sync_event", ascend_now_ns() - _t0); return 0; }
  using Fn = aclError (*)(aclrtEvent); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtSynchronizeEvent")); ascend_overhead_note("dev_sync_event_passthru", ascend_now_ns() - _t0); return fn ? fn(event) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclrtSynchronizeEventWithTimeout(aclrtEvent event, int32_t timeout) {
  if (emulate()) { (void)timeout; ascend_device_sync_event(event); return ACL_SUCCESS; }
  using Fn = aclError (*)(aclrtEvent, int32_t); auto fn = reinterpret_cast<Fn>(real_lookup("aclrtSynchronizeEventWithTimeout"));
  return fn ? fn(event, timeout) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode) {
  using Fn = aclError (*)(aclrtStream, aclmdlRICaptureMode); auto fn = reinterpret_cast<Fn>(real_lookup("aclmdlRICaptureBegin"));
  aclError rc = fn ? fn(stream, mode) : ACL_ERROR_API_NOT_SUPPORT;
  if (rc == ACL_SUCCESS && emulate()) ascend_graph_capture_begin(stream);
  return rc;
}
extern "C" aclError aclmdlRICaptureEnd(aclrtStream stream, aclmdlRI* model) {
  using Fn = aclError (*)(aclrtStream, aclmdlRI*); auto fn = reinterpret_cast<Fn>(real_lookup("aclmdlRICaptureEnd"));
  aclError rc = fn ? fn(stream, model) : ACL_ERROR_API_NOT_SUPPORT;
  if (rc == ACL_SUCCESS && emulate() && model) ascend_graph_capture_end(stream, *model);
  return rc;
}
extern "C" aclError aclmdlRIExecuteAsync(aclmdlRI model, aclrtStream stream) {
  if (emulate()) { ascend_graph_replay(model, stream); return ACL_SUCCESS; }
  using Fn = aclError (*)(aclmdlRI, aclrtStream); auto fn = reinterpret_cast<Fn>(real_lookup("aclmdlRIExecuteAsync"));
  return fn ? fn(model, stream) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclmdlRIExecute(aclmdlRI model, int32_t timeout) {
  if (emulate()) { (void)timeout; ascend_graph_replay(model, nullptr); ascend_device_sync_all(); return ACL_SUCCESS; }
  using Fn = aclError (*)(aclmdlRI, int32_t); auto fn = reinterpret_cast<Fn>(real_lookup("aclmdlRIExecute"));
  return fn ? fn(model, timeout) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" aclError aclmdlRIDestroy(aclmdlRI model) {
  ascend_graph_destroy(model);
  using Fn = aclError (*)(aclmdlRI); auto fn = reinterpret_cast<Fn>(real_lookup("aclmdlRIDestroy"));
  return fn ? fn(model) : ACL_ERROR_API_NOT_SUPPORT;
}
extern "C" void* ascend_device_wrapper_for_symbol(const char* name) {
  if (!name) return nullptr;
  if (!std::strcmp(name, "aclrtSynchronizeStream")) return reinterpret_cast<void*>(&aclrtSynchronizeStream);
  if (!std::strcmp(name, "aclrtSynchronizeStreamWithTimeout")) return reinterpret_cast<void*>(&aclrtSynchronizeStreamWithTimeout);
  if (!std::strcmp(name, "aclrtSynchronizeDevice")) return reinterpret_cast<void*>(&aclrtSynchronizeDevice);
  if (!std::strcmp(name, "aclrtSynchronizeDeviceWithTimeout")) return reinterpret_cast<void*>(&aclrtSynchronizeDeviceWithTimeout);
  if (!std::strcmp(name, "aclrtRecordEvent")) return reinterpret_cast<void*>(&aclrtRecordEvent);
  if (!std::strcmp(name, "aclrtStreamWaitEvent")) return reinterpret_cast<void*>(&aclrtStreamWaitEvent);
  if (!std::strcmp(name, "aclrtStreamWaitEventWithTimeout")) return reinterpret_cast<void*>(&aclrtStreamWaitEventWithTimeout);
  if (!std::strcmp(name, "aclrtSynchronizeEvent")) return reinterpret_cast<void*>(&aclrtSynchronizeEvent);
  if (!std::strcmp(name, "aclrtSynchronizeEventWithTimeout")) return reinterpret_cast<void*>(&aclrtSynchronizeEventWithTimeout);
  if (!std::strcmp(name, "aclmdlRICaptureBegin")) return reinterpret_cast<void*>(&aclmdlRICaptureBegin);
  if (!std::strcmp(name, "aclmdlRICaptureEnd")) return reinterpret_cast<void*>(&aclmdlRICaptureEnd);
  if (!std::strcmp(name, "aclmdlRIExecuteAsync")) return reinterpret_cast<void*>(&aclmdlRIExecuteAsync);
  if (!std::strcmp(name, "aclmdlRIExecute")) return reinterpret_cast<void*>(&aclmdlRIExecute);
  if (!std::strcmp(name, "aclmdlRIDestroy")) return reinterpret_cast<void*>(&aclmdlRIDestroy);
  return nullptr;
}
'''


def virtual_time_source() -> str:
    return r'''// Ascend virtual-time module with an asynchronous device model.
//
// ASCEND_EMULATOR_TIME_MODE controls how each intercepted operator accounts for its
// predicted latency:
//   "jump"  (default) accumulate virtual time only, no sleep (fast simulation)
//   "sleep"           reproduce vLLM's *async* control flow: ops are handed to
//                      a per-stream virtual device clock without blocking the
//                      host thread; the host only blocks at synchronization
//                      points (aclrtSynchronizeStream/Device/Event). This keeps
//                      stream-order serialization and cross-stream overlap.
//
// Both modes accumulate a process-local work counter, printed periodically
// (ASCEND_EMULATOR_TIME_LOG_INTERVAL_MS, default 5000; 0 disables) and at exit, and
// readable via ascend_virtual_time_us()/ascend_virtual_time_reset().
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include <pthread.h>
static std::atomic<unsigned long long> g_virtual_time_us{0};
static std::atomic<bool> g_reported{false};
extern "C" void ascend_tensor_registry_stats(unsigned long long*, unsigned long long*, unsigned long long*);
extern "C" void ascend_memory_registry_stats(unsigned long long*, unsigned long long*);

// Virtual device state (sleep mode): each stream has a "busy until" timestamp
// (monotonic microseconds) — the moment all work submitted to it is done.
// Different streams overlap; a timeline is only as long as its own chain.
static std::mutex g_dev_mutex;
static std::unordered_map<void*, unsigned long long> g_stream_busy;
static std::unordered_map<void*, unsigned long long> g_event_time;
static std::unordered_map<void*, unsigned long long> g_capture_work;
static std::unordered_map<void*, unsigned long long> g_graph_work;
static std::atomic<unsigned int> g_active_captures{0};
static std::atomic<unsigned long long> g_graph_captures{0};
static std::atomic<unsigned long long> g_graph_replays{0};
static std::atomic<unsigned long long> g_graph_replay_us{0};
static char g_default_stream_key;  // sentinel for the NULL/default stream

static bool time_mode_sleep() {
  static const bool cached = [] {
    const char* v = std::getenv("ASCEND_EMULATOR_TIME_MODE");
    return v && *v && (std::strcmp(v, "sleep") == 0 || std::strcmp(v, "real") == 0 || std::strcmp(v, "1") == 0);
  }();
  return cached;
}

static unsigned long long steady_now_us() {
  return static_cast<unsigned long long>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
}

static void sleep_us(unsigned long long us) {
  if (us == 0) return;
  struct timespec req{};
  req.tv_sec = static_cast<time_t>(us / 1000000ULL);
  req.tv_nsec = static_cast<long>((us % 1000000ULL) * 1000ULL);
  while (nanosleep(&req, &req) != 0) {
    if (errno != EINTR) break;
  }
}

static void sleep_until_us(unsigned long long target) {
  unsigned long long now = steady_now_us();
  if (target > now) sleep_us(target - now);
}

static void* stream_key(void* stream) { return stream ? stream : static_cast<void*>(&g_default_stream_key); }

// Main operator accounting entry. The wrappers pass the aclrtStream the native
// call would have been enqueued on, so the device model can honor stream order
// and cross-stream overlap instead of summing latencies on the host thread.
extern "C" void ascend_op_latency(unsigned long long duration_us, void* stream) {
  if (g_active_captures.load(std::memory_order_relaxed) != 0) {
    std::lock_guard<std::mutex> lock(g_dev_mutex);
    auto capture = g_capture_work.find(stream_key(stream));
    if (capture != g_capture_work.end()) {
      capture->second += duration_us;
      return;
    }
  }
  g_virtual_time_us.fetch_add(duration_us, std::memory_order_relaxed);
  if (!time_mode_sleep()) return;
  if (duration_us == 0) return;
  std::lock_guard<std::mutex> lock(g_dev_mutex);
  unsigned long long& busy = g_stream_busy[stream_key(stream)];
  busy = std::max(busy, steady_now_us()) + duration_us;
}

extern "C" void ascend_time_jump_us(unsigned long long duration_us) {
  ascend_op_latency(duration_us, nullptr);
}

// Synchronous host APIs (e.g. aclrtMemcpy, aclrtMemset) block the calling
// thread in native code; sleep mode reproduces that directly.
extern "C" void ascend_host_blocking(unsigned long long duration_us) {
  g_virtual_time_us.fetch_add(duration_us, std::memory_order_relaxed);
  if (time_mode_sleep()) sleep_us(duration_us);
}

extern "C" void ascend_device_sync(void* stream) {
  if (!time_mode_sleep()) return;
  unsigned long long target = 0;
  { std::lock_guard<std::mutex> lock(g_dev_mutex); target = g_stream_busy[stream_key(stream)]; }
  sleep_until_us(target);
}

extern "C" void ascend_device_sync_all(void) {
  if (!time_mode_sleep()) return;
  unsigned long long target = 0;
  { std::lock_guard<std::mutex> lock(g_dev_mutex);
    for (const auto& entry : g_stream_busy) target = std::max(target, entry.second); }
  sleep_until_us(target);
}

extern "C" void ascend_device_record_event(void* event, void* stream) {
  if (!time_mode_sleep()) return;
  std::lock_guard<std::mutex> lock(g_dev_mutex);
  g_event_time[event] = g_stream_busy[stream_key(stream)];
}

extern "C" void ascend_device_wait_event(void* stream, void* event) {
  if (!time_mode_sleep()) return;
  std::lock_guard<std::mutex> lock(g_dev_mutex);
  unsigned long long at = g_event_time[event];
  unsigned long long& busy = g_stream_busy[stream_key(stream)];
  busy = std::max(busy, at);
}

extern "C" void ascend_device_sync_event(void* event) {
  if (!time_mode_sleep()) return;
  unsigned long long target = 0;
  { std::lock_guard<std::mutex> lock(g_dev_mutex); target = g_event_time[event]; }
  sleep_until_us(target);
}

extern "C" unsigned long long ascend_virtual_time_us(void) {
  return g_virtual_time_us.load(std::memory_order_relaxed);
}

extern "C" void ascend_virtual_time_reset(void) {
  g_virtual_time_us.store(0, std::memory_order_relaxed);
}

extern "C" void ascend_graph_capture_begin(void* stream) {
  std::lock_guard<std::mutex> lock(g_dev_mutex);
  g_capture_work[stream_key(stream)] = 0;
  g_active_captures.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void ascend_graph_capture_end(void* stream, void* graph) {
  std::lock_guard<std::mutex> lock(g_dev_mutex);
  auto it = g_capture_work.find(stream_key(stream));
  unsigned long long duration = it == g_capture_work.end() ? 0 : it->second;
  if (it != g_capture_work.end()) g_capture_work.erase(it);
  if (g_active_captures.load(std::memory_order_relaxed) != 0) g_active_captures.fetch_sub(1, std::memory_order_relaxed);
  g_graph_work[graph] = duration;
  g_graph_captures.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void ascend_graph_replay(void* graph, void* stream) {
  unsigned long long duration = 0;
  { std::lock_guard<std::mutex> lock(g_dev_mutex); auto it = g_graph_work.find(graph); if (it != g_graph_work.end()) duration = it->second; }
  g_graph_replays.fetch_add(1, std::memory_order_relaxed);
  g_graph_replay_us.fetch_add(duration, std::memory_order_relaxed);
  ascend_op_latency(duration, stream);
}

extern "C" void ascend_graph_destroy(void* graph) {
  std::lock_guard<std::mutex> lock(g_dev_mutex);
  g_graph_work.erase(graph);
}

extern "C" void ascend_collective_barrier(const char*) {
}

static bool log_enabled() {
  static const bool value = [] { const char* v = std::getenv("ASCEND_EMULATOR_LOG_TIME"); return !(v && std::strcmp(v, "0") == 0); }();
  return value;
}

static void log_time(bool force) {
  if (!log_enabled()) return;
  unsigned long long us = g_virtual_time_us.load(std::memory_order_relaxed);
  if (!force) {
    static std::atomic<unsigned long long> last{0};
    if (us == last.load(std::memory_order_relaxed)) return;
    last.store(us, std::memory_order_relaxed);
  }
  std::fprintf(stderr, "[ascend-virtual-time] mode=%s total_us=%llu (%.3f s)\n",
               time_mode_sleep() ? "sleep" : "jump",
               static_cast<unsigned long long>(us), us / 1000000.0);
}

// Interceptor host-overhead accounting (ASCEND_EMULATOR_LOG_OVERHEAD=1 dumps at exit).
static std::mutex g_ov_mutex;
static std::unordered_map<std::string, std::pair<long long, long long>> g_overhead_ns;
static bool overhead_log_enabled() {
  static const bool value = [] { const char* v = std::getenv("ASCEND_EMULATOR_LOG_OVERHEAD"); return v && std::strcmp(v, "0") != 0; }();
  return value;
}
extern "C" void ascend_overhead_note(const char* name, long long ns) {
  if (!overhead_log_enabled() || !ns) return;
  std::lock_guard<std::mutex> lock(g_ov_mutex);
  auto& p = g_overhead_ns[name];
  p.first += ns; p.second += 1;
}
extern "C" long long ascend_now_ns(void) {
  if (!overhead_log_enabled()) return 0;
  return static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
static void dump_overhead() {
  if (!overhead_log_enabled()) return;
  unsigned long long live = 0, created = 0, destroyed = 0;
  unsigned long long ranges = 0, handles = 0;
  ascend_tensor_registry_stats(&live, &created, &destroyed);
  ascend_memory_registry_stats(&ranges, &handles);
  std::fprintf(stderr, "[ascend-overhead] tensor_registry: live=%llu created=%llu destroyed=%llu\n", live, created, destroyed);
  std::fprintf(stderr, "[ascend-overhead] memory_registry: ranges=%llu handles=%llu\n", ranges, handles);
  std::fprintf(stderr, "[ascend-overhead] graphs: captures=%llu replays=%llu replay_us=%llu\n",
               g_graph_captures.load(std::memory_order_relaxed),
               g_graph_replays.load(std::memory_order_relaxed),
               g_graph_replay_us.load(std::memory_order_relaxed));
  std::lock_guard<std::mutex> lock(g_ov_mutex);
  for (const auto& kv : g_overhead_ns) {
    std::fprintf(stderr, "[ascend-overhead] %s: %.3f ms / %lld calls (%.0f ns avg)\n",
                 kv.first.c_str(), kv.second.first / 1e6, static_cast<long long>(kv.second.second),
                 kv.second.second ? static_cast<double>(kv.second.first) / kv.second.second : 0.0);
  }
}

static void report_exit() {
  bool expected = false;
  if (!g_reported.compare_exchange_strong(expected, true)) return;
  log_time(true);
  dump_overhead();
}

static long long interval_ms() {
  static const long long value = [] { const char* v = std::getenv("ASCEND_EMULATOR_TIME_LOG_INTERVAL_MS"); if (!v || !*v) return 5000LL; long long x = std::atoll(v); return x < 0 ? 5000LL : x; }();
  return value;
}

static void* reporter_main(void*) {
  long long ms = interval_ms();
  while (ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    log_time(false);
    if (overhead_log_enabled()) dump_overhead();
  }
  return nullptr;
}

struct AscendTimeRegistrar {
  AscendTimeRegistrar() {
    std::atexit(report_exit);
    if (interval_ms() > 0) {
      pthread_t t;
      if (pthread_create(&t, nullptr, reporter_main, nullptr) == 0) pthread_detach(t);
    }
  }
  ~AscendTimeRegistrar() { report_exit(); }
};
static AscendTimeRegistrar g_registrar;
'''


def default_cann_include() -> Path:
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        candidate = Path(home) / "aarch64-linux" / "include"
        if candidate.is_dir():
            return candidate
    for candidate in sorted(Path("/usr/local/Ascend").glob("cann-*/aarch64-linux/include"), reverse=True):
        if candidate.is_dir():
            return candidate
    return Path("/usr/local/Ascend/cann-8.5.1/aarch64-linux/include")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--cann-include", type=Path, default=default_cann_include())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    generated = args.output / "generated"
    generated.mkdir(exist_ok=True)
    aclnn, hccl = event_symbols(args.profile_manifest.resolve())
    # Derive the missing GetWorkspaceSize twins from each execute symbol so the
    # workspace call (which carries the concrete aclTensor args) can be wrapped
    # for shape extraction. The profiler only records a subset of workspace
    # API calls, but the symbols are name-derived plain ACLNN interfaces.
    aclnn = sorted(set(aclnn) | {s + EXECUTE_SUFFIX for s in aclnn if not s.endswith(EXECUTE_SUFFIX)})
    defaults = model_defaults(args.model_manifest.resolve(), aclnn)
    model_known, model_low_confidence = model_status(args.model_manifest.resolve(), aclnn)
    latency_table, table_symbols, feature_masks = latency_table_source(
        args.model_manifest.resolve(), aclnn
    )
    compute, unsupported = compute_source(
        aclnn, args.cann_include.resolve(), defaults, model_known, model_low_confidence,
        table_symbols, feature_masks,
    )
    workspace_passthrough = [item for item in unsupported if item["symbol"].endswith(EXECUTE_SUFFIX)]
    unsupported = [item for item in unsupported if not item["symbol"].endswith(EXECUTE_SUFFIX)]
    # Execute symbols without public headers still use the uniform ACLNN
    # execute signature; only unknown workspace entries become passthrough notes.
    unmodeled = sorted(set(aclnn) - model_known - {s for s in aclnn if s.endswith(EXECUTE_SUFFIX)})
    (generated / "compute_interposer.cpp").write_text(compute, encoding="utf-8")
    (generated / "communication_interposer.cpp").write_text(
        communication_source(defaults, model_known, model_low_confidence), encoding="utf-8"
    )
    (generated / "memory_interposer.cpp").write_text(memory_source(defaults), encoding="utf-8")
    (generated / "virtual_time.cpp").write_text(virtual_time_source(), encoding="utf-8")
    (generated / "device_interposer.cpp").write_text(device_interposer_source(), encoding="utf-8")
    (generated / "latency_table.cpp").write_text(latency_table, encoding="utf-8")
    (generated / "rf_forest.cpp").unlink(missing_ok=True)  # legacy name, superseded by latency_table.cpp
    (generated / "latency_lookup.json").write_text(json.dumps(defaults, indent=2) + "\n", encoding="utf-8")
    cmake = """cmake_minimum_required(VERSION 3.16)
project(ascend_emulator CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
add_library(ascend_emulator SHARED compute_interposer.cpp communication_interposer.cpp memory_interposer.cpp device_interposer.cpp latency_table.cpp virtual_time.cpp)
target_include_directories(ascend_emulator PRIVATE $ENV{ASCEND_HOME_PATH}/aarch64-linux/include)
target_link_libraries(ascend_emulator PRIVATE dl pthread)
"""
    (generated / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "manifest_type": "ascend_emulator",
        "model_id": args.model_id,
        "profile_manifest": str(args.profile_manifest.resolve()),
        "model_manifest": str(args.model_manifest.resolve()),
        "generated_files": sorted(path.name for path in generated.iterdir()),
        "observed_acl_nn_symbols": aclnn,
        "observed_hccl_symbols": hccl,
        "supported_acl_nn_symbols": sorted(set(aclnn) - {item["symbol"] for item in unsupported}),
        "unmodeled_acl_nn_symbols": unmodeled,
        "low_confidence_model_symbols": sorted(set(aclnn) & model_low_confidence),
        "shape_table_symbols": sorted(table_symbols),
        "unsupported": unsupported,
        "workspace_passthrough": workspace_passthrough,
        "strict_default": True,
    }
    (generated / "emulator_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (generated / "coverage_report.json").write_text(json.dumps({
        "unsupported": unsupported,
        "workspace_passthrough": workspace_passthrough,
        "observed_hccl_symbols": hccl,
        "unmodeled_acl_nn_symbols": unmodeled,
        "low_confidence_model_symbols": sorted(set(aclnn) & model_low_confidence),
        "shape_table_symbols": sorted(table_symbols),
        "strict_requires_independent_sweep": bool(model_low_confidence),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(generated), "aclnn": len(aclnn), "hccl": len(hccl), "unsupported": len(unsupported), "shape_table": len(table_symbols)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
