---
name: pypto-emulate
description: 构建、审计并运行严格的 PyPTO task 级 sleep 和 a2a3 device-delay plan。
---

# PyPTO Emulation

本 skill 消费 profile manifest 和 latency model，生成严格 simulation plan，并提供两个隔离后端：

- `a2a3sim` sleep：无 NPU 的 control-flow 基线，child 使用 host delay；
- `a2a3` device-delay：真实 NPU hardware-in-the-loop，child 使用同 ABI device delay kernel。

native profile 和 baseline 永不加载这两个 hook。

## Build plan

```bash
python .agents/skills/pypto-simulator/pypto-emulate/scripts/build_simulation_plan.py \
  --profile-manifest /tmp/prefill/profile_manifest.json \
  --profile-manifest /tmp/decode/profile_manifest.json \
  --model-manifest /tmp/pypto-model/model_manifest.json \
  --backend a2a3sim --sampled-token-id 0 \
  --require-baseline-confidence --strict \
  --output /tmp/pypto-plan
```

每个 graph task 保留 decision，runtime replacement surface 按稳定的 `program + func_id` 聚合。
Planner 必须记录 task、subslot、control output、capture role、candidate/excluded role 和 source
provenance。`--prefer-capture-role` 用于同一 callable 跨生命周期复用时显式选择 request/warmup
样本；工具不能按样本数量猜测。

strict 模式拒绝：

- missing model 或 runtime-unmapped task；
- unsupported/extrapolated feature；
- block count、subslot、control output 冲突；
- 未达到 confidence 门禁的 replacement point；
- 非法 token 或 memory contract。

## Run sleep

```bash
python .agents/skills/pypto-simulator/pypto-emulate/scripts/run_sleep_simulator.py \
  --plan /tmp/pypto-plan/simulation_plan.json \
  --memory-profile /tmp/memory-profile.json \
  --telemetry /tmp/pypto-plan/telemetry.jsonl -- \
  pypto-serving <包含 --platform a2a3sim 的 serving command>
```

launcher 通过临时 `sitecustomize.py` 注入 spawned worker，只替换 outlined child callable，保留：

- parent orchestration、TensorMap、TaskId 和依赖；
- AIC/AIV/MIX/SPMD resource shape；
- early dispatch、同步和 completion；
- request allocation/reclaim、ACK/FIN 和输出生命周期。

plan 外 child 立即失败，不允许 zero-duration fallback 或静默执行真实业务 kernel。telemetry 必须
至少包含一个 modeled stub，且不能出现 unmodeled execution。`--sim-aicpu-threads` 和 custom
runtime 只用于 host-contention 诊断，不能作为 5% 校准旋钮。

## Run device-delay

```bash
python .agents/skills/pypto-simulator/pypto-emulate/scripts/run_native_delay_simulator.py \
  --plan /tmp/pypto-plan-native/simulation_plan.json \
  --memory-profile /tmp/memory-profile.json \
  --audit /tmp/device-delay/audit.jsonl \
  --memory-audit /tmp/device-delay/memory-audit.jsonl \
  --npu-memory-telemetry /tmp/device-delay/npu-memory.json \
  --npu-memory-device 0 --zero-duration -- \
  pypto-serving <包含 --platform a2a3 的 serving command>
```

`--zero-duration` 只做 replacement reachability gate，不是性能结果。非零验证必须使用同一 physical
NPU 的：

```text
native-A -> device-delay -> native-B
```

每个 callable、subslot 和 block-rank 都必须有 model。device child 保持
`void kernel(__gm__ int64_t *args)` ABI，使用 50 MHz `get_sys_cnt()`；MIX AIV 需区分 AIV0/AIV1。
选择器必须驻留 ELF `.text`，不能生成未注册的 `.rodata` jump table。

## Weight/KV contract

提供 strict memory profile 时，sleep 和 device-delay 都必须保持原生：

- allocator、logical extent、canonical identity、borrow、in-flight free；
- KV sizing、capacity、free-memory 和 request lifetime；
- weight/KV materialization phase duration。

sleep 可跳过大 payload 初始化；device-delay 通过精确 one-shot allocation token 返回有效 NPU VA，仅
reserve virtual address，不提交 physical HBM。未知 allocation 不得按尺寸阈值猜测。bytes 不匹配、
token 未消费、backend 实际成本超过 modeled duration、logical memory 超预算或 memory audit 非零
physical backing 时立即失败。

模型特有的 memory layout 和 control output 只在 `references/<model>-adapter.md` 描述。

## Streaming validation

使用通用 streaming benchmark 统计：

- TTFT：request start 到首 token；
- TPOT：`(last_token-first_token)/(output_tokens-1)`；
- E2EL：request start 到 terminal output。

一次 callback 产生多个 token 时必须按 `completion_tokens` 增量保存 token boundary。每个 case 和全局
的 mean/p50/p95 都必须验收，不能只报告 E2E 或全局平均。

完整 bracket、memory audit 和模型适配说明见 `../references/manifest-contract.md`、
`../references/model-adapter-contract.md` 以及对应的 `references/<model>-validation.md`。
