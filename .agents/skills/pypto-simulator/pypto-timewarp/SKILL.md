---
name: pypto-timewarp
description: 从已通过 strict coverage 的 PyPTO sleep plan 派生共享虚拟时钟路线。
---

# PyPTO Timewarp

本 skill 是 sleep 路线之上的独立无 NPU 加速层。sleep 仍是正确性 oracle；timewarp 不替代 native
profile、sleep bracket 或 a2a3 device-delay bracket。

## 运行边界

- parent orchestration、TensorMap、TaskId、MIX/SPMD、early dispatch、同步、ACK/FIN、completion
  和 request lifecycle 继续执行；
- child stub 保留原 ABI、telemetry 和 control output，但不做 wall-clock sleep；
- program duration 来自 profile task DAG、subslot、sorted block-rank occupancy 和资源竞争；
- worker 在 `StepResult` seal 后提交 step frontier；engine 在输出可见前推进共享虚拟时钟；
- output producer/observer gate 防止后续 jump 越过调用方尚未读取的 token timestamp；
- host 控制面超过 modeled deadline 时 offset 不增加，strict launcher 记录并拒绝该 run。

当前 V1 的 DAG/resource schedule 在 plan 编译阶段完成，runtime 尚未把每个 task frontier 在线写回
simpler scheduler。大图和严格性能验收必须等待 runtime event integration 完成。

## 派生 plan

先生成并通过 strict coverage 的 sleep plan，再派生 timewarp plan：

```bash
python .agents/skills/pypto-simulator/pypto-timewarp/scripts/build_timewarp_plan.py \
  --sleep-plan /tmp/pypto-plan/simulation_plan.json \
  --output /tmp/pypto-plan/timewarp_plan.json \
  --aic-clusters 20
```

builder 读取 task predecessor、block count、active subslot、block-rank duration 和 AIC/AIV/MIX
资源，按 deterministic list scheduling 生成每个 program 的 frontier。它拒绝 capture role 歧义、
独立 DAG 不一致、缺 predecessor、cycle、未建模 callable、非法 duration 和 block vector 维度
冲突。Timewarp plan 只能交给 timewarp launcher，不能交给 sleep 或 native-delay launcher。

## 运行

```bash
python .agents/skills/pypto-simulator/pypto-timewarp/scripts/run_timewarp_simulator.py \
  --plan /tmp/pypto-plan/timewarp_plan.json \
  --memory-profile /tmp/memory-profile.json \
  --telemetry /tmp/timewarp/telemetry.jsonl \
  --trace /tmp/timewarp/clock.jsonl \
  --snapshot /tmp/timewarp/snapshot.json -- \
  python <streaming benchmark> --manifest <model workload manifest> -- \
  pypto-serving <a2a3sim serving command>
```

launcher 为本次 run 创建私有 Unix socket、mmap clock 和 TimeKeeper，并通过临时 `sitecustomize.py`
注入 hook。它不会访问 NPU，也不会修改全局 `time.monotonic`、`time.perf_counter` 或 timeout。
只有显式 `PYPTO_TIMEWARP_CLOCK` 存在时，benchmark 才读取虚拟时钟。

`--runtime-lib-dir` 必须直接或通过 `lib/` 包含
`a2a3/sim/tensormap_and_ringbuffer/libhost_runtime.so`；非法路径立即失败。

## TimeKeeper 协议

共享时钟为：

```text
virtual_now = CLOCK_MONOTONIC + non_decreasing_global_offset
```

TimeKeeper 是唯一 writer，mmap reader 使用 seqlock。推进 absolute deadline 时只执行：

```text
offset = max(offset, deadline - CLOCK_MONOTONIC)
```

deadline 已经过期时不增加 offset，并记录：

- `deadline_overrun_count`；
- `deadline_overrun_total_ns`；
- `deadline_overrun_max_ns`。

`--allow-deadline-overrun` 只用于保存诊断结果，不能用于 acceptance。

step key 为 `{dp_rank}:{step_id}`。worker lifecycle 调用 `begin_step`，L3 dispatch 追加 graph
frontier，encoded `StepResult` 入队时 `seal_step`，engine 通过 `wait_step` 等待 seal 和 output gate
清空。多个并发 request 的非 terminal output 分别建立 producer/observer gate；terminal output 不
建立该 gate。

初始化和 weight/KV materialization 复用 sleep 的 strict allocation/capacity/lifetime contract。
backend 已消耗的时间从 modeled startup duration 中扣除，剩余时间推进虚拟时钟；backend 超过目标
duration 时 fail closed。

## 验收

执行：

```text
sleep-A -> timewarp -> sleep-B
```

每个 case 和全局都必须比较 TTFT、TPOT、E2EL 的 mean/p50/p95，并检查：

- token count、finish reason、completion order；
- `offset_ns`、`jumped_ns`、graph event、completed step；
- output gate 是否归零；
- modeled deadline overrun；
- strict stub telemetry、unmodeled execution 和 timeout；
- timewarp wall runtime 是否低于 sleep。

只有全部统计量在 5% 内且输出顺序一致，timewarp 才能通过 sleep acceptance。Native 5% 验收仍
单独执行。当前只支持单卡、单 replica、无 TP、无多卡通信。

## 当前限制

- runtime callback 当前不携带 task、epoch、subslot、physical lane identity；
- simpler scheduler 尚未维护在线 dependency frontier 和 physical lane occupancy；
- 仍需修复大 outlined graph 的 custom runtime child abort 和 buffer lifecycle；
- 多 replica、TP/PP、跨节点、HCCL 和原生 `ExecutionMode.LATENCY_SIM` 尚未实现。

协议详见 `references/protocol.md`；通用模型输入见 `../references/model-adapter-contract.md`。
