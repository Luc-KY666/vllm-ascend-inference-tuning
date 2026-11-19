# Timewarp 协议与正确性契约

## 时钟

共享时钟为：

```text
virtual_now = CLOCK_MONOTONIC + non_decreasing_global_offset
```

mmap 文件保存 `{sequence, offset_ns}`。TimeKeeper 是唯一 writer，benchmark 是 seqlock reader。
推进 absolute deadline 时只执行：

```text
offset = max(offset, deadline - CLOCK_MONOTONIC)
```

Wall time 始终自然前进。RPC、orchestration、scheduler 和 IPC 已消耗的时间会减少剩余 jump，不能
再次叠加。Deadline 已经过期时 offset 不变；TimeKeeper 记录每个带 graph step 的
`deadline_overrun_ns`。Strict launcher 发现任一 overrun 就保留 snapshot/trace 并失败，不能把该
结果解释为已消除 host 控制面开销。`--allow-deadline-overrun` 仅用于诊断。

## Program Timeline

`build_timewarp_plan.py` 从 sleep plan 引用的 normalized profile 读取：

- task ID 和本地 predecessor；
- `block_num`、active subslot 和 func ID；
- 每个 subslot 的 block-rank duration；
- AIC cluster 数和 AIV0/AIV1 配对。

它按 dependency-ready time 和稳定 TaskId 顺序进行 list scheduling。纯 AIC block 使用 AIC pool，
纯 AIV block 使用全部 AIV pool，MIX block 原子选择一个 `{AIC,AIV0,AIV1}` cluster，并从各 lane
availability 与 predecessor frontier 的最大值开始。Task finish 是所有 block/lane finish 的最大值；
program finish 是所有 task finish 的最大值。

Block-rank 仍只表示排序后的 physical-block duration distribution，不恢复原 block identity。Builder
要求 vector 长度精确等于 `block_num`，不允许补齐、截断或外推。

## Runtime Step

每个 replica 的 step key 是 `{dp_rank}:{step_id}`。当前只接受单 replica，key 仍包含 dp rank 以避免
无意碰撞。

1. Worker FIFO device lane 执行 `_apply_command_lifecycle` 时 `begin_step`。
2. `_run_l3` 或 `_submit_l3` 在原生 dispatch 前提交 `{step, program, duration_ns}`。
3. TimeKeeper 维护单一 device frontier：`start=max(virtual_now, prior_device_frontier)`。
4. 原生 a2a3sim orchestration、scheduler 和 no-wait child 实际执行。
5. Async output reclaim 在自己的线程绑定原 step key；后续 sampling program 继续追加同一 frontier。
6. Worker output queue 在发布 encoded `StepResult` 前 `seal_step`。
7. Engine `_get_live_result` 收到 FIFO result 后执行 `wait_step`；只有前一输出没有待确认 observer
   generation 时，TimeKeeper 才推进到该 step frontier。
8. Request queue 在放入每个非 terminal `TokenOutput` 前调用 `output_expected`。
9. 异步输出生成器把 output 交给调用方；调用方读取虚拟时间并请求下一项或关闭生成器时，生成器
   调用 `output_delivered`。
10. 最后一个 ACK 清除 gate 后，后续 step 才能增加 offset。

Pipelined step 可以在前一个 output 被 engine 消费前提交；它们共享 device frontier，因此不会时间倒退。
Producer-to-observer gate防止后续step jump抢在调用方给当前token读取时间戳之前。Gate按实际非terminal
queue output计数，多个并发request必须分别ACK。Terminal output不建立gate，因为它可能需要后续
worker release step才能完成；其E2EL自然包含该FIFO release边界。Release-only 或 error step可以没有
graph event，但仍必须seal和deliver。

## 初始化与内存

不属于 request step 的 warmup program 在 graph RPC 返回前直接推进时钟到其 program frontier。Exact
weight/KV materialization 继续执行 sleep 路线的 allocation/capacity/lifecycle contract；backend 已
消耗的 wall time从 target duration 中扣除，剩余部分作为 startup virtual duration 推进。Backend
cost超过 measured target 时 fail closed。

## 隔离

Timewarp 有独立目录、plan mode、installer、launcher、环境变量和 TimeKeeper。它不修改：

- `install_sleep_hook.py`；
- `run_sleep_simulator.py`；
- sleep plan 的 `execution_mode=sleep`；
- native/device-delay launcher；
- Python 全局 `time.monotonic`、`time.perf_counter` 或 timeout 行为。

`run_streaming_benchmark.py` 只有在 `PYPTO_TIMEWARP_CLOCK` 存在时读取 mmap offset；默认分支仍直接
调用 `time.perf_counter()`。

## 失败策略

以下情况 fatal：socket/clock 缺失、重复 begin/seal、unknown step、sealed step 后追加 graph、
step wait timeout、unknown program、非法 duration、未建模 child、空 telemetry、DAG/role/block
contract 不一致，以及 strict 模式下任一 modeled deadline overrun。任何 RPC 错误都返回明确错误并
终止 workload，不能退化到独立进程时钟或 wall sleep。
