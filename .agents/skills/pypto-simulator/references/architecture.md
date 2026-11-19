# PyPTO 模拟器架构

## 目标

复用 `pypto-serving` 的 request、scheduler、executor、runner 和 PyPTO orchestration 控制流，只在
outlined simpler task / child dispatch 边界替换 profile 已覆盖的设备工作。目标是 timing 行为，不是
模型数值输出。

```text
native PyPTO capture
  -> normalized profile
  -> task/subslot/block latency model
  -> simulation plan
  -> sleep / timewarp / device-delay backend
  -> unchanged serving lifecycle and metrics
```

当前验证只覆盖单卡、单 replica、无 TP、无真实多卡通信/HCCL。多 rank、TP、PP、跨节点和通信 actor
barrier 仍是待实现范围。

## 必须保留的语义

- orchestration、scope、TensorMap RAW/WAW 和显式 `deps=`；
- TaskId 完成、fanin/fanout release、early dispatch、`sync_start`；
- AIC、AIV、MIX、SPMD resource shape；
- window notify/wait counter、epoch 和同步完成边界；
- serving allocation、dispatch、reclaim、ACK/FIN、request state；
- control output 的 ABI 和 terminal/non-terminal 生命周期。

## 可以替换的工作

- outlined compute task；
- 有 profile 的 communication/window task；
- 访问虚拟 weight/KV/window backing 的大型操作；
- 被替换 producer 的控制输出 provider。

同步本身只保留状态和依赖，除非 profile 明确提供独立 runtime-overhead model。不能在
`worker.run()` 外层整体 sleep，也不能在 task 内部 DSL statement 层拦截。

## 三条 backend 路线

### Sleep

`a2a3sim` 保留 CPU parent orchestration 和 scheduler，在 child callable 处执行 host delay。它是
控制流和输出生命周期基线，但 Linux 线程、mutex、future、timer wakeup 和 simulator scheduler
开销不等同于真实 NPU。

### Timewarp

timewarp 复用 sleep 的 child boundary。child 只做 telemetry 和 control output，不做 wall sleep。
builder 从 task predecessor、subslot、sorted block-rank duration 和 AIC/AIV/MIX resource 竞争离线
编译 program frontier；TimeKeeper 用共享
`CLOCK_MONOTONIC + non_decreasing_global_offset` 将 frontier 映射到 StepResult 和 token output
边界。

当前 timewarp 仍是 V1：offline timeline 已有 task DAG，但 runtime 尚未把每个 task 的 dependency
frontier、physical lane occupancy 和 completion event 写回 simpler scheduler。它必须先通过
`sleep-A -> timewarp -> sleep-B`，不能仅用 plan makespan 验收。

### Device-delay

`a2a3` 保留真实 AICPU scheduler、TensorMap、dispatch/ACK/FIN 和 completion，在相同 child boundary
使用 CCEC 编译的 delay kernel。child ABI 为 `void kernel(__gm__ int64_t *args)`，使用 50 MHz
`get_sys_cnt()`。该路线需要真实 NPU，是高保真控制面参考，不是无 NPU simulator。

Weight/KV 的 device-delay allocation 通过 one-shot exact token 进入 VA interposer，只 reserve valid
NPU virtual address，不提交 physical HBM；原 allocator 仍维护 logical extent、canonical identity、
provenance、borrow、free 和 committed bytes。

## 数据契约

`pypto-common/pypto_contract.py` 是 schema validator。关键身份分层如下：

```text
source.workload_id
  -> graph task identity
  -> program + func_id callable identity
  -> subslot identity
  -> sorted physical-block rank
```

同一 graph 的重复 task 不能代替独立 native workload。跨 warmup/request 生命周期的 callable 复用
必须显式选择 capture role，并记录候选、选择和排除结果。

Model manifest 必须保留 raw observations、feature range、confidence、interpolation/extrapolation
状态、subslot 和 block distribution。任何未建模 callable、feature 外推、block 维度冲突、control
output 冲突或 memory point 不匹配都必须 fail closed。

## 新模型接入边界

通用实现只消费 schema 和 adapter metadata。模型 adapter 负责：

- program/phase/feature/capture role；
- semantic tensor role；
- exact memory allocation/materialization；
- tokenizer/streaming workload；
- control-output ABI。

这些内容放在 `references/<model>-adapter.md`，不要复制到通用 planner、hook 或 launcher。参见
`references/model-adapter-contract.md`。

## 长期 runtime 方向

完整无 NPU 的 runtime 需要：

1. `ExecutionMode.LATENCY_SIM`；
2. virtual tensor/window registry；
3. compile-time task metadata；
4. runtime-level task dispatch/completion event hook；
5. dependency frontier、physical lane occupancy 和 ACK/FIN timeline；
6. distributed window、rank rendezvous 和通信 overlap；
7. strict latency resolver 和 control-output provider。

当前 Python timewarp 只覆盖 graph/step/output barrier，不能声称已经在线重建全部 runtime task DAG。
