# PyPTO Simulator

面向 `pypto-serving` 的 task 级延迟模拟流程。目标是复用真实 PyPTO 控制流，只在 outlined
`simpler task` / child kernel dispatch 边界替换计算、通信和大内存操作；不把整个
`worker.run()` 或整张 graph 替换成一次固定 sleep。

本目录的流程与模型无关。模型接入只需要提供符合 manifest contract 的 native profile、模型适配
metadata 和 serving workload；当前 Qwen3-14B 是第一个验证适配，所有 Qwen 数值、token ABI、
memory bytes 和验证产物都放在 `references/` 或 fixture 中。

## 目标与边界

- 保留 orchestration、TensorMap、TaskId、依赖、MIX/SPMD、early dispatch、同步、ACK/FIN、
  completion 和 request 生命周期。
- 以稳定的 task identity、`program + func_id`、subslot、block count 和动态 feature 建立可审计
  延迟模型。
- 对未建模 task、非法 feature、外推、block 维度冲突、memory point 不匹配和 control-output
  冲突 fail closed。
- `sleep` 是无 NPU 的控制流正确性基线；`timewarp` 是无 NPU 的虚拟时间加速路线；`a2a3`
  device-delay 是需要真实 NPU 的高保真参考路线。三者不能互相冒充验收真值。
- 最终验收必须覆盖小请求和大请求，在 sleep、timewarp 两种模式下分别将 TTFT、TPOT、E2EL
  的每个 case 与全局 mean/p50/p95 控制在 5% 以内；device-delay/native bracket 单独验证。

## 目录职责

```text
pypto-simulator/
|-- README.md                         通用入口、边界和最短流程
|-- HANDOFF_REPORT.md                 当前进展、缺口和交接计划
|-- OPTIMIZATION_LOG.md               历史实验与失败证据，不作为操作手册
|-- pypto-common/                     schema、validator 和共享身份/分类逻辑
|-- pypto-profile/                    native capture、导出和 memory evidence
|-- pypto-latency-model/              feature point、模型拟合和 coverage confidence
|-- pypto-emulate/                    sleep plan、sleep runtime 和 a2a3 device-delay
|-- pypto-timewarp/                   sleep plan 派生的虚拟时钟路线
|-- references/                       通用协议、架构、manifest 和模型适配说明
|-- examples/                         contract fixture，不是硬件证据
```

四个子目录是可独立发现的 skill。公共 schema、references 和 examples 不是额外 skill。

## 标准流程

```text
native PyPTO dispatch
  -> pypto-profile
  -> normalized profile manifest
  -> pypto-latency-model
  -> model manifest / latency lookup
  -> pypto-emulate build plan
  -> sleep validation
  -> pypto-timewarp build/run and sleep bracket
  -> optional a2a3 device-delay bracket
```

### 1. Profile

先对每个需要模拟的 program、phase、shape 和生命周期采集至少三个独立 native workload。必须
保留 `deps.json`、level-4 swimlane、callable identity、capture role 和运行期污染审计。正式硬件
采集必须使用同一 physical NPU，并在启动前保证连续 60 秒空卡。

入口：[`pypto-profile/SKILL.md`](pypto-profile/SKILL.md)

### 2. Latency model

将 normalized manifests 按稳定 latency signature 聚合。raw task 数和独立 workload 数必须分开
统计；同一 graph 的重复 task 不能冒充独立样本。默认只接受 exact point 或测量域内插值，禁止
无证据外推。模型必须保留 workload provenance、observed range、subslot 和 sorted block-rank
occupancy distribution。

入口：[`pypto-latency-model/SKILL.md`](pypto-latency-model/SKILL.md)

### 3. Sleep / device-delay plan

`sleep` 和 `a2a3` device-delay 共用模型和 strict coverage，但 backend、hook 和 launcher 完全
隔离。sleep 只作用于 `a2a3sim`；device-delay 只由显式 launcher 启用并使用真实 `a2a3`。native
baseline 和 profile 进程不得加载任一 hook。

入口：[`pypto-emulate/SKILL.md`](pypto-emulate/SKILL.md)

### 4. Timewarp

只允许从已经通过 strict coverage 的 sleep plan 派生。builder 离线计算 task DAG、subslot、block
rank 和 AIC/AIV/MIX resource frontier；运行时 child 不做 wall sleep，serving 控制流仍自然执行。
TimeKeeper 使用共享 `CLOCK_MONOTONIC + non_decreasing_global_offset`，并在 StepResult、token
producer/observer 边界推进虚拟时间。

入口：[`pypto-timewarp/SKILL.md`](pypto-timewarp/SKILL.md)

## 新模型接入

新模型不应复制一套 Qwen 脚本。适配层只负责以下事项：

1. 提供稳定的 program、task、callable 和 feature identity；
2. 标注 `weight`、`kv_cache`、`window`、`control` 等 semantic tensor role；
3. 提供模型特有的 allocation、materialization、control-output 和 tokenizer/streaming adapter；
   serving runner 通过 `PYPTO_MODEL_RUNNER_MODULE` / `PYPTO_MODEL_RUNNER_CLASS` 注入，避免通用 hook
   直接 import 模型类；
4. 将模型特有命令、固定 shape、token ABI、memory bytes 和 known limitation 放入
   `references/<model>-adapter.md`；
5. 用通用 profile/model/plan/validation 脚本生成证据，而不是在通用脚本中增加模型分支。

参见 [`references/manifest-contract.md`](references/manifest-contract.md)、
[`references/model-adapter-contract.md`](references/model-adapter-contract.md) 和
[`references/qwen-adapter.md`](references/qwen-adapter.md)。

## 验收规则

- sleep：`sleep-A -> sleep-B` bracket，确认模拟请求顺序、token 数、finish reason 和每 case/global
  TTFT、TPOT、E2EL mean/p50/p95。
- timewarp：`sleep-A -> timewarp -> sleep-B`，同时检查 virtual offset、graph event、completed
  step、output gate、真实 wall runtime 和 deadline overrun。
- device-delay：同一 physical NPU 上 `native-A -> device-delay -> native-B`，单独报告 native drift。
- 任一 case 或任一统计量超过 5% 都是失败，不能用全局平均掩盖局部失真。
- 当前测试范围只有单卡、单 replica、无 TP、无真实多卡通信/HCCL；不能把单卡结果外推到 TP。

详细协议和历史证据分别见：

- [`references/architecture.md`](references/architecture.md)
- [`references/manifest-contract.md`](references/manifest-contract.md)
- [`references/pypto-execution-and-simulation-guide.md`](references/pypto-execution-and-simulation-guide.md)
- [`references/native-qwen-validation.md`](references/native-qwen-validation.md)
- [`HANDOFF_REPORT.md`](HANDOFF_REPORT.md)
- [`OPTIMIZATION_LOG.md`](OPTIMIZATION_LOG.md)

## 本地回归

```bash
PTOAS_ROOT=/root/revati/.venvs/pypto-simulator \
PYTHONPATH=/root/revati/pypto-serving:/root/revati/pypto-serving/pypto-lib \
/root/revati/.venvs/pypto-simulator/bin/python -m pytest -q \
  .agents/skills/pypto-simulator/pypto-emulate/tests \
  .agents/skills/pypto-simulator/pypto-profile/tests \
  .agents/skills/pypto-simulator/pypto-latency-model/tests \
  .agents/skills/pypto-simulator/pypto-timewarp/tests
```

fixture 只能验证 schema、planner、hook 和 failure contract，不能替代 NPU profile 或性能验收。
