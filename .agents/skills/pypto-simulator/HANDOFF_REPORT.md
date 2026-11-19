# PyPTO Simulator 交接报告

## 1. 最终目标

构建一套面向任意 `pypto-serving` 模型的 task 级 simulator 流程：

```text
native profile
  -> normalized task/subslot/block evidence
  -> feature-point latency model
  -> strict sleep plan
  -> sleep / timewarp execution
  -> streaming bracket and attribution
```

最终验收目标是：在小请求和大请求上，sleep 与 timewarp 两种模式都相对同一 workload 的 native
reference 将以下指标的每个 case 和全局 mean/p50/p95 控制在 5% 以内：

- TTFT：request start 到首 token；
- TPOT：`(last_token - first_token) / (output_tokens - 1)`；
- E2EL：request start 到 terminal output。

同时必须保持 token 数、finish reason、completion order、request lifecycle、TaskId 依赖、ACK/FIN
和 memory lifecycle 一致。device-delay 是独立的高保真 hardware-in-the-loop 参考路线，不能替代
sleep/timewarp 的无 NPU acceptance。

**当前测试范围明确只有单卡、单 replica、无 TP。尚未覆盖 tensor parallel、pipeline parallel、多卡
HCCL、跨节点通信或多 rank barrier。任何当前结果都不能外推到 TP。**

## 2. 当前目录结构

```text
pypto-simulator/
|-- README.md                         通用入口和验收规则
|-- HANDOFF_REPORT.md                 本报告
|-- OPTIMIZATION_LOG.md               历史实验、失败证据和版本演进
|-- pypto-common/                     schema、validator、身份和分类
|-- pypto-profile/                    native task/memory profile
|-- pypto-latency-model/              feature-point latency model
|-- pypto-emulate/                    sleep 与 a2a3 device-delay
|-- pypto-timewarp/                   虚拟时钟路线
|-- references/                       通用协议、schema、架构、模型 adapter
`-- examples/                         fixture，不是硬件证据
```

模型特有内容应放在 `references/<model>-adapter.md` 和对应 validation reference 中。通用脚本不得
增加按模型名分支。当前 Qwen 内容包括：

- `references/qwen-adapter.md`：Qwen control output、feature 和 memory adapter；
- `references/native-qwen-validation.md`：Qwen native/device-delay 证据；
- `references/qwen-streaming-benchmark.json`：Qwen workload fixture；
- `references/qwen/qwen_control.py`、`references/qwen/run_validation.py`：Qwen 专用 control/validation executable；
- `examples/qwen-decode-task-profile.json`：contract fixture。

已删除 `build_output/`，该目录是约 52 MB 的重复 JIT 生成物，不是 skill 输入或可复用源码。

## 3. 已完成

### 通用契约和 profile

- schema-v1 profile、model、simulation plan 和 memory profile contract；
- `source.workload_id`、capture role、独立 workload confidence；
- `program + func_id` callable aggregation；
- AIC/AIV/MIX subslot、block count、sorted block-rank occupancy distribution；
- dependency、external predecessor、control-output 和 strict coverage；
- exact median、测量域内插值、extrapolation rejection 和 low-confidence rejection；
- profile ordinal bootstrap：首次非目标 dispatch 也生成 executable child artifact；
- profile launcher 支持 `--npu-device` 运行期 PID/telemetry audit；
- weight/KV exact memory profile、capacity、allocation、free、materialization 和 lifecycle contract。

### Sleep / device-delay

- sleep 路线保留完整 simpler orchestration、TensorMap、TaskId、依赖、ACK/FIN 和 request lifecycle；
- native profile、native baseline、sleep 和 device-delay hook 进程隔离；
- a2a3 device-delay child 使用 50 MHz `get_sys_cnt()` 和原 ABI；
- weight/KV 通过精确 one-shot allocation token 虚拟化 NPU VA，不提交 physical HBM；
- strict memory smoke 已通过，低 HBM physical peak 约 `9,312 MB`；
- 固定小 workload 的 V7 device-delay bracket 已通过：mean/p50/p95 误差
  `-0.085%/-0.207%/+0.085%`，native drift `+1.071%/+0.920%/+0.714%`；
- 已确认 `a2a3sim` zero-work host control-plane floor 与 native 存在结构性差异，不能用 zero-work
  `a2a3sim` 结果冒充硬件精度。

### Streaming benchmark

已实现按 token boundary 统计 TTFT、TPOT、E2EL，并覆盖多输入长度、输出长度和并发度的 workload
manifest。当前较新的多样化 streaming bracket 仍为 `accepted=false`：

- TPOT 各 case 已在 5% 内；
- batched prefill 的 TTFT 高估；
- 全局 TTFT 约 `+20%`，E2EL 约 `+10%`；
- 主要问题集中在 request feature coverage 和 batched prefill，而不是 decode TPOT。

### Timewarp

- 从 strict sleep plan 派生 timewarp plan；
- 离线按 task DAG、subslot、block-rank 和 AIC/AIV/MIX resource 竞争计算 program frontier；
- child stub 不做 wall sleep，但保留 telemetry、control output 和 strict coverage；
- TimeKeeper 使用 `CLOCK_MONOTONIC + non_decreasing_global_offset`；
- 通过 step begin/seal/deliver 和 producer-to-observer gate 对齐 StepResult 与 token output；
- 记录并 strict 拒绝 modeled deadline overrun；
- runtime root/lib root 校验和 socket/mmap 生命周期已实现；
- serial Qwen timewarp smoke 已执行 58/58 modeled stubs、0 unmodeled、6 graph events、4 completed
  steps，输出 token `[2813, 2813]`。

## 4. 当前欠缺

### Acceptance 尚未完成

- 尚未完成完整的 `sleep-A -> timewarp -> sleep-B` streaming bracket；
- 尚未证明 sleep 和 timewarp 在小请求、大请求、不同并发度下都满足每 case/global 的 5% 门禁；
- V7 通过的是固定小 workload 的 device-delay bracket，不是 sleep/timewarp acceptance；
- batch16 timewarp 在 17,939-task prefill graph 处触发 custom runtime child `SIGABRT`，尚未进入真实
  request step；
- strict deadline overrun gate 已实现，但还需要在 workload matrix 上重新跑并统计通过率。

### Runtime 精度模型尚未闭合

当前 timewarp 是离线 task schedule 加在线 program/step barrier：

- delay callback 仍只有 `duration_ns`，没有 task、epoch、subslot、physical lane identity；
- runtime scheduler 尚未维护真实 dependency frontier；
- physical lane occupancy 和 ACK/FIN completion 尚未形成 runtime-level event timeline；
- graph-level makespan、runtime child event 和 `StepResult` deadline 尚未完全统一；
- 因此当前 timewarp 不能声称已经在线重建完整 runtime task DAG。

### Model and platform coverage

- 当前 Qwen feature matrix 仍需要 serial、batch4、batch16 的独立 request profile point；
- 当前 batch16 profile4/5/6 是候选 evidence，采集时尚未启用新的运行期 PID monitor，需要重采集或
  补充可证明的污染审计；
- request dispatch 级动态 feature selection 仍不完整；
- request kernel 内 HBM/weight/KV access latency 与 physical-block latency 尚未完全拆分；
- 真实多卡通信、HCCL、TP、PP、跨节点和多 replica 未覆盖；
- 原生 `ExecutionMode.LATENCY_SIM`、通用 virtual tensor/window registry 尚未实现。

## 5. 下一步计划

### P0：先闭合 runtime timewarp

1. 扩展 runtime timewarp event ABI，至少携带：`task_id`、epoch、program、subslot、block rank、
   physical lane/core、dispatch/complete 类型和 step identity。
2. 在 scheduler dispatch/completion 路径维护：
   - predecessor frontier；
   - physical lane availability；
   - task completion/ACK/FIN；
   - cross-subslot frontier。
3. 让 runtime event timeline 与 Python TimeKeeper 的 graph/step deadline 对齐。
4. 修复 batch16 17,939-task graph 的 child `SIGABRT`，重点检查 simulator child mailbox、buffer
   ownership、canonical identity、KV cleanup 和 worker lifecycle。
5. 对 serial、batch4、batch16 分别执行 strict overrun smoke；任何 overrun 都保留 trace/snapshot，
   不用 `--allow-deadline-overrun` 进入验收。

### P1：重建可审计的 workload evidence

1. NPU 空闲后，用 `run_serving_profile.py --npu-device <id>` 重采集 serial、batch4、batch16 的
   prefill/decode/request capture。
2. 每个 feature point 至少三个独立 workload，并保存 observed PID、monitor error、child returncode
   和 capture role。
3. 补齐 input token `2/8/16/32/128`、output token `5/8/12/16`、concurrency `1/4/16` 的 request
   feature coverage。
4. 将 warmup、prefill、decode、sampling 的 feature selection 分开，禁止数量更多的 warmup task
   覆盖 request model。
5. 对 request kernel 独立建立 `{operation, bytes}`、weight/KV access 和 physical-block model，
   但不把 E2E 误差作为 duration 缩放依据。

### P2：完成 sleep/timewarp acceptance

对每个 workload case 执行：

```text
sleep-A -> timewarp -> sleep-B
```

每段都需要：

- 相同 model/config、相同 prompt token 数和 output token 数；
- 相同 physical environment；
- 每 case 和全局 TTFT/TPOT/E2EL mean/p50/p95；
- sleep A/B drift；
- output order、finish reason、token boundary；
- `graph_events`、`completed_steps`、`offset_ns`、`jumped_ns`；
- deadline overrun、unmodeled stub、timeout 和 runtime telemetry。

只有所有 case、全局、三项指标和两种统计门禁都通过，才能宣布 sleep/timewarp 完成。任何单个
batched prefill 或大请求 case 失败，都必须继续定位，不能由全局平均覆盖。

### P3：再扩展 TP

1. 先定义 distributed window、rank identity、remote store/put、notify/wait 和 HCCL event contract；
2. 为每个 rank 建立独立 profile workload 和 communication latency model；
3. 将 `dp_rank:step_id` 扩展为完整 replica/rank/step identity；
4. 保持单卡 acceptance 作为 baseline，再执行同 physical topology 的多卡
   `native-A -> simulator -> native-B`；
5. 在 TP 完成前，所有文档和结果继续明确标注“单卡、无 TP”。

## 6. 验证命令

```bash
PTOAS_ROOT=/root/revati/.venvs/pypto-simulator \
PYTHONPATH=/root/revati/pypto-serving:/root/revati/pypto-serving/pypto-lib \
/root/revati/.venvs/pypto-simulator/bin/python -m pytest -q \
  .agents/skills/pypto-simulator/pypto-emulate/tests \
  .agents/skills/pypto-simulator/pypto-profile/tests \
  .agents/skills/pypto-simulator/pypto-latency-model/tests \
  .agents/skills/pypto-simulator/pypto-timewarp/tests
```

本次整理前的完整回归结果为 `62 passed`。文档整理后应重新运行该回归、Ruff、`py_compile` 和
`git diff --check`。

## 7. 重要产物索引

- 固定 device-delay bracket：`/tmp/pypto-qwen-native-delay-v7-vmem-bracket-npu2/`
- 多样化 streaming bracket：`/tmp/pypto-qwen-streaming-v8e-npu2/`
- batch16 model candidate：`/tmp/pypto-qwen-model-npu2-v7-batch16-v2/`
- batch16 timewarp failure：`/tmp/pypto-qwen-timewarp-batch16-v1/`
- strict memory profile：`/tmp/pypto-qwen-memory-profile-v2-strict.json`
- 历史实验：`OPTIMIZATION_LOG.md`

上述 `/tmp` 产物是实验环境中的外部 evidence，不属于本目录源码；正式交接应同时保存其 manifest、
SHA256、运行命令和 PID audit。
