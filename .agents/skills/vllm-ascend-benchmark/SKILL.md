---
name: vllm-ascend-benchmark
description: 在已冻结的 vLLM-Ascend 服务和负载合同上使用官方 vllm bench serve 执行极限或生产级在线压测，并交付含 TTFT、TPOT、E2EL P50/P95 与可选 goodput 的独立 benchmark artifact。
---

# vLLM-Ascend 压测技能

执行单元为一次独立 benchmark attempt。输入为任务、Goal Plus provider 或 verifier
冻结的服务接口、模型身份、部署摘要、负载场景、压测模式、统计配置和输出 surface；
测量入口为目标环境安装的官方 `vllm bench serve`，输出为一次性的
`benchmark_result.json`。

职责边界限定为单次 benchmark attempt。Search 创建、candidate 调度、设备分配、冻结参数
修改、最优判定和 promotion 归属上层流程。服务生命周期遵循 provider 合同；允许当前
verifier 管理服务时，前置步骤调用
[vLLM-Ascend 部署技能](../vllm-ascend-deployment/SKILL.md)，预启动或 ThinkThread
路径要求输入服务接口已交付并完成身份验收。

## 输入与路由

开始前读取任务的 `goal.md`、`workflow.md`、`inputs.md`、冻结的 SearchSpec、provider
配置和已有 baseline artifact，建立本次 attempt 的规范输入。至少确认：

- served model name、base URL、endpoint、tokenizer 来源和服务身份；
- 部署配置摘要、配置指纹、provider、设备和端口归属；
- `selection`、`scenario`、`benchmark_mode` 以及数据集适配器参数；
- `dataset_root`、数据集名称/路径、split、revision/hash 和适配器专用参数；
- 请求数、并发、warmup、到达率、burstiness、采样参数、输出长度和重放顺序；
- goodput SLO 是否由任务合同提供；
- raw benchmark、日志、报告和临时目录的允许写入位置。

`selection` 使用[负载契约](references/workload.md)定义的 `preset`、`hf` 或 `custom`。
`preset` 下的 `scenario` 可取 `random`、`long_prefill`、`long_decode` 或
`prefill_decode_balance`；`hf` 和 `custom` 使用合同直接提供的数据集配置。
设备、端口和压测数量仍由任务合同提供。`benchmark_mode` 必须明确为：

- `stress`：极限压力模式，用于突发、饱和、容量拐点和高并发稳定性观察；
- `production`：生产级模式，用于冻结到达率、并发和样本窗口下的 open-loop 评估。

## 执行流程

1. 确认服务已经通过部署层的 `/health`、`/v1/models` 和最小请求身份验收。服务身份无
   效时不运行 benchmark。
2. 读取[负载契约](references/workload.md)，确认 `selection`、`scenario`、模式、请求
   和结果目录均已冻结，并按适配器分支解析数据集：
   - `preset` 根据 `scenario` 派生 random、HF 或 ShareGPT 配置；
   - `hf` 使用合同提供的 HF 数据集名称、路径、split 和可选参数；
   - `custom` 使用合同提供的本地 JSONL 和 custom 专用参数；
   - 校验路径、格式、样本数、版本/hash 和当前 vLLM CLI 参数兼容性；
   - 将解析后的数据集路径、来源、版本/hash、适配器参数和解析状态写入本次 attempt。
   数据集解析未完成时不执行压测。
3. 执行 [官方测量方法](references/measurement.md)：
   - 用当前安装版本的 `vllm bench serve --help=all` 检查参数能力；
   - 构造唯一的 `vllm bench serve` 命令；
   - 显式请求 TTFT、TPOT、E2EL 的 P50/P95；
   - 任务提供 SLO 时显式启用 `--goodput`；
   - 记录实际命令、版本、服务身份和 raw result 路径。
4. 只消费本次命令生成的、唯一且可解析的官方 raw JSON。按
   [报告结构](references/report-schema.md)核对模型身份、请求数、完成数、失败数、
   指标字段和 goodput 字段。
5. 将官方 raw JSON 中已经计算出的 TTFT、TPOT、E2EL P50/P95、吞吐和 goodput 原样
   映射到 `benchmark_result.json`。指标来源限定为官方 raw JSON；日志、SSE、请求时间、
   事件数和部分样本均不参与指标重算。
6. 运行时 OOM 判定仅适用于正式 `vllm bench serve` 请求阶段。该阶段发生 OOM 时，按
   [测量方法](references/measurement.md)将 attempt 标记为 `status=error`、
   `failure.stage=benchmark`、`failure.type=runtime_oom` 和 `oom.phase=benchmark`。
   静态预检、服务启动和最小请求验收阶段的 OOM 归入部署报告。
7. 无论成功、输入无效、官方 CLI 不兼容、服务失败、请求不完整、OOM、raw result 缺失或
   清理失败，都写出可解析的 `benchmark_result.json`；失败报告中的性能字段为 `null`，
   并保留阶段、错误类型、OOM 证据、raw artifact 路径和清理状态。

官方 `vllm bench serve` 的 workload、stream/usage、百分位、goodput 和 raw result
能力构成本次测量门禁。能力不完整时，attempt 状态为 `blocked` 或 `error`。测量后端为
官方 benchmark；固定请求继续使用任务合同指定的 workload。

## 输出语义

`benchmark_result.json` 只表示一次测量 attempt：

- `success`：服务身份有效、官方 benchmark 完整结束、请求和 raw result 通过校验、所需
  指标完整、清理符合合同；
- `invalid`：输入、workload、身份、请求计数、raw result 或 goodput 合同不满足；
- `blocked`：目标 vLLM CLI 或服务能力未满足冻结合同的测量门禁；
- `error`：环境、服务、资源、输出或清理故障，无法把结果归因于候选性能。
- 正式压测期间的 OOM 必须通过 `failure.type=runtime_oom` 和
  `oom.type=runtime_oom` 显式报告；OOM 证据和清理状态分别记录。

`service_identity_valid=true` 不是 benchmark 成功；benchmark 成功也不是 candidate 通过。
Goal Plus Main 只能使用独立 verifier 产出的报告和 raw artifact 计算 Search 指标。

## 边界

- NPU 与端口由 provider 提供；设备调度事实不从 `npu-smi` 快照推断。
- TP/DP、设备、端口、模型、请求、统计方法和质量门槛沿用冻结合同；预置场景、HF
  数据集和 custom 数据集均通过负载契约声明并记录最终适配器参数。
- 每次 attempt 使用独立目录和输出文件，既有 attempt 保持不变。
- 官方正式 benchmark 仅包含合同定义的正式测量窗口；warmup、profile、服务启动和编译
  归入准备或诊断阶段。
- 报告内容限于任务合同允许的字段，不包含全机 HBM、进程列表或资源快照。
