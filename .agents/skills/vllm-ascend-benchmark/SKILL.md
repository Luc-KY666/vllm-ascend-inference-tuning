---
name: vllm-ascend-benchmark
description: 在已冻结的 vLLM-Ascend 服务和负载合同上执行在线压测，解析 TTFT、TPOT、E2EL 的 P50/P95，并输出可供 Goal Plus 读取的一次性 benchmark artifact。
---

# vLLM-Ascend 压测技能

本 skill 负责一次独立测量 attempt，不负责创建 Search、调度 candidate、选择设备或
判定最终参数最优。它消费任务冻结的服务接口、模型配置、负载、统计口径和输出目录，
并产出可复算的 `benchmark_result.json`。当当前 provider 允许本次 verifier 管理服务
生命周期时，先调用部署 skill 完成参数校验、容量预检和服务身份验收；预启动服务则
只消费已交付接口。

部署服务由 [vLLM-Ascend 部署技能](../vllm-ascend-deployment/SKILL.md) 或任务 provider
管理。git_worktree/direct 任务可以在同次持锁 verifier 内启动、检查、压测和清理；
ThinkThread 或预启动任务只能访问已交付的 base URL，不得在短 verifier 中自行启动或
停止服务。

## 输入与工作负载

开始前读取任务 `goal.md`、`workflow.md`、`inputs.md`、冻结 SearchSpec 和已有 baseline
artifact，确认：

- `model_name`、服务 endpoint、tokenizer 来源和服务身份。
- 当前 attempt 的部署配置摘要和配置指纹。
- `scenario`：`random`、`long_prefill`、`long_decode` 或
  `prefill_decode_balance`。
- 数据集 ID/路径、请求数、并发、warmup、到达率、burstiness、采样参数和输出长度。
- 请求是否必须严格重放固定文件，以及失败请求、超时、流中断和不可估计 TPOT 的规则。
- 原始流、vLLM raw JSON、规范化报告、日志和临时目录的允许写入位置。

场景和请求到达模式只提供命名空间，不提供模型、数据集路径、端口、设备或数量默认
值，详见 [负载契约](references/workload.md)。

## 执行流程

### 1. 冻结并核对测量入口

1. 确认服务已经通过部署层的健康和模型身份检查。
2. 用安装版本的 `vllm bench serve --help=all` 核对 endpoint、backend、tokenizer、
   数据集和结果参数。
3. 如果官方 benchmark 能重放任务冻结的请求、流式 usage 和统计口径，冻结其版本、
   命令和输出路径。
4. 如果不能完整重放，使用本 skill 的 `scripts/streaming_metrics.py`。调用方负责
   HTTP 请求、单调时钟、原始字节块保存和请求顺序；helper 不启动服务、不管理资源锁。
5. 记录本次工具、服务、模型、配置和 workload 的摘要或 hash，避免不同口径的样本混排。

### 2. 执行 warmup 与正式请求

- warmup 只确认服务能够完成有效生成，不计入正式指标。
- 正式请求按任务冻结顺序、并发和轮次执行；不因单次失败而丢弃样本或重试到通过。
- 流式测量必须记录请求发送时间、每个收到的原始字节块及其单调时间戳、结束时间。
- 只把非空内容事件作为首 token 和末 token 依据；role、空 delta、usage-only 事件不
  计为内容 token。
- 输出 token 数来自已核验的 usage 或任务冻结 tokenizer，不用 SSE 事件数替代。
- 服务端引擎 batch 与客户端请求并发是不同概念；没有服务端证据时不能声称实际 batch。

### 3. 计算指标

使用 `streaming_metrics.py` 或等价的冻结实现，计算：

- TTFT：`t_first - t_send`。
- TPOT：输出 token 数大于 1 且至少有两个内容事件时，
  `(t_last - t_first) / (completion_tokens - 1)`。
- E2EL：`t_end - t_send`。
- P50/P95：对完整请求样本使用 nearest-rank，即第 `ceil(p * N)` 个有序值。

TPOT 不可估计的样本从 TPOT 分布中排除，但必须记录数量。若任何正式请求失败、流
协议不完整、usage 缺失或可估计 TPOT 样本不足，不能把对应指标填为 0，也不能把部分
结果报告为成功。

### 4. 规范化原始结果

对包含部署身份证据的 raw benchmark 包装对象或流式摘要运行：

```text
python <skill-root>/scripts/normalize_benchmark_result.py \
  --input <唯一原始结果包装或流式摘要> \
  --model <冻结的 served model name> \
  --expected-requests <冻结请求数> \
  --output <任务允许的 benchmark_result.json>
```

包装对象可以把原始结果放在 `raw_result` 中，并在同一对象的 `service` 中记录部署 skill
返回的 `served_model_name` 和 `service_identity_valid=true`。直接使用没有服务身份字段的
raw benchmark 会被标记为 invalid；不能因为 raw JSON 有性能字段就跳过部署验收。

工具会校验：

- 原始结果输入恰好一个；缺失或多个候选都返回 invalid。
- 模型身份与冻结的 served model name 一致。
- 请求数、完成数和失败数与本次 workload 一致；缺少请求数或完成数也会失败。
- 部署服务身份存在且为 `true`。
- TTFT、TPOT、E2EL 的 P50/P95 可解析，且均来自本次结果。
- 原始文件路径和 SHA256 记录在报告中，不把未经核验的 glob、日志或候选自报字段当作
  指标来源。

## 输出语义

`benchmark_result.json` 只表示一次 attempt 的测量结果。它必须能区分：

- `success`：服务身份有效、请求完整、指标完整、清理状态符合合同。
- `invalid`：输入、身份、请求数量、流式协议或 raw result 校验失败。
- `error`：环境、服务、资源、输出或清理错误，无法将本次结果归因于候选性能。

部署层的 `service_identity_valid` 不是 benchmark 成功；benchmark 成功也不是 candidate
通过。Goal Plus Main 只能使用独立 verifier 产出的 artifact 计算 Search 指标，不能把
本文件的报告字段直接当作 promotion verdict。

## 资源与生命周期边界

- 不自行选择 NPU，不从 `npu-smi` 快照推断调度结果。
- 不改变冻结的 TP/DP、设备、端口、请求、数据集、统计方法或质量门槛。
- 不在报告中写入全机 HBM、健康状态、进程列表或清理前后资源快照。
- 不覆盖已有 attempt；每次重试使用新的目录、文件名或 attempt 标识。
- 不把 profiling 数据混入正式性能指标；profile 只作独立诊断。
