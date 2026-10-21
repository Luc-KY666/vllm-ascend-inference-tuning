# Benchmark 报告结构

`benchmark_report.json` 是一次独立官方 benchmark attempt 的事实记录，不表达参数搜索、候选
通过或最终推荐。完整结构唯一维护在
[`benchmark-report-template.json`](benchmark-report-template.json)；本文件只说明约束。

## 顶层契约

报告必须使用 `vllm-ascend-benchmark-report-v2`，并包含以下固定区块：

- `attempt`：attempt/provider 标识、配置和 workload digest、开始和更新时间；
- `preflight`：服务路径、执行上下文，以及唯一的环境 probe 和 deployment report 路径；
- `service`：服务来源、base URL、endpoint、served model、tokenizer 和身份验收结果；
- `workload`：冻结的数据集、请求窗口、生成参数、默认值来源和百分位配置；
- `counts`、`metrics`、`goodput`、`throughput`：只保存官方 raw JSON 中的必要测量结果；
- `source`：命令、raw JSON 路径/SHA256、版本和 CLI 能力摘要；
- `failure`、`oom`、`cleanup`、`timings`、`errors`：失败、OOM、清理和耗时事实。

报告对象从模板加载后在内存中维护，只能通过 `scripts/report_io.py` 的
`atomic_write_report` 写入 `<attempt-dir>/benchmark_report.json`。报告写入节点为初始化、
preflight/workload 完成、raw result 校验完成、清理完成或受控失败退出。

## Artifact 归属

- `preflight.environment_probe` 是本次唯一 `environment_probe.json` 的路径。Path A 必须引用
  deployment report 的 `artifacts.probe.environment_probe`；Path B 由本次唯一 probe 生成。
- `preflight` 中记录的执行上下文必须来自该唯一 probe。只有 `status=pass`、
  `sandbox.status=not_detected`、`execution_context=current` 才允许阶段跨多个普通 Bash 调用；
  提权或 `execution_context=elevated` 时，后续依赖该环境的阶段必须复用同一个提权 Bash；
  `sandbox.status=unknown` 或 `execution_context=unknown` 不得按无隔离处理，必须以 preflight
  failure 结束。Bash 调用次数不写入报告，也不单独构成失败；服务交接失败、服务不可达或
  提权生命周期丢失才记录 attempt 错误。
- `preflight.deployment_report` 是 Path A deployment report 或 Path B Provider 提供部署摘要的唯一路径；
  只有 Provider 明确没有部署报告时才为 `null`。
- `source.raw_result` 只能指向本次命令生成的 `raw_benchmark_result.json`，并且必须同时保存
  `source.raw_result_sha256`。没有可解析 raw JSON 时两者都为 `null`。
- benchmark 不在报告中复制 probe、deployment、服务日志或命令 transcript；完整 probe/help/部署
  日志由对应 skill 的外置 artifact 负责保存。
- `failure.artifacts` 和 `oom.evidence[*].artifact` 使用逻辑引用或当前 attempt 的必要 artifact
  标识，不嵌入日志正文。普通命令诊断最多保留有限尾部。
- 同一个非空 artifact 路径只能有一个 canonical owner；不得在 `service` 或其他区块重复保存
  deployment report 路径。

## Deployment v2 接口

Path A 只接受 `schema_version=vllm-ascend-deployment-report-v2`，不提供 v1 兼容适配：

- probe/help 路径只能从 `deployment_report.artifacts.probe` 读取；不得从 `environment` 或旧
  `inputs.*` 字段猜路径；
- 设备和端口只能从 `deployment_report.stages.resources.devices`、`stages.resources.ports` 读取；
- 服务 URL、PID、model identity 和验收结果只能从 `deployment_report.stages.service` 读取；
- 只有 deployment report 为 `service_ready` 且环境、资源、服务阶段通过时才能进入测量；
- `resource_preflight_failed`、输入失败、静态 OOM、服务验收失败或 probe 失败必须立即写出
  benchmark failure，不等待、轮询或重新探测 NPU/端口。

## 状态

- `running`：报告已初始化但 attempt 尚未完成；
- `success`：服务身份有效、官方 benchmark 完整结束、raw JSON 唯一且必需计数/指标通过；
- `invalid`：Provider 输入、负载、数据集、身份、请求计数或 goodput 合同不满足；
- `blocked`：环境或 CLI 不具备冻结合同要求的能力，正式 benchmark 未执行；
- `error`：服务、运行时、输出、请求完整性或报告交付故障，结果不能归因于性能。

`service.service_identity_valid=true` 只代表服务验收通过，不代表 benchmark 成功。清理状态
独立记录，不得用清理结果覆盖已经完成的性能区块。

## 指标和 goodput

`metrics` 数值必须直接映射官方 raw JSON 的 TTFT、TPOT、E2EL P50/P95；不得从日志、SSE、请求
时间戳、部分样本或自定义公式重算。请求计数沿用官方结果，不移除失败样本。

Goodput 规则：

- `reported`：Provider 至少提供一个 SLO，命令启用 `--goodput`，raw JSON 含合法
  `request_goodput`；
- `not_requested`：Provider 未提供 SLO，命令不传 `--goodput`，`request_goodput` 为 `null`；
- `missing`：Provider 要求 goodput，但 raw JSON 缺失、不可解析或无法归属本次命令；attempt
  不能为 `success`。

`workload.resolution.defaulted_fields` 必须列出实际应用的默认值。Provider 未提供 temperature
时记录默认 `0` 并传入 CLI；Provider 显式冻结为 `null` 时不传参数，也不得把 CLI 未设置值解释
为零。

## 失败和 OOM

失败报告仍保留模板全部顶层区块；无法信任的性能字段保持 `null`。`failure` 至少包含
`stage`、`type`、`message`，并可用 `artifacts` 列出逻辑引用。`errors` 记录有限的阶段错误，
不得保存完整 stdout/stderr、服务日志正文或 transcript。

正式请求期间的显存不足必须使用：

```text
status=error
failure.stage=benchmark
failure.type=runtime_oom
oom.detected=true
oom.phase=benchmark
oom.type=runtime_oom
```

静态预检、服务启动和最小请求阶段的 OOM 归属 deployment report；benchmark 尚未进入正式请求
时不得生成 `runtime_oom`。OOM attempt 的 TTFT、TPOT、E2EL、吞吐和 goodput 全部保持 `null`。

主要错误类型包括：

`service_identity_invalid`、`runtime_oom`、`cli_incompatible`、`workload_not_replayable`、
`raw_result_not_found`、`raw_result_ambiguous`、`raw_result_invalid`、`model_mismatch`、
`request_count_mismatch`、`benchmark_incomplete`、`metrics_missing`、`goodput_missing`、
`dataset_contract_invalid`、`dataset_mode_invalid`、
`dataset_not_found`、`dataset_prepare_failed`、`dataset_invalid`、`dataset_insufficient`、
`custom_dataset_invalid`、`hf_dataset_invalid`、`cleanup_failed`、`sandbox_approval_denied`、
`sandbox_escalation_failed`、`sandbox_isolation_unknown`、`environment_preflight_failed`。

## 禁止项

- 不生成或引用旧的 `benchmark_result.json`；
- 不读取 deployment v1 的 `inputs`、`model`、`deployment`、`resources`、`preflight` 或 `checks`；
- 不在 benchmark report 中保存完整 stdout、stderr、服务日志或 command transcript；
- 不创建额外的 benchmark 阶段 JSON 交接文件；
- 不自动修改 TP、DP、并发、上下文、端口或 workload 后重试。
