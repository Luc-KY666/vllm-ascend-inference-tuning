# 并行任务输入

本文件说明输入来源和确认规则。模型路径、数据集根目录和候选预算集中维护于本任务
入口参数区，本文件引用该冻结来源。

## 固定任务值

| 输入 | 冻结值 |
| --- | --- |
| served model name | `qwen3-8b` |
| workload selection | `preset` |
| workload scenario | `long_prefill` |
| benchmark mode | `production` |
| maximum concurrency | `64` |
| request rate | `5 req/s` |
| dataset name | `likaixin/InstructCoder` |
| dataset split | `train` |
| dataset output length | `128` |
| optimization metric | official TTFT P95 |
| metric direction | minimize |
| adjustable parameters | `TP`, `DP`, `max_num_seqs`, `max_num_batched_tokens` |
| provider | `git_worktree` + `direct runtime` + `local verifier` |
| promotion | `artifact_only` |
| search concurrency | `max_parallel=2` |

## 动态确认值

以下输入由 provider、目标 CLI、模型配置或用户授权提供：

- 实际模型路径、官方 `config.json` 和 tokenizer 来源；
- vLLM、vLLM-Ascend、torch、torch_npu、Python、CANN 和 ATB 版本；
- activation wrapper、扩展探针和清理命令；
- provider 授权的两条设备 lane、可见性环境变量、物理锁、两个端口和临时输出目录；
- `gpu_memory_utilization`、`max_model_len`、dtype、量化、block size 和最低上下文；
- vLLM 默认的 `max_num_seqs`、`max_num_batched_tokens`；
- benchmark 默认的请求数、warmup、rounds、burstiness、采样和输出参数；
- 当前 CLI 是否支持 HF adapter、production 到达率、64 并发和官方 percentile 输出；
- dataset root 下预设数据的实际解析路径、revision/hash、样本数和缓存状态。

缺少任何正式部署或 benchmark 必需事实时，Main 必须报告 `unknown`、`blocked` 或
`incomplete`，并等待目标环境补齐事实。

## Baseline 来源

部署 Skill recommendation-mode 负责给出最小 TP 推荐。推荐结果仅作为诊断输入；
部署通过以完整部署验收为准。完整 baseline 的 TP 使用推荐值，DP 为 `1`，两个批处理参数使用目标
vLLM 当前 CLI 默认值。其他字段按模型配置、provider 合同和 CLI 默认值冻结。

## 并行资源

每个 candidate 必须得到独立的 provider 资源事实：

- 一个独立授权设备组；
- 一个独立端口；
- 一个独立 attempt 输出目录；
- 一个独立服务进程和日志边界；
- provider 可核验的锁与临时目录。

并行证据由正式 benchmark 时间区间、provider 回执、锁和清理状态共同构成，四类
信息必须共同证明两条 lane 的独立执行。

## 输出与证据

每个 attempt 使用独立目录，至少保留：

- `deployment-input.json`
- `static-oom-input.json`
- `static-oom-report.json`
- `deployment_report.json`
- `benchmark_result.json`
- 官方 raw benchmark JSON 及 SHA256
- 命令、版本、服务日志、清理状态和配置/workload 摘要

只把 provider 允许的 artifact 导出到 Goal Plus Evidence。输出目录限定为 provider
允许的 artifact surface；模型目录、环境目录和共享锁目录归入运行资源范围。
