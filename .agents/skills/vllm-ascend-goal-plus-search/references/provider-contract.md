# Provider 输入契约

本文件说明 Provider 如何把一次由外部调用者冻结的 Search/attempt 合同映射为部署与压测
Skill 的输入，并定义 Provider 反馈给 Search 的事实结构。搜索状态和执行判定沿用对应公开合同。

## Probe 绑定

Search 级 probe、help 输出和调用约束见[搜索流程](search-flow.md)。Provider 的职责是把
Search 绑定的 probe artifact 引用、SHA256、执行上下文和 shell fingerprint 注入每个 attempt，并让
deployment/benchmark Skill 引用这些事实。Search 绑定时，执行 Skill 以既有 artifact 完成
probe 校验；校验失败或上下文变化时，当前 Search 进入流程定义的阻塞状态并按流程创建新的 Search。

全局 probe、verifier Evidence 和运行时输出均由 Goal Plus 管理。Provider 只接收或返回 Goal Plus
提供的不透明句柄和 artifact 引用，不创建目录、不命名目录，也不为 task 暴露另一套持久化位置。

## 合同边界

Search 建立及 Attempt 生命周期见[搜索流程](search-flow.md)。Provider 接收已冻结的 SearchSpec
和 provider 回执，并按下文映射输入、隔离输出和最终反馈。

## Provider 反馈给 Search

Provider 向 Search 交付执行 Skill 校验后的最终部署/压测报告、raw result 和 Evidence 索引。
valid attempt 的反馈由以下结构组成：部署报告的 schema 标识为
`vllm-ascend-deployment-report-v3`，服务事实来自 `model`、`resources`、`service`、`launch`；
压测报告的 schema 标识为 `vllm-ascend-benchmark-report-v3`，负载和结果来自 `model`、`load`、
`dataset`、`arrival`、`parameters`、`result`。Search 的唯一评分字段由 SearchSpec 指定，
直接映射到 `result` 中的官方指标百分位字段；报告主体承载本轮官方字段，raw result 和摘要归入
Evidence 索引。每个 baseline、candidate 和 promotion attempt 只交付一个有效单轮值，该值直接
作为 Search 层评分；不追加测量或取多个样本平均，不改变 deployment/benchmark 公共报告 schema。

## 输入来源

一次 attempt 的输入由四类来源组成：

1. 冻结常量：模型身份、固定部署事实、workload、指标、统计口径和 provider 合同；
2. 候选字段：范围限定为外部 edit surface 内的部署字段和值；
3. Provider 注入：可见资源、设备组、端口候选、物理锁和 `GOAL_PLUS_VERIFIER_TMPDIR`；
4. 目标环境事实：命令行能力、模型配置、tokenizer、版本、数据类型、量化和上下文能力。

第 1、3、4 类输入保持冻结；Provider 事实以受信任的 Goal Plus/provider 回执为准。
Verifier 还必须为 vLLM 服务进程固定注入 `VLLM_RPC_BASE_PATH=/tmp`。它只用于缩短 IPC
transport socket 路径，不改变 `GOAL_PLUS_VERIFIER_TMPDIR` 或任何 Goal Plus 报告、日志和
Evidence 句柄；持久化输出仍必须写入 Goal Plus/provider 注入的位置。

## 部署输入映射

部署 Skill 的规范输入由 provider 组装并交给其 `scripts/input_contract.py`：

| Provider 输入 | 来源 | 约束 |
| --- | --- | --- |
| 模型路径 | 冻结常量 | 必须是可读的本地模型目录 |
| 服务身份 | 冻结常量 | 独立于路径定义 |
| 可见资源 | provider 注入 | 必须与本次设备组和资源锁一致 |
| 端口候选 | provider 注入 | 范围限定为候选集合 |
| 偏好端口 | provider 注入 | 必须属于候选集合 |
| `GOAL_PLUS_VERIFIER_TMPDIR` | Goal Plus/provider 注入 | 本次 verifier 使用的运行时输出句柄，原样透传 |
| `VLLM_RPC_BASE_PATH` | verifier 固定运行时环境 | 必须固定为 `/tmp`；仅用于 vLLM IPC transport，不作为报告或 Evidence 输出目录 |
| `runtime.<field>` | 外部候选或推荐 | 字段名和值域由 SearchSpec 提供 |
| 资源上限 | SearchSpec/环境事实 | 由外部合同冻结 |

部署成功后向 Provider 交还选定资源、服务地址、服务进程、服务身份和部署报告；压测路径
沿用这些事实完成复核和测量。

## 输出交接

Verifier 必须使用 Goal Plus/provider 注入的运行时输出句柄，并将 deployment、benchmark 返回的
报告、raw result、日志和 Evidence 引用交还公开接口。Verifier 不得在被验收 workspace 创建
临时或持久化输出，也不得自行推断、拼接或替换 Goal Plus 的存储位置；句柄未就绪时按公开合同
报告 `error`。

## 压测输入映射

压测 Skill 接收冻结的服务事实和 workload 合同：

- 服务路径 A：在同一个受保护调用中完成部署、健康验收、官方测量、raw result 校验和清理；
- 服务路径 B：Provider 提供已部署服务地址、服务身份、tokenizer、部署报告和 Goal Plus 输出句柄，
  压测只负责复核和测量；
- 数据集遵守 benchmark contract：preset 传场景，由 benchmark 派生数据集 ID 并使用默认缓存
  解析；hf 传数据集 ID；custom 传本地数据集路径；缓存位置沿用 benchmark 的默认解析；
- workload 必须显式提供选择方式、场景、模式、数据集来源、请求数、预热、并发、到达率、
  轮次、采样和生成参数；
- 最终报告、raw result、持久化版本及 Evidence 均遵守 Goal Plus 的公开输出句柄合同。

字段缺失、矛盾或冻结条件未满足时反馈为 `invalid`；字段值以 Provider 事实和冻结合同为准。
