# Provider 输入契约

本文件说明 Provider 如何把一次由外部调用者冻结的 Search/attempt 合同映射为部署与压测
Skill 的输入，并定义 Provider 反馈给 Search 的事实结构。搜索状态和执行判定沿用对应公开合同。

## Probe 绑定

Search 级 probe、help 输出和重复调用策略见[搜索流程](search-flow.md)。Provider 的职责是把
Search 绑定的 probe 路径、SHA256、执行上下文和 shell fingerprint 注入每个 attempt，并让
deployment/benchmark Skill 引用这些事实。Search 绑定时，执行 Skill 以既有 artifact 完成
probe 校验；校验失败或上下文变化时，当前 Search 进入流程定义的阻塞状态并按流程创建新的 Search。

## 合同边界

Search 建立及 Attempt 生命周期见[搜索流程](search-flow.md)。Provider 接收已冻结的 SearchSpec
和 provider 回执，并按下文映射输入、隔离输出和最终反馈。

## Provider 反馈给 Search

Provider 向 Search 交付执行 Skill 校验后的最终部署/压测报告、raw result 和 Evidence 索引。
valid attempt 的反馈由以下结构组成：部署报告的 schema 标识为
`vllm-ascend-deployment-report-v3`，服务事实来自 `model`、`resources`、`service`、`launch`；
压测报告的 schema 标识为 `vllm-ascend-benchmark-report-v3`，负载和结果来自 `model`、`load`、
`dataset`、`arrival`、`parameters`、`result`。Search 的唯一评分字段由 SearchSpec 指定，
直接映射到 `result` 中的官方指标百分位字段；报告主体承载官方字段，raw result 和摘要归入
Evidence 索引。

## 输入来源

一次 attempt 的输入由四类来源组成：

1. 冻结常量：模型身份、固定部署事实、workload、指标、统计口径和 provider 合同；
2. 候选字段：范围限定为外部 edit surface 内的部署字段和值；
3. Provider 注入：可见资源、设备组、端口候选、物理锁、`GOAL_PLUS_VERIFIER_TMPDIR` 和
   `GOAL_PLUS_VERIFIER_DIAGNOSTICS_DIR`；
4. 目标环境事实：命令行能力、模型配置、tokenizer、版本、数据类型、量化和上下文能力。

第 1、3、4 类输入保持冻结；Provider 事实以受信任的 Goal Plus/provider 回执为准。

## 部署输入映射

部署 Skill 的规范输入由 provider 组装并交给其 `scripts/input_contract.py`：

| Provider 输入 | 来源 | 约束 |
| --- | --- | --- |
| 模型路径 | 冻结常量 | 必须是可读的本地模型目录 |
| 服务身份 | 冻结常量 | 独立于路径定义 |
| 可见资源 | provider 注入 | 必须与本次设备组和资源锁一致 |
| 端口候选 | provider 注入 | 范围限定为候选集合 |
| 偏好端口 | provider 注入 | 必须属于候选集合 |
| `GOAL_PLUS_VERIFIER_TMPDIR` | Goal Plus/provider 注入 | 本次 attempt 的唯一绝对临时目录，必须在 source/worktree 外 |
| `GOAL_PLUS_VERIFIER_DIAGNOSTICS_DIR` | Goal Plus/provider 注入 | 持久化 Evidence 目录，必须在 source/worktree 外 |
| `runtime.<field>` | 外部候选或推荐 | 字段名和值域由 SearchSpec 提供 |
| 资源上限 | SearchSpec/环境事实 | 由外部合同冻结 |

部署成功后向 Provider 交还选定资源、服务地址、服务进程、服务身份和部署报告；压测路径
沿用这些事实完成复核和测量。

## 输出与持久化

Goal Plus/provider 注入 `GOAL_PLUS_VERIFIER_TMPDIR`，作为部署和压测报告、raw result、日志、
脚本、`.work` 及其他 attempt 文件的归档根目录。每个 attempt 使用该目录下的唯一子目录，
各 lane 保持独立。

持久化 Verifier 的脚本、报告、日志、生命周期记录和其他 Evidence 写入
`GOAL_PLUS_VERIFIER_DIAGNOSTICS_DIR`；目录位置由 Goal Plus/provider 提供。Evidence 至少记录
两个实际目录、文件摘要 hash，以及 verifier 结束时 worktree 无新增运行产物的结果。目录状态
未就绪或位于 source/worktree 内时，attempt 状态记为 `error`。

## 压测输入映射

压测 Skill 接收冻结的服务事实和 workload 合同：

- 服务路径 A：在同一个受保护调用中完成部署、健康验收、官方测量、raw result 校验和清理；
- 服务路径 B：Provider 提供已部署服务地址、服务身份、tokenizer、部署报告和 attempt 目录，
  压测只负责复核和测量；
- 数据集遵守 benchmark contract：preset 传场景，由 benchmark 派生数据集 ID 并使用默认缓存
  解析；hf 传数据集 ID；custom 传本地数据集路径；缓存根目录沿用 benchmark 的默认解析；
- workload 必须显式提供选择方式、场景、模式、数据集来源、请求数、预热、并发、到达率、
  轮次、采样和生成参数；
- 最终报告和 raw result、持久化版本及 Evidence 均遵守“输出与持久化”中的目录合同。

字段缺失、矛盾或冻结条件未满足时反馈为 `invalid`；字段值以 Provider 事实和冻结合同为准。
