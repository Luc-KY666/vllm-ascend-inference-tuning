# Provider 输入契约

本文件说明 Provider 如何把一次由外部调用者冻结的 Search/attempt 合同映射为部署与压测
Skill 的输入。它不创建第二份搜索真相，也不改写执行 Skill 的判定。

## Probe 绑定

Search 级 probe、help 输出和禁止重复调用的规则只定义在[搜索流程](search-flow.md)；本契约
不复制那套流程。Provider 的职责是把流程冻结的 probe 路径、SHA256、执行上下文和 shell
fingerprint 注入每个 attempt，并让 deployment/benchmark Skill 引用这些事实。执行 Skill
若按普通 attempt 合同要求调用 probe，Search 绑定时必须跳过本地调用，改为校验既有 artifact；
校验失败或上下文变化时阻塞当前 Search，由流程规则决定创建新的 Search。

## 冻结门禁

创建 Search 前，Goal Plus/Main 必须把以下事实写入 SearchSpec 或 durable Evidence：

- 模型、tokenizer、后端版本、激活 wrapper 和代码版本；
- 模型路径、服务身份、固定运行参数和模型配置摘要；
- workload、数据集选择及其 ID/本地路径语义、请求顺序、预热、轮次、并发、采样、统计公式和失败处理；
- 外部提供的可调字段白名单、离散值、非法组合和 edit surface；
- 指标字段、优化方向、绝对/相对门槛、平局规则和停止条件；
- candidate 总预算、并发、超时、provider、设备池、设备组、端口和资源锁；
- Search 级环境事实的 provider 绑定（路径、SHA256、执行上下文和复用规则见[搜索流程](search-flow.md)）；
- verifier 位置、artifact 导出方式和服务清理/保留合同。

缺少指标、方向、预算、独立 verifier 或真实资源授权时，不创建 Search；只能调用执行 Skill
生成诊断或单次测量 artifact，并记录真实缺口。

## 输入来源

一次 attempt 的输入由四类来源组成：

1. 冻结常量：模型身份、固定部署事实、workload、指标、统计口径和 provider 合同；
2. 候选字段：只允许改写外部 edit surface 内的部署字段和值；
3. Provider 注入：可见资源、设备组、端口候选、物理锁、attempt 输出目录和 verifier 临时目录；
4. 目标环境事实：命令行能力、模型配置、tokenizer、版本、数据类型、量化和上下文能力。

候选不得改写第 1、3、4 类输入。Provider 不使用普通环境变量、旧日志或候选自报值替代
受信任的 Goal Plus/provider 回执。

## 部署输入映射

部署 Skill 的规范输入由 provider 组装并交给其 `scripts/input_contract.py`：

| Provider 输入 | 来源 | 约束 |
| --- | --- | --- |
| 模型路径 | 冻结常量 | 必须是可读的本地模型目录 |
| 服务身份 | 冻结常量 | 不从路径推导 |
| 可见资源 | provider 注入 | 必须与本次设备组和资源锁一致 |
| 端口候选 | provider 注入 | 只检查候选集合，不扫描未授权范围 |
| 偏好端口 | provider 注入 | 必须属于候选集合 |
| attempt 输出目录 | Goal Plus/provider 注入 | 部署和压测本次写入范围 |
| `runtime.<field>` | 外部候选或推荐 | 字段名和值域由 SearchSpec 提供 |
| 资源上限 | SearchSpec/环境事实 | 由外部合同冻结 |

部署成功后向 Provider 交还选定资源、服务地址、服务进程、服务身份和部署报告；压测路径
复用这些事实，不重新猜测资源归属。

## 输出与持久化

Goal Plus/provider 必须为 deployment 和 benchmark 注入本次 attempt 的输出目录。部署和压测
产生的报告、raw result、日志、脚本及其他文件只能写入该目录，不得自行选择或创建其他输出
目录。

持久化 Verifier 的脚本、报告、日志和其他证据按照 Goal Plus verifier 存放契约处理；存放
位置由 Goal Plus/provider 提供，task 和本契约不推断额外路径。

## Attempt 运行时与清理

Provider 必须为每个 attempt 提供统一服务启动 wrapper，并让 probe、serve、benchmark 和清理
使用同一执行上下文。若 Goal Plus verifier 用
`env["PYTHONPATH"] = str(cwd)` 覆盖环境，wrapper 必须从仍存活的 `ASCEND_HOME_PATH` 或
`ASCEND_TOOLKIT_LATEST_HOME` 重建 CANN 的 `python/site-packages` 与
`opp/built-in/op_impl/ai_core/tbe`，再与候选源路径组成服务子进程的 `PYTHONPATH`。

wrapper 必须设置 `PYTHONDONTWRITEBYTECODE=1`。服务停止时除了进程组清理，还必须遍历
`/proc` 清理 engine/worker 后代，并在 attempt 结算前确认没有孤儿进程、NPU 残留或未释放锁。
客户端超时不等于 verifier 已取消；下一次 session 或 steering 前必须先通过 wait、recovery
或 status 确认前一 attempt 已终态。

## 压测输入映射

压测 Skill 接收冻结的服务事实和 workload 合同：

- 服务路径 A：在同一个受保护调用中完成部署、健康验收、官方测量、raw result 校验和清理；
- 服务路径 B：Provider 提供已部署服务地址、服务身份、tokenizer、部署报告和 attempt 目录，
  压测只负责复核和测量；
- 数据集必须遵守 benchmark contract：preset 只传场景，由 benchmark 派生数据集 ID 并使用
  默认缓存解析；hf 传数据集 ID；只有 custom 传本地数据集路径；不得把外部缓存根目录作为
  额外 workload 输入；
- workload 必须显式提供选择方式、场景、模式、数据集来源、请求数、预热、并发、到达率、
  轮次、采样和生成参数；
- 最终报告和 raw result 必须写入 Goal Plus/provider 注入的当前 attempt 输出目录。

缺失、矛盾或无法冻结的字段报告为 `invalid`，不得用宿主默认值替代 Provider 事实。

## 指标与评分

独立 verifier 只从官方 raw result 读取 SearchSpec 指定的唯一字段，不重算、不插值、不以
零值填补失败。最后一个 stdout JSON 必须包含该字段、方向、attempt 身份和
`outcome=valid|invalid|error`。

同一字段和统计口径必须用于 baseline、candidate 与 promotion；服务健康、HTTP 成功和
worker 自报分数不能成为排名依据。

## Attempt 差异

| 阶段 | 可变内容 | 目的 |
| --- | --- | --- |
| baseline | 推荐值或冻结默认 | 建立可比较 origin |
| candidate | edit surface 内的候选值 | 搜索候选空间 |
| promotion | 获选值，可使用新 attempt 目录 | 独立复验与交付 |

baseline 失败、环境故障和候选失败必须分开归因。每个候选配置只提交一次；重复测量只能
作为诊断，不能重试到通过。

## 交付清单

至少交付：

- baseline、候选和最终配置摘要；
- candidate/iteration/attempt 索引；
- 部署报告、压测报告、官方 raw result 及摘要 hash；
- 指标、方向、门槛、实际数值和失败候选；
- 设备、端口、锁、服务生命周期和清理证据；
- promotion 复验结果和 artifact；
- 未覆盖项、环境故障和仍需用户执行的真实 E2E。
