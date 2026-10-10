---
name: vllm-ascend-inference-tuning
description: 面向 Goal Plus Main 编排 vLLM-Ascend 的部署、独立压测和有界参数搜索，生成可审查的最优配置推荐与证据索引。
---

# vLLM-Ascend 推理调优工作流

本 skill 是顶层编排层，负责把部署和压测能力接入 Goal Plus 的 Goal/Search/candidate
流程。它不实现第二套调度器、资源锁、Runtime、MCP、服务生命周期控制器或搜索器。

模型、设备、workload、可调参数、指标方向、性能门槛、预算和 provider 必须由当前任务
及已安装 Goal Plus 的公开 `SearchSpec` 冻结。没有完整搜索合同时，本 skill 只能完成
准备、部署或测量，并明确报告阻塞；不能自行选用 E2EL、吞吐或其他默认优化目标。

## 先读取本轮资料

按以下顺序读取并记录事实：

1. 工作区根 `AGENTS.md`、`TASKS.md`。
2. `.agents/task/goal.md` 或 `.agents/task/goal-pi.md`、`workflow.md`、`inputs.md`、
   `modules.md` 和任务级 benchmark/request 文件。
3. 任务实际业务仓库中的规则、环境入口、服务脚本和测试。
4. Goal Plus provider、SearchSpec、设备池、candidate、verifier 和 output surface 的
   实际公开配置。
5. [搜索合同](references/search-contract.md)、[部署 skill](../vllm-ascend-deployment/SKILL.md)
   和[压测 skill](../vllm-ascend-benchmark/SKILL.md)。

不得把 `build.json`、候选自报 JSON、旧日志或普通环境变量当作 Goal Plus 运行期真相。

## 搜索合同门禁

启动 Search 前，Main 必须确认以下字段已经冻结，并将身份和 hash 写入 TASKS 或
Goal Plus durable Evidence：

- 模型、tokenizer、vLLM/vLLM-Ascend、激活 wrapper 和代码版本。
- `model_path`、`served_model_name`、dtype、TP/DP、上下文、批处理和量化参数。
- 固定 workload、请求顺序、warmup、rounds、并发、采样、失败处理和统计公式。
- candidate 允许修改的参数白名单、离散取值、非法组合和 edit surface。
- 优化指标、方向、绝对 SLO、相对退化门槛、平局规则和停止条件。
- `max_parallel`、candidate 总预算、单次超时、设备池、provider、端口和资源锁。
- verifier 输出位置、artifact 导出方式、清理/保留服务合同和最终复验次数。

缺少指标、方向、门槛、预算、独立 verifier 或真实设备授权时：

1. 不创建或派发参数 Search。
2. 可以调用部署/压测 skill 生成诊断或一次测量 artifact。
3. 在 TASKS 中记录缺口、已完成阶段和恢复所需的输入。
4. 不输出“最优”“提升”“通过”或“应发布”的结论。

具体字段和推荐报告见[搜索合同](references/search-contract.md)。

## 标准执行顺序

### 1. 建立 baseline

- 核对 provider、模型身份、环境 wrapper、设备分配、端口、锁和输出隔离。
- 使用部署 skill 完成真实环境和服务身份验收。
- 使用同一任务 workload 完成 baseline 正式测量，保存原始流/raw JSON、配置摘要、
  hash、计数、指标和清理状态。
- baseline 必须通过任务质量、协议、样本完整性、资源和清理门槛；单次 HTTP 成功或
  健康状态不能替代性能 baseline。

### 2. 创建和分配 candidate

- 只使用 Goal Plus 的公开 Search/candidate 接口。
- 由 GP 管理 candidate workspace、设备 lease、物理锁、iteration、超时和状态。
- Main 冻结基线、输入、verifier、reference、统计口径和 edit surface；candidate 只能
  修改白名单参数。
- 不让 worker 创建嵌套搜索器、第二个调度器或自己的全局最佳记录。

### 3. 部署和测量

每个 candidate 只执行 GP 分配的一次 attempt：

1. 读取 candidate 的冻结参数。
2. 在 provider 允许服务生命周期由本轮 verifier 管理时，调用部署 skill 做输入和容量
   预检并启动服务；预启动/ThinkThread 路径只核对 worker 交付的部署报告和服务接口。
3. 验证健康、模型身份和最小请求。
4. 调用压测 skill 执行冻结 workload。
5. 导出独立 benchmark artifact、原始证据和清理状态。
6. 将 attempt ID、配置摘要、失败阶段和输出路径交回 GP/Main。

git_worktree/direct 的独立 verifier 必须在同一次持锁执行中完成启停、请求和清理；
ThinkThread 的短 verifier 只访问 worker 预启动并交付的服务接口。

### 4. 分析、选择和再搜索

- 只读取独立 verifier 产出的 artifact，不读取 candidate 自报分数作为事实。
- 由 GP 按冻结的 `metric_name`、`metric_direction`、门槛和排序规则结算。
- Main 审查逐请求样本、绝对指标、配置差异、资源证据和失败类型，再决定下一批假设。
- 批次之间不得改变 workload、指标口径、设备约束或 verifier；规则变化必须新建或重
  新冻结 Search。
- 只有满足任务停止条件时才进入 selection/promotion。

### 5. 最终独立复验

- 通过公开 selection 获取 GP 选择的 artifact，核对实际配置、服务接口和 publication
  内容一致。
- 在任务合同指定的设备和 provider 上独立重放 baseline 与获选配置，覆盖完整 workload。
- 按任务合同清理服务或交付最终保留服务；核对 PID、端口、设备、配置和停止责任。
- 无收益时，只有实际选择并独立验证 baseline 后，才能报告 baseline 保留。
- 不能修改报告字段、重命名 artifact 或伪造 `valid` 来替代 selection、promotion 和
  最终 verifier。

## 报告边界

顶层交付至少包含：

- 冻结的 baseline 和最终配置摘要；
- candidate/iteration/attempt 索引；
- 部署报告和 benchmark 报告路径及 SHA256；
- 任务指标、方向、门槛、实际数值和失败候选；
- 设备/锁/端口/服务生命周期证据；
- 最终独立复验结果、清理或服务保留状态；
- 未覆盖项、环境故障和仍需用户执行的真实 E2E。

部署报告的 `service_identity_valid` 只能证明服务身份。benchmark 报告的 `status=success`
只能证明一次测量完整。Goal Plus 的最终结论必须来自冻结合同下的独立 verifier 和真实
运行状态。

## 禁止事项

- 不把公共 skill 改造成具体模型、数据集、设备或机器路径的任务合同。
- 不自动为缺省任务选择优化指标、TP/DP 搜索空间、SLO、候选预算或 provider。
- 不绕过 GP 设备池、lane lease、物理锁、verifier 临时目录和 output surface。
- 不以 fixture、模拟响应、静态报告或单次服务健康代替真实部署和性能证据。
- 不修改 `.codex/`、`.pi/`，不写入模型目录，不自动删除用户现场。
