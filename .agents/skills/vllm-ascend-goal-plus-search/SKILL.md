---
name: vllm-ascend-goal-plus-search
description: 以 Goal Plus Provider 身份编排一次 vLLM-Ascend 部署参数搜索：建立 baseline、冻结外部搜索合同、创建 run、物化候选、运行独立 verifier、选择与提升，并把部署、测量和资源事实交给对应 Skill。
---

# vLLM-Ascend Goal Plus 搜索 Provider 技能

本 skill 是 Goal Plus 与 vLLM-Ascend 执行 Skill 之间的桥梁：把外部调用者冻结的有界搜索
接入 Goal、Search、candidate、Evidence 和 promotion 公开接口。它不实现第二套调度器、租约、
Runtime、MCP、服务控制器或搜索器。

## 入口与唯一规范来源

启动时读取根 `AGENTS.md`、task 的 `goal.md`、`goal-pi.md`、`workflow.md`、`input.md`（存在时）
和业务仓事实，再按需读取：

- [搜索流程](references/search-flow.md)：Search 生命周期、唯一 probe/help、attempt 状态、
  固定续跑循环、freeze 门禁、verifier 等待/恢复、Attempt 运行时与清理、selection、promotion
  和最终审查；它是流程与 probe 规则的唯一权威来源；
- [Provider 输入契约](references/provider-contract.md)：SearchSpec 输入映射、目录隔离、
  artifact/Evidence 和 Provider 反馈字段的唯一权威来源；
- `../vllm-ascend-probes/SKILL.md`、`../vllm-ascend-deployment/SKILL.md`、
  `../vllm-ascend-benchmark/SKILL.md`：环境、部署和压测的执行合同。

需要工具参数、回执、幂等、恢复、失败分类或报告字段时直接查阅上述来源；task 和本桥接文件不
复制流程细节。

## 桥接边界

- 模型、指标、候选值、资源、预算和 workload 由外部 SearchSpec 冻结；不猜测事实或使用普通环境变量、
  旧日志、fixture 和候选自报值替代 provider 回执。
- preset 只冻结 workload 场景；数据集 ID/路径由 benchmark 按场景合同派生，task 不另造 dataset root。
- Provider 注入候选参数、设备组、端口、锁和输出目录；目录隔离、Evidence 持久化及反馈字段见
  [Provider 输入契约](references/provider-contract.md)，不在入口重复展开。
- 全局 probe 结果和 verifier Evidence 由 Goal Plus 在 workspace 根目录创建并管理持久化 Evidence
  目录；verifier 输出使用 Goal Plus 注入的 `GOAL_PLUS_VERIFIER_TMPDIR`，不得写入被验收 workspace。
  Evidence、临时目录和其他产物的具体位置由 Goal Plus 管理，task 不自行指定路径。
- Search source 必须是独立、clean 的 Git 目录；不绕过设备池、lane lease、物理锁或独立 verifier。
