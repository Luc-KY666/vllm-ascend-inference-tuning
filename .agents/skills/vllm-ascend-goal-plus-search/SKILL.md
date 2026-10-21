---
name: vllm-ascend-goal-plus-search
description: 以 Goal Plus Provider 身份编排一次 vLLM-Ascend 部署参数搜索：建立 baseline、冻结外部搜索合同、创建 run、物化候选、运行独立 verifier、选择与提升，并把部署、测量和资源事实交给对应 Skill。
---

# vLLM-Ascend Goal Plus 搜索 Provider 技能

本 skill 是 Goal Plus 与 vLLM-Ascend 部署、压测 Skill 之间的通用桥梁。它把外部调用者
冻结的有界搜索接入 Goal、Search、candidate、Evidence 和 promotion 公开接口，不实现
第二套调度器、租约、Runtime、MCP、服务控制器或搜索器。

## 入口与职责

启动时先读取工作区根 `AGENTS.md`、当前 task 的 `goal.md`、`goal-pi.md`、
`workflow.md`、`input.md`（存在时）和业务仓事实，再读取以下执行 Skill：

- `../vllm-ascend-probes/SKILL.md`
- `../vllm-ascend-deployment/SKILL.md`
- `../vllm-ascend-benchmark/SKILL.md`

普通部署、测量或调查只调用执行 Skill 完成一次 attempt；只有具备可评分目标、确定性
独立 verifier 和隔离 edit surface 时才创建 Search。task 只声明目标、可调字段、benchmark
合同和用户输入，不能复制本 skill 的通用执行流程。

## 唯一规范来源

- [搜索流程](references/search-flow.md)：定义完整 Search 生命周期、triage、SearchSpec
  冻结、Search 级唯一 probe/help、candidate session、verifier、续跑、selection、promotion、
  终审和失败恢复。这里是流程与 probe 规则的唯一权威来源。
- [Provider 输入契约](references/provider-contract.md)：定义 SearchSpec/provider 输入、
  deployment/benchmark 映射、attempt artifact、运行时 wrapper、输出授权和交付清单。这里
  是 provider 与执行 Skill 之间输入和 artifact 规则的唯一权威来源。

需要准确工具参数、回执、幂等、恢复或失败分类时，直接查阅上述 reference；不要在 task 或
其他桥接文件中重新定义同一规则。

## 桥接边界

- 模型、指标、参数、候选值、资源、预算和 workload 必须由外部 SearchSpec 冻结；本 skill
  不猜测这些事实，也不读取普通环境变量、旧日志或候选自报值替代 provider 回执。
- 数据集选择以及 `dataset ID`/本地路径语义必须遵守 `vllm-ascend-benchmark` contract：
  preset 只冻结 workload 场景，由 benchmark 按场景合同派生数据集 ID 并解析默认缓存；hf
  才由调用者提供数据集 ID，只有 custom 允许本地数据集路径。task 或 provider 不得另造
  外部数据集根目录输入。
- Goal Plus/provider 必须注入候选参数、设备组、端口、锁和本次 deployment/benchmark 输出
  目录；部署和压测只能在该注入目录下写入文件。持久化 Verifier 按 Goal Plus 的 verifier
  存放契约处理，不由 task 或本 skill 猜测路径。
- Search source 必须是独立且 clean 的 Git 目录。
- 不绕过 Goal Plus 设备池、lane lease、物理锁或 verifier 临时目录，不以 fixture、模拟响应、
  单次健康检查或 worker 自报分数代替独立性能 Evidence。
