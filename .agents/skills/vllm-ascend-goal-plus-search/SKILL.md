---
name: vllm-ascend-goal-plus-search
description: 以 Goal Plus Provider 身份编排一次 vLLM-Ascend 部署参数搜索：建立 baseline、冻结外部搜索合同、创建 run、物化候选、运行独立 verifier、选择与提升，并把部署、测量和资源事实交给对应 Skill。
---

# vLLM-Ascend Goal Plus 搜索 Provider 技能

本 skill 负责完整的 vLLM-Ascend 参数搜索流程，并把部署和压测交给对应执行 Skill。它消费
Goal Plus 提供的 SearchSpec、状态句柄和运行时输入，不创建或管理 Goal Plus 的 task、Goal、
FrozenSpec、run、attempt、Evidence 或输出结构；也不实现第二套调度器、租约、Runtime、MCP、
服务控制器或搜索器。

## 入口与唯一规范来源

启动时读取 Goal Plus 提供的当前 task/Search 合同和业务仓事实，再按需读取：

- [搜索流程](references/search-flow.md)：Search 生命周期、唯一 probe/help、attempt 状态、
  固定续跑循环、freeze 流程、verifier 等待/恢复、selection、promotion 和最终审查；它是流程
  与 probe 规则的唯一权威来源；
- [Provider 输入契约](references/provider-contract.md)：SearchSpec 输入映射、部署/压测事实交接、
  artifact 引用和 Provider 反馈字段的唯一权威来源；
- `../vllm-ascend-probes/SKILL.md`、`../vllm-ascend-deployment/SKILL.md`、
  `../vllm-ascend-benchmark/SKILL.md`：环境、部署和压测的执行合同。

需要工具参数、回执、幂等、恢复、失败分类或报告字段时直接查阅上述来源；不在本 skill 中复制
Goal Plus 的任务结构或存储布局。

## 桥接边界

- 模型、指标、候选值、资源、预算和 workload 由外部 SearchSpec 冻结；不猜测事实或使用普通环境变量、
  旧日志、fixture 和候选自报值替代 provider 回执。
- preset 只冻结 workload 场景；数据集 ID/路径由 benchmark 按场景合同派生，task 不另造数据来源。
- Provider/Goal Plus 注入候选参数、设备组、端口、锁以及不透明的运行时输出句柄；skill 将这些
  输入原样交给 verifier、deployment 和 benchmark，不创建目录、不选择路径、不定义目录层级。
- 每个 verifier attempt 启动 vLLM 服务前必须向 verifier/服务进程环境注入
  `VLLM_RPC_BASE_PATH=/tmp`，将 vLLM IPC transport socket 限制在短路径下；该变量只影响
  IPC transport，不能替换、覆盖或重定向 Goal Plus 注入的报告、日志和 Evidence 输出句柄。
- Goal、Search、FrozenSpec、run、candidate、attempt、Evidence 以及所有输出的命名、生命周期和
  持久化方式由 Goal Plus 管理；skill 只使用公开接口返回的 ID、状态和 artifact 引用。
- Search source 必须是独立、clean 的 Git 目录；不绕过设备池、lane lease、物理锁或独立 verifier。

## 搜索主流程与执行 Skill 交接

完整流程由 [搜索流程](references/search-flow.md) 定义，主线只做以下工作：

1. 消费 Goal Plus 提供的 SearchSpec、资源/锁句柄、唯一 probe/help artifact 引用和输出句柄；
2. 完成 triage、baseline、freeze、candidate 规划、独立 verifier、结果结算、selection、promotion
   和 final review；
3. 在每个 verifier attempt 中调用 [`vllm-ascend-deployment`](../vllm-ascend-deployment/SKILL.md)，
   传入冻结的模型、候选部署参数、Provider 资源事实、Search 级 probe/help 引用和 Goal Plus
   提供的运行时输出句柄，并在服务进程环境中设置 `VLLM_RPC_BASE_PATH=/tmp`，接收部署报告
   与服务事实；
4. 将部署返回的服务地址、served model identity、tokenizer 和部署报告交给
   [`vllm-ascend-benchmark`](../vllm-ascend-benchmark/SKILL.md)，在同一已启动服务上按冻结 workload
   仅执行一次正式 benchmark，接收该次官方报告、raw result 引用和唯一评分字段，并直接将该次
   结果作为该配置的搜索评分；不追加测量、不对多个样本取平均；
5. 将部署/压测结果和清理状态交回 Goal Plus 公开接口，由 Goal Plus 完成状态、Evidence、选择、
   提升和最终报告的持久化。任一 verifier 失败时，先保留当次 serve/benchmark 日志和最后状态
   的 Evidence 引用，再执行服务清理；完整日志不写入最终报告正文。

上述流程不得创建或拼接 Goal Plus 的任务、freeze、run、attempt 或输出路径；deployment 和
benchmark 也只能使用 Goal Plus/provider 注入的输出句柄。
