$goal-plus mode=autonomous max_parallel=2 workspace_provider=git_worktree promotion_mode=artifact_only strategy=agent_guided 请调用 Boom 场景流程与领域工具搜索 Qwen3.8-27B 的 vLLM 参数，在授权 NPU 上独立部署并测量单/多请求性能，分析后继续搜索，选优复验并交付配置与证据，最后停服。

读取本任务 workflow.md。Main 根据本轮授权、实际环境和用户目标准备可运行的 origin 配置，冻结独立 verifier、负载、指标、资源与预算；不要沿用旧报告的分数。worker 只交候选配置，verifier 负责试验服务的启动、请求、测量和清理。至少根据一轮真实 Evidence 提出下一轮假设；按公开选择结果复验，未获得有效收益则如实交付经过复验的 baseline。缺少授权、有效测量或清理证据时报告未完成。
