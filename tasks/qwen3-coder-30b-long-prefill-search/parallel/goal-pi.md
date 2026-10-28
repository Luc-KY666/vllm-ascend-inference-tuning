/goal-plus mode=autonomous max_parallel=2 workspace_provider=git_worktree promotion_mode=artifact_only strategy=agent_guided 完成 Qwen3-Coder-30B long-prefill 的并行部署参数搜索。

## 优化目标

模型服务身份为 `qwen3-coder-30b`。benchmark 固定为 preset 模式的 `long_prefill` 场景；
数据集 ID 和适配器由 benchmark 按场景 contract 从默认 `~/.cache` 派生解析，用户不重复
输入数据集 ID 或路径。只接受 `vllm-ascend-deployment` v3 与
`vllm-ascend-benchmark` v3 artifact；以官方 raw result 的 `TTFT_P95` 为唯一评分字段，
v3 报告中固定映射为 `result.ttft_ms.p95`，并最小化它。baseline、candidate 和
promotion 必须保持相同的模型、场景、指标口径和 workload。

## 可调参数

`input.md` 中的 `tp`、`dp`、`max_num_seqs` 和 `max_num_batched_tokens` 是唯一可调部署
字段，候选必须满足 `tp * dp <= max_tp_dp`。候选取值由用户在 `input.md` 提供；不得修改
benchmark 的其他 workload 参数、指标、阈值或资源合同。通用执行流程由
`.agents/skills/vllm-ascend-goal-plus-search/SKILL.md` 统一负责。

## 用户输入

`input.md` 是唯一输入来源。用户 workload 输入仅为 `num_prompts`、`max_concurrency` 和
`request_rate`；三者在 baseline、candidate、promotion 中保持一致。其余 workload 参数、
默认值、benchmark mode、数据集适配器、生成参数和百分位配置由
`vllm-ascend-benchmark` 及其 references 指定，不由 task 猜测或覆写。

并行模式使用 `max_parallel=2`；每批包含两个 candidate，`max_attempts` 为整个 Search
的共享候选预算。
