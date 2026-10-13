# Qwen3-8B Long-Prefill Search: Serial

本任务使用一个 candidate lane 顺序搜索 Qwen3-8B 的 vLLM 部署参数。
入口配置为 `git_worktree`、`direct runtime`、`local verifier`、`artifact_only`，
`max_parallel=1`。所有候选共享入口中声明的 `budget.max_attempts`，该预算只统计
候选搜索 attempt。

任务使用 `vllm-ascend-deployment` 完成部署预检和服务身份验收，使用
`vllm-ascend-benchmark` 通过官方 `vllm bench serve` 执行 production long-prefill
测量，使用 `vllm-ascend-inference-tuning` 编排 baseline、candidate、selection 和
最终复验。

## 构建事实

本任务是 parent-only 文档任务。模型路径、数据集目录、设备、端口、环境 wrapper、
Python、输出目录和 benchmark workload 都在真实执行时由用户或 Goal Plus provider
注入并核验。

## 启动文件

- `goal.md`：Codex 的 `$goal-plus` 入口。
- `goal-pi.md`：Pi 的 `/goal-plus` 入口。
- `workflow.md`：本任务的阶段顺序和门禁。
- `inputs.md`：运行期输入和来源。
- `search.md`：SearchSpec、负载、指标和预算合同。
- `modules.md`：三个 vLLM-Ascend Skill 的职责边界。

运行期占位值集中维护于对应入口文件的参数区，其他文档引用该来源。
