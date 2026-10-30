$goal-plus mode=autonomous max_parallel=1 workspace_provider=git_worktree promotion_mode=artifact_only strategy=agent_guided 完成 Qwen3-30B-A3B long-prefill 的串行部署参数搜索。

## 优化目标

模型服务身份为 `qwen3-30b-a3b`。benchmark 固定为 preset 模式的 `long_prefill` 场景；
数据集 ID 和适配器由 benchmark 按场景 contract 从默认 `~/.cache` 派生解析，用户不重复
输入数据集 ID 或路径。只接受 `vllm-ascend-deployment` v3 与
`vllm-ascend-benchmark` v3 artifact；以官方 raw result 的 `TTFT_P95` 为唯一评分字段，
v3 报告中固定映射为 `result.ttft_ms.p95`，并最小化它。baseline、candidate 和
promotion 必须保持相同的模型、场景、指标口径和 workload。

## 可调参数

`input.md` 中的 `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens`、`enable_expert_parallel`、
`enable_prefix_caching`、`distributed_executor_backend`、`async_scheduling` 和 `compilation_config`
是唯一可调部署字段，候选必须满足
`tp * dp <= max_tp_dp`。
其中 `enable_expert_parallel` 控制 `--enable-expert-parallel`，`enable_prefix_caching=false`
时传递 `--no-enable-prefix-caching`；`distributed_executor_backend` 传递给
`--distributed_executor_backend`；值为 `mp` 时传递 `--distributed_executor_backend mp`，
值为 `null` 时省略该参数。`async_scheduling=true` 传递 `--async-scheduling`，
`compilation_config` 仅允许 `NONE`、`FULL_DECODE_ONLY`、`PIECEWISE` 和
`FULL_AND_PIECEWISE`，并传递给 `--compilation-config`。
`additional_config` 固定为 `{"enable_flashcomm1": true, "weight_nz_mode": 2}`，
`gpu_memory_utilization` 固定为 `0.95`，`max_model_len` 固定为 `40960` 并按
`--max-model-len 40960` 传递；三者不属于 Search 候选。候选取值由用户在
`input.md` 提供；不得修改 benchmark 的其他 workload 参数、指标、阈值或资源合同。通用执行流程由
`.agents/skills/vllm-ascend-goal-plus-search/SKILL.md` 统一负责。

## Baseline 部署参数

baseline origin 固定使用以下部署参数：`--trust-remote-code`、`--max-num-seqs 100`、
`--max-model-len 40960`、`--max-num-batched-tokens 16384`、`--tensor-parallel-size 4`、
`--enable-expert-parallel`、`--distributed_executor_backend mp`、
`--no-enable-prefix-caching`、`--async-scheduling`、
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`、
`--additional-config '{"enable_flashcomm1":true,"weight_nz_mode":2}'` 和
`--gpu-memory-utilization 0.95`。baseline 不传 `--data-parallel-size`，按 deployment contract
使用默认 `dp=1`；candidate 仍可在上方 edit surface 中调整可调字段。

## 用户输入

`input.md` 是唯一输入来源。用户 workload 输入仅为 `num_prompts`、`max_concurrency` 和
`request_rate`；三者在 baseline、candidate、promotion 中保持一致。其余 workload 参数、
默认值、benchmark mode、数据集适配器、生成参数和百分位配置由
`vllm-ascend-benchmark` 及其 references 指定，不由 task 猜测或覆写。

串行模式使用 `max_parallel=1`；`max_attempts` 为整个 Search 的共享候选预算。
