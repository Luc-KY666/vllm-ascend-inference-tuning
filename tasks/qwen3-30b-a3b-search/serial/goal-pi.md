/goal-plus mode=autonomous max_parallel=1 workspace_provider=git_worktree promotion_mode=artifact_only strategy=agent_guided 完成 Qwen3-30B-A3B long-prefill 的串行部署参数搜索。

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

串行模式使用 `max_parallel=1`。用户输入的 `max_attempts` 是 candidate attempt 的共享预算，
不因串行模式复制或重置。Goal Plus 必须在候选空间仍有未测试的合法配置且不存在硬阻塞时，
持续派发不重复的 candidate，尽可能把已结算的 candidate attempt 数量推进到
`max_attempts`；不能因为已经找到当前最优值、单次候选无效或一次候选报错而提前结束。
每个已派发并进入 `valid`、`invalid` 或 `error` 终态的 candidate attempt 只计数一次；客户端
查询超时、恢复原 attempt 和清理过程不得重复计数，也不得为了消耗预算重复测试同一配置。
只有候选空间已穷尽、资源/环境等硬阻塞无法继续，或 Goal Plus 的安全停止条件触发时，才允许
少于 `max_attempts` 结束，并在结果中记录 `attempts_used/max_attempts` 与提前停止原因。

设备组必须按 `input.md` 中 `authorized_ascend_npu_devices` 的声明顺序由 `tp * dp` 确定：
对任意相同的 `tp * dp = k`，baseline、candidate 和 promotion 都必须使用完全相同、顺序也
相同的前 `k` 张物理卡。不得因实时 free/busy、lane 或调度顺序改用另一组同数量卡；固定组
不可用时应报告资源阻塞/失败，不得静默换卡。每次 attempt 的 `ASCEND_RT_VISIBLE_DEVICES`、
资源锁和部署报告都必须记录这组实际物理设备。
