# 并行任务合同

通用执行流程统一由 `vllm-ascend-goal-plus-search/SKILL.md` 及其 references 负责；本文件
只声明本任务的目标、可调参数、benchmark 合同、用户输入和并行模式差异。

## 优化目标

- 模型服务身份为 `qwen3-30b-a3b`，模型路径来自 `input.md`。
- 场景为 preset 模式的 `long_prefill`；数据集 ID 和适配器由 benchmark 按场景 contract
  派生并从默认 `~/.cache` 解析，不接受用户重复传入数据集 ID 或路径。
- 只接受 `vllm-ascend-deployment` v3 与 `vllm-ascend-benchmark` v3 artifact；部署和压测报告分别
  必须通过对应 v3 schema 校验，拒绝 v2 或旧中间区块。
- 唯一评分字段为官方 raw result 中冻结的 `TTFT_P95`，在 benchmark v3 报告中映射为
  `result.ttft_ms.p95`，优化方向为最小化。
- baseline、candidate 和 promotion 使用同一模型、场景、指标口径和三个 workload 输入。

## 可调参数

`input.md` 中的 `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens`、
`enable_expert_parallel`、`enable_prefix_caching`、`distributed_executor_backend`、
`async_scheduling` 和 `compilation_config` 是唯一部署 edit surface。候选必须满足
`tp * dp <= max_tp_dp`；其中 `enable_expert_parallel` 映射为
`--enable-expert-parallel`，`enable_prefix_caching=false` 映射为 `--no-enable-prefix-caching`，
`distributed_executor_backend=mp` 映射为 `--distributed_executor_backend mp`，
`distributed_executor_backend=null` 时省略该参数，
`async_scheduling=true` 映射为 `--async-scheduling`，
`compilation_config` 仅允许 `NONE`、`FULL_DECODE_ONLY`、`PIECEWISE` 和
`FULL_AND_PIECEWISE`，传递给 `--compilation-config <json>`。
`trust_remote_code` 固定启用并按 `--trust-remote-code` 传递；`additional_config` 固定为
`{"enable_flashcomm1": true, "weight_nz_mode": 2}`，
`gpu_memory_utilization` 固定为 `0.95`，`max_model_len` 固定为 `40960` 并按
`--max-model-len 40960` 传递；这些字段不属于 edit surface。不得修改模型、benchmark
skill 其余 workload 参数、指标、阈值或资源合同。候选取值由用户在 `input.md` 提供，task
不推导额外默认值或搜索范围。

baseline origin 固定使用 `--trust-remote-code`、`--max-num-seqs 100`、
`--max-model-len 40960`、`--max-num-batched-tokens 16384`、`--tensor-parallel-size 4`、
`--enable-expert-parallel`、`--distributed_executor_backend mp`、
`--no-enable-prefix-caching`、`--async-scheduling`、
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`、
`--additional-config '{"enable_flashcomm1":true,"weight_nz_mode":2}'` 和
`--gpu-memory-utilization 0.95`；不传 `--data-parallel-size` 时按 deployment contract 使用
默认 `dp=1`。candidate 仍可在 edit surface 中调整可调字段。

## Benchmark 与用户输入

用户只填写 `input.md` 的 `num_prompts`、`max_concurrency` 和 `request_rate` 三个
workload 字段。三个字段必须为有效值，并在 baseline、candidate、promotion 中保持一致；
其余 workload 参数、默认值、benchmark mode、数据集适配器、生成参数和百分位配置由
`vllm-ascend-benchmark` 及其 references 指定和记录。

## 并行模式

- Goal Plus 参数为 `max_parallel=2`，每个 batch 同时包含两个 candidate。
- 两个 candidate 必须使用各自的候选值；其资源隔离和执行编排由 provider/bridge skill
  负责，task 不声明设备、端口或目录分配规则。
- `max_attempts` 是整个 Search 的共享候选预算，不按 lane 复制或重置 workload 合同。
- 具体 Search、部署、压测、报告和交付步骤只按 bridge skill 执行。
