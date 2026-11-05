# 串行任务合同

通用执行流程统一由 `vllm-ascend-goal-plus-search/SKILL.md` 及其 references 负责；本文件
集中声明本任务的完整目标、部署 edit surface、baseline、benchmark 合同、用户输入、执行
职责和串行模式差异。

## 执行职责与完成条件

- Main 必须根据本轮授权、实际环境和本文件准备可运行的 origin 配置，并冻结独立 verifier、
  负载、指标、资源和预算；不得沿用旧报告的分数。
- worker 只交候选配置；verifier 负责试验服务的启动、请求、测量和清理。
- 至少根据一轮真实 Evidence 提出下一轮假设；按公开选择结果复验。未获得有效收益时，
  如实交付经过复验的 baseline。
- 缺少授权、有效测量或清理证据时，报告必须标记为未完成。

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

每个 baseline、candidate 和 promotion attempt 仅执行一次正式 benchmark。该次官方 raw result
直接作为该 attempt 的评分；不追加测量、不对多个样本取平均，失败直接按 `invalid` 或 `error`
结算。

## 串行模式

- Goal Plus 参数为 `max_parallel=1`，同一时刻只有一个 active candidate。
- `max_attempts` 是整个 Search 的 candidate attempt 共享预算；每个 candidate 使用用户授权的
  候选值，不因串行模式复制或重置 workload 合同。只要仍有未测试的合法配置且没有硬阻塞，
  Goal Plus 必须继续提交不重复 candidate，尽可能消耗到 `max_attempts`，不能因当前已有最优
  结果、单次 `invalid` 或 `error` 提前停止。
- 每个已派发并结算为 `valid`、`invalid` 或 `error` 的 candidate attempt 计数一次；查询超时、
  原 attempt 恢复、清理和幂等重试不重复计数，也不能重复测试同一配置来填满预算。候选空间
  穷尽、资源/环境硬阻塞或安全停止条件触发时才允许提前结束，并记录
  `attempts_used/max_attempts` 和停止原因；baseline/promotion 的独立复验不挪用 candidate
  预算。
- 设备组由 `input.md` 的授权设备列表顺序和 `k = tp * dp` 的前缀确定。相同 `k` 的所有配置
  （包括 baseline、candidate、promotion）必须使用同一组、同一物理顺序和同一
  `ASCEND_RT_VISIBLE_DEVICES`；当前授权列表中 `k=2` 固定为 `0,1`，`k=4` 固定为 `0,1,2,3`。
  不得因为实时 free/busy、lane 或调度顺序换用其他同数量卡；固定组不可用时报告资源阻塞/失败。
- 具体 Search、部署、压测、报告和交付步骤只按 bridge skill 执行。
