# 并行任务合同

通用执行流程由 `vllm-ascend-goal-plus-search/SKILL.md` 及其 references 负责；本文件集中声明
本任务的目标、部署 edit surface、baseline、benchmark 合同、用户输入和并行模式差异。

## 优化目标

模型服务身份为 `qwen3.8-27b`，模型路径来自 `input.md`。benchmark 固定为 preset 的
`prefill_decode_balance` 场景和 `production` 模式，使用 ShareGPT 数据集；数据集 ID 和适配器由
benchmark 按场景 contract 从默认 `~/.cache` 派生，用户不重复填写数据集路径。

只接受 `vllm-ascend-deployment` v3 与 `vllm-ascend-benchmark` v3 artifact。唯一评分字段为
官方 raw result 的 `request_goodput`，映射为 `result.goodput.request_goodput`（req/s），
优化方向为最大化。候选还必须满足严格的 `TTFT_P95 < 500 ms` 和 `TPOT_P95 < 25 ms`，
对应报告字段为 `result.ttft_ms.p95` 和 `result.tpot_ms.p95`。

benchmark 必须显式传递 `--goodput ttft:500 tpot:25`，固定发送 500 个正式请求、最大并发 64、
请求到达率 4 req/s，并使用 P50/P95 统计。baseline、candidate 和 promotion 必须保持相同的
模型、场景、SLO、指标口径和 workload。

## 可调参数

`input.md` 中的 `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens`、
`enable_prefix_caching` 和 `compilation_config` 是唯一可调部署字段，候选必须满足
`tp * dp <= max_tp_dp`。

`enable_prefix_caching=false` 传递 `--no-enable-prefix-caching`。compilation_config 仅允许
`NONE`、`FULL_DECODE_ONLY`、`PIECEWISE` 和 `FULL_AND_PIECEWISE`，并传递给
`--compilation-config`。`max_model_len=133000`、`trust_remote_code`、
`gpu_memory_utilization=0.90`、`additional_config={"enable_cpu_binding":true}` 和
`async_scheduling=true` 是固定参数，不进入候选。

## Baseline 部署参数

baseline origin 固定使用：`--data-parallel-size 1`、`--tensor-parallel-size 2`、
`--max-num-seqs 32`、`--max-model-len 133000`、`--max-num-batched-tokens 16384`、
`--trust-remote-code`、`--gpu-memory-utilization 0.90`、`--no-enable-prefix-caching`、
`--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}`、
`--additional-config {"enable_cpu_binding":true}` 和 `--async-scheduling`。

candidate 只能在上方 edit surface 中调整 TP、DP、max-num-seqs、max-num-batched-tokens、
compilation-config 和 prefix-caching；不得修改固定部署参数或 benchmark workload。

## 用户输入

`input.md` 是唯一输入来源。`num_prompts`、`max_concurrency` 和 `request_rate` 在 baseline、
candidate、promotion 中保持一致；其余 workload 默认值、数据集适配器、生成参数和百分位配置
由 `vllm-ascend-benchmark` 及其 references 指定。

## 单次测量

每个 baseline、candidate 和 promotion attempt 仅执行一次正式 benchmark。该次官方 raw result
直接作为该 attempt 的评分；不追加测量、不对多个样本取平均，失败直接按 `invalid` 或 `error`
结算。

## 并行模式

- Goal Plus 使用 `max_parallel=2`，每个 batch 同时包含两个 candidate。
- `max_attempts` 是整个 Search 的共享预算，不按 lane 复制；每个 active candidate 使用独立
  设备组、端口、锁和输出句柄。
- 每个 candidate 必须使用不重复的候选值；资源隔离和执行编排由 provider/bridge skill 负责。
