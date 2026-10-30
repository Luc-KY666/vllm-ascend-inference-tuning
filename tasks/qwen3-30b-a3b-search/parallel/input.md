# 并行任务输入

本文件是本任务唯一的用户输入来源。启动前替换下表中的占位符；目标文件和 workflow 只
引用这些字段，不复制其实际值。

| 输入 | 用户填写值 |
| --- | --- |
| `model_path` | `<模型路径>` |
| `max_attempts` | `<最大尝试次数>` |
| `authorized_ascend_npu_devices` | `["0", "1", "2", "3", "4", "5", "6", "7"]` |
| `max_tp_dp` | `<TP * DP 最大值>` |
| `tp` | `<候选值，例如 2>` |
| `dp` | `<候选值，例如 1>` |
| `max_num_seqs` | `<候选值，例如 16-256>` |
| `max_num_batched_tokens` | `<候选值，例如 64-8192>` |
| `enable_expert_parallel` | `true|false` |
| `enable_prefix_caching` | `true|false` |
| `distributed_executor_backend` | `mp|null` |
| `async_scheduling` | `true` |
| `compilation_config` | `{"cudagraph_mode":"NONE"}`; `{"cudagraph_mode":"FULL_DECODE_ONLY"}`; `{"cudagraph_mode":"PIECEWISE"}`; `{"cudagraph_mode":"FULL_AND_PIECEWISE"}` |
| `num_prompts` | `<请求数>` |
| `max_concurrency` | `<最大并发数>` |
| `request_rate` | `<请求到达速率>` |

固定部署参数（不可调，baseline、candidate 和 promotion 必须一致）：

| 参数 | 固定值 |
| --- | --- |
| `trust_remote_code` | 启用，传递 `--trust-remote-code` |
| `additional_config` | `{"enable_flashcomm1": true, "weight_nz_mode": 2}` |
| `gpu_memory_utilization` | `0.95` |
| `max_model_len` | `40960` |

Baseline origin 部署参数（baseline 固定；candidate 仍可在可调字段中搜索）：

| CLI 参数 | baseline 值 |
| --- | --- |
| `--max-num-seqs` | `100` |
| `--max-num-batched-tokens` | `16384` |
| `--tensor-parallel-size` | `4` |
| `--enable-expert-parallel` | 启用 |
| `--distributed_executor_backend` | `mp` |
| `--no-enable-prefix-caching` | 启用 |
| `--async-scheduling` | 启用 |
| `--compilation-config` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` |

约束：

- `model_path` 必须是目标环境可读路径；不得从宿主默认值推断。
- 本任务的 preset 负载类型固定为 `long_prefill`；数据集 ID 和适配器由 benchmark 按场景
  contract 派生并从默认 `~/.cache` 解析，用户不填写数据集 ID 或路径。
- `max_attempts` 和 `max_tp_dp` 必须是正整数。
- `authorized_ascend_npu_devices` 必须是无重复的非负十进制设备 ID 列表。
- `max_tp_dp` 不得大于授权设备数量。
- `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens`、`enable_expert_parallel`、
  `enable_prefix_caching`、`distributed_executor_backend`、`async_scheduling` 和
  `compilation_config` 是本任务的可调字段；
  用户必须在本表提供有限、无重复的候选值，Goal Plus 冻结后只能在这些值中选择。
- `enable_expert_parallel` 和 `enable_prefix_caching` 必须使用布尔候选值；前者为 `true` 时
  传递 `--enable-expert-parallel`，后者为 `false` 时传递 `--no-enable-prefix-caching`。
- `distributed_executor_backend` 只能是 `mp` 或 `null`；前者按
  `--distributed_executor_backend mp` 传递，后者不传该参数。`async_scheduling` 必须使用
  布尔候选值，为 `true` 时传递 `--async-scheduling`，为 `false` 时省略该 flag。
- `compilation_config` 候选必须是 JSON object，且 `cudagraph_mode` 只能取
  `NONE`、`FULL_DECODE_ONLY`、`PIECEWISE` 或 `FULL_AND_PIECEWISE`；按
  `--compilation-config <json>` 传递。
- `trust_remote_code` 固定启用并按 `--trust-remote-code` 传递；`additional_config` 固定为 `{"enable_flashcomm1": true, "weight_nz_mode": 2}`，`gpu_memory_utilization` 固定为 `0.95`，`max_model_len` 固定为 `40960` 并按 `--max-model-len 40960` 传递；这些字段不进入 Search 候选空间。
- `num_prompts` 和 `max_concurrency` 必须是正整数；`request_rate` 必须显式填写，并按
  `vllm-ascend-benchmark` 的 workload contract 校验。
- 这三个 workload 输入在 baseline、candidate 和 promotion 中保持一致；其他 workload
  参数、默认值、benchmark mode、数据集适配器、生成参数和百分位配置由 benchmark skill
  及其 references 指定和记录，本文件不复制第二套 workload 合同。
- 每个候选必须满足 `tp * dp <= max_tp_dp`，其余模型架构、命令行和部署硬约束由 deployment
  Skill 复核；非法组合不得派发。
- baseline 固定使用 `tp=4`，按 `--tensor-parallel-size 4` 传递；`max_tp_dp` 和授权设备数必须
  支持该 baseline，baseline 不使用 deployment Skill 的最小 TP recommendation。
- 每个 active candidate 必须取得属于授权池的独立设备组、端口、锁、输出目录和 verifier 临时目录。
- `max_attempts` 是整个并行 Search 的共享候选预算，不按 lane 复制。
- 缺少输入、保留占位符、路径不可读、资源组不足或约束不满足时只能报告 `blocked`/`incomplete`。
