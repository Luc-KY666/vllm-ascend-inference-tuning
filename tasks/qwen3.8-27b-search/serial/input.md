# 串行任务输入

本文件是本任务唯一的用户输入来源。启动前替换下表中的占位符；目标文件和 workflow 只
引用这些字段，不复制其实际值。

| 输入 | 用户填写值 |
| --- | --- |
| `model_path` | `/data/disk/models/Qwen3.8-27B` |
| `max_attempts` | `20` |
| `authorized_ascend_npu_devices` | `["0", "1", "2", "3", "4", "5", "6", "7"]` |
| `max_tp_dp` | `2` |
| `tp` | `1, 2` |
| `dp` | `1, 2` |
| `max_num_seqs` | `16-256` |
| `max_num_batched_tokens` | `2048-32768` |
| `enable_prefix_caching` | `true|false` |
| `compilation_config` | `{"cudagraph_mode":"NONE"}`; `{"cudagraph_mode":"FULL_DECODE_ONLY"}`; `{"cudagraph_mode":"PIECEWISE"}`; `{"cudagraph_mode":"FULL_AND_PIECEWISE"}` |
| `num_prompts` | `500` |
| `max_concurrency` | `64` |
| `request_rate` | `4` |

固定 benchmark 负载合同（baseline、candidate 和 promotion 必须一致）：

| 参数 | 固定值 |
| --- | --- |
| `selection` | `preset` |
| `scenario` | `prefill_decode_balance` |
| `benchmark_mode` | `production` |
| `goodput.ttft` | `500` ms |
| `goodput.tpot` | `25` ms |
| `statistics.percentiles` | `[50, 95]` |

固定部署参数（不可调，baseline、candidate 和 promotion 必须一致）：

| 参数 | 固定值 |
| --- | --- |
| `trust_remote_code` | 启用，传递 `--trust-remote-code` |
| `additional_config` | `{"enable_cpu_binding": true}` |
| `gpu_memory_utilization` | `0.90` |
| `max_model_len` | `133000` |
| `async_scheduling` | 启用，传递 `--async-scheduling` |

Baseline origin 部署参数（baseline 固定；candidate 仍可在可调字段中搜索）：

| CLI 参数 | baseline 值 |
| --- | --- |
| `--data-parallel-size` | `1` |
| `--tensor-parallel-size` | `2` |
| `--max-num-seqs` | `32` |
| `--max-model-len` | `133000` |
| `--max-num-batched-tokens` | `16384` |
| `--no-enable-prefix-caching` | 启用 |
| `--compilation-config` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` |
| `--additional-config` | `{"enable_cpu_binding":true}` |
| `--gpu-memory-utilization` | `0.90` |
| `--trust-remote-code` | 启用 |
| `--async-scheduling` | 启用 |

约束：

- `model_path` 必须是目标环境可读路径；不得从宿主默认值推断。
- 本任务的 preset 负载类型固定为 `prefill_decode_balance`；数据集为 ShareGPT，数据集 ID 和适配器由 benchmark 按场景
  contract 派生并从默认 `~/.cache` 解析，用户不填写数据集 ID 或路径。
- `max_attempts` 和 `max_tp_dp` 必须是正整数。
- `authorized_ascend_npu_devices` 必须是按物理设备顺序声明、无重复的非负十进制设备 ID 列表；
  列表顺序是本任务的稳定卡组顺序，不得在搜索过程中重排。
- `max_tp_dp` 不得大于授权设备数量。
- `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens`、`enable_prefix_caching` 和
  `compilation_config` 是本任务的可调字段；
  用户必须在本表提供有限、无重复的候选值，Goal Plus 冻结后只能在这些值中选择。
- `enable_prefix_caching` 必须使用布尔候选值，为 `false` 时传递
  `--no-enable-prefix-caching`；该模型不启用 expert parallel。
- `compilation_config` 候选必须是 JSON object，且 `cudagraph_mode` 只能取
  `NONE`、`FULL_DECODE_ONLY`、`PIECEWISE` 或 `FULL_AND_PIECEWISE`；按
  `--compilation-config <json>` 传递。
- `max_model_len=133000`、`trust_remote_code`、`gpu_memory_utilization=0.90`、
  `additional_config={"enable_cpu_binding":true}` 和 `async_scheduling=true` 是固定部署参数，
  分别按 `--max-model-len 133000`、`--trust-remote-code`、`--gpu-memory-utilization 0.90`、
  `--additional-config '{"enable_cpu_binding":true}'` 和 `--async-scheduling` 传递；这些字段不进入 Search 候选空间。
- `num_prompts` 和 `max_concurrency` 必须是正整数；`request_rate` 必须显式填写，并按
  `vllm-ascend-benchmark` 的 workload contract 校验。
- `goodput.ttft=500` ms 和 `goodput.tpot=25` ms 是严格 SLO 上限；聚合的 `TTFT_P95` 或
  `TPOT_P95` 达到/超过对应上限的候选不能作为最终配置。
- 这三个 workload 输入在 baseline、candidate 和 promotion 中保持一致；其他 workload
  参数、默认值、benchmark mode、数据集适配器、生成参数和百分位配置由 benchmark skill
  及其 references 指定和记录，本文件不复制第二套 workload 合同。
- 每个候选必须满足 `tp * dp <= max_tp_dp`，其余模型架构、命令行和部署硬约束由 deployment
  Skill 复核；非法组合不得派发。
- baseline 固定使用 `dp=1`、`tp=2`，分别按 `--data-parallel-size 1` 和 `--tensor-parallel-size 2` 传递；`max_tp_dp` 和授权设备数必须
  支持该 baseline，baseline 不使用 deployment Skill 的最小 TP recommendation。
- 设备组按 `tp * dp` 从 `authorized_ascend_npu_devices` 的前缀确定：`k = tp * dp` 的所有
  配置统一使用前 `k` 张卡，并保持 `ASCEND_RT_VISIBLE_DEVICES` 的顺序不变。例如当前授权
  列表下，所有 `tp * dp = 2` 的配置都必须使用 `0,1`，所有 `tp * dp = 4` 的配置都必须
  使用 `0,1,2,3`；`tp=1, dp=2` 与 `tp=2, dp=1` 也必须共享 `0,1`。baseline、candidate
  和 promotion 必须复用同一映射。provider 负责注入这些固定设备组、端口、锁、输出目录、
  verifier 临时目录和实际服务命令；不得按实时 free/busy 或调度顺序替换同数量的其他卡。
  固定设备组不可用时只能报告资源阻塞/失败，不能静默换卡。
- 缺少输入、保留占位符、路径不可读或约束不满足时只能报告 `blocked`/`incomplete`。
