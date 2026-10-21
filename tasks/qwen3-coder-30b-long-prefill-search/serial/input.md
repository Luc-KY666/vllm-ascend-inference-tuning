# 串行任务输入

本文件是本任务唯一的用户输入来源。启动前替换下表中的占位符；目标文件和 workflow 只
引用这些字段，不复制其实际值。

| 输入 | 用户填写值 |
| --- | --- |
| `model_path` | `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct` |
| `max_attempts` | `10` |
| `authorized_ascend_npu_devices` | `0-7` |
| `max_tp_dp` | `2` |
| `tp` | `2` |
| `dp` | `1` |
| `max_num_seqs` | `16-256` |
| `max_num_batched_tokens` | `64-8192` |
| `num_prompts` | `500` |
| `max_concurrency` | `64` |
| `request_rate` | `4` |

约束：

- `model_path` 必须是目标环境可读路径；不得从宿主默认值推断。
- 本任务的 preset 负载类型固定为 `long_prefill`；数据集 ID 和适配器由 benchmark 按场景
  contract 派生并从默认 `~/.cache` 解析，用户不填写数据集 ID 或路径。
- `max_attempts` 和 `max_tp_dp` 必须是正整数。
- `authorized_ascend_npu_devices` 必须是无重复的非负十进制设备 ID 列表。
- `max_tp_dp` 不得大于授权设备数量。
- `tp`、`dp`、`max_num_seqs` 和 `max_num_batched_tokens` 是本任务的可调字段；用户必须在
  本表提供有限、无重复的候选值，Goal Plus 冻结后只能在这些值中选择。
- `num_prompts` 和 `max_concurrency` 必须是正整数；`request_rate` 必须显式填写，并按
  `vllm-ascend-benchmark` 的 workload contract 校验。
- 这三个 workload 输入在 baseline、candidate 和 promotion 中保持一致；其他 workload
  参数、默认值、benchmark mode、数据集适配器、生成参数和百分位配置由 benchmark skill
  及其 references 指定和记录，本文件不复制第二套 workload 合同。
- 每个候选必须满足 `tp * dp <= max_tp_dp`，其余模型架构、命令行和部署硬约束由 deployment
  Skill 复核；非法组合不得派发。
- baseline 不在本表填写固定 `tp`；baseline 的 `tp` 必须来自 deployment Skill 的最小 TP
  recommendation，并保留 recommendation artifact。
- 设备组、端口、锁、输出目录、verifier 临时目录和实际服务命令由 provider 注入。
- 缺少输入、保留占位符、路径不可读或约束不满足时只能报告 `blocked`/`incomplete`。
