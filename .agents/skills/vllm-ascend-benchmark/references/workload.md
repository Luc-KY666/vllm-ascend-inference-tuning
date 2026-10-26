# 压测负载契约

Provider 冻结一次 attempt 的负载选择、数据集、请求窗口和 benchmark 模式。模型服务、
设备、端口和输出目录由 Provider 的服务输入冻结；除下文明确列出的可选字段默认值外，
本契约不推导其他宿主默认值。

## 负载选择

输入必须声明 `selection`：

| selection | scenario | 数据源 |
| --- | --- | --- |
| `preset` | `random`、`long_prefill`、`long_decode` 或 `prefill_decode_balance` | 本契约派生 |
| `hf` | `null` | Provider 提供的 Hugging Face 数据集 |
| `custom` | `null` | Provider 提供的本地 JSONL |

`preset` 不允许隐式 scenario。`scenario` 只在 `selection=preset` 时存在；`hf` 和
`custom` 不使用 scenario。

预置场景固定为：

| scenario | mode | 数据集 | 典型用途 |
| --- | --- | --- | --- |
| `random` | `random` | vLLM 合成输入/输出 | 协议、长度和基线验证 |
| `long_prefill` | `hf` | `likaixin/InstructCoder`, `train` | 长输入、短输出、TTFT |
| `long_decode` | `hf` | `AI-MO/NuminaMath-CoT`, `train` | 短输入、长输出、TPOT |
| `prefill_decode_balance` | `sharegpt` | ShareGPT JSON | 真实对话分布和综合服务 |

## 数据集输入

本契约不接受外部 `dataset_root`。使用 Hugging Face 数据集的分支统一通过数据集 ID
解析默认缓存；执行环境从标准 Hugging Face cache（默认位于 `~/.cache`，例如
`~/.cache/huggingface`）读取已有数据，缺失时由 `datasets` 按数据集 ID 下载。不要要求
调用者额外提供 cache 根目录，也不要把 preset 或 hf 数据集转换为本地 JSON/JSONL 后使用
`--dataset-name custom` 加载。

- `preset+random`：不需要数据集 ID，`--dataset-path` 不传；
- 其他 `preset`：由场景合同派生数据集 ID，并生成对应的 `--dataset-path`（不得是本地
  cache 路径）；调用者只提供 `scenario`，不重复填写该 ID；
- `hf`：调用者只提供 Hugging Face 数据集 ID，`--dataset-path` 和 `--hf-name` 使用该 ID，
  不接受本地目录作为数据集输入；
- `custom`：调用者必须提供本地 JSON/JSONL 路径，禁止远程下载或用 Hugging Face ID 替代。

对于非 random 的 preset，数据集 ID 由场景合同解析并写入 `dataset.name`/`dataset.path`；
调用者不再填写 cache 根目录。对于 `hf`，调用者只填写一个 Hugging Face 数据集 ID，解析器
将同一 ID 用于 `dataset.name`、`dataset.path`、`--hf-name` 和 `--dataset-path`。

`selection=hf` 至少包含：

```json
{
  "mode": "hf",
  "name": "<hf dataset name>",
  "path": "<same Hugging Face dataset ID>",
  "split": "<split>",
  "subset": "<optional subset>",
  "revision_or_hash": "<resolved revision or hash>",
  "output_len": "<optional positive integer>",
  "text_column": "<optional column>"
}
```

`selection=custom` 至少包含：

```json
{
  "mode": "custom",
  "path": "<local JSONL>",
  "revision_or_hash": "<file hash>",
  "output_len": "<optional non-negative integer>",
  "skip_chat_template": "<boolean>"
}
```

custom 的 `dataset.path` 是本地文件路径；这是三种选择中唯一允许使用本地数据集路径的分支。

Custom JSONL 每行必须是 JSON object，并包含字符串字段 `prompt`。不执行远程下载；路径
不可读、抽样行不可解析或抽样记录缺少 prompt 时，workload 状态为 `invalid`。不得通过全量
计数来判断样本是否充足。

HF 数据集必须使用 Provider 冻结的 Hugging Face 数据集 ID。执行前必须完成 split、格式、
有限样本可读性和 revision/hash 校验；禁止为了得到全量 `num_rows` 而遍历数据集到 EOF。
无法由当前 vLLM HF adapter 稳定解析的数据集应转换为 custom JSONL。

## 有界数据集校验

压测前只做轻量可复现的输入校验：确认声明的 split 存在，读取首个不超过 8 条（绝不超过
32 条）的样本并验证记录格式、必需字段和 tokenizer 可读性，然后记录
`validation_sample_count`、抽样索引和错误。优先使用 dataset metadata 或 streaming iterator
读取抽样；不得调用会物化整个 split 的 `len(dataset)`、`count()` 或遍历到 EOF 的计数逻辑。
抽样不足以证明 Provider 要求的最小窗口时，只能在合同明确给出阈值的情况下提前读取到该
阈值，仍不得扫描剩余的 108,391 等全部样本。Custom JSONL 也采用同一有界抽样规则；完整
请求数量由官方 benchmark 在正式测量阶段负责。

## CLI 适配器参数

解析完成后，只能使用与当前分支对应的参数。以下是 benchmark 内部生成的 CLI 参数，不是
调用者需要逐项填写的输入；preset 调用者只提供 `scenario`，其数据集 ID 由场景合同派生：

```text
# preset/random
--dataset-name random
--random-input-len <input length>
--random-output-len <output length>
--random-range-ratio <range ratio>

# preset/long_prefill or long_decode
--dataset-name hf
--dataset-path <dataset id>
--hf-name <derived name>
--hf-split <derived split>
[--hf-output-len <output length>]

# preset/prefill_decode_balance
--dataset-name sharegpt
--dataset-path <dataset id>
[--sharegpt-output-len <output length>]

# selection=hf
--dataset-name hf
--dataset-path <dataset id>
--hf-name <dataset id>
--hf-split <dataset.split>
[--hf-subset <dataset.subset>]
[--hf-output-len <dataset.output_len>]
[--hf-text-column <dataset.text_column>]

# selection=custom
--dataset-name custom
--dataset-path <dataset.path>
[--custom-output-len <dataset.output_len>]
[--custom-skip-chat-template | --skip-chat-template]
```

可选参数必须先由当前 `vllm bench serve --help=all` 证实。两个 skip-chat-template 别名
只能选择一个。Custom 不得传递 HF 或 ShareGPT 专用参数。

对于 `likaixin/InstructCoder`，`--hf-name` 和 `--dataset-path` 都必须保留为
`likaixin/InstructCoder`，前者负责选择 InstructCoder adapter，后者作为 Hugging Face 数据集
ID 触发默认 cache 解析。不要用 cache 目录名替代数据集 ID，也不要为确认 adapter 分支读取
vLLM 源码。

## Benchmark 模式与冻结字段

`benchmark_mode` 必须为 `stress` 或 `production`：

- `stress` 用于突发、饱和、容量拐点和高并发稳定性；
- `production` 用于有限到达率、并发和样本窗口下的 open-loop 评估。

Provider 必须冻结以下字段：

```json
{
  "selection": "preset|hf|custom",
  "scenario": "random|long_prefill|long_decode|prefill_decode_balance|null",
  "benchmark_mode": "stress|production",
  "dataset": {
    "mode": "random|hf|sharegpt|custom",
    "name": "<derived or provider-defined>",
    "path": "<dataset ID, or local path only for custom>",
    "split": "<split or null>",
    "subset": "<subset or null>",
    "revision_or_hash": "<resolved value>",
    "output_len": "<integer or null>",
    "text_column": "<column or null>",
    "skip_chat_template": "<boolean or null>"
  },
  "requests": {
    "num_prompts": "<positive integer>",
    "max_concurrency": "<positive integer>",
    "num_warmups": "<non-negative integer>",
    "request_rate": "<contract value>",
    "burstiness": "<contract value>",
    "rounds": "<positive integer>"
  },
  "generation": {
    "seed": "<non-negative integer>",
    "temperature": "<显式值 | null(不传参) | 未提供时默认 0>",
    "top_p": "<contract value>",
    "stream": "<contract value>",
    "include_usage": "<contract value>"
  },
  "goodput": {
    "ttft": "<optional milliseconds>",
    "tpot": "<optional milliseconds>",
    "e2el": "<optional milliseconds>"
  },
  "statistics": {"percentiles": [50, 95]}
}
```

最终报告将 `requests` 中的请求数、warmup、最大并发、到达率、burstiness 和轮次镜像到
`arrival`，以便不读取 raw result 也能复现本次压测的请求到达合同。缺失或矛盾的
到达字段仍属于 `invalid`，不能用宿主默认值补齐。

goodput 约束只记录 Provider 实际提供的字段。合同不完整、数据集未解析完成、模式关系
错误或 CLI 不支持所需适配器参数时，不进入正式 benchmark，并在最终报告中使用
`invalid` 或 `blocked`。`temperature=null` 表示命令不传 `--temperature`，由服务端决定默认值；
不得把它解释成 `0`。

Provider 未显式冻结以下可选字段时使用默认值，并在内存中的 workload context 记录；实际传给
CLI 的 seed、temperature 和来源，以及其他 generation 配置写入最终
`benchmark_report.json.parameters`：

- `seed=42`：显式传入 `--seed 42`，保证请求生成可复现；
- `temperature=0`：显式传入 `--temperature 0`，贪心且可重复；
- `num_warmups=10`：warmup 请求不计入正式请求统计；
- `burstiness=1.0`；
- `output_len=128`：仅在 `selection=preset` 且 `scenario=long_prefill` 时作为
  `--hf-output-len` 的默认值；其余 selection/scenario 不设置默认 `output_len`（保持
  `null`），由 Provider 显式提供或遵循 CLI 与数据集默认。

`temperature` 默认值仅覆盖“Provider 未提供该字段”的情况；Provider 显式冻结为 `null`
时仍表示不传 `--temperature`，由服务端决定。
