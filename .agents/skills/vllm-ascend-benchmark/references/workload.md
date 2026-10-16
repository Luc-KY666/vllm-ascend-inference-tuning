# 压测负载契约

Provider 冻结一次 attempt 的负载选择、数据集、请求窗口和 benchmark 模式。模型服务、
设备、端口和输出目录由 Provider 的服务输入冻结；本契约不推导宿主默认值。

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

`dataset_root` 的使用规则如下：

- `preset+random`：必须为 `null`；
- 其他 preset：必须提供，用于预置数据集和 Hugging Face cache；
- `hf`：可选，提供时所有 cache 必须位于该目录下；
- `custom`：必须为 `null`，数据位置由 `dataset.path` 提供。

`selection=hf` 至少包含：

```json
{
  "mode": "hf",
  "name": "<hf dataset name>",
  "path": "<local path or dataset id>",
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

Custom JSONL 每行必须是 JSON object，并包含字符串字段 `prompt`。不执行远程下载；路径
不可读、抽样行不可解析或抽样记录缺少 prompt 时，workload 状态为 `invalid`。不得通过全量
计数来判断样本是否充足。

HF 数据集可使用本地路径或 Provider 信任的远程数据集 ID。执行前必须完成 split、格式、
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

解析完成后，只能使用与当前分支对应的参数：

```text
# preset/random
--dataset-name random
--random-input-len <input length>
--random-output-len <output length>
--random-range-ratio <range ratio>

# preset/long_prefill or long_decode
--dataset-name hf
--dataset-path <resolved path>
--hf-name <derived name>
--hf-split <derived split>
[--hf-output-len <output length>]

# preset/prefill_decode_balance
--dataset-name sharegpt
--dataset-path <resolved path>
[--sharegpt-output-len <output length>]

# selection=hf
--dataset-name hf
--dataset-path <dataset.path>
--hf-name <dataset.name>
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

对于 `likaixin/InstructCoder`，`--hf-name` 必须保留为 `likaixin/InstructCoder`，因为它负责
选择 InstructCoder adapter；`--dataset-path` 可以同时指向本地 cache。不要用 cache 目录名
替代 `--hf-name`，也不要为确认 adapter 分支读取 vLLM 源码。

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
  "dataset_root": "<path or null>",
  "dataset": {
    "mode": "random|hf|sharegpt|custom",
    "name": "<derived or provider-defined>",
    "path": "<resolved path or id>",
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
    "temperature": "<contract value; explicit or null>",
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

goodput 约束只记录 Provider 实际提供的字段。合同不完整、数据集未解析完成、模式关系
错误或 CLI 不支持所需适配器参数时，不进入正式 benchmark，并在最终报告中使用
`invalid` 或 `blocked`。`temperature=null` 表示命令不传 `--temperature`，由服务端决定默认值；
不得把它解释成 `0`。
