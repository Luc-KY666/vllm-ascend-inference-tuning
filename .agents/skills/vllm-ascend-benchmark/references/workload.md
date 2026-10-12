# 压测负载契约

负载契约定义负载选择、数据集适配模式、预置场景、数据集准备和压测模式。
模型、设备、端口、请求数量和宿主路径由任务合同冻结。

## 负载选择

输入合同必须声明 `selection`，取值如下：

| `selection` | `scenario` | 数据集配置 |
| --- | --- | --- |
| `preset` | `random`、`long_prefill`、`long_decode` 或 `prefill_decode_balance` | 由预置场景派生 |
| `hf` | `null` | 由合同提供 Hugging Face 数据集参数 |
| `custom` | `null` | 由合同提供本地 custom JSONL 参数 |

`preset` 没有隐式默认值。四个预置场景是可选的负载模板；`hf` 和 `custom` 是
用户定义数据源入口。`scenario` 只在 `selection=preset` 时存在。

`dataset_root` 的适用范围如下：

| 选择分支 | `dataset_root` |
| --- | --- |
| `preset` + `random` | `null` |
| `preset` + `long_prefill`/`long_decode`/`prefill_decode_balance` | 必填 |
| `hf` | 可选；提供时用于 Hugging Face 缓存 |
| `custom` | `null`；数据位置由 `dataset.path` 提供 |

## 预置场景

| `scenario` | 数据集模式 | 数据集 | 负载特征 |
| --- | --- | --- | --- |
| `random` | `random` | vLLM 合成输入/输出 | 可控长度、协议和基线验证 |
| `long_prefill` | `hf` | `likaixin/InstructCoder` | 长输入、短输出，关注 prefill 和 TTFT |
| `long_decode` | `hf` | `AI-MO/NuminaMath-CoT` | 短输入、长输出，关注 decode 和 TPOT |
| `prefill_decode_balance` | `sharegpt` | `ShareGPT_V3_unfiltered_cleaned_split.json` | 真实多轮对话分布和综合服务能力 |

预置场景的适配模式、数据集身份和专用参数由 `scenario` 派生。`dataset_root` 是
三个文件型/远程预置数据集的存储输入；`random` 不需要 `dataset_root`。

## 预置数据集路径

`dataset_root` 由任务 Main 或 provider 注入，不设置宿主路径默认值。以下路径均为
相对于 `dataset_root` 的规范路径：

| 场景 | 模式 | 数据集 | split/文件 | 规范相对路径 | 专用参数 |
| --- | --- | --- | --- | --- | --- |
| `long_prefill` | `hf` | `likaixin/InstructCoder` | `train` | `hf/likaixin__InstructCoder` | `hf_output_len=128` |
| `long_decode` | `hf` | `AI-MO/NuminaMath-CoT` | `train` | `hf/AI-MO__NuminaMath-CoT` | 不设置 `hf_output_len` |
| `prefill_decode_balance` | `sharegpt` | `ShareGPT_V3_unfiltered_cleaned_split.json` | JSON 文件 | `sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json` | 通常不设置 `sharegpt_output_len` |

Hugging Face 缓存根目录为：

```text
HF_HOME=<dataset_root>/hf-cache
```

缓存根目录和物化数据集路径是两个不同的目录。`HF_HOME` 保存 Hugging Face 工具缓存；
`resolved_path` 是传给 benchmark adapter 的可读数据集路径。

ShareGPT 官方来源为：

```text
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

provider 可以通过合同提供受信任的 Hugging Face endpoint 或镜像；镜像只改变下载来源，
不改变数据集身份、规范相对路径或版本校验。

## Hugging Face 数据集

`selection=hf` 时，合同必须提供以下字段：

```json
{
  "selection": "hf",
  "scenario": null,
  "dataset_root": "<optional-cache-root>",
  "dataset": {
    "mode": "hf",
    "name": "<hf-name>",
    "path": "<local-dataset-path-or-hf-dataset-id>",
    "split": "<split>",
    "subset": "<optional-subset>",
    "revision_or_hash": "<revision-or-hash>",
    "output_len": "<optional-positive-integer>",
    "text_column": "<optional-column>"
  }
}
```

字段语义：

- `name` 对应 vLLM HF adapter 的 `hf-name`，用于选择数据集格式化方式；
- `path` 对应 `dataset-path`，可以是可读本地目录/文件或远程 Hugging Face 数据集 ID；
- `split` 是必填的 dataset split；
- `subset`、`text_column` 和 `output_len` 只有当前 vLLM CLI 支持时才传递；
- `dataset_root` 存在时，Hugging Face 缓存必须位于该目录下；未提供本地缓存根目录时，
  使用任务/provider 已冻结的环境缓存配置；
- `revision_or_hash` 用于锁定远程 revision 或本地数据指纹，不能使用未解析的占位值
  作为测量证据。

HF adapter 的参数分支为：

```text
--dataset-name hf
--dataset-path <dataset.path>
--hf-name <dataset.name>
--hf-split <dataset.split>
[--hf-subset <dataset.subset>]
[--hf-output-len <dataset.output_len>]
[--hf-text-column <dataset.text_column>]
```

可选参数必须先通过当前环境的 `vllm bench serve --help=all` 确认。`hf` 数据集路径为
远程 ID 时执行下载或缓存解析；路径为本地数据时执行可读性、split、格式、样本数和
revision/hash 校验。

`hf` 模式要求当前 vLLM HF adapter 能够识别该数据集的记录格式；无法由该 adapter
稳定解析的数据集应转换为 custom JSONL 后使用 `selection=custom`。

## Custom 数据集

`selection=custom` 时，合同必须提供本地 JSONL 路径：

```json
{
  "selection": "custom",
  "scenario": null,
  "dataset_root": null,
  "dataset": {
    "mode": "custom",
    "path": "<local-custom-jsonl>",
    "revision_or_hash": "<file-hash>",
    "output_len": "<optional-non-negative-integer>",
    "skip_chat_template": "<true-or-false>"
  }
}
```

Custom JSONL 的每行必须是一个 JSON object，并至少包含字符串字段 `prompt`：

```jsonl
{"prompt": "第一条压测请求"}
{"prompt": "第二条压测请求"}
```

Custom adapter 的输入字段是文本 prompt；对话消息数组应在进入 benchmark 前转换为已
渲染文本，并按 `skip_chat_template` 选择是否跳过服务端 chat template。

`output_len` 用于固定所有请求的输出长度；未提供时命令不追加
`--custom-output-len`，实际行为由当前 vLLM CLI 的默认值决定，并记录在本次 workload
摘要中。`skip_chat_template=true` 表示 prompt 已经完成对话模板渲染，或服务入口为
completions；该选项只在当前 CLI 提供对应参数时启用。

Custom adapter 的参数分支为：

```text
--dataset-name custom
--dataset-path <dataset.path>
[--custom-output-len <dataset.output_len>]
[--custom-skip-chat-template | --skip-chat-template]
```

`--custom-skip-chat-template` 是 Python benchmark CLI 常见的参数名；
`--skip-chat-template` 是部分 Rust/新 CLI 的参数名。实际命令只能使用当前
`vllm bench serve --help=all` 暴露的一个参数名，不能同时传递两个别名。

Custom 模式不传递以下 HF/ShareGPT 专用参数：

```text
--hf-name
--hf-split
--hf-subset
--hf-output-len
--sharegpt-output-len
```

Custom 数据集不通过 Hugging Face 自动下载。路径不存在、不可读、不是 JSONL、存在
无法解析的行、缺少 `prompt` 或有效样本数不足时，workload 准备失败。

## 数据集解析流程

压测前按 `selection` 选择解析路径：

1. `preset`：根据 `scenario` 派生适配模式和数据集配置；需要时在 `dataset_root` 下
   复用或下载预置数据集。
2. `hf`：读取合同中的 `name`、`path`、`split`、subset、revision/hash 和可选参数；
   根据路径类型执行本地校验或 Hugging Face 下载/缓存解析。
3. `custom`：读取合同中的本地 JSONL，校验文件指纹、JSON 行、`prompt` 字段和有效样本数；
   不执行远程下载。
4. 为本次 attempt 生成解析后的 dataset artifact，记录 `mode`、名称、路径、split、
   revision/hash、样本数和解析状态。
5. benchmark 命令只使用解析后的 dataset artifact；数据集解析失败时不执行正式测量。

下载或物化过程先写临时位置，完成格式、样本数和版本校验后再交付规范路径，避免半成品
被后续 attempt 复用。

## 压测模式

`benchmark_mode` 必须是 `stress` 或 `production`，并由任务合同冻结完整参数。

| 模式 | 目的 | 到达方式 | 合同要求 |
| --- | --- | --- | --- |
| `stress` | 极限压力、突发饱和、容量拐点和高并发稳定性 | 任务允许的饱和或高到达率配置 | 请求数、并发上限、warmup、到达率、burstiness 和停止条件 |
| `production` | 生产级 open-loop 负载下的延迟、吞吐和 goodput | 冻结的有限到达率或生产流量模型 | 流量窗口、请求数、并发、warmup、SLO、失败处理和重复规则 |

## 必须冻结的字段

```json
{
  "selection": "preset|hf|custom",
  "scenario": "random|long_prefill|long_decode|prefill_decode_balance|null",
  "benchmark_mode": "stress|production",
  "dataset_root": "<path-or-null>",
  "dataset": {
    "mode": "random|hf|sharegpt|custom",
    "name": "<derived-or-contract-defined>",
    "path": "<resolved-path-or-id>",
    "split": "<split-or-null>",
    "subset": "<subset-or-null>",
    "revision_or_hash": "<resolved-revision-or-hash>",
    "output_len": "<integer-or-null>",
    "skip_chat_template": "<boolean-or-null>"
  },
  "requests": {
    "num_prompts": "<task-defined>",
    "max_concurrency": "<task-defined>",
    "num_warmups": "<task-defined>",
    "request_rate": "<task-defined>",
    "burstiness": "<task-defined>",
    "rounds": "<task-defined>"
  },
  "generation": {
    "temperature": "<task-defined>",
    "top_p": "<task-defined>",
    "stream": "<task-defined>",
    "include_usage": "<task-defined>"
  },
  "goodput": {
    "ttft": "<optional-task-defined-milliseconds>",
    "tpot": "<optional-task-defined-milliseconds>",
    "e2el": "<optional-task-defined-milliseconds>"
  },
  "statistics": {
    "percentiles": [50, 95]
  }
}
```

`selection`、`scenario`、`dataset.mode` 和各适配器专用字段必须满足上述分支关系。
合同不完整、适配器参数不兼容或数据集未解析完成时，测量状态为 `blocked` 或
`invalid`。
