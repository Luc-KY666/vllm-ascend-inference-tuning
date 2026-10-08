# 数据集配置

本节定义负载场景到数据集的映射、数据集默认值、路径准备和显式覆盖规则，并生成后续 `vllm bench serve` 所需的配置。数据集模式由输入契约中的 `scenario` 直接决定，不使用环境变量选择数据集。

## 1. 场景与数据集映射

| `scenario` | `DATASET_MODE` | 数据集 | 负载特征 |
|------------|----------------|--------|----------|
| `random` | `random` | vLLM random | 可控输入/输出长度，用于服务可用性和基线验证 |
| `long_prefill` | `hf` | `likaixin/InstructCoder` | 真实代码长输入、短输出，侧重 prefill 和 TTFT |
| `long_decode` | `hf` | `AI-MO/NuminaMath-CoT` | 数学题短输入、CoT 长输出，侧重 decode、TPOT 和 ITL |
| `prefill_decode_balance` | `sharegpt` | `ShareGPT_V3_unfiltered_cleaned_split.json` | 真实多轮对话分布，侧重综合服务能力 |

`DATASET_MODE`、数据集 ID、数据集路径和数据集专用参数均由 `scenario` 派生。输入场景不合法时，生成 `status=input_invalid` 的结果 JSON，不启动服务或 benchmark。

## 2. Random 场景

当 `scenario=random` 时使用以下默认值：

```text
DATASET_MODE=random
DATASET_NAME=random
DATASET_PATH=null
RANDOM_INPUT_LEN=256
RANDOM_OUTPUT_LEN=1024
RANDOM_RANGE_RATIO=0.3
NUM_PROMPTS=200
MAX_CONCURRENCY=64
NUM_WARMUPS=10
REQUEST_RATE=inf
BURSTINESS=1.0
TEMPERATURE=0
PERCENTILE_METRICS=ttft,tpot,e2el
METRIC_PERCENTILES=90,99
```

random 场景的输入长度、输出长度、`random_range_ratio`、样本数、并发数、请求速率和 warmup 仅在显式覆盖时改变。覆盖后必须检查：

- `RANDOM_INPUT_LEN` 和 `RANDOM_OUTPUT_LEN` 是正整数。
- `0 <= RANDOM_RANGE_RATIO < 1`。
- `floor(RANDOM_INPUT_LEN * (1 - RANDOM_RANGE_RATIO)) >= 1`。
- `NUM_PROMPTS`、`MAX_CONCURRENCY`、`NUM_WARMUPS` 满足当前 vLLM CLI 的参数约束。
- `REQUEST_RATE` 是 `inf` 或正数，`BURSTINESS` 是正数。

random 场景使用 dummy 权重的规则由[模型拉起与压测](model-launch-and-benchmark.md)执行，本节只记录数据集参数。

## 3. 长 Prefill 场景

当 `scenario=long_prefill` 时使用 `hf` 数据集 `likaixin/InstructCoder`：

```text
DATASET_MODE=hf
DATASET_NAME=hf
HF_DATASET=likaixin/InstructCoder
HF_DATASET_PATH=likaixin/InstructCoder
HF_HOME=
HF_SPLIT=train
HF_OUTPUT_LEN=128
NUM_PROMPTS=200
MAX_CONCURRENCY=64
NUM_WARMUPS=10
REQUEST_RATE=inf
BURSTINESS=1.0
TEMPERATURE=0
PERCENTILE_METRICS=ttft,tpot,e2el
METRIC_PERCENTILES=90,99
```

`likaixin/InstructCoder` 包含真实代码和编辑指令。输入由真实代码文件决定，通常明显长于普通对话提示；默认将输出上限设为 128 tokens，以突出 prefill 和 TTFT。`HF_OUTPUT_LEN` 可以显式覆盖为其他正整数。

`HF_DATASET` 是 Hugging Face 数据集 ID，必须通过 `--hf-name "${HF_DATASET}"` 传入，用于选择 vLLM 的数据集格式化器。`HF_DATASET_PATH` 是实际数据源，必须通过 `--dataset-path "${HF_DATASET_PATH}"` 传入；它可以是可读的本地数据集目录、文件或远程数据集 ID。使用本地数据集时，`HF_DATASET_PATH` 必须设置为本地路径。不得省略 `--hf-name`，也不得仅通过 `--dataset-path` 推断数据集格式。

## 4. 长 Decode 场景

当 `scenario=long_decode` 时使用 `hf` 数据集 `AI-MO/NuminaMath-CoT`：

```text
DATASET_MODE=hf
DATASET_NAME=hf
HF_DATASET=AI-MO/NuminaMath-CoT
HF_DATASET_PATH=AI-MO/NuminaMath-CoT
HF_HOME=
HF_SPLIT=train
HF_OUTPUT_LEN=
NUM_PROMPTS=200
MAX_CONCURRENCY=64
NUM_WARMUPS=10
REQUEST_RATE=inf
BURSTINESS=1.0
TEMPERATURE=0
PERCENTILE_METRICS=ttft,tpot,e2el
METRIC_PERCENTILES=90,99
```

`AI-MO/NuminaMath-CoT` 使用数学题作为输入、带 chain-of-thought 的解答作为真实输出。默认不设置 `--hf-output-len`，保留数据集的真实输出长度；如需限制最长输出，可显式提供正整数 `HF_OUTPUT_LEN`。实际输出长度仍受服务的 `max_model_len` 和当前数据集过滤规则约束。

`HF_DATASET` 是 Hugging Face 数据集 ID，必须通过 `--hf-name "${HF_DATASET}"` 传入，用于选择 AIMO/NuminaMath 数据集格式化器。`HF_DATASET_PATH` 是实际数据源，必须通过 `--dataset-path "${HF_DATASET_PATH}"` 传入；它可以是可读的本地数据集目录或远程数据集 ID。使用本地数据集时，`HF_DATASET_PATH` 必须设置为本地路径。不得省略 `--hf-name`，也不得仅通过 `--dataset-path` 推断数据集格式。

两个 HF 场景都通过 `huggingface_hub`/`datasets` 自动加载。首次运行需要网络或已有缓存：

```bash
export HF_ENDPOINT=https://hf-mirror.com  # Hugging Face 不可达时设置
export HF_HOME=<run_dir>/dataset/hf_cache
```

`HF_ENDPOINT` 只影响 `huggingface_hub`/`datasets`，不会改变场景到数据集的映射，也不会重写 `wget` 或 `curl` 的 URL。

## 5. Prefill Decode 均衡场景

当 `scenario=prefill_decode_balance` 时使用 `sharegpt` 数据集：

```text
DATASET_MODE=sharegpt
DATASET_NAME=sharegpt
SHAREGPT_DATASET_PATH=<本地 JSON 文件>
SHAREGPT_OUTPUT_LEN=
NUM_PROMPTS=200
MAX_CONCURRENCY=64
NUM_WARMUPS=10
REQUEST_RATE=inf
BURSTINESS=1.0
TEMPERATURE=0
PERCENTILE_METRICS=ttft,tpot,e2el
METRIC_PERCENTILES=90,99
```

ShareGPT 的输入和输出长度来自真实多轮对话分布。vLLM 默认过滤 `max_prompt_len=1024`、`max_total_len=2048`；通常不设置 `SHAREGPT_OUTPUT_LEN`，显式设置时追加 `--sharegpt-output-len`。`SHAREGPT_DATASET_PATH` 必须是目标机器上存在且可读的 JSON 文件。

准备数据集时可以使用官方源；网络不可达时显式切换到镜像源：

```bash
mkdir -p <run_dir>/dataset
wget -P <run_dir>/dataset \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# 或使用镜像源
wget -P <run_dir>/dataset \
  https://hf-mirror.com/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

下载完成后必须校验文件可读且为 JSON 数组；文件缺失、为空或解析失败时，不启动 benchmark：

```bash
python3 -c "import json, sys; p=sys.argv[1]; d=json.load(open(p, encoding=\"utf-8\")); assert isinstance(d, list) and d; print(len(d), \"entries\")" \
  <run_dir>/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
```

## 6. 本地数据集校验缓存

本地数据集目录或文件校验属于可复用的本机环境探测，应写入[模型拉起与压测](model-launch-and-benchmark.md)定义的环境画像文件。后续运行优先读取画像，只有路径未记录、路径指纹变化或显式刷新画像时才重新校验。

本地路径指纹至少包含：

```text
absolute_path
path_type=file|directory
size_bytes
mtime_ns
inode 或等价文件标识（可获取时）
```

校验规则：

- HF 场景的 `HF_DATASET_PATH` 若是本地路径，必须校验路径存在且可读，并记录 `hf_name`、`split`、路径指纹和校验时间。目录结构是否能被 `datasets` 精确加载可以记录为 `unknown`，不得为了画像生成而执行无界 streaming 样本探测。
- HF 场景的 `HF_DATASET_PATH` 若是远程数据集 ID，只在画像中记录 `source_type=remote_id` 和数据集 ID；不做本地路径校验，也不把网络可达性缓存为长期事实。
- ShareGPT 场景的 `SHAREGPT_DATASET_PATH` 必须是本地 JSON 文件；画像中记录 JSON 数组可解析性和条目数。路径指纹一致时，后续运行不重复解析整个 JSON 文件。
- random 场景没有本地数据集路径校验，但 random 参数约束仍需每次按本次输入或覆盖值检查。

如果画像中的本地路径校验结果为失败且路径指纹未变化，后续运行可以直接复用该失败结论并在启动 benchmark 前失败；路径指纹变化时必须重新校验并更新画像。环境画像写入失败不得影响已经完成的本次数据集校验结果，但必须在诊断日志中记录。

## 7. 输出给压测阶段的统一字段

数据集配置完成后，向压测阶段提供以下最终字段：

```text
SCENARIO
DATASET_MODE
DATASET_NAME
DATASET_PATH / HF_DATASET / HF_DATASET_PATH / SHAREGPT_DATASET_PATH
HF_HOME
HF_SPLIT
HF_OUTPUT_LEN
SHAREGPT_OUTPUT_LEN
NUM_PROMPTS
MAX_CONCURRENCY
NUM_WARMUPS
REQUEST_RATE
BURSTINESS
TEMPERATURE
PERCENTILE_METRICS=ttft,tpot,e2el
METRIC_PERCENTILES=90,99
```

random 场景额外提供：

```text
RANDOM_INPUT_LEN
RANDOM_OUTPUT_LEN
RANDOM_RANGE_RATIO
```

最终使用的值必须原样写入输出 JSON 的 `dataset` 字段。HF 场景必须同时记录 `dataset.hf_name` 和实际传给 `--dataset-path` 的 `dataset.path`。对于非 random 场景，输出 JSON 的 `dataset.scenario` 必须记录 `long_prefill`、`long_decode` 或 `prefill_decode_balance`；对于 random 场景，省略 `dataset.scenario` 字段。

数据条数使用以下优先级确定：

1. 原始 benchmark JSON 的 `completed`。
2. 没有 `completed` 时使用原始 JSON 的 `num_prompts`。
3. benchmark 未执行或无法解析时写 `null`。
