# 部署输入契约

ordinary deployment 的 Provider 输入，以及 PD 每个实例的节点输入，都由 agent 根据最终的
`vllm serve` 启动命令组装为 JSON，再由 `scripts/input_contract.py` 一次规范化。脚本返回
stdout JSON；指定 `--runtime-output` 时额外写出供 static OOM 使用的临时 runtime artifact，
不写部署报告。规范化结果保存在内存中，供静态 OOM、TP 决策和服务启动复用。

## 调用方式

probe 阶段必须先生成当前执行上下文的 `vllm-serve-help-all.txt`。输入校验使用同一份
artifact，不重新执行 `vllm serve --help`：

```text
python <skill-root>/scripts/input_contract.py \
  --input /absolute/attempt-dir/deployment-input.json \
  --serve-help-file /absolute/attempt-dir/vllm-serve-help-all.txt \
  --runtime-output /absolute/attempt-dir/.work/runtime.json
```

`--input -` 可从 stdin 读取 JSON。TP 推荐回写使用已经规范化的 JSON，从 stdin 读取，不
需要再次传 help 文件：

```text
python <skill-root>/scripts/input_contract.py \
  --resolve-tp <recommended-tp> \
  --resolution-source static_oom \
  --input - < normalized-input.json
```

旧的 `--model-path`、`--tp`、`--dp`、`--extra-arg` 等逐项 CLI 不再是 ordinary 合同。

## JSON 输入

`serve.argv` 是 `vllm serve` 之后的完整 argv，不包含 `vllm` 和 `serve`。第一项必须是
模型路径 positional 参数；后续每一项必须是 help 文件中声明的 option 或其 value。

```json
{
  "serve": {
    "argv": [
      "/absolute/model/path",
      "--served-model-name", "qwen3-30b-a3b",
      "--tensor-parallel-size", "2",
      "--data-parallel-size", "1",
      "--max-num-seqs", "32",
      "--max-num-batched-tokens", "4096",
      "--gpu-memory-utilization", "0.95",
      "--enable-expert-parallel",
      "--distributed-executor-backend", "mp",
      "--no-enable-prefix-caching",
      "--async-scheduling",
      "--compilation-config", "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}",
      "--additional-config", "{\"enable_flashcomm1\":true,\"weight_nz_mode\":2}"
    ]
  },
  "model": {
    "config_path": "/absolute/model/path/config.json"
  },
  "runtime": {
    "min_required_context_len": 4096,
    "official_min_tp": 1
  },
  "resources": {
    "visible_devices": ["0", "1"],
    "role_device_count": 2,
    "card_memory_bytes": 68719476736,
    "port_candidates": [8117, 8118],
    "preferred_port": 8117
  },
  "service": {},
  "outputs": {
    "attempt_dir": "/absolute/attempt-dir"
  }
}
```

`resources.card_memory_bytes` 是 static OOM 所需的单卡显存事实，agent 必须在组装每份
`deployment_input.json` 时填写为正整数。该值必须来自 Provider 冻结的硬件事实或当前环境
probe，不能由 agent 根据模型大小、设备数量或显存比例推算，也不能省略后交给 static OOM
补默认值。PD 模式下每个 Prefill/Decode 节点都填写自己设备池对应的卡型显存值；
`role_device_count` 只表示角色级 DP 总设备数，不替代 `card_memory_bytes`。

模型路径、served model name、TP、DP、mns、mnbt、显存比例、dtype、quantization、block
size、max model length、tokenizer 和 port 等启动值必须来自 `serve.argv`。`runtime` 只
携带没有直接对应 vLLM 启动 flag 的校验元数据，例如 `allowed_tp`、`official_min_tp` 和
`min_required_context_len`；不得在 `runtime` 中重复提供由 `serve.argv` 派生的字段。

`--config` 和 `--model` 第一版禁止使用，因为它们会把实际启动参数隐藏在第二个文件或
第二个模型来源中，input contract 无法保证命令与校验事实一致。

## Help 参数检查

脚本从 probe artifact 中解析 option 声明和短别名。每个 `serve.argv` option 必须存在于
该文件，例如 `--tensor-parallel-size`、`-tp`、`--enable-expert-parallel` 和
`--no-enable-prefix-caching`。普通 option 的 value 不在 input-contract 的校验范围内；
只要 option 名称存在，就不会根据其 value 做类型、枚举、JSON 或业务语义判断。模型路径仍
必须是第一项 positional token，隐藏参数来源 `--config` 和 `--model` 仍会被拒绝。

长 option 的下划线拼写会在校验时规范化为连字符，例如
`--distributed_executor_backend` 等价于 help 中的 `--distributed-executor-backend`；输出的
`serve.argv` 保留 agent 提供的原始拼写。

支持 `--flag=value` 形式。对于硬约束参数，脚本只提取其相邻 value 并执行原有约束；其它
option 的 value 原样留在 `serve.argv`，具体语义仍由服务启动阶段负责。

help 文件必须可读、包含 `usage: vllm serve` 和至少一个 option 声明。规范化结果只记录
其 SHA256，不复制 help 正文。

## 硬约束和规范化字段

以下字段从真实 vLLM flag 提取到 `runtime`：

| vLLM flag | normalized 字段 |
| --- | --- |
| `--tensor-parallel-size`, `-tp` | `runtime.tp` |
| `--data-parallel-size`, `-dp` | `runtime.dp` |
| `--max-num-seqs` | `runtime.max_num_seqs` |
| `--max-num-batched-tokens` | `runtime.max_num_batched_tokens` |
| `--gpu-memory-utilization` | `runtime.gpu_memory_utilization` |
| `--max-model-len` | `runtime.max_model_len` |
| `--block-size` | `runtime.block_size` |
| `--dtype` | `runtime.dtype` |
| `--quantization`, `-q` | `runtime.quantization` |

保留原有硬约束：

- `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens` 和 `block_size` 为正整数，且
  `0 < gpu_memory_utilization <= 1`；
- `max_num_batched_tokens >= max_num_seqs`；
- `tp` 不超过当前实例的 `resources.visible_devices` 数量；
- `tp * dp` 不超过 `resources.role_device_count`（未提供时回退到可见设备数量）或
  `resources.max_tp_dp`；
- `num_attention_heads` 能被 TP 整除；
- `allowed_tp`、`official_min_tp`、设备 ID 和端口候选的格式与归属；
- `preferred_port`、实际 port 必须属于 `port_candidates`；
- `outputs.attempt_dir` 必须是绝对路径。

`max_model_len` 允许 vLLM 支持的 `auto`、`-1` 或人类可读后缀，`dtype` 和
`quantization` 也不在 input-contract 中做类型或 choices 判断；这些值只要对应 option
存在就原样保留。

缺少 TP 时保留 `runtime.tp=null` 并返回 `input_status=missing_parameters`，交给静态 OOM
阶段推荐最小 TP。缺少 DP、mns 或 mnbt 时继续沿用原有默认值和 `missing_parameters` 状态：

```text
dp=1, max_num_seqs=32, max_num_batched_tokens=4096,
gpu_memory_utilization=0.9, min_required_context_len=4096, block_size=128
```

## 模型 config 校验

模型目录必须可读；`config.json` 必须存在、可读、能解析为 JSON object。原始字节只读取
一次，同时计算 SHA256 和配置摘要。摘要保留 static OOM 所需的架构字段，包括
`hidden_size`、`num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、
`head_dim`、`vocab_size`、`intermediate_size`、`num_experts`、`torch_dtype`、
`tie_word_embeddings` 和 `has_quantization_config`；存在 `text_config` 时使用其覆盖值。

## 规范化输出

输出 schema 升级为 `vllm-ascend-deployment-input-v3`，继续包含 `model`、`runtime`、
`service`、`resources`、`outputs`、`resolution` 和 `diagnostics`，并新增：

```json
{
  "serve": {
    "argv": [],
    "help_digest": "sha256..."
  }
}
```

`runtime` 是 static OOM 双输入中的规范化部分；原始 argv 和 help 正文不需要再次写入临时
阶段文件。PD 的每个实例使用本节同一 schema：`visible_devices` 是本地设备，
`role_device_count` 是该 Prefill/Decode DP 角色的总设备数。

使用 `--runtime-output` 时，脚本额外写出临时 `vllm-ascend-runtime-v1` artifact：

```json
{
  "schema_version": "vllm-ascend-runtime-v1",
  "model": {
    "config": {},
    "config_digest": "sha256..."
  },
  "runtime": {},
  "resources": {}
}
```

该 artifact 只包含已校验的 runtime、config 摘要/digest 和资源；若原始 model 输入提供
`weight_bytes` 或 `weight_gib`，也会保留给 static OOM 使用。输入无效时不得写出 artifact。
static OOM 必须把它与原始 `deployment_input.json` 一起传入，不得由 agent 重新拼装等价
payload。

## TP 解析

静态 OOM 返回推荐值后，调用方只对已缓存对象执行轻量回写：

```python
updated = resolve_recommended_tp(normalized, recommended_tp, "static_oom")
```

该操作只检查缓存的 `allowed_tp`、本地设备数量、角色设备数、DP、`max_tp_dp` 和
attention heads，不重读 config、不重读 help、不再次执行完整输入校验。它将 `runtime.tp`
更新为推荐值，并在 `serve.argv` 中追加或更新 `--tensor-parallel-size <tp>`，供后续服务
启动直接复用。

## PD 模式

PD 不再调用专用输入脚本。agent 将每个 Prefill/Decode 实例展开为独立的
`deployment_input_<role><rank>.json`，并分别调用本脚本；P/D 设备池、端口唯一性、connector
和跨实例拓扑规则由 agent 按 [`pd-deployment.md`](pd-deployment.md) 处理。
