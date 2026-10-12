# 静态 OOM 与最小 TP 预检

静态预检是一次部署 attempt 的容量诊断，不是设备调度器、模型文件检查器或运行时
verifier。部署流程先从唯一的 `deployment-input.json` 派生
`static-oom-input.json`，再调用本文件规定的脚本。

## 标准入口

唯一标准调用方式是：

```text
python <skill-root>/scripts/static_oom_precheck.py \
  --input <attempt-dir>/static-oom-input.json \
  --output <attempt-dir>/static-oom-report.json
```

脚本只消费已经规范化的 JSON，不读取模型目录、不扫描 tokenizer 或权重文件、不解析
端口和 CLI、不枚举或选择设备，也不启动服务。脚本输出的
`status`、`risk_level`、`recommendation` 和 `diagnostics` 必须原样保留在
`static-oom-report.json`，再由部署报告引用。

`static-oom-input.json` 的字段类型、数值范围、字段关系和来源由部署输入契约及其上游
派生流程负责确认。脚本不重复实现这些输入校验；它只处理字段缺失、无法从官方配置或
显式事实完成公式计算的情况，并返回 `unknown`。因此，违反输入契约的 JSON 不属于
脚本的公共输入。

## 输入

`static-oom-input.json` 的结构如下；尖括号是由部署流程填入的事实，不是脚本默认值：

```json
{
  "schema_version": "vllm-ascend-static-oom-input-v1",
  "model": {
    "config": {
      "hidden_size": "<正整数>",
      "num_hidden_layers": "<正整数>",
      "num_attention_heads": "<正整数>",
      "num_key_value_heads": "<正整数，可省略>",
      "head_dim": "<正整数，可省略>",
      "vocab_size": "<配置推导权重时必填>",
      "intermediate_size": "<配置推导权重时必填>",
      "tie_word_embeddings": "<布尔值，可省略>",
      "torch_dtype": "<可解析 dtype>",
      "text_config": {}
    },
    "weight_bytes": "<明确的正整数，可选>",
    "weight_gib": "<明确的正数，可选>"
  },
  "hardware": {
    "card_memory_bytes": "<正整数>",
    "device_count": "<正整数>"
  },
  "runtime": {
    "tp": "<正整数或 null>",
    "dp": "<正整数>",
    "max_num_seqs": "<正整数>",
    "gpu_memory_utilization": "<(0, 1] 内的有限数>",
    "min_required_context_len": "<正整数>",
    "block_size": "<正整数>",
    "dtype": "<可选的 dtype 覆盖>",
    "quantization": null,
    "allowed_tp": ["<正整数>"],
    "official_min_tp": "<正整数>"
  }
}
```

模型配置支持官方配置中标准的 `text_config`：读取时以其中同名字段覆盖外层配置。
`num_key_value_heads` 缺省时按 attention heads 处理；`head_dim` 缺省时只有在
`hidden_size` 能被 attention heads 整除时才推导。其他架构字段不能靠猜测补齐。

权重大小只能提供 `weight_bytes` 或 `weight_gib` 其中一个。显式量化配置存在而没有
明确权重大小时，结果为 `unknown`；显式权重大小可以继续执行容量检查，但不会假设
量化比例。静态脚本不从文件系统推断权重大小。

## 计算口径

对于能够从官方配置可靠识别的 dense decoder，配置推导的权重需求使用：

```text
attention =
  hidden_size * num_attention_heads * head_dim
  + 2 * hidden_size * num_key_value_heads * head_dim
  + num_attention_heads * head_dim * hidden_size

mlp = 3 * hidden_size * intermediate_size
norm = 2 * hidden_size
embedding = vocab_size * hidden_size
lm_head = 0 if tie_word_embeddings else embedding

total_params = embedding + lm_head
             + num_hidden_layers * (attention + mlp + norm)
weight_bytes = total_params * dtype_bytes
```

缺少架构、dtype、词表或中间层字段，或配置包含无法可靠解释的复杂量化/MoE 信息时，
不使用估算值，返回 `unknown`。显式 `weight_bytes` 或 `weight_gib` 时，只需保留
attention 和 KV 所需的架构字段。

对每个候选 TP，容量预算为：

```text
memory_budget = floor(card_memory_bytes * gpu_memory_utilization)
weight_per_card = ceil(weight_bytes / tp)
kv_heads_per_rank = max(1, ceil(num_key_value_heads / tp))
kv_bytes_per_token =
  2 * num_hidden_layers * kv_heads_per_rank * head_dim * kv_dtype_bytes
context_tokens =
  ceil(min_required_context_len / block_size) * block_size
minimum_kv_bytes =
  max_num_seqs * context_tokens * kv_bytes_per_token
```

`weight_per_card >= memory_budget` 或
`weight_per_card + minimum_kv_bytes > memory_budget` 时，该 TP 为 `reject`。
可可靠计算且未超预算时为 `pass`。`dp` 只检查 `tp * dp <= device_count`，不把 DP
倍乘到单卡权重或 KV Cache。

当前 TP 和推荐候选都必须满足 `tp` 不大于且能整除 attention heads。不能证明结构兼容
时返回 `unknown`，不自动调整 TP。

## 结果与最小 TP 推荐

顶层 `status` 只表示当前 `runtime.tp` 的静态结论：

- `pass`：当前 TP 满足设备和最低容量要求。
- `reject`：当前 TP 明确超出设备数量或显存预算；部署流程不得启动服务。
- `unknown`：缺少硬件、架构、运行参数或可靠权重事实，不能完成静态证明。

当 `runtime.tp` 为 `null` 时，当前配置状态为未冻结，顶层保持 `unknown`。脚本使用
`allowed_tp`、`official_min_tp`、`device_count` 和容量判定按升序尝试候选，第一个
`pass` 作为 `recommendation.recommended_tp`，推荐状态为 `available`。候选全部因
设备或容量拒绝时为 `unavailable`；存在无法判断的候选且没有通过值时保持 `unknown`。
推荐值必须由任务 Main 或用户写回新的规范输入后才能启动服务。

`static-oom-report.json` 只表示这一次预检 attempt。它不代表服务已健康、压测已完成、
candidate 已通过、参数最优或最终收益。
