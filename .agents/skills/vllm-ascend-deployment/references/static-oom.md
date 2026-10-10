# 静态 OOM 与最小 TP 预检

静态预检是容量诊断，不是设备调度器，也不是运行时 verifier。它的输入必须来自本轮
任务和 GP 已冻结的模型、设备容量及运行参数。

## 输入

纯计算工具接受 JSON 文件或 stdin。最小结构如下：

```json
{
  "model": {
    "config": {
      "vocab_size": 151936,
      "hidden_size": 4096,
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "intermediate_size": 11008,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16"
    }
  },
  "hardware": {
    "card_memory_bytes": 34359738368,
    "device_count": 1
  },
  "runtime": {
    "tp": 1,
    "dp": 1,
    "max_num_seqs": 1,
    "gpu_memory_utilization": 0.9,
    "min_required_context_len": 4096,
    "block_size": 16,
    "allowed_tp": [1, 2, 4, 8, 16]
  }
}
```

也可以在 `model` 中提供 `weight_bytes` 或 `weight_gib`，此时工具不从架构字段推导
总权重；KV Cache 仍需要足够的架构字段才能计算。缺失关键字段时输出 `unknown`，不
为了通过预检而猜测。

## 计算口径

对常见 dense decoder，工具按配置估算：

```text
attention =
  hidden_size * num_attention_heads * head_dim
  + hidden_size * num_key_value_heads * head_dim * 2
  + num_attention_heads * head_dim * hidden_size

mlp = 3 * hidden_size * intermediate_size
norm = 2 * hidden_size
embedding = vocab_size * hidden_size
lm_head = 0 if tie_word_embeddings else vocab_size * hidden_size

total_params = embedding + lm_head
             + num_hidden_layers * (attention + mlp + norm)
```

检测到常见 MoE 字段时，工具把所有 expert 权重视为常驻，并使用 `num_experts`、
`moe_intermediate_size` 及共享 expert 字段计算；无法可靠识别层分布或量化元数据时
返回 `unknown`。

显存预算和最低上下文 KV Cache 按以下规则计算：

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

如果 `weight_per_card >= memory_budget`，或
`weight_per_card + minimum_kv_bytes > memory_budget`，结果为 `reject`。
能可靠计算且未超出预算时为 `pass`；dtype、架构、量化、卡容量或 KV 字段不完整时为
`unknown`。

`dp` 只用于检查 `tp * dp <= device_count`，不把 DP 倍乘到单卡权重或 KV 需求。工具
不会选择设备，只报告设备数量约束。

## 推荐最小 TP

当没有冻结 `tp` 时，工具按 `allowed_tp` 升序尝试满足：

1. `tp >= official_min_tp`；
2. `tp * dp <= device_count`；
3. 静态容量结果为 `pass`。

第一个满足条件的值作为 `recommended_tp`。如果无法计算、设备数量不足或没有满足者，
推荐状态分别为 `unknown` 或 `unavailable`。推荐值必须由 Main/用户确认后才可写入
启动配置。
