# 静态 OOM 与最小 TP 预检

## PD 模式

PD 模式调用 `scripts/pd_static_oom_precheck.py` 一次，分别检查 Prefill 和 Decode。
两侧都通过时整体检查通过；任一侧拒绝或无法判断时停止部署。未指定角色 TP 时，
在该角色设备池的可行候选中选择最小 TP；不能把 P/D 的卡数或 TP 相加替代角色级检查。

静态预检是容量诊断，每个 deployment attempt 只执行一次。它不启动服务、不绑定端口、
不重新读取模型目录或中间 JSON。agent 将规范化输入和硬件事实在内存中组成 payload，脚本
只从 stdin 或显式 `--input` 文件读取，并把结果打印到 stdout：

```text
python <skill-root>/scripts/static_oom_precheck.py < normalized-oom-input.json
```

或：

```text
python <skill-root>/scripts/static_oom_precheck.py --input /path/to/input.json
```

脚本结果必须直接汇总到内存中的静态 OOM 上下文；如果执行器需要保留原始输入或 stdout，可
临时写入 `<attempt-dir>/.work/static_oom/`。清理阶段必须删除这些临时文件，最终报告只
在 `launch.parameters` 中保留静态 OOM 结论影响的 TP，不得写入预检输入、结果或证据。

## 推荐模式

当 TP 缺失时，payload 必须显式包含：

```json
{
  "recommendation_mode": true,
  "runtime": {
    "tp": null,
    "dp": 1,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096
  }
}
```

脚本按 `allowed_tp` 升序选择第一个明确通过的值。TP 写回只更新内存中的启动配置，记录
`static_oom_reused=true`、`recheck_performed=false`，不得因写回再次调用静态预检。

## 输入和计算

输入至少包含 `model.config`、可靠的 `weight_bytes` 或 `weight_gib`、
`hardware.card_memory_bytes`、`hardware.device_count`、`runtime.tp`、`runtime.dp`、
`max_num_seqs`、显存比例、最低上下文、block size、`allowed_tp` 和 `official_min_tp`。
缺少 KV Cache 字段、dtype、权重需求或硬件容量时返回 `unknown`。

### 输入 JSON 结构

各字段有严格的层级归属，必须按下述结构放置，不能平铺到顶层：

- `weight_bytes` / `weight_gib` 属于 `model`；
- `allowed_tp`、`official_min_tp` 属于 `runtime`；
- `card_memory_bytes`、`device_count` 属于 `hardware`。

```json
{
  "recommendation_mode": true,
  "model": {
    "model_path": "/absolute/model/path",
    "served_model_name": "model-name",
    "weight_bytes": 61066575656,
    "config": {
      "vocab_size": 151936,
      "head_dim": 128,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "num_attention_heads": 32,
      "num_experts": 128,
      "hidden_size": 2048,
      "num_hidden_layers": 48,
      "intermediate_size": 6144,
      "num_key_value_heads": 4
    }
  },
  "hardware": {
    "card_memory_bytes": 68719476736,
    "device_count": 4
  },
  "runtime": {
    "tp": null,
    "dp": 1,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096,
    "gpu_memory_utilization": 0.9,
    "min_required_context_len": 4096,
    "block_size": 128,
    "allowed_tp": [1, 2, 4],
    "official_min_tp": 1
  }
}
```

MoE 模型（`config.num_experts` 非空）必须提供明确的 `model.weight_bytes` 或
`model.weight_gib`，脚本不会从配置推导 MoE 权重需求；缺失时返回 `unknown`。

Dense decoder 的口径为：

```text
memory_budget = floor(card_memory_bytes * gpu_memory_utilization)
weight_per_card = ceil(weight_bytes / tp)
kv_heads_per_rank = max(1, ceil(num_key_value_heads / tp))
kv_bytes_per_token = 2 * num_hidden_layers * kv_heads_per_rank * head_dim * kv_dtype_bytes
context_tokens = ceil(min_required_context_len / block_size) * block_size
minimum_kv_bytes = max_num_seqs * context_tokens * kv_bytes_per_token
```

`pass` 表示当前 TP 明确满足最低容量，`reject` 表示明确超出设备或显存约束，`unknown`
表示事实不足。推荐结果不能单独表示服务已启动或健康。
