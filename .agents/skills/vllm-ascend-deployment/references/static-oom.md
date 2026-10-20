# 静态 OOM 与最小 TP 预检

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

脚本不接受 `--output` 或 `--input-artifact`，不创建静态 OOM 输入/报告文件。agent 直接把
结果写入 `stages.static_oom.input`、`result`、`recommended_tp` 和 `oom`，必要诊断尾部写入
`diagnostics_tail`。

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

脚本按 `allowed_tp` 升序选择第一个明确通过的值。TP 写回只更新
`stages.tp_resolution`，记录 `static_oom_reused=true`、`recheck_performed=false`，不得因
写回再次调用静态预检。

## 输入和计算

输入至少包含 `model.config`、可靠的 `weight_bytes` 或 `weight_gib`、
`hardware.card_memory_bytes`、`hardware.device_count`、`runtime.tp`、`runtime.dp`、
`max_num_seqs`、显存比例、最低上下文、block size、`allowed_tp` 和 `official_min_tp`。
缺少 KV Cache 字段、dtype、权重需求或硬件容量时返回 `unknown`。

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
