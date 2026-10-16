# 静态 OOM 与最小 TP 预检

静态预检是容量诊断，不启动服务、不绑定端口、不读取运行中的设备进程，也不替代服务
验收。部署流程从 `deployment.json` 和官方 `config.json` 派生
`static-oom-input.json`，再调用唯一脚本入口：

```text
python <skill-root>/scripts/static_oom_precheck.py \
  --input <attempt-dir>/static-oom-input.json \
  --output <attempt-dir>/static-oom-report.json
```

## 推荐模式

当 `deployment.json.input_status=missing_parameters` 且 TP 缺失时，派生输入必须显式写入：

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

脚本不会从 vLLM CLI 的 `None` 或帮助文本推断这些参数。推荐结果为
`recommendation.recommended_tp` 时，调用方必须使用 `input_contract.py --resolve-tp` 原子
更新同一个 `deployment.json`，记录 `resolution.mode=auto_recommended`，然后重新执行
契约校验和后续部署。

## 输入

`static-oom-input.json` 至少包含：

- `model.config`、可靠的 `weight_bytes` 或 `weight_gib`；
- `hardware.card_memory_bytes` 和 `hardware.device_count`；
- `runtime.tp`、`runtime.dp`、`max_num_seqs`、显存比例、最低上下文、block size、
  `allowed_tp` 和 `official_min_tp`。

`config` 由部署流程从 `deployment.json.model.config_path` 读取；脚本本身不扫描模型目录
或权重文件。`hidden_size`、`num_attention_heads` 和 `head_dim` 按配置中的独立字段参与
计算，不再要求 `hidden_size == num_attention_heads * head_dim`，以兼容分组/多查询注意力
等投影维度不同的模型。缺少 KV Cache 所需字段、dtype、权重需求或硬件容量时仍返回
`unknown`。

## 计算口径

Dense decoder 的权重和最低 KV Cache 计算保持以下规则：

```text
memory_budget = floor(card_memory_bytes * gpu_memory_utilization)
weight_per_card = ceil(weight_bytes / tp)
kv_heads_per_rank = max(1, ceil(num_key_value_heads / tp))
kv_bytes_per_token = 2 * num_hidden_layers * kv_heads_per_rank * head_dim * kv_dtype_bytes
context_tokens = ceil(min_required_context_len / block_size) * block_size
minimum_kv_bytes = max_num_seqs * context_tokens * kv_bytes_per_token
```

`weight_per_card >= memory_budget` 或 `weight_per_card + minimum_kv_bytes > memory_budget`
时，候选 TP 为 `reject`。`tp * dp > device_count` 或 TP 不能整除 attention heads 时，
候选也不可用。DP 不倍乘单卡权重或 KV Cache。

## 结果

- `pass`：当前 TP 明确满足设备和最低容量约束；
- `reject`：当前 TP 明确超出设备或显存约束，部署不得启动；
- `unknown`：输入事实不足以证明通过或拒绝。

`runtime.tp=null` 时，顶层状态保持 `unknown`，脚本按 `allowed_tp` 升序评估，第一个
`pass` 作为最小推荐。候选全部拒绝时 `recommendation.status=unavailable`；存在未知候选
且没有通过值时保持 `unknown`。推荐报告不能单独表示服务已启动或健康。
