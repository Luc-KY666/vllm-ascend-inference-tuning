# 静态 OOM 与最小 TP 预检

## PD 模式

PD 不再调用专用静态 OOM 脚本。agent 对每个 Prefill/Decode 实例分别保存
`deployment_input.json` 和 input contract 生成的 `runtime.json`，调用同一个
`scripts/static_oom_precheck.py`。脚本使用每个实例自己的本地设备数、显存事实和允许 TP
候选；不能把 P/D 的卡数或 TP 相加替代节点级检查。

节点 payload 的 `runtime.dp` 固定为 `1`，用于计算当前实例自己的单卡容量。实际命令中的
角色级 `--data-parallel-size` 已由 `input_contract.py` 的 `resources.role_device_count`
和 agent 的拓扑校验负责，不参与本地容量计算。

所有节点都必须通过或得到可解析的最小 TP 推荐；任一节点拒绝、无法判断或没有可行 TP
时停止部署。缺失 TP 的节点分别执行推荐和 `resolve_recommended_tp()`，不得用一个节点的
推荐覆盖另一个节点，也不得因 TP 写回重复执行静态预检。

静态预检是容量诊断。ordinary 每个 deployment attempt 执行一次；PD 每个节点执行一次。
它不启动服务、不绑定端口、不重新读取模型目录。`input_contract.py` 先输出临时
`runtime.json`，static OOM 再把原始 `deployment_input.json` 与该 runtime artifact 合并；
agent 不再手工拼装 OOM payload。脚本把结果打印到 stdout：

```text
python <skill-root>/scripts/static_oom_precheck.py \
  --deployment-input <deployment-input.json> \
  --runtime <runtime.json>
```

`runtime.json` 的 schema 固定为 `vllm-ascend-runtime-v1`，由 input contract 生成，包含
规范化 `runtime`、config 摘要、config digest 和资源。`deployment_input.json` 提供模型权重
事实与原始资源。脚本结果必须直接汇总到内存中的静态 OOM 上下文；如果执行器需要保留
原始输入或 stdout，可临时写入 `<attempt-dir>/.work/static_oom/`。清理阶段必须删除这些
临时文件，最终报告只在 `launch.parameters` 中保留静态 OOM 结论影响的 TP，不得写入预检
输入、结果或证据。

## 推荐模式

当 TP 缺失时，runtime artifact 中必须保留 `runtime.tp=null`，例如：

```json
{
  "recommendation_mode": true,
  "runtime": {
    "tp": null,
    "dp": 2,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096
  }
}
```

脚本按 `allowed_tp` 升序选择第一个明确通过的值。PD 模式在脚本内部将本地容量判断的
有效 `dp` 设为 `1`；普通模式保留 runtime 中的实际 DP。TP 写回只更新内存中的启动配置，
记录 `static_oom_reused=true`、`recheck_performed=false`，不得因写回再次调用静态预检。

## 输入和计算

下面的 `model`、`hardware`、`runtime` 结构是 `static_oom_precheck.py` 将原始
`deployment_input.json` 与 input contract 生成的 `runtime.json` 合并后的内部 payload，
不是 agent 需要另行组装的 deployment input 格式。deployment input 的字段定义以及
`resources.card_memory_bytes` 的填写要求唯一以 [`input-contract.md`](input-contract.md)
为准；static OOM 只读取并计算，不推导或补写该硬件事实。

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
