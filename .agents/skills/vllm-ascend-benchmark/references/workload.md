# 压测负载契约

本文件只定义可复用的场景命名和必须显式冻结的字段，不携带具体模型、数据集目录、
设备、端口或数量。

## 场景

| 场景 | 语义 |
| --- | --- |
| `random` | 使用 vLLM 支持的合成输入/输出长度，适合协议和可控基线 |
| `long_prefill` | 真实长输入、相对短输出，关注 TTFT 和 prefill 排队 |
| `long_decode` | 相对短输入、长输出，关注 TPOT 和 decode 稳态 |
| `prefill_decode_balance` | 真实多轮对话分布，观察综合服务能力 |

场景不决定数据集 ID 或本地路径。任务必须在合同中写明 `dataset_name`、`dataset_path`
或远程数据集 ID、split、数据版本/hash 和访问权限。

## 必须冻结的参数

```json
{
  "scenario": "random",
  "benchmark_mode": "task-defined",
  "dataset": {
    "name": "task-defined",
    "path": "task-defined",
    "revision_or_hash": "task-defined"
  },
  "requests": {
    "num_prompts": 0,
    "max_concurrency": 0,
    "num_warmups": 0,
    "request_rate": null,
    "burstiness": null,
    "rounds": 1
  },
  "generation": {
    "temperature": null,
    "top_p": null,
    "max_tokens": null,
    "stream": true,
    "include_usage": true
  },
  "statistics": {
    "minimum_decode_samples": 1,
    "percentiles": [50, 95]
  }
}
```

上面 `0`、`null` 和 `task-defined` 都是占位符，不是默认值。任务必须冻结正数请求
数量、warmup、并发和适用的到达率；如果使用固定请求文件，还要冻结文件 hash、请求
顺序、并发分组和失败规则。

## 入口选择

使用 `vllm bench serve` 前必须核对：

- 当前版本是否支持任务需要的 backend、endpoint、tokenizer、数据集和结果参数；
- 是否能完整重放固定请求的模型名、stream、usage、采样和长度；
- 结果 JSON 是否包含本任务要求的完成数、失败数和 TTFT/TPOT/E2EL 分布。

任一条件不满足时，改用任务冻结的 HTTP 流式客户端和 `streaming_metrics.py`，不能用
随机负载结果替代固定请求。

## 失败处理

- 失败请求计入总数，不从分母删除。
- 超时、HTTP 错误、服务断流、usage 缺失和异常 finish reason 必须保留原始证据。
- 不重试到通过，不用部分样本填充缺失 percentile。
- warmup 与正式样本分离；不能把 warmup 统计混进正式结果。
- benchmark 期间发生服务 OOM 或资源清理失败时，报告错误阶段，不伪造性能数值。
