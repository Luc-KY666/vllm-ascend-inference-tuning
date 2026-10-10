# 流式测量方法

## 单请求样本

每个正式请求至少保存以下元数据：

```json
{
  "request_id": "<唯一 ID>",
  "sent_at": 0.0,
  "ended_at": 0.0,
  "wire_chunks": "<单独保存的原始字节块文件>",
  "wire_sha256": "<原始流 hash>",
  "model": "<served model name>",
  "status": "passed"
}
```

原始流建议使用 JSONL 保存每个 chunk 的 base64 字节和单调时间戳。解析后的样本只引用
原始文件和 hash，不在汇总报告中重复嵌入完整 SSE。

## 指标

对完整请求：

```text
TTFT = first_non_empty_content_at - sent_at
E2EL  = ended_at - sent_at
TPOT  = (last_non_empty_content_at - first_non_empty_content_at)
        / (completion_tokens - 1)
```

只有 completion tokens 大于 1 且至少存在两个非空内容事件时，TPOT 才可估计。TPOT 的
P50/P95 仅使用可估计样本，同时记录 `decode_estimable` 和 `decode_unestimable`。

P50/P95 使用 nearest-rank：

```text
rank = ceil(percentile * sample_count)
value = sorted_values[rank - 1]
```

不能平均各轮 P95 作为整体 P95。多轮吞吐应保留每轮窗口及整体窗口，不用启动、编译和
warmup 时间替代稳态吞吐。

## 最小完整性门禁

正式测量成功至少要求：

- 所有预定请求都有结果；
- 没有失败或超时请求；
- 每个成功流包含正常 finish、usage、非空内容和完整 `[DONE]`；
- TTFT 与 E2EL 的样本数等于正式请求数；
- TPOT 可估计样本数达到任务冻结的 `minimum_decode_samples`；
- 服务身份、模型名、配置摘要和 workload hash 与本次 attempt 一致。

不满足门禁时仍应输出失败报告和原始证据，但性能字段必须为 `null`。
