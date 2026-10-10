# 压测报告结构

`benchmark_result.json` 是一次独立 measurement attempt 的 artifact。它不包含 Goal Plus
的最终 selection 或 promotion 状态。

```json
{
  "schema_version": "vllm-ascend-benchmark-v1",
  "status": "success",
  "attempt": {
    "id": "<任务生成的唯一 attempt ID>",
    "provider": "git_worktree",
    "config_digest": "<部署配置摘要>",
    "workload_digest": "<负载摘要>"
  },
  "service": {
    "base_url": "<任务提供的服务接口>",
    "served_model_name": "<服务模型名>",
    "service_identity_valid": true
  },
  "workload": {
    "scenario": "task-defined",
    "dataset": "<任务定义>",
    "requested_count": 0,
    "warmups": 0,
    "max_concurrency": 0,
    "rounds": 0
  },
  "counts": {
    "requested": 0,
    "completed": 0,
    "failed": 0,
    "decode_estimable": 0,
    "decode_unestimable": 0
  },
  "metrics": {
    "TTFT": {"unit": "ms", "P50": null, "P95": null},
    "TPOT": {"unit": "ms", "P50": null, "P95": null},
    "E2EL": {"unit": "ms", "P50": null, "P95": null}
  },
  "throughput": {
    "window_seconds": null,
    "requests_per_second": null,
    "output_tokens_per_second": null
  },
  "source": {
    "format": "vllm_bench_serve",
    "raw_result": "<唯一原始结果路径>",
    "raw_result_sha256": "<sha256>",
    "raw_streams": ["<每个原始流的相对路径和 hash 索引>"]
  },
  "cleanup": {
    "status": "passed",
    "artifact_exported": true
  },
  "errors": []
}
```

失败时保留相同顶层结构，`status` 使用 `invalid` 或 `error`，指标为 `null`，并在
`errors` 中写明阶段、类型和可复现的原始证据路径。报告不能因为指标缺失而删除计数和
失败原因。
