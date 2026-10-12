# 部署报告结构

`deployment_report.json` 是一次部署 attempt 的结构化事实记录。输出路径由任务或
Goal Plus provider 注入；报告不能替代独立 verifier，也不表示压测成功、参数最优或
candidate promotion。

```json
{
  "schema_version": "vllm-ascend-deployment-v1",
  "status": "<recommendation_only|input_invalid|startup_failed|healthcheck_failed|service_identity_invalid|service_ready|cleanup_failed>",
  "model": {
    "model_path": "<deployment-input.json 中的模型路径或标识>",
    "served_model_name": "<deployment-input.json 中的服务身份>",
    "config_digest": "<官方配置摘要>"
  },
  "deployment": {
    "tp": "<实际使用的 TP 或 null>",
    "dp": "<实际使用的 DP 或 null>",
    "max_num_seqs": "<实际值或 null>",
    "max_num_batched_tokens": "<实际值或 null>",
    "gpu_memory_utilization": "<实际值或 null>",
    "max_model_len": "<实际值或 null>",
    "dtype": "<实际值或 null>",
    "quantization": null
  },
  "resources": {
    "provider": "<provider 标识>",
    "devices": ["<本次实际使用的设备>"],
    "port": "<本次实际使用的端口或 null>",
    "resource_lock": "<任务允许记录的锁标识或 null>"
  },
  "inputs": {
    "deployment_input": "<attempt-dir>/deployment-input.json",
    "static_oom_input": "<attempt-dir>/static-oom-input.json",
    "static_oom_report": "<attempt-dir>/static-oom-report.json"
  },
  "preflight": {
    "status": "<来自 static-oom-report.json 的 status>",
    "risk_level": "<来自 static-oom-report.json 的 risk_level>",
    "recommended_tp": "<来自 static-oom-report.json 的推荐值或 null>",
    "artifact": "<attempt-dir>/static-oom-report.json"
  },
  "service_identity_valid": "<true|false>",
  "checks": {
    "environment": "<passed|failed|not_run>",
    "process": "<passed|failed|not_run>",
    "health": "<passed|failed|not_run>",
    "models_api": "<passed|failed|not_run>",
    "minimal_request": "<passed|failed|not_run>",
    "cleanup": "<passed|failed|not_applicable|not_run>"
  },
  "artifacts": {
    "command": "<命令记录路径>",
    "server_log": "<服务日志路径>",
    "pid": "<PID 记录路径>",
    "raw_checks": "<原始检查路径>"
  },
  "timing": {
    "startup_ms": "<数值或 null>",
    "healthcheck_ms": "<数值或 null>",
    "minimal_request_ms": "<数值或 null>",
    "cleanup_ms": "<数值或 null>"
  },
  "error": "<错误阶段和原因，或 null>"
}
```

约束：

- `inputs` 中的三个路径必须指向同一次 attempt 的 artifact。不存在的 artifact 不能
  通过修改报告字段伪造为已生成。
- `preflight` 只摘要 `static-oom-report.json`，不得重新计算或把 `reject`、`unknown`
  改成成功。
- `service_identity_valid=true` 只能在健康接口、`/v1/models`、served model identity
  和最小请求均通过后记录；它不表示模型质量或性能通过。
- 未执行的检查使用 `not_applicable` 或 `not_run`，不能填 `true`。
- 失败时保留日志、原始检查和错误阶段。重试必须使用新的 attempt 或唯一输出目录，
  不能覆盖已有报告。
