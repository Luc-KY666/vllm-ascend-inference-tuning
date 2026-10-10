# 部署报告结构

部署报告是一次 attempt 的结构化事实记录，不能作为 Goal Plus 最终 candidate verdict。
具体输出路径由任务或 GP 注入。

```json
{
  "schema_version": "vllm-ascend-deployment-v1",
  "status": "service_ready",
  "model": {
    "model_path": "<按任务规则记录的路径或标识>",
    "served_model_name": "<服务模型名>",
    "config_digest": "<配置摘要>"
  },
  "deployment": {
    "tp": 1,
    "dp": 1,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 1,
    "gpu_memory_utilization": null,
    "max_model_len": null,
    "dtype": null,
    "quantization": null
  },
  "resources": {
    "provider": "git_worktree",
    "devices": ["<本次实际使用的设备>"],
    "port": null,
    "resource_lock": "<仅记录任务允许的标识>"
  },
  "preflight": {
    "status": "pass",
    "risk_level": "none",
    "recommended_tp": null,
    "artifact": "<预检结果路径>"
  },
  "service_identity_valid": true,
  "checks": {
    "environment": "passed",
    "process": "passed",
    "health": "passed",
    "models_api": "passed",
    "minimal_request": "passed",
    "cleanup": "not_applicable"
  },
  "artifacts": {
    "command": "<命令记录路径>",
    "server_log": "<日志路径>",
    "pid": "<PID 记录路径>",
    "raw_checks": "<原始检查路径>"
  },
  "timing": {
    "startup_ms": null,
    "healthcheck_ms": null,
    "minimal_request_ms": null,
    "cleanup_ms": null
  },
  "error": null
}
```

约束：

- `service_identity_valid` 只表示服务健康和身份检查，不表示性能或质量通过。
- `devices` 只能记录实际使用的设备，不记录全机资源快照、HBM 占用或进程列表。
- 未执行的检查使用 `not_applicable` 或 `not_run`，不能填 `true`。
- 失败时保留日志和错误阶段；不能把失败字段删掉后当作成功报告。
- 重试必须有新的 attempt 或唯一标识，不能覆盖已有报告和原始日志。
