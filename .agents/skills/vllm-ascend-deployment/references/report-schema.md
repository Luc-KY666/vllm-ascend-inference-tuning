# 部署报告结构

`deployment_report.json` 是一次独立 deployment attempt 的最终事实记录。它必须保留
规范输入、推荐前后配置、各阶段 artifact、服务验收和清理状态。

```json
{
  "schema_version": "vllm-ascend-deployment-report-v1",
  "status": "<recommendation_only|input_invalid|environment_preflight_failed|resource_preflight_failed|static_oom_rejected|startup_failed|healthcheck_failed|service_identity_invalid|service_ready|cleanup_failed>",
  "model": {
    "model_path": "<deployment.json.model.model_path>",
    "served_model_name": "<deployment.json.model.served_model_name>",
    "config_path": "<deployment.json.model.config_path>",
    "config_digest": "<sha256>"
  },
  "deployment": {
    "before_resolution": {"tp": null},
    "resolved": {
      "tp": 2,
      "dp": 1,
      "max_num_seqs": 32,
      "max_num_batched_tokens": 4096,
      "gpu_memory_utilization": 0.9,
      "max_model_len": null,
      "dtype": "auto",
      "quantization": null
    },
    "resolution": {
      "mode": "input|auto_recommended|none",
      "recommended_tp": 2,
      "source": "<static-oom-report.json>"
    }
  },
  "resources": {
    "visible_devices": ["0", "1"],
    "selected_devices": ["0", "1"],
    "port_candidates": [8117],
    "selected_port": 8117
  },
  "inputs": {
    "deployment": "<attempt-dir>/deployment.json",
    "sandbox_detection": "<attempt-dir>/sandbox-detection.json",
    "ascend_preflight": "<attempt-dir>/ascend-preflight.json",
    "npu_resource_probe": "<attempt-dir>/npu-resource-probe.json",
    "port_resource_probe": "<attempt-dir>/port-resource-probe.json",
    "static_oom_input": "<attempt-dir>/static-oom-input.json",
    "static_oom_report": "<attempt-dir>/static-oom-report.json"
  },
  "preflight": {
    "environment": "<pass|failed|not_run>",
    "static_oom": "<pass|reject|unknown|not_run>",
    "resources": "<pass|failed|not_run>",
    "recommended_tp": 2
  },
  "service": {
    "url": "http://127.0.0.1:8117",
    "pid": 1234,
    "handoff": "<provider|not_handed_off>"
  },
  "checks": {
    "process": "<passed|failed|not_run>",
    "health": "<passed|failed|not_run>",
    "models_api": "<passed|failed|not_run>",
    "served_model_identity": "<passed|failed|not_run>",
    "minimal_request": "<passed|failed|not_run>",
    "cleanup": "<passed|failed|not_applicable|not_run>"
  },
  "service_identity_valid": false,
  "artifacts": {
    "command": "<path|null>",
    "server_log": "<path|null>",
    "pid": "<path|null>",
    "raw_checks": "<path|null>"
  },
  "failure": null,
  "oom": {
    "detected": false,
    "phase": "none",
    "type": null,
    "evidence": []
  },
  "error": null
}
```

## 约束

- `inputs` 中的路径必须指向当前 attempt 的实际 artifact；未执行阶段为 `null`，不得
  伪造生成成功。
- `npu_resource_probe` 和 `port_resource_probe` 是本次 attempt 唯一的服务资源事实；它们由
  [resource-probes.md](resource-probes.md) 在启动前记录一次可用卡 `npu-smi info`、端口探测、
  原始证据和 agent 判定结果。`sandbox_detection` 中的 `npu-smi` 只用于隔离判定，不得当作
  可用卡快照。服务启动直接使用资源 probe 返回的设备和端口，不记录或执行二次资源 probe。
- `input_invalid`、`environment_preflight_failed`、`resource_preflight_failed`、
  `static_oom_rejected` 和 `recommendation_only` 均不得启动服务。
- 静态 OOM `reject` 必须记录 `failure.stage=static_preflight`、
  `failure.type=static_oom_rejected`、`oom.phase=static_preflight` 和
  `oom.type=static_oom_rejected`。
- 启动或最小请求阶段 OOM 必须记录 `failure.stage=service_startup`、
  `failure.type=service_startup_oom`、`oom.phase=service_startup` 和
  `oom.type=service_startup_oom`。
- `service_identity_valid=true` 只能在进程、健康接口、`/v1/models`、served model
  identity 和最小请求全部通过后记录。
- 失败报告必须包含 `failure.stage`、`failure.type`、`failure.message` 和相关 artifact；
  OOM evidence 至少包含当前日志或预检报告、匹配摘要和时间信息。
- 重试使用新的唯一输出目录，不覆盖已有 attempt。
