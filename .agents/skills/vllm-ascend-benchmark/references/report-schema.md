# Benchmark 报告结构

`benchmark_result.json` 是一次独立官方 benchmark attempt artifact。它不表达参数搜索或
最终推荐。所有性能指标必须来自本次 `vllm bench serve` 生成的唯一
raw JSON。

## 顶层结构

成功或失败都保留相同的顶层区块：

```json
{
  "schema_version": "vllm-ascend-benchmark-v2",
  "status": "success",
  "attempt": {
    "id": "<unique attempt id>",
    "provider": "<provider id>",
    "config_digest": "<deployment/config digest or null>",
    "workload_digest": "<workload digest or null>"
  },
  "preflight": {
    "status": "pass|failed|not_run",
    "provider_path": "managed_deployment|predeployed",
    "execution_context": "current|elevated|unknown",
    "sandbox_detection": "<path or null>",
    "benchmark_preflight": "<path or null>",
    "deployment_report": "<path or null>"
  },
  "service": {
    "source": "managed_deployment|predeployed",
    "base_url": "<service URL>",
    "endpoint": "<endpoint>",
    "served_model_name": "<served model name>",
    "tokenizer": "<resolved tokenizer source>",
    "deployment_report": "<path or null>",
    "service_identity_valid": true
  },
  "workload": {
    "selection": "preset|hf|custom",
    "scenario": "random|long_prefill|long_decode|prefill_decode_balance|null",
    "benchmark_mode": "stress|production",
    "dataset_root": "<path or null>",
    "dataset": {
      "source": "scenario_preset|contract_defined|vllm_random",
      "mode": "hf|sharegpt|random|custom",
      "name": "<dataset name or null>",
      "path": "<resolved path or id>",
      "split": "<split or null>",
      "subset": "<subset or null>",
      "revision_or_hash": "<resolved value or null>",
      "output_len": "<integer or null>",
      "text_column": "<column or null>",
      "skip_chat_template": "<boolean or null>",
      "validation": {
        "split_verified": false,
        "validation_sample_count": 0,
        "sample_indices": [],
        "full_count_attempted": false
      },
      "resolution": "reused|downloaded|cached|local|not_required"
    },
    "requested_count": null,
    "warmups": null,
    "max_concurrency": null,
    "request_rate": null,
    "burstiness": null,
    "rounds": null
  },
  "counts": {"requested": null, "completed": null, "failed": null},
  "metrics": {
    "source": "vllm_bench_serve",
    "TTFT": {"unit": "ms", "P50": null, "P95": null},
    "TPOT": {"unit": "ms", "P50": null, "P95": null},
    "E2EL": {"unit": "ms", "P50": null, "P95": null}
  },
  "goodput": {
    "status": "reported|not_requested|missing",
    "constraints_ms": {},
    "request_goodput": null,
    "unit": "req/s"
  },
  "throughput": {
    "window_seconds": null,
    "requests_per_second": null,
    "output_tokens_per_second": null
  },
  "failure": null,
  "oom": {
    "detected": false,
    "phase": "none",
    "type": null,
    "evidence": []
  },
  "source": {
    "format": "vllm_bench_serve",
    "command": "<complete command or controlled reference>",
    "raw_result": "<unique raw JSON path or null>",
    "raw_result_sha256": "<SHA256 or null>",
    "vllm_version": "<version or null>",
    "vllm_ascend_version": "<version or unknown>",
    "cli_capabilities": "<capability summary>"
  },
  "cleanup": {
    "status": "passed|failed|not_owned|not_run",
    "artifact_exported": false
  },
  "errors": []
}
```

`service.source=managed_deployment` 表示本次 benchmark 调用了部署 skill；
`service.source=predeployed` 表示服务由 Provider 交付，本 skill 只验证和测量。

`preflight` 是本次 benchmark 使用的环境和执行上下文事实。路径 A 必须引用部署报告提供
的 `sandbox_detection`、`execution_context` 和 `deployment_report`，且不得重复执行沙箱或
NPU/端口资源探测；路径 B 必须在本次 attempt 中写出 `sandbox_detection` 和
`benchmark_preflight`。`execution_context=elevated` 时，服务验收和正式 benchmark 必须
沿用同一提权上下文。

`source.vllm_version` 和 `source.vllm_ascend_version` 必须来自
`importlib.metadata.version("vllm")` 与 `importlib.metadata.version("vllm-ascend")`；
metadata 不存在时保留 `null/unknown`，不得访问模块 `__version__` 或通过扫描
`site-packages` 猜测 editable install 的源码位置。

## 状态

- `success`：服务身份有效、官方 benchmark 完整结束、raw result 唯一、请求和必需指标
  通过校验；`cleanup` 独立表达服务清理结果。
- `invalid`：Provider 输入、负载、身份、请求计数、数据集或 goodput 合同不满足。
- `blocked`：环境或目标 vLLM CLI 不具备冻结合同要求的能力，正式 benchmark 未执行。
- `error`：服务、资源、运行时、输出或报告交付故障，无法把结果归因于候选性能。

`service_identity_valid=true` 不是 benchmark 成功；benchmark 成功也不代表任何候选通过。

## 指标和 goodput

`metrics` 数值只能直接映射 raw JSON 中官方计算的 TTFT、TPOT、E2EL P50/P95。不得插值、
重算、从日志推断或用零补齐缺失值。请求计数沿用官方结果，不移除失败样本。

Goodput 状态规则：

- `reported`：Provider 提供至少一个 SLO，命令启用 `--goodput`，raw JSON 有合法
  `request_goodput`；
- `not_requested`：Provider 未提供 SLO，命令未启用 `--goodput`，`request_goodput` 为
  `null`；
- `missing`：Provider 要求 goodput，但 raw JSON 缺失、为 `null`、不可解析或无法证明
  属于本次 attempt。此时 attempt 不能为 `success`。

`constraints_ms` 只记录 Provider 实际提供的 `ttft`、`tpot`、`e2el`，单位为毫秒。请求
吞吐不得映射为 goodput。

## 失败、OOM 和清理

失败报告仍保留相同顶层结构：

- 性能字段不可信时为 `null`；
- `counts` 尽可能保留 raw JSON 的请求、完成和失败计数；
- raw result 存在且可读时保留路径和 SHA256，否则为 `null`；
- `failure` 至少包含 `stage`、`type`、`message` 和 `artifacts`；
- `errors` 记录更细的阶段、类型、消息和 artifact；
- `oom` 始终存在；没有 OOM 使用 `detected=false`、`phase=none`、`type=null`；
- 清理结果只写入 `cleanup`，不覆盖性能区块。

正式 benchmark 请求期间的 OOM 使用：

```json
{
  "status": "error",
  "failure": {
    "stage": "benchmark",
    "type": "runtime_oom",
    "message": "正式压测请求期间检测到运行时显存不足",
    "artifacts": ["<benchmark log>", "<service log>"]
  },
  "oom": {
    "detected": true,
    "phase": "benchmark",
    "type": "runtime_oom",
    "evidence": [
      {
        "artifact": "<current attempt artifact>",
        "pattern": "out of memory",
        "observed_at": "<timestamp or null>"
      }
    ]
  }
}
```

静态预检、服务启动和最小请求验收阶段的 OOM 属于部署报告；benchmark 未启动时不得生成
`runtime_oom`。

主要错误类型包括：

`service_identity_invalid`、`runtime_oom`、`cli_incompatible`、`workload_not_replayable`、
`raw_result_not_found`、`raw_result_ambiguous`、`raw_result_invalid`、`model_mismatch`、
`request_count_mismatch`、`benchmark_incomplete`、`metrics_missing`、`goodput_missing`、
`dataset_contract_invalid`、`dataset_mode_invalid`、`dataset_root_invalid`、
`dataset_not_found`、`dataset_prepare_failed`、`dataset_invalid`、`dataset_insufficient`、
`custom_dataset_invalid`、`hf_dataset_invalid`、`cleanup_failed`、`sandbox_approval_denied`、
`sandbox_escalation_failed`、`sandbox_isolation_unknown`、`environment_preflight_failed`。
