# 压测报告结构

`benchmark_result.json` 是一次独立的 official benchmark attempt artifact，不包含
Goal Plus 的 selection、promotion 或最终参数结论。指标来源必须是本次
`vllm bench serve` 生成的唯一 raw JSON。

## 成功报告

```json
{
  "schema_version": "vllm-ascend-benchmark-v2",
  "status": "success",
  "attempt": {
    "id": "<任务生成的唯一 attempt ID>",
    "provider": "<provider>",
    "config_digest": "<部署配置摘要>",
    "workload_digest": "<负载摘要>"
  },
  "service": {
    "base_url": "<任务提供的服务接口>",
    "served_model_name": "<服务模型名>",
    "service_identity_valid": true
  },
  "workload": {
    "selection": "<preset|hf|custom>",
    "scenario": "<random|long_prefill|long_decode|prefill_decode_balance|null>",
    "benchmark_mode": "<stress|production>",
    "dataset_root": "<任务输入中的数据集目录或 null>",
    "dataset": {
      "source": "scenario_preset|contract_defined|vllm_random",
      "mode": "hf|sharegpt|random|custom",
      "name": "<场景派生或合同提供的数据集名>",
      "path": "<解析后的本地路径或合同远程 ID>",
      "split": "<split 或 null>",
      "subset": "<subset 或 null>",
      "revision_or_hash": "<版本、hash 或 null>",
      "output_len": "<输出长度或 null>",
      "text_column": "<文本列或 null>",
      "skip_chat_template": "<boolean 或 null>",
      "resolution": "reused|downloaded|cached|local|not_required"
    },
    "requested_count": "<冻结请求数>",
    "warmups": "<冻结 warmup 数>",
    "max_concurrency": "<冻结并发>",
    "request_rate": "<冻结到达率>",
    "burstiness": "<冻结 burstiness>",
    "rounds": "<冻结轮次>"
  },
  "counts": {
    "requested": null,
    "completed": null,
    "failed": null
  },
  "metrics": {
    "source": "vllm_bench_serve",
    "TTFT": {"unit": "ms", "P50": null, "P95": null},
    "TPOT": {"unit": "ms", "P50": null, "P95": null},
    "E2EL": {"unit": "ms", "P50": null, "P95": null}
  },
  "goodput": {
    "status": "not_requested",
    "constraints_ms": {},
    "request_goodput": null,
    "unit": "req/s"
  },
  "failure": null,
  "oom": {
    "detected": false,
    "phase": "none",
    "type": null,
    "evidence": []
  },
  "throughput": {
    "window_seconds": null,
    "requests_per_second": null,
    "output_tokens_per_second": null
  },
  "source": {
    "format": "vllm_bench_serve",
    "command": "<完整命令或受控命令引用>",
    "raw_result": "<唯一 raw JSON 路径>",
    "raw_result_sha256": "<SHA256>",
    "vllm_version": "<版本>",
    "vllm_ascend_version": "<版本或 unknown>",
    "cli_capabilities": "<能力检查摘要>"
  },
  "cleanup": {
    "status": "passed",
    "artifact_exported": true
  },
  "errors": []
}
```

`metrics` 中的数值直接来自官方 raw JSON；报告只负责字段映射和完整性记录，不生成、
插值或重算这些值。
如果官方 JSON 使用不同字段名，按目标 vLLM 版本的实际输出合同映射，并在
`source.cli_capabilities` 或错误信息中记录版本差异；字段映射以版本合同为准。

## Goodput 状态

- `reported`：任务提供了至少一个 goodput SLO，官方 raw JSON 包含本次
  `request_goodput`；
- `not_requested`：任务没有提供 goodput SLO，未传 `--goodput`；
- `missing`：任务要求 goodput，但官方 raw JSON 缺失、为 `null`、不可解析或无法证明
  来自本次 attempt。此状态对应非 `success` 报告。

`constraints_ms` 只记录任务实际提供的 `ttft`、`tpot`、`e2el` 约束，单位为毫秒。没有
提供的约束保持缺失状态。`request_goodput` 单位固定为 `req/s`，请求吞吐不映射为
goodput。

## 失败报告

失败时保留相同顶层结构：

- `status` 使用 `invalid`、`blocked` 或 `error`；
- `metrics` 中所有不可信数值为 `null`；
- `goodput.status` 保留 `reported`、`not_requested` 或 `missing` 的实际状态；
- `counts` 尽量保留官方 raw JSON 中的请求、完成和失败计数；
- `source.raw_result` 和 hash 在文件存在且可读时保留，否则为 `null`；
- `failure` 在 `status` 非 `success` 时给出本次 attempt 的主失败分类；成功时为 `null`；
- `oom` 始终存在。没有 OOM 时使用 `detected=false`、`phase=none`、`type=null` 和空的
  `evidence`；
- `errors` 至少记录 `stage`、`type`、`message` 和相关 artifact 路径；
- 清理失败单独记录在 `cleanup.status`，与性能状态分别表达。

## OOM 失败报告

OOM 报告同时使用 `failure` 和 `oom` 两个区块。`failure` 用于下游按阶段和类型分流，
`oom` 用于保留统一的检测结果和证据：

```json
{
  "status": "error",
  "failure": {
    "stage": "benchmark",
    "type": "runtime_oom",
    "message": "正式压测请求期间检测到运行时显存不足",
    "artifacts": ["<benchmark-log>", "<service-log>"]
  },
  "oom": {
    "detected": true,
    "phase": "benchmark",
    "type": "runtime_oom",
    "evidence": [
      {
        "artifact": "<benchmark-log>",
        "pattern": "out of memory",
        "observed_at": "<时间戳或 null>"
      }
    ]
  },
  "metrics": {
    "source": "vllm_bench_serve",
    "TTFT": {"unit": "ms", "P50": null, "P95": null},
    "TPOT": {"unit": "ms", "P50": null, "P95": null},
    "E2EL": {"unit": "ms", "P50": null, "P95": null}
  }
}
```

字段规则：

- `failure` 为失败 attempt 的主分类对象，至少包含 `stage`、`type`、`message` 和
  `artifacts`。`errors` 可继续记录更细的诊断条目。
- `oom.detected=true` 时，`oom.phase` 和 `oom.type` 只能分别使用 `benchmark` 和
  `runtime_oom`。该状态只表示正式 `vllm bench serve` 请求期间发生的运行时 OOM。
- 静态预检、服务启动和最小请求验收阶段的 OOM 由部署报告记录；benchmark 未启动时不
  生成 `runtime_oom`。
- `evidence` 至少包含一个当前 attempt 的日志或进程 artifact、匹配摘要和时间信息；不能
  使用历史 attempt 日志推断本次 OOM。
- OOM attempt 的 TTFT、TPOT、E2EL、吞吐和 goodput 数值不得由部分样本、日志或自定义
  公式补齐。`cleanup.status` 独立表达清理是否成功。

必须区分以下错误类型：

- `service_identity_invalid`
- `runtime_oom`
- `cli_incompatible`
- `workload_not_replayable`
- `raw_result_not_found`
- `raw_result_ambiguous`
- `raw_result_invalid`
- `model_mismatch`
- `request_count_mismatch`
- `benchmark_incomplete`
- `metrics_missing`
- `goodput_missing`
- `dataset_contract_invalid`
- `dataset_mode_invalid`
- `dataset_root_invalid`
- `dataset_not_found`
- `dataset_prepare_failed`
- `dataset_invalid`
- `dataset_insufficient`
- `custom_dataset_invalid`
- `hf_dataset_invalid`
- `cleanup_failed`

部署成功、服务健康和 `service_identity_valid=true` 只表示服务层状态；官方 benchmark
完整性由 raw result 和指标门禁确认。
