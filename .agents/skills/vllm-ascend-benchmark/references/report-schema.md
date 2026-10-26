# Benchmark 最终报告结构

`benchmark_report.json` 是一次独立官方 benchmark attempt 的 v3 最终事实记录。报告只保存
用户需要复现和比较本次压测的模型、负载、数据集、请求到达合同和官方测量结果；服务验收、
probe、命令、raw 文件、OOM 证据、清理、artifact、版本和阶段计时/时间戳信息均不进入报告。

## 顶层字段

报告只允许包含：

- `schema_version`：固定为 `vllm-ascend-benchmark-report-v3`；
- `status`：`success`、`invalid`、`blocked` 或 `error`；
- `model.name`：本次压测模型名，与部署报告配合使用；
- `load`：`mode`、`selection` 和 `scenario`；
- `dataset`：数据集模式、名称、路径、split、subset、revision/hash 和适配器字段；
- `arrival`：请求到达模式、请求数、warmup、最大并发、到达率、burstiness 和轮次；
- `parameters`：实际传给 `vllm bench serve` 的配置参数；特殊适配器参数放在 `additional`；
- `result`：请求计数、TTFT/TPOT/E2EL P50/P95、吞吐和 goodput；
- `failure`：成功时为 `null`，失败时包含 `stage`、`type` 和 `reason`；
- `errors`：按发生顺序追加的错误数组。

不得添加 `attempt`、`preflight`、`service`、`workload`、`source`、`counts`、`metrics`、
`throughput`、`oom`、`cleanup`、`artifacts` 或 `timings` 等中间区块。

## 负载和结果

`load.mode` 记录 `stress` 或 `production`，`load.selection` 记录 `preset`、`hf` 或
`custom`，`load.scenario` 只在 preset 分支使用。`arrival` 镜像负载契约冻结的请求到达信息，
不得使用宿主默认值替换缺失字段。

`parameters` 记录实际传给 `vllm bench serve` 的值，包括 backend、tokenizer、base URL、endpoint、
seed 及其来源、temperature 及其来源、top_p、stream、include_usage、请求/并发参数、百分位参数
和可选 goodput SLO。`parameters.additional` 记录其他实际传入的适配器专用参数，但不得保存完整
命令、raw 文件路径、日志或版本信息。

`result` 中的 TTFT、TPOT、E2EL 只能直接来自官方 raw JSON；不得从日志、SSE、请求时间戳、
部分样本或自定义公式重算。没有 SLO 时不生成 `result.goodput`；提供 SLO 后才写入该对象，
其规则为：

- `reported`：至少提供一个 SLO，`slo_ms` 非空且 raw JSON 提供合法 `request_goodput`；
- `missing`：要求 goodput 但官方结果缺失或不可归属，报告不能为 `success`。

## 错误和写出

`errors` 每项至少包含 `stage`、`type`、`message`、`recoverable` 和 `resolved`，可选
`repair_action`。错误消息不得包含完整 stdout/stderr、日志正文、raw 路径、临时路径或时间戳。
失败终态还必须在 `failure` 中提供最终失败阶段和原因。

报告对象在内存中组装，清理 `.work/` 后通过 `scripts/report_io.py` 的
`atomic_write_report` 写出一次。写出完成后只调用一次 `scripts/report_schema.py`；中间阶段不
写报告、不运行 schema 校验，校验失败不重跑 attempt。
