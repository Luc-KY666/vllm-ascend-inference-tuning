---
name: vllm-ascend-benchmark
description: 在外部 Provider 冻结的服务和负载合同上使用官方 vllm bench serve 完成一次独立压测，并交付 benchmark_report.json。
---

# vLLM-Ascend 压测技能

本 skill 执行一次独立 benchmark attempt。Provider 冻结服务来源、模型身份、负载、
benchmark 模式、统计口径和唯一输出目录；本 skill 负责环境预检、服务验收、官方测量、
结果核验和报告交付。Search、参数搜索、候选调度和最终推荐不属于本 skill。

## 最终报告和临时数据

benchmark 侧唯一 canonical 结构化输出是 `<attempt-dir>/benchmark_report.json`。最终报告使用
[`references/benchmark-report-template.json`](references/benchmark-report-template.json) 的 v3
结构，只保留：

- `model`：本次压测使用的模型名；
- `load`：负载模式、selection 和场景；
- `dataset`：数据集模式、名称、路径、split、revision/hash 和适配器字段；
- `arrival`：请求到达模式、请求数、warmup、并发、到达率、burstiness 和轮次；
- `parameters`：实际传给 `vllm bench serve` 的配置参数，至少包括 backend、tokenizer、base URL、
  endpoint、seed、temperature、top_p、stream、include_usage、请求/并发参数、百分位参数和可选
  goodput SLO；其他实际传入的适配器参数放入 `additional`；
- `result`：请求计数、TTFT/TPOT/E2EL、吞吐和按要求计算的 goodput；未提供 SLO 时不生成
  `result.goodput`；
- `failure` 和 `errors`：失败阶段、失败原因和按发生顺序追加的错误。

报告不得包含 preflight、服务验收、probe/help/log 路径、raw result 路径或摘要、命令、
版本、OOM、清理、artifact、attempt、source 或阶段执行耗时/时间戳字段。阶段输入、数据集抽样、命令
输出和指标中间结果只保存在内存或 `<attempt-dir>/.work/<stage>/`；清理阶段必须删除 `.work/`
的全部内容。

`benchmark_report.json` 不在初始化时创建，也不从模板复制出 running 文件。agent 在内存中
收集服务、负载和 raw result 事实，清理完成后一次性组装最终对象，并通过
`scripts/report_io.py` 的 `atomic_write_report` 原子写出一次。写出完成后仅调用一次
`scripts/report_schema.py`；中间阶段禁止写报告和调用 schema 校验。校验返回非零时将其作为
本次 attempt 的报告交付错误结束，不重跑阶段、不新建 attempt、不得再次写报告或重复校验。

官方 `raw_benchmark_result.json`、probe/help、服务日志和脚本可以按 Provider 的外部审计合同
留在 attempt 目录中，但不得被最终报告引用；临时阶段文件不得留存。

## Provider 输入

Provider 选择以下一种服务路径，并同时提供公共负载输入。

### 路径 A：本次 attempt 尚未部署服务

至少提供：

- `resources.visible_devices`、`resources.port_candidates` 或冻结端口；
- `model.model_path`、`model.served_model_name`；
- 可选的 `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens`、dtype、量化和显存利用率；
- `outputs.attempt_dir`：只属于本次 attempt 的绝对输出目录；
- 与部署共用的 `environment_probe.json`、`vllm serve --help` 和
  `vllm bench serve --help` 执行上下文。

本路径必须先调用 [vLLM-Ascend 部署技能](../vllm-ascend-deployment/SKILL.md)完成服务启动、健康
检查、模型身份验收和最小请求验收；部署 Skill 必须按照自身流程在清理完成后一次性写出
`<attempt-dir>/deployment_report.json`，并只执行一次 deployment report schema 校验。
benchmark 只能读取该文件作为路径 A 的部署事实，不能自行拼装或替代该报告。只接受部署报告
`status=success`；从 `deployment_report.model`、`deployment_report.resources`、
`deployment_report.service` 和 `deployment_report.launch` 读取模型、实际设备/端口、接口和启动
参数。路径 A 的服务由本次 benchmark attempt 负责生命周期收尾：只要本次尝试启动过服务，
benchmark 就必须在正式压测命令返回后停止该服务并释放其设备、端口和关联进程；不得把停止动作
留给调用方、shell 退出或 deployment/provider 的隐含约定。部署报告缺失、schema 校验失败或
状态非 `success` 时立即记录 benchmark 失败并退出，不执行正式压测，但仍须清理已经启动的
路径 A 服务。

### 路径 B：Provider 已提供部署好的服务

至少提供：

- `service.base_url`、`service.endpoint`、`service.backend` 和 tokenizer 来源；
- `service.served_model_name`；
- 受 Provider 认可的部署报告或部署摘要；
- `outputs.attempt_dir`：只属于本次 attempt 的绝对输出目录。

本路径不启动、不停止、不改写已有服务。先验证 URL/端口可达、健康接口、`/v1/models`、
served model identity 和最小请求；任一项不通过时不执行正式压测。

### 公共负载输入

Provider 还必须提供[负载契约](references/workload.md)中的 selection、scenario、
benchmark_mode、数据集、请求数、warmup、最大并发、到达率、burstiness、轮次和可选 goodput
SLO。缺失、矛盾或无法冻结的输入必须报告为 `invalid` 或 `blocked`，不得用宿主默认值替代
Provider 事实。报告保留冻结后的 `load`、`dataset`、`arrival` 和实际 CLI `parameters`；不把
完整命令、raw 路径或阶段诊断写入报告。

## 执行权限和外部事实

两条服务路径都复用 Provider 冻定的唯一 `environment_probe.json` 和同一执行上下文。benchmark
不得从最终 deployment report 猜测 probe/help 路径，也不得在测量阶段重新探测沙箱、CANN、NPU
或端口工具。路径 A 的部署报告只提供模型、资源、服务接口和启动参数；probe/help 和服务日志
由 Provider 单独交付。

probe 要求提权或执行上下文为 `elevated` 时，服务验收、数据集校验、正式 benchmark、清理和
最终报告写入必须继续使用同一个提权 Bash、同一网络命名空间和环境快照。只有 probe 通过、无
沙箱隔离且上下文为 `current` 时才允许跨普通 Bash 调用。状态失败、未知或提权失败时写出
受控失败报告，不进入正式 benchmark。

## 临时脚本归档

本次 benchmark 生成的临时 Bash/Python 脚本、命令包装器和诊断脚本必须保存到
`<attempt-dir>/scripts/`，不得存储在 `/tmp` 或只留在 shell 历史中。脚本属于外部审计 artifact，
不属于 `.work/` 清理范围。

## CLI 与运行时语义

- CLI 能力检查由 probe 只执行一次 `vllm bench serve --help=all`；完整输出作为外部 artifact
  保留，benchmark 阶段不重复加载 help。
- `--base-url` 只包含 scheme、host 和 port；`--endpoint` 只包含 HTTP 路径，并与 backend
  匹配。执行前用最终拼接 URL 验收服务。
- 当 deployment report 或服务输入的 endpoint 为 `/v1/chat/completions` 时，正式命令必须显式
  使用 `--backend openai-chat`；不得省略该参数、使用 `vllm` 默认值或根据模型名称猜测 backend。
- `temperature`、`seed`、`top_p`、`stream`、`include_usage` 等生成参数严格按 [负载契约](references/workload.md)
  解析，实际传给 CLI 的值和来源写入最终报告的 `parameters`；不能把未传入 CLI 的宿主默认值
  伪装成实际参数。
- 正式命令固定使用 `--save-result --result-dir <attempt-dir> --result-filename
  raw_benchmark_result.json`，不使用 `--save-detailed`。raw JSON 只作为内存中的测量证据，
  不把路径、SHA256 或命令复制到最终报告。

## 执行流程

1. 路径 A 先调用 deployment skill，并等待其按 deployment skill 合同写出并校验唯一的
   `<attempt-dir>/deployment_report.json`；读取该 v3 报告的模型、资源、服务和 launch 字段。
   路径 B 读取 Provider 的服务输入。两条路径都复用唯一 probe 和执行上下文，不创建
   `benchmark_report.json`。路径 A 的部署报告未写出、不可解析、schema 校验失败或状态不是
   `success` 时，追加 benchmark 错误并进入清理和最终失败报告流程，不启动正式压测。
2. 验证服务 URL、健康接口、模型身份和最小请求。服务交接或明确瞬态连接故障最多在同一
   attempt 内修复一次，只重跑服务阶段并复用已完成的部署、probe 和冻结输入。
3. 按[负载契约](references/workload.md)解析 selection、scenario、数据集、请求到达窗口和
   `benchmark_mode`，对数据集做有界抽样验证；所有中间结果只保存在内存或 `.work/`。
4. 复用 probe 已取得的 `vllm bench serve --help=all`，根据服务 endpoint 构造并执行唯一官方
   benchmark 命令；`/v1/chat/completions` 必须显式加入 `--backend openai-chat`。
   显式请求 TTFT、TPOT、E2EL 的 P50/P95；Provider 提供 goodput SLO 时追加 `--goodput`。
5. 只选择本次命令产生的唯一、可解析 raw JSON，核验模型身份、请求计数、完成/失败计数、
   数据集摘要、到达参数、指标和 goodput；将冻结的负载字段、实际 CLI `parameters` 和最终
   结果写入内存中的报告对象。
6. 正式请求阶段发生 OOM、请求不完整、指标缺失或 raw result 无法归属当前命令时，记录失败
   阶段和原因；不可信性能字段保持 `null`。NPU 或端口不可用时立即停止、清理并退出；其他
   明确瞬态故障最多在同一 attempt 内修复一次，不改变冻结的模型、设备、端口或负载参数。
7. 无论成功、输入无效、CLI 不兼容、服务失败、请求不完整、OOM 或清理失败，都追加错误。
   路径 A 必须先停止本次 attempt 启动的服务并释放资源，再删除 `.work/`、组装最终 v3 对象并
   只写出一次 `benchmark_report.json`；随后仅调用一次 `scripts/report_schema.py`。路径 B 不得
   由本 skill 停止或修改 Provider 已提供的服务。

## 失败边界

失败报告必须填写 `failure.stage`、`failure.type`、`failure.reason`；每次执行错误都追加到
`errors`，错误项不得包含时间戳、完整 stdout/stderr、日志正文、raw 路径或临时文件路径。
最终报告不包含阶段执行耗时或时间戳，也不把清理状态或服务验收过程作为独立区块写入报告。

## 参考资料

- [报告模板](references/benchmark-report-template.json)
- [报告约束](references/report-schema.md)
- [负载契约](references/workload.md)
- [官方测量方法](references/measurement.md)
- [报告 I/O](scripts/report_io.py)
- [报告校验](scripts/report_schema.py)
