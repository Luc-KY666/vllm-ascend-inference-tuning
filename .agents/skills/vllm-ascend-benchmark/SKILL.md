---
name: vllm-ascend-benchmark
description: 在外部 Provider 冻结的服务和负载合同上使用官方 vllm bench serve 完成一次独立压测，并交付 benchmark_result.json。
---

# vLLM-Ascend 压测技能

本 skill 执行一次独立 benchmark attempt。Provider 冻结服务来源、模型身份、负载、
benchmark 模式、统计口径和唯一输出目录；本 skill 负责环境预检、服务验收、官方测量、
结果核验和报告交付。Search、参数搜索、候选调度和最终推荐不属于本 skill。

## Provider 输入

Provider 选择以下一种服务路径，并同时提供公共负载输入。

### 路径 A：本次 attempt 尚未部署服务

至少提供：

- `resources.visible_devices`：Provider 授权且可见的 NPU ID；
- `resources.port_candidates` 或冻结端口；
- `model.model_path`：可读的本地模型目录；
- `model.served_model_name`：服务身份中的模型名；
- 可选的模型运行参数，如 `tp`、`dp`、`max_num_seqs`、`max_num_batched_tokens`、
  `dtype`、量化和显存利用率；
- `outputs.attempt_dir`：只属于本次 attempt 的绝对输出目录。

本路径调用[vLLM-Ascend 部署技能](../vllm-ascend-deployment/SKILL.md)完成服务启动、
健康检查、模型身份验收和最小请求验收。部署成功后，当前 skill 接管该服务的 benchmark
阶段和按部署合同约定的清理；部署失败时不执行压测。

路径 A 的服务生命周期必须由本次 attempt 持有的长生命周期 shell 管理。调用部署 skill
时，部署、服务验收和后续 benchmark 必须在同一个 shell/session、网络命名空间和受保护环境
快照中串行完成；部署返回 `service_ready` 不能表示该 shell 已经可以退出。若使用了提权
shell，benchmark orchestrator 必须保留该提权 shell 的 session handle，并让它一直存活到
正式 `vllm bench serve`、raw result 校验、清理和 `benchmark_result.json` 写入全部完成后再
释放。不得在短命 shell 中启动服务后以 detached/`nohup` 方式把服务交给后续 shell，也不得
通过新的提权 shell 重新连接服务。

### 路径 B：Provider 已提供部署好的服务

至少提供：

- `service.base_url` 和必要的 endpoint；
- `service.served_model_name`；
- `service.tokenizer` 来源；
- `service.deployment_report`：对应部署报告路径或受 Provider 认可的部署摘要；
- `outputs.attempt_dir`：只属于本次 attempt 的绝对输出目录。

本路径不启动、不停止、不改写已有服务。先验证 URL/端口可达、`/health`、`/v1/models`、
served model identity 和最小请求；任一项不通过时不执行正式压测。

### 公共负载输入

Provider 还必须提供[负载契约](references/workload.md)中的：

- `selection`：`preset`、`hf` 或 `custom`；`preset` 必须同时提供 `scenario`；
- `benchmark_mode`：`stress` 或 `production`；
- 数据集路径、名称、split、revision/hash 和适配器专用参数；
- 请求数、warmup、最大并发、到达率、burstiness、轮次和生成参数；
- 可选的 goodput SLO（`ttft`、`tpot`、`e2el`，单位毫秒）；
- 允许写入的 raw benchmark、日志和报告路径，且最终报告必须位于 attempt 目录内。

缺失、矛盾或无法冻结的输入必须在 workload 或服务验收阶段报告为 `invalid`，不得用宿主
默认值替代 Provider 事实。路径 A 未单独提供 tokenizer 时，使用模型路径作为 tokenizer
来源，并把解析后的来源写入报告。

## 执行权限

Ascend 驱动设备节点、wildcard TCP bind 和 Provider 服务所在的网络命名空间可能不暴露给
默认沙箱。两条服务路径的沙箱责任不同：路径 A 由[vLLM-Ascend 部署技能](../vllm-ascend-deployment/SKILL.md)
在部署 attempt 开始时完成唯一一次统一隔离检测；路径 B 由本 skill 在 benchmark attempt
开始时完成唯一一次统一隔离检测。对应的 `sandbox-detection.json` 是后续阶段的唯一事实，
不得重复检测或从新的沙箱快照推翻它。

检测到隔离或状态不确定时，必须默认通过执行平台的
`sandbox_permissions=require_escalated` 请求启动提权 Bash，不先发送人工确认消息或停止
attempt。平台弹窗负责授权；授权成功后，环境预检、服务验收（路径 B）、CLI 检查和正式
`vllm bench serve` 必须复用同一个提权上下文。路径 A 还必须复用部署报告所记录的上下文，
并直接使用部署 skill 返回的 `selected_devices` 和 `selected_port`，不重新探测 NPU 或端口。
路径 A 的提权上下文是 attempt 级生命周期 owner：部署 skill 启动的 shell/session 在
`service_ready` 后继续阻塞或等待 benchmark 完成信号，直到清理结果和最终报告落盘；benchmark
阶段不得切换到另一个 `exec`、Bash、网络命名空间或环境快照。shell/session 提前退出、句柄
丢失或服务因此不可达时，当前 attempt 必须报告环境/服务生命周期错误，性能字段保持 `null`，
并使用新的 attempt 重试。

平台拒绝、提权失败或状态无法确认时写出结构化 preflight failure；不得把沙箱内的 `EPERM`、
驱动错误、零设备数或端口不可达当作真实 Provider 资源结论。路径 A 的部署失败沿用部署
报告；路径 B 写出 benchmark 的失败报告且不执行正式测量。

## CLI 与运行时语义

以下语义直接作为 workload 合同的一部分冻结，执行时不通过读取 vLLM 源码临时确认：

- CLI 能力检查只执行一次 `vllm bench serve --help=all`；禁止先运行裸 `--help`、按子命令
  重复加载 help，或因为输出截断而启动第二次 serve 进程。所有需要的参数必须从这份完整 help
  artifact 解析。
- `--base-url` 只包含 scheme、host 和 port（例如 `http://127.0.0.1:8117`），`--endpoint`
  只包含以 `/` 开头的 HTTP 路径（例如 `/v1/completions` 或 `/v1/chat/completions`），不能把
  完整 URL 或重复的 `/v1` 拼进 endpoint。endpoint 必须与 `--backend` 的请求函数匹配：
  completions backend 的最终 URL 以 `completions` 结尾，chat backend 以 `chat/completions`
  结尾（其他 backend 同理遵循其 help 中的后缀）。执行前用最终拼接出的 URL 做一次服务验收；
  不要为了确认 endpoint 语义去读取 vLLM 源码。
- `generation.temperature` 必须由 Provider 显式冻结并在支持时传入 `--temperature`。不传该参数
  时 CLI 值为未设置，最终默认由服务端/模型 API 决定，不能假定为 `0` 或贪心采样；报告必须记录
  是否传入。需要可重复的贪心请求时显式传 `--temperature 0`。
- 对 `selection=hf` 或预置长 Prefill，使用 `--dataset-name hf` 和语义数据集 ID 的
  `--hf-name`。`likaixin/InstructCoder` 必须通过 `--hf-name likaixin/InstructCoder` 触发
  `InstructCoderDataset` adapter，即使 `--dataset-path` 指向本地 cache；不要用本地目录名
  替代该 ID，也不要为此读取源码。
- 版本用 `importlib.metadata.version("vllm")` 与
  `importlib.metadata.version("vllm-ascend")`；源码位置用模块 `__file__` 或
  `importlib.util.find_spec`。可编辑安装可能位于 `/vllm-workspace`，禁止在
  `site-packages` 下用 `find` 猜路径，也不要访问 `__version__` 属性。

## 执行流程

1. 按[Ascend 环境探测](references/ascend-probes.md)的路径规则确定执行上下文。路径 A
   调用部署 skill，由其完成沙箱、Ascend 环境、NPU 和端口预检；本 skill 只验收部署报告，
   不重复执行这些探测。路径 A 必须先建立 attempt 级长生命周期 shell/session，再在该
   session 内调用部署 skill；该 session 只有在 benchmark、清理和报告交付结束后才能回收。
   路径 B 完成一次沙箱检测和 benchmark 环境预检，确认当前上下文可运行官方
   `vllm bench serve`，并将 artifact 写入 attempt 目录。路径 A 不再执行 `npu-smi info`；
   路径 B 的 `npu-smi info`（若需要用于本地沙箱隔离判定）只执行一次并复用其 artifact。
2. 按 Provider 服务路径处理服务：路径 A 仅在部署报告为 `service_ready` 时接管服务；
   在同一长生命周期 session 内验证进程、健康接口、`/v1/models`、served model identity
   和最小请求。路径 B 在已确定的同一执行上下文中验证交付的 URL、端口可达性、健康接口、
   模型身份和最小请求。服务阶段失败时写出失败报告，不进入测量。
3. 按[负载契约](references/workload.md)解析 `selection`、`scenario`、数据集和
   `benchmark_mode`，验证数据集 split、版本/hash 和有限样本可读性，并准备官方 CLI 参数。
   HF 数据集不得为了计算 108,391 等全量样本数而遍历到 EOF；只读取 split 元数据并抽样一个
   有界窗口（建议首 8 条，最多 32 条），记录 `validation_sample_count`、索引和错误。Custom
   JSONL 同样使用有界抽样，除非 Provider 另行要求阈值检查；解析未完成或抽样不可读时不执行
   benchmark。
4. 在已确定的执行上下文中按[官方测量方法](references/measurement.md)复用已经取得的
   `vllm bench serve --help=all` artifact，
   构造并执行一条唯一命令。显式请求 TTFT、TPOT、E2EL 的 P50/P95；Provider 提供
   goodput SLO 时追加 `--goodput`。
5. 只选择本次命令产生的唯一、可解析 raw JSON，按[报告结构](references/report-schema.md)
   核验模型身份、请求计数、完成/失败计数、指标和 goodput，再写出唯一
   `benchmark_result.json`。
6. 路径 A 在 raw result 校验、清理 artifact 和 `benchmark_result.json` 写入完成前不得释放
   lifecycle owner shell。若 shell 在正式请求期间消失，先记录服务日志、进程和连接错误，
   再按请求不完整处理；不得把连接失败的零值指标当作有效性能结果。
7. 正式请求阶段发生 OOM 时，记录当前 attempt 的日志或进程 artifact，并报告
   `status=error`、`failure.type=runtime_oom`、`oom.phase=benchmark`。服务启动或最小
   请求阶段的 OOM 由部署报告负责。
8. 无论成功、输入无效、CLI 不兼容、服务失败、请求不完整、raw result 缺失、OOM 或清理
   失败，都写出可解析的 `benchmark_result.json`。不可信性能字段保持 `null`，清理状态
   与性能状态分开记录。

## 输出与边界

- `benchmark_result.json` 只表示本次 attempt，不表达参数最优、候选通过或最终推荐。
- 性能指标只能直接映射官方 raw JSON；不得从日志、SSE、请求时间戳或部分样本重算。
- `service_identity_valid=true` 只表示服务验收通过，不表示 benchmark 成功。
- 每次 attempt 使用独立目录，既有 attempt artifact 不得覆盖。
- 路径 A 的服务进程不得脱离 attempt lifecycle owner shell；清理和最终报告完成后才能回收
  提权 shell。服务 shell 提前回收造成的连接失败必须作为 attempt 错误记录，不能重用其
  部分结果。
- `preflight` 必须引用本次使用的沙箱检测、benchmark 环境预检和（路径 A）部署报告；
  路径 A 的 `selected_devices`、`selected_port` 只能来自部署报告。
- `npu-smi info` 只允许出现在沙箱隔离检测和部署 skill 在启动 `vllm serve` 前的可用卡探测中；
  benchmark 的服务验收、数据集校验和正式测量不得再次采集。
- NPU、端口、模型、请求参数和质量门槛均来自 Provider；benchmark 本身不扫描或替换未授权资源。
- 具体状态、失败类型、OOM 证据、goodput 和清理字段以 `references/report-schema.md` 为准。
