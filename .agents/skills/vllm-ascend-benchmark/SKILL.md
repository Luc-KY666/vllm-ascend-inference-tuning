---
name: vllm-ascend-benchmark
description: 在外部 Provider 冻结的服务和负载合同上使用官方 vllm bench serve 完成一次独立压测，并交付 benchmark_report.json。
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
阶段和按部署合同约定的清理；部署失败时不执行压测。部署报告为
`resource_preflight_failed`（Provider 可见卡中空闲数量不足 `TP * DP`，或端口不可用）时，
本次 benchmark attempt 必须立即写出失败报告（`status=error`，性能字段为 `null`）并退出，
同样不得等待设备释放、不得轮询或重新探测 NPU/端口。

路径 A 的 Bash 生命周期由唯一 `environment_probe.json` 中的执行上下文决定。只有在 probe
要求提权，或最终 `execution_context=elevated` 时，部署、服务验收、正式
`vllm bench serve`、raw result 校验、清理和 `benchmark_report.json` 写入才必须在同一个
提权 Bash、同一网络命名空间和受保护环境快照中串行完成；部署返回 `service_ready` 不能
表示该提权 Bash 可以退出。若 probe 报告 `status=pass`、`sandbox.status=not_detected` 且
`execution_context=current`，上述阶段允许使用多个普通 Bash 调用，但每次交接都必须读取
deployment v2 的 service-ready checkpoint，并重新验收服务 URL、PID、模型身份和健康状态。
不得把 Bash 调用次数本身当作错误；服务交接失败、服务不可达或提权生命周期丢失时才记录
attempt 错误。

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
- 允许写入的 raw benchmark 和报告路径，且最终报告必须位于 attempt 目录内。

缺失、矛盾或无法冻结的输入必须在 workload 或服务验收阶段报告为 `invalid`，不得用宿主
默认值替代 Provider 事实。路径 A 未单独提供 tokenizer 时，使用模型路径作为 tokenizer
来源，并把解析后的来源写入报告。

## 执行权限

两条服务路径都直接复用 [vllm-ascend-probes](../vllm-ascend-probes/SKILL.md) 生成的唯一
`environment_probe.json`。该 artifact 是 benchmark 的唯一环境事实，包含 CANN、Ascend
runtime、依赖、两个 vLLM CLI、HTTP、沙箱原始证据和已验证的端口探测工具。benchmark 不再
自行执行环境、CANN、sandbox、端口工具发现或 `vllm bench serve --help=all`。

路径 A 从部署报告的 `artifacts.probe.environment_probe` 引用同一个
`environment_probe.json`，从 `artifacts.probe.bench_help` 复用
`bench-serve-help-all.txt`，并从 probe 文件本身复用 `status`、`sandbox.status`、
`execution_context`、shell fingerprint 和 `port_probe_tool`。路径 B 在 benchmark attempt
开始时调用 probe 一次。probe 初始状态为 `escalation_required`，或最终执行上下文为
`elevated` 时，服务验收、数据集校验、正式 benchmark、清理和报告写入都必须继续使用同一
个提权 Bash；不得再次请求提权、启动第二个 Bash、重新发现端口工具或切换网络命名空间。
只有 `status=pass`、`sandbox.status=not_detected` 且 `execution_context=current` 才允许
后续阶段跨普通 Bash 调用执行。`status=failed`、`sandbox.status=unknown` 或上下文仍为
`unknown` 时写出结构化 preflight failure，不进入正式 benchmark；不得把状态不确定当作
无沙箱隔离。

平台拒绝、提权失败或状态无法确认时写出结构化 preflight failure；不得把沙箱内的 `EPERM`、
驱动错误、零设备数或端口不可达当作真实 Provider 资源结论。路径 A 的部署失败沿用部署
报告的事实与结论，并在当前执行上下文允许的 Bash 中立即写出 benchmark 自己的失败报告后退出，
不等待设备或端口状态变化；路径 B 写出 benchmark 的失败报告且不执行正式测量。

## 报告、raw result 和脚本

attempt 开始时从 [`references/benchmark-report-template.json`](references/benchmark-report-template.json)
加载报告对象，创建 `<attempt-dir>/scripts/`，并立即通过
`scripts/report_io.py` 的 `atomic_write_report` 写出 `benchmark_report.json` 的 `running`
checkpoint。报告对象始终由当前执行上下文持有，不读取中间报告 JSON 作为阶段输入。

只在以下节点原子写入报告：初始化；preflight、服务验收和 workload 冻结完成；raw result
校验完成；清理完成或受控失败退出。普通命令输出只保留有限诊断尾部，不写入完整 stdout、stderr、
服务日志或 command transcript。

正式命令固定使用 `--save-result --result-dir <attempt-dir> --result-filename
raw_benchmark_result.json`，且不使用 `--save-detailed`。该 raw JSON 是本次官方测量的唯一证据，
报告只记录其路径和 SHA256，以及从中直接提取的必要指标。

### 临时脚本归档

agent 为本次 benchmark 生成的所有临时 Bash/Python 脚本、命令包装器和可复用诊断脚本，
必须保存到 Provider 提供的 `<attempt-dir>/scripts/` 目录下。目录不存在时可以在 attempt
初始化阶段创建；脚本应使用稳定、可识别的文件名（如 `benchmark_command.sh`、
`validate_dataset.py`、`extract_metrics.py`），并保留必要的参数说明，便于后续同类任务
复用和审计。

这些脚本不得存储在 `/tmp`，也不得只留在 shell 历史或未命名的临时文件中。`/tmp` 仅可用
于真正短生命周期的运行时临时数据（例如受控的 socket 或进程临时文件），任务完成后应清理。

归档到 `<attempt-dir>/scripts/` 的脚本应包括但不限于：

- 完整的 benchmark 命令（含所有参数和环境变量设置）；
- 数据集校验脚本；
- 指标提取或后处理脚本；
- 服务验收检查脚本；
- 其他可复用的诊断或辅助脚本。

## CLI 与运行时语义

以下语义直接作为 workload 合同的一部分冻结，执行时不通过读取 vLLM 源码临时确认：

- CLI 能力检查由 `vllm-ascend-probes` 只执行一次 `vllm bench serve --help=all`；禁止 benchmark
  阶段先运行裸 `--help`、按子命令重复加载 help，或因为输出截断而启动第二次 serve 进程。
  完整输出落盘为 `<attempt-dir>/bench-serve-help-all.txt`，所有需要的参数必须从这份 artifact
  解析。
- `--base-url` 只包含 scheme、host 和 port（例如 `http://127.0.0.1:8117`），`--endpoint`
  只包含以 `/` 开头的 HTTP 路径（例如 `/v1/completions` 或 `/v1/chat/completions`），不能把
  完整 URL 或重复的 `/v1` 拼进 endpoint。endpoint 必须与 `--backend` 的请求函数匹配：
  completions backend 的最终 URL 以 `completions` 结尾，chat backend 以 `chat/completions`
  结尾（其他 backend 同理遵循其 help 中的后缀）。执行前用最终拼接出的 URL 做一次服务验收；
  不要为了确认 endpoint 语义去读取 vLLM 源码。
- `generation.temperature` 由 Provider 显式冻结时按冻结值传入 `--temperature`；Provider
  未显式冻结时使用[负载契约](references/workload.md)的默认值 `--temperature 0`（贪心、
  可重复），并在 workload artifact 中标记为 defaulted。仅当 Provider 显式冻结为 `null`
  时命令不传 `--temperature`，由服务端/模型 API 决定，不能把 CLI 未设置值假定为 `0`；
  报告必须记录实际传入值及其来源。
- 对 `selection=hf` 或预置长 Prefill，使用 `--dataset-name hf` 和语义数据集 ID 的
  `--hf-name`。`likaixin/InstructCoder` 必须通过 `--hf-name likaixin/InstructCoder` 触发
  `InstructCoderDataset` adapter，即使 `--dataset-path` 指向本地 cache；不要用本地目录名
  替代该 ID，也不要为此读取源码。
- 版本用 `importlib.metadata.version("vllm")` 与
  `importlib.metadata.version("vllm-ascend")`；源码位置用模块 `__file__` 或
  `importlib.util.find_spec`。可编辑安装可能位于 `/vllm-workspace`，禁止在
  `site-packages` 下用 `find` 猜路径，也不要访问 `__version__` 属性。

## 执行流程

1. 路径 A 调用部署 skill，并直接复用部署报告
   `artifacts.probe.environment_probe` 和 `artifacts.probe.bench_help` 指向的文件；本 skill 不重复执行环境、CANN、
   NPU 或端口工具探测。若 probe 的执行上下文为 `elevated`，必须把整个服务生命周期
   （启动→验收→压测→清理）放进同一个提权 Bash，该调用只有在 benchmark、清理和报告交付
   结束后才能退出；若为 `current` 且无沙箱隔离，则允许部署和 benchmark 分属不同普通 Bash
   调用，但必须通过 deployment v2 service-ready checkpoint 完成服务交接。
   路径 B 在 benchmark attempt 开始时调用一次 [vllm-ascend-probes](../vllm-ascend-probes/SKILL.md)，
   只有 `environment_probe.json.status=pass` 才能继续；后续阶段复用该 artifact 和同一个
   执行上下文。
2. 按 Provider 服务路径处理服务：路径 A 仅在部署报告为 `service_ready` 时接管服务；
   在同一提权 Bash 内，或在 current 上下文的当前普通 Bash 中验证进程、健康接口、`/v1/models`、
   served model identity 和最小请求。路径 B 在已确定的同一执行上下文中验证交付的 URL、端口
   可达性、健康接口、模型身份和最小请求；current 上下文允许该验证与后续 benchmark 分属
   不同普通 Bash。服务阶段失败时写出失败报告，不进入测量。
3. 按[负载契约](references/workload.md)解析 `selection`、`scenario`、数据集和
   `benchmark_mode`，验证数据集 split、版本/hash 和有限样本可读性，并准备官方 CLI 参数。
   HF 数据集不得为了计算 108,391 等全量样本数而遍历到 EOF；只读取 split 元数据并抽样一个
   有界窗口（建议首 8 条，最多 32 条），记录 `validation_sample_count`、索引和错误。Custom
   JSONL 同样使用有界抽样，除非 Provider 另行要求阈值检查；解析未完成或抽样不可读时不执行
   benchmark。
4. 在已确定的执行上下文中按[官方测量方法](references/measurement.md)复用 probe 已取得的
   `vllm bench serve --help=all` artifact，
   构造并执行一条唯一命令。显式请求 TTFT、TPOT、E2EL 的 P50/P95；Provider 提供
   goodput SLO 时追加 `--goodput`。
5. 只选择本次命令产生的唯一、可解析 `raw_benchmark_result.json`，按[报告结构](references/report-schema.md)
   核验模型身份、请求计数、完成/失败计数、指标和 goodput，再写出唯一
   `benchmark_report.json`。
6. 路径 A 在 elevated 上下文中，必须在 raw result 校验、清理 artifact 和
   `benchmark_report.json` 写入完成前保持同一个提权 Bash；若该调用在正式请求期间退出，
   先记录服务日志、进程和连接错误，再按请求不完整处理。current 上下文允许 Bash 边界变化，
   但服务交接失败或连接失败仍按请求不完整处理；不得把连接失败的零值指标当作有效性能结果。
7. 正式请求阶段发生 OOM 时，记录当前 attempt 的日志或进程 artifact，并报告
   `status=error`、`failure.type=runtime_oom`、`oom.phase=benchmark`。服务启动或最小
   请求阶段的 OOM 由部署报告负责。
8. 无论成功、输入无效、CLI 不兼容、服务失败、请求不完整、raw result 缺失、OOM 或清理
   失败，都写出可解析的 `benchmark_report.json`。不可信性能字段保持 `null`，清理状态
   与性能状态分开记录。

## 输出与边界

- `benchmark_report.json` 只表示本次 attempt，不表达参数最优、候选通过或最终推荐。
- attempt 目录中的持久化文件仅限报告、官方 `raw_benchmark_result.json`、允许的 probe/deployment
  artifact 和 `scripts/` 归档；benchmark 不生成额外的 JSON 阶段交接文件。
- 性能指标只能直接映射官方 raw JSON；不得从日志、SSE、请求时间戳或部分样本重算。
- `service_identity_valid=true` 只表示服务验收通过，不表示 benchmark 成功。
- 每次 attempt 使用独立目录，既有 attempt artifact 不得覆盖。
- 路径 A 在 elevated 上下文中，服务进程不得脱离持有整个服务生命周期的提权 Bash；清理和
  最终报告完成后该调用才能退出。在 current 且无沙箱隔离的上下文中，服务可以由部署 Bash
  交接给后续普通 Bash，但必须通过 deployment v2 service-ready checkpoint 和服务验收完成
  交接。提权 Bash 提前退出、服务交接失败或连接失败必须作为 attempt 错误记录，不能重用其
  部分结果。
- `preflight` 必须引用本次唯一的 `environment_probe.json` 和（路径 A）部署报告；路径 A
  的 probe 路径只能从 `deployment_report.artifacts.probe.environment_probe` 读取，完整
  benchmark help 只能从 `deployment_report.artifacts.probe.bench_help` 读取；
  路径 A 的设备和端口只能从 `deployment_report.stages.resources.devices`、
  `deployment_report.stages.resources.ports` 读取；服务 URL、PID 和模型身份只能从
  `deployment_report.stages.service` 读取。部署报告必须严格为 v2，不得读取旧的
  `inputs.*`、`model`、`deployment`、`resources`、`preflight` 或 `checks` 别名。
- `npu-smi info` 只允许出现在沙箱隔离检测和部署 skill 在启动 `vllm serve` 前的可用卡探测中；
  benchmark 的服务验收、数据集校验和正式测量不得再次采集。
- NPU、端口、模型、请求参数和质量门槛均来自 Provider；benchmark 本身不扫描或替换未授权资源。
- 具体状态、失败类型、OOM 证据、goodput 和清理字段以 `references/report-schema.md` 为准。

## 参考资料

- [报告模板](references/benchmark-report-template.json)
- [报告约束](references/report-schema.md)
- [负载契约](references/workload.md)
- [官方测量方法](references/measurement.md)
- [报告 I/O](scripts/report_io.py)
- [报告校验](scripts/report_schema.py)
