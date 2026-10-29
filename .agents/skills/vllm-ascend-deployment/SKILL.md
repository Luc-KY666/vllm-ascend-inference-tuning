---
name: vllm-ascend-deployment
description: 在外部 Provider 冻结的模型、设备、端口和输出合同内，完成 vLLM-Ascend 环境预检、静态 OOM 检查、普通或者PD分离服务启动、健康验收和部署报告。
---

# vLLM-Ascend 部署技能

本 skill 负责一次独立 deployment attempt。Provider 提供模型、可见 NPU、端口候选、运行
参数和 attempt 目录；本 skill 不下载模型、不扫描未授权资源、不负责搜索调度或性能评估。
本 skill 不得自行创建新的 attempt；需要新的 attempt 时只能由外部调用方重新发起。

## 部署模式

- 未指定 PD、Prefill/Decode、KV Transfer 或 Mooncake：使用 `ordinary`，启动一个
  `vllm serve`。
- 指定 `deployment.mode=single_node_pd` 或明确要求 PD 分离：使用 `single_node_pd`，
  启动 Prefill、Decode、Proxy；池化 `MultiConnector` 额外启动 Mooncake Master。

普通模式保持现有 v3 报告合同；PD 模式最终公共服务仍写入 v3 的 `service.base_url`，
指向 Proxy，P/D 角色命令和实例信息写入 `launch.parameters`。

## 最终报告和临时数据

部署侧唯一 canonical 结构化输出是 `<attempt-dir>/deployment_report.json`。最终报告使用
[`references/deployment-report-template.json`](references/deployment-report-template.json) 的 v3
结构，只保留：

- `model`：模型名称和权重路径；
- `resources`：实际使用的 NPU 和端口；
- `service`：backend、base URL、endpoint、健康/模型接口和 tokenizer；
- `launch`：所有实际传给 `vllm serve` 的参数，统一写入 `parameters`；
- `failure` 和 `errors`：失败阶段、失败原因和按发生顺序追加的错误。

报告不得包含阶段对象、probe/help/log 路径、原始输入、静态 OOM 结果、清理结果、PID、
artifact 索引、阶段执行耗时或时间戳字段。阶段输入、诊断输出和命令结果只保存在内存或
`<attempt-dir>/.work/<stage>/`；清理阶段必须删除 `.work/` 的全部内容。

`deployment_report.json` 不在初始化时创建，也不从模板复制出 running 文件。agent 在内存中
收集所有阶段事实，服务停止、资源释放和 `.work/` 清理完成后一次性组装最终对象，并通过
`scripts/report_io.py` 的 `atomic_write_report` 原子写出一次。写出完成后仅调用一次
`scripts/report_schema.py`；中间阶段禁止写报告和调用 schema 校验。校验返回非零时将其作为
本次 attempt 的报告交付错误结束，不重跑阶段、不新建 attempt、不得再次写报告或重复校验。

probe/help、服务日志、PID、脚本和官方 raw result 可以按 Provider 的外部审计合同留在 attempt
目录中，但不得被最终报告引用；临时阶段文件不得留存。

## 临时脚本归档

agent 为本次部署生成的临时 Bash/Python 脚本、命令包装器和可复用诊断脚本，必须保存到
Provider 提供的 `<attempt-dir>/scripts/` 目录下。目录不存在时可以在 attempt 初始化阶段
创建；脚本应使用稳定、可识别的文件名，并保留必要的参数说明，便于后续同类任务复用和审计。
这些脚本不得存储在 `/tmp`，也不得只留在 shell 历史或未命名的临时文件中；脚本不属于
`.work/` 清理范围。

## Provider 输入

Provider 至少提供：

- ordinary 模式的模型路径、served model name 和其它 vLLM 启动值统一放在
  `serve.argv`；`model.config_path` 仅可选地覆盖 config 校验路径；
- static OOM 所需的可靠权重事实放在 `model.weight_bytes` 或 `model.weight_gib`；MoE/量化
  模型缺少明确权重时 static OOM 返回 `unknown`；
- `resources.visible_devices`、`resources.port_candidates`、`resources.card_memory_bytes`，可选
  `preferred_port`；其中 `card_memory_bytes` 必须由 agent 从 Provider 冻结的硬件事实或当前
  环境 probe 填写为正整数，不能推算或省略。PD 每个节点还提供
  `resources.role_device_count`，表示该角色 DP 组的总设备数；它不替代单卡显存字段；
- ordinary 模式必须由 agent 根据最终 `vllm serve` 命令组装 `serve.argv`，并交给
  `scripts/input_contract.py` 与 probe 生成的 `vllm-serve-help-all.txt` 一起校验；
- `serve.argv` 的第一项是模型路径，TP、DP、mns、mnbt 和其它 vLLM 启动参数都从该数组
  提取；缺少 TP 时进入最小 TP 推荐；
- 可选 `service.backend`、`service.endpoint` 和 `service.tokenizer`；缺省时分别为
  `vllm`、`/v1/chat/completions` 和模型路径；
- PD 模式额外提供 `deployment.mode`、`deployment.connector` 以及 `pd.prefill`、
  `pd.decode`、`pd.proxy` 配置，字段见 [`references/pd-deployment.md`](references/pd-deployment.md)；
- `outputs.attempt_dir`，且必须是本次 attempt 的绝对路径；agent 生成的临时脚本统一归档到
  该目录下的 `scripts/` 子目录。

输入字段、默认值和硬约束见 [input-contract.md](references/input-contract.md)。

## 执行流程

1. **初始化内存上下文和环境 probe。** 只在内存中加载 canonical 模板作为最终字段骨架，
   不写 `deployment_report.json`。调用 [vllm-ascend-probes](../vllm-ascend-probes/SKILL.md)
   生成唯一的环境、sandbox、`vllm serve --help` 和 `vllm bench serve --help` 外置文件；
   只在内存中保留继续执行所需的摘要。probe 失败时记录失败阶段和错误，清理后走统一的最终
   报告写出路径。
2. **输入校验。** probe 成功后，agent 根据最终 ordinary `vllm serve` argv 组装 JSON，调用
   `scripts/input_contract.py --input <deployment-input.json> --serve-help-file
   <vllm-serve-help-all.txt> --runtime-output <.work>/runtime.json`。脚本先验证每个 option 是否存在于同一份 help artifact，
   普通 option 的 value 不在 input-contract 校验范围内，再对 TP、DP、mns、mnbt 等硬约束
   执行原有检查。PD 将每个 Prefill/Decode 实例展开为独立的
   `deployment_input_<role><rank>.json`，逐个调用同一个 `input_contract.py`；PD 设备池、
   端口、connector 和跨实例拓扑由 agent 校验。规范化结果只保存在内存；必要时把原始
   stdout 临时写入 `.work/`。每个节点的 input contract 调用只读取一次该节点的
   `config.json` 原始字节，同时完成解析和 SHA256。校验成功或缺少 TP 时额外输出一个
   runtime artifact，只包含规范化 runtime、config 摘要、digest 和资源；缺少 TP 时保留
   规范化对象，交给该节点的静态 OOM 推荐。
3. **静态 OOM。** ordinary 和 PD 都使用双输入调用：
   `scripts/static_oom_precheck.py --deployment-input <deployment-input.json> --runtime
   <runtime.json>`。脚本从 deployment input 读取模型权重和资源事实，从 runtime artifact
   读取已校验 runtime/config；agent 不再手工构造 OOM payload。PD 脚本根据 deployment mode
   将本地容量判断的有效 `runtime.dp` 设为 `1`，角色级 DP 仍由输入校验和 agent 拓扑校验
   处理。每个缺失 TP 的节点独立获得最小可行 TP；任一节点 reject、unknown 或无可行推荐
   都停止部署。双输入和输出如需落盘只能写入 `.work/static_oom/`，每个节点只执行一次。
4. **TP 决策。** 推荐阶段不启动服务、不占用 NPU 或端口。得到可行最小 TP 后，使用
   `input_contract.py --resolve-tp <tp> --resolution-source static_oom --input -` 仅对已缓存的
   normalized object 做轻量约束检查；不得再次执行完整输入校验、重新读取 `config.json` 或
   help 文件，或调用 OOM 预检。PD 对每个节点独立执行同样的轻量回写；除更新内存中的
   `runtime.tp` 外，同时把推荐的 `--tensor-parallel-size` 写回该节点的 `serve.argv`。
   同一角色的最终 TP 必须一致，否则在服务启动前报告拓扑不一致并失败。最终 TP 写入
   内存中的 `launch.parameters`，复用原有静态 OOM 结果。
5. **资源检查。** 复用 probe 已验证的端口探测工具和同一执行上下文，检查 Provider 可见
   NPU 和端口。用户指定了可见设备或端口时，只能在指定范围内选择；未指定时，从探测到的
   空闲设备和端口候选中选择。把实际选中的设备和端口写入内存中的 `resources`。
   若资源在服务启动时发生竞态占用或变为不可用，停止已启动的服务并在同一 attempt 内
   最多重新探测、选择并启动一次；只重做资源选择，复用环境 probe、输入校验、静态 OOM
   和 TP 决策结果。重选不得等待释放、扩大设备或端口范围，探测工具不可用或状态无法确认
   时直接失败。第二次资源选择或启动仍失败，记录错误、清理现场并退出；不得自行创建新的
   attempt。
6. **服务和验收。** ordinary 启动一个 `vllm serve`；PD 将 Prefill、Decode、Proxy 和可选
   Mooncake Master 作为同一启动组，按 [`references/pd-deployment.md`](references/pd-deployment.md)
   的依赖顺序启动。Prefill/Decode 只做进程、端口和 `/health` readiness；Proxy 不请求
   `/health`，只通过发送两个生成接口请求确认健康。所有业务请求都必须发送到 Proxy。把实际使用的所有 CLI 参数归一化
   到内存中的 `launch.parameters`。启动等待按短间隔轮询进程状态、日志新增内容和 P/D 的
   `/health` 接口；Proxy 不使用 health readiness，直接通过两个生成请求验收；
   发现明确错误、OOM、进程退出或服务验收
   失败时立即停止等待。普通模式必须依次通过 `/health`、`GET /v1/models`、
   `POST /v1/completions` 和 `POST /v1/chat/completions`；模型接口和两个生成接口都必须
   返回正确 served model identity，生成接口还必须返回非空文本。PD 模式的 Proxy 不检查
   `/v1/models` 或 `/health`，只通过 Proxy 验证两个生成接口；Prefill、Decode 仍检查
   `/health`。PD 生成请求必须在有界日志窗口中证明 Prefill 节点集合至少一个、
   Decode 节点集合至少一个出现同一请求的处理记录，不能只依据 Proxy 的 HTTP 成功判断链路。
   Completion 请求使用 `model`、唯一 marker、`max_tokens=1`、`stream=false`；chat 请求
   使用同一 `model`、带 marker 的单条 user message、`max_tokens=1`、`stream=false`。
   以上验收通过后，把最终服务接口写入内存中的 `service`。服务启动/健康阶段的明确非资源瞬态错误最多在
   同一 attempt 内修复一次，只重跑服务阶段并复用 probe、输入、OOM、TP、设备和端口结果。
7. **清理和最终报告。** 停止服务、释放边界内资源、删除 `.work/`，将模型、实际资源、
   服务接口、启动参数、失败信息和错误数组组装为最终 v3 对象；成功或受控失败都只写出一次
   `deployment_report.json`，随后仅调用一次 `scripts/report_schema.py`。

## 失败边界

输入失败、环境失败、静态 OOM reject/unknown、资源失败和服务验收失败都不得伪造成功。失败
报告必须填写 `failure.stage`、`failure.type`、`failure.reason`；每次执行错误都追加到
`errors`，错误项不得包含时间戳、完整 stdout/stderr、日志正文或临时文件路径。

资源竞态只允许按第 5 步在当前 attempt 内重选一次；第二次仍失败则追加错误并终止。
其它服务启动/验收瞬态错误仍最多修复一次。不得通过修改冻结的 TP、DP、设备、端口或模型
参数制造“修复”，也不得自行创建新的 attempt。最终报告不包含阶段执行耗时或时间戳。

## PD 分离部署

P/D 设备池必须互不重叠。未指定角色 TP 时，分别为每个节点选择静态 OOM 可行的最小 TP，
并要求同一角色最终 TP 一致；P/D 的 `/health` 只检查 HTTP 状态码 `200`，不检查响应体。
Proxy 脚本、命令模板、KV connector、Mooncake、后端列表、启动顺序、链路验收和清理规则见
[`references/pd-deployment.md`](references/pd-deployment.md)。

## 参考资料

- [报告模板](references/deployment-report-template.json)
- [报告约束](references/report-schema.md)
- [输入契约](references/input-contract.md)
- [静态 OOM 规则](references/static-oom.md)
- [资源预检](references/resource-probes.md)
- [PD 分离部署](references/pd-deployment.md)
- [报告校验脚本](scripts/report_schema.py)
- [报告 I/O](scripts/report_io.py)
