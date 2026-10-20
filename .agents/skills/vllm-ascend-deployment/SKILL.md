---
name: vllm-ascend-deployment
description: 在外部 Provider 冻结的模型、设备、端口和输出合同内，完成 vLLM-Ascend 环境预检、静态 OOM 检查、服务启动、健康验收和部署报告。
---

# vLLM-Ascend 部署技能

本 skill 负责一次独立 deployment attempt。Provider 提供模型、可见 NPU、端口候选、运行
参数和 attempt 目录；本 skill 不下载模型、不扫描未授权资源、不负责搜索调度或性能评估。

## 唯一报告和内存聚合

部署侧唯一结构化输出是 `<attempt-dir>/deployment_report.json`。attempt 开始时必须复制
[`references/deployment-report-template.json`](references/deployment-report-template.json)，并在
内存中持有报告对象。报告管理不读取中间 JSON，也不为阶段交接创建部署侧 JSON。

报告使用 `vllm-ascend-deployment-report-v2`，由以下固定区块组成：

- `input`：Provider 原始 payload、规范化对象和配置路径/SHA256/架构摘要；
- `environment`：环境预检的结构化事实、摘要、端口工具和有界诊断尾部；
- `stages.input_validation`、`static_oom`、`tp_resolution`、`resources`、`service`、`cleanup`：
  各阶段的结构化结果；
- `artifacts.probe`：probe skill 外置生成的环境、sandbox 和完整 help 文件的唯一索引；
- `artifacts.raw_logs`：完整服务日志和命令 transcript 的路径及必要元数据；
- `failure`、`timings`：失败事实和耗时。

`environment` 不得包含 probe/sandbox/help 路径；同一个非空路径在报告中只能出现一次。
`failure.artifacts` 只能使用 `artifacts.*` 逻辑引用。`model_config.parsed` 只保存静态 OOM
和运行所需的架构摘要，不复制完整 `config.json`。

部署侧不得生成或引用 `deployment.json`、`static-oom-input.json`、`static-oom-report.json`、
`npu-resource-probe.json`、`port-resource-probe.json`、`acceptance-checks.json` 或
`cleanup.json`。完整 probe/help/服务日志仍可由对应技能或服务流程外置保存。

## Checkpoint 和原子写入

只在以下关键节点调用 `scripts/report_io.py` 的 `atomic_write_report`：

1. 复制模板，填充 `attempt`，写入 `status=running`；
2. 输入校验、静态 OOM 和 TP 决策完成；
3. 资源检查和服务验收完成；
4. 清理完成或受控失败退出。

辅助脚本是无状态的，只负责同目录临时文件、刷新、`os.replace`、有界尾部和 SHA256；agent
负责持有对象和阶段编排。普通命令 stdout/stderr 尾部最多 4 KiB，服务日志尾部最多 8 KiB。
强制终止时保留最近一次 `status=running` 检查点，并由 `attempt.updated_at` 表明尚未完成。

## 临时脚本归档

agent 为本次部署生成的临时 Bash/Python 脚本、命令包装器和可复用诊断脚本，必须保存到
Provider 提供的 `<attempt-dir>/scripts/` 目录下。目录不存在时可以在 attempt 初始化阶段
创建；脚本应使用稳定、可识别的文件名，并保留必要的参数说明，便于后续同类任务复用和审计。
这些脚本不得存储在 `/tmp`，也不得只留在 shell 历史或未命名的临时文件中。`/tmp` 仅可用于
真正短生命周期的运行时临时数据（例如受控的 socket 或进程临时文件），任务完成后应清理。

## Provider 输入

Provider 至少提供：

- `model.model_path`、`model.served_model_name`；
- `resources.visible_devices`、`resources.port_candidates`，可选 `preferred_port`；
- 可选 `runtime.tp`、`runtime.dp`、`runtime.max_num_seqs` 和
  `runtime.max_num_batched_tokens`；缺少 TP 时进入最小 TP 推荐；
- `outputs.attempt_dir`，且必须是本次 attempt 的绝对路径；agent 生成的临时脚本统一归档到
  该目录下的 `scripts/` 子目录。

输入字段、默认值和硬约束见 [input-contract.md](references/input-contract.md)。

## 执行流程

1. **初始化报告和环境 probe。** 复制 canonical 模板并立即写入 running checkpoint。调用
   [vllm-ascend-probes](../vllm-ascend-probes/SKILL.md) 生成唯一的
   `environment_probe.json`、`sandbox-detection.json`、`vllm serve --help` 和
   `vllm bench serve --help` 外置文件。将 probe 的结构化摘要写入 `environment`，将路径只
   写入 `artifacts.probe`。只有 probe 通过才继续；失败写入 `failure` 并完成失败 checkpoint。
2. **输入校验。** 调用 `scripts/input_contract.py`。它只输出规范化 JSON，不创建目录或文件；
   agent 将对象写入 `input.normalized` 和 `stages.input_validation`。读取 `config.json` 的
   原始字节只进行一次，同时完成解析和 SHA256；将有限架构摘要写入 `input.model_config.parsed`。
3. **静态 OOM。** 从内存中的规范对象、配置摘要和资源事实组成 payload，通过 stdin 调用
   `scripts/static_oom_precheck.py`。脚本只输出结果，不接受 `--output` 或
   `--input-artifact`。结果直接写入 `stages.static_oom`；每个 attempt 只执行一次。
4. **TP 决策。** 推荐阶段不启动服务、不占用 NPU 或端口。若得到可行最小 TP，使用
   `input_contract.py --resolve-tp <tp> --resolution-source static_oom` 通过 stdin 重验并返回
   新对象，更新 `stages.tp_resolution.before/after`。必须记录
   `static_oom_reused=true`、`recheck_performed=false`，不得再次调用 OOM 预检。
5. **资源检查。** 复用 probe 已验证的端口探测工具和同一执行上下文，检查 Provider 可见
   NPU 和端口。结构化结果直接写入 `stages.resources`，不创建资源 JSON。设备不足、端口
   不可用或状态无法确认时立即写失败报告并退出，不等待资源释放或自动换卡/换端口。
6. **服务和验收。** 使用已选设备和端口启动 `vllm serve`，完整命令 transcript、服务日志
   和 PID 文件只保留在 `artifacts.raw_logs`，必要尾部写入 `stages.service.diagnostics_tail`。
   启动等待不得单纯依赖固定 `timeout` 阻塞：在有界启动截止时间内按短间隔轮询进程状态、
   服务日志新增内容和健康接口。日志按上次读取位置增量读取，不要每次轮询重新读取完整日志。
   一旦发现 `ERROR`、`FATAL`、`Traceback`、OOM、进程退出或其他明确启动失败证据，立即停止
   等待并记录失败，不要继续等待剩余 timeout；健康接口确认就绪后立即结束轮询并继续验收
   `/v1/models`、served model identity 和最小请求，结果写入 `stages.service`。只有轮询截止
   仍未就绪时才记录 `startup_timeout`，同时保留日志尾部和进程状态。服务启动或最小请求
   OOM 必须记录受控 `failure`。
7. **清理和最终 checkpoint。** 停止服务、释放边界内资源并记录 `stages.cleanup`。成功或
   受控失败都更新 `attempt.updated_at`，写出唯一最终 `deployment_report.json`。

## 失败边界

输入失败、环境失败、静态 OOM reject/unknown、资源失败和服务验收失败都不得伪造成功；
未执行阶段保留 `not_run`、`skipped` 或 `null`。受控失败必须填写 `failure.stage`、
`failure.type`、`failure.message`，并使用逻辑 artifact 引用。服务启动阶段必须优先使用日志
轮询结果判断失败；`timeout` 只是最后的上限保护，不能替代运行时日志和进程检查。
`service_ready` 只代表部署层健康、模型身份和最小请求通过，不代表性能 benchmark 或参数最优。

## 参考资料

- [报告模板](references/deployment-report-template.json)
- [报告约束](references/report-schema.md)
- [输入契约](references/input-contract.md)
- [静态 OOM 规则](references/static-oom.md)
- [资源预检](references/resource-probes.md)
- [报告校验脚本](scripts/report_schema.py)
- [无状态报告 I/O](scripts/report_io.py)
