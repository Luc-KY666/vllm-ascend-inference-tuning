# 部署报告结构

`deployment_report.json` 是 deployment skill 唯一负责的结构化输出。完整的
`environment_probe.json`、`sandbox-detection.json`、两个 help 文件和服务日志仍由探针或
服务流程外置保存。规范模板唯一维护在
[`deployment-report-template.json`](deployment-report-template.json)；本文件只说明约束，
不再复制第二份完整 JSON。

## 顶层契约

报告必须使用 `vllm-ascend-deployment-report-v2`，并包含 `status`、`attempt`、`input`、
`environment`、`stages`、`artifacts`、`failure` 和 `timings`。`stages` 固定包含：

- `input_validation`
- `static_oom`
- `tp_resolution`
- `resources`
- `service`
- `cleanup`

各阶段将结构化结果直接写入报告。agent 在内存中持有完整对象，只在初始化、输入/静态 OOM/
TP 决策完成、资源/服务验收完成、清理或受控失败四个关键节点写回。

## 路径归属

- `environment` 只保存环境事实、摘要、已验证的端口工具和有界诊断尾部。不得出现
  `probe_path`、`sandbox_path` 或 probe/help 路径。
- `artifacts.probe` 是 probe 文件路径的唯一来源：`environment_probe`、
  `sandbox_detection`、`serve_help`、`bench_help`。
- `artifacts.raw_logs` 只保存完整服务日志或命令 transcript 的路径及必要元数据；日志尾部
  写入对应阶段，普通命令尾部最多 4 KiB，服务日志尾部最多 8 KiB。
- 同一个非空路径在报告中只能出现一次。`failure.artifacts` 只能写逻辑引用，例如
  `artifacts.raw_logs.server_log`，不得重复绝对路径。
- `model_config.parsed` 只保存静态 OOM 和运行所需的架构摘要，不复制完整 `config.json`。

## 阶段规则

`stages.input_validation` 保存校验结果、规范化输入和错误/警告；`stages.static_oom` 保存
预检输入、结果、推荐 TP 和 OOM 证据；TP 写回只更新 `stages.tp_resolution`，必须记录
`static_oom_reused=true`、`recheck_performed=false`。资源、服务验收和清理结果分别写入
`resources`、`service` 和 `cleanup`。

以下部署侧中间 JSON 不得创建或在报告中引用：
`deployment.json`、`static-oom-input.json`、`static-oom-report.json`、
`npu-resource-probe.json`、`port-resource-probe.json`、`acceptance-checks.json`、
`cleanup.json`。

## 生命周期和失败

每次 checkpoint 使用同目录临时文件、刷新并 `os.replace` 原子替换最终报告。受控失败必须
填写 `failure` 并完成最终 checkpoint；强制终止只保留最近一次 `status=running` 的检查点，
由 `attempt.updated_at` 表明报告尚未完成。未执行阶段使用 `not_run`、`skipped` 或 `null`，
不得伪造成功。

benchmark 只接受 v2，设备和端口从 `stages.resources` 读取，服务 URL、PID 和模型身份从
`stages.service` 读取，probe 路径从 `artifacts.probe` 读取。不提供 v1 兼容适配器；历史
v1 报告仅作为既有 artifact 保留。
