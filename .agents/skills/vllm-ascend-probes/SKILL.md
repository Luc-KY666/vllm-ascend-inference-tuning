---
name: vllm-ascend-probes
description: 在一次固定的执行上下文中快速探测 CANN/Ascend runtime、Python 依赖、vllm serve、vllm bench serve 和可复用的端口探测工具，并生成唯一 environment_probe.json。
---

# vLLM-Ascend 环境探测

本 skill 只负责一次 attempt 的环境能力确认，不启动或停止模型服务，不分配 NPU，不选择
Provider 端口。部署和 benchmark 都只读取本 skill 生成的 `environment_probe.json`，不得
重复实现 CANN、sandbox、依赖或 vLLM CLI 检查。

## 固定执行规则

环境探测阶段不读取 vLLM 源码、不猜测安装路径、不调用裸 `--help`、不重试失败命令，也不
要求 agent 临场选择探测方式。直接在当前长生命周期 Bash 中执行一次脚本：

```text
set -euo pipefail
PYTHON_BIN="$(command -v python3 || command -v python)"
"$PYTHON_BIN" <skill-root>/scripts/probe_environment.py \
  --attempt-dir <attempt-dir> \
  --provider-path managed_deployment|predeployed \
  --execution-context current \
  --probe-phase initial \
  --timeout 30 \
  --cli-timeout 180 \
  --output <attempt-dir>/environment_probe.json \
  --sandbox-output <attempt-dir>/sandbox-detection.json \
  --serve-help-output <attempt-dir>/vllm-serve-help-all.txt \
  --bench-help-output <attempt-dir>/bench-serve-help-all.txt
```

脚本固定检查 Python、`vllm`、`torch`、`torch_npu`、`vllm_ascend`、NPU runtime、
`npu-smi info`、`libascendcl.so`、`libatb.so`、Python HTTP 能力、`vllm serve --help=all`、
`vllm bench serve --help=all` 和端口探测工具。CANN 检查全部在本 skill 内完成；调用方只
判断 `environment_probe.json.status`。

普通命令默认超时为 30 秒；`vllm serve --help=all` 和 `vllm bench serve --help=all` 使用独立的
180 秒超时，以覆盖 vLLM-Ascend 导入和初始化时间。命令的完整规则、artifact schema、失败类型和 shell 生命周期见
[probe-contract.md](references/probe-contract.md)。

## 沙箱与提权

初始 probe 返回 `status=escalation_required` 时，先保留当前
`sandbox-detection.json`，再使用执行平台的 `sandbox_permissions=require_escalated` 启动
唯一一个提权 Bash。在该 Bash 中用完全相同的参数执行 `--execution-context elevated
--probe-phase elevated`，随后部署、资源检查、服务验收、benchmark、清理和报告写入都必须
继续使用该 shell。不得再次请求提权、启动第二个 Bash 或重新连接服务。

提权拒绝、提权失败、提权后隔离仍无法确认或 shell 句柄丢失时，probe 失败并停止后续流程。
沙箱内的 `EPERM`、零设备或端口不可达不能直接作为宿主资源结论。

## 输出

- `environment_probe.json` 是唯一环境检验结果；它包含依赖模块可用性、两个 vLLM CLI、HTTP、
  端口工具和执行上下文。成功探测只写结构化状态、版本、路径和必要参数，不把命令 stdout/stderr
  嵌入结果；失败时仅保留有限长度的 `diagnostic_tail`，完整的 vLLM help 仍写入单独 artifact。
- `sandbox-detection.json` 只保存沙箱探测的必要证据、失败诊断和初始/提权两次摘要快照。
- `vllm-serve-help-all.txt` 与 `bench-serve-help-all.txt` 保存完整命令输出，后续阶段必须
  复用，不得重复调用对应 help。
- 端口工具选择结果位于 `environment_probe.json.port_probe_tool`，选定工具只保留在该对象的
  顶层字段中；候选列表只记录名称、状态及失败诊断。资源阶段必须复用该工具；工具失效时报告
  `port_probe_tool_unavailable`，不在同一 attempt 中重新发现工具。
