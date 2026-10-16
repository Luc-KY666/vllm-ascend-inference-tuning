# Benchmark 环境与沙箱探测

环境探测只确认本次 attempt 能够在冻结的执行上下文中运行官方 benchmark。它不分配 NPU、
不绑定 Provider 端口、不启动或停止模型服务。路径 A 的 NPU、端口和服务资源事实由部署
skill 负责；路径 B 只验证交付服务的网络可达性，不把远程服务的端口或 NPU 当作本地资源
扫描。

## 路径和 artifact

每次 attempt 必须有一份唯一的 `sandbox-detection.json`：

- 路径 A：由[vLLM-Ascend 部署技能](../../vllm-ascend-deployment/references/ascend-probes.md)
  在部署 attempt 开始时生成。benchmark 读取部署报告中的路径、`execution_context` 和
  `approval`，不得重新执行沙箱检测；部署报告中的环境和资源 artifact 是唯一事实。
- 路径 B：由本文件在 benchmark attempt 开始时生成，并写入
  `<attempt-dir>/sandbox-detection.json`。

两条路径都写出 `<attempt-dir>/benchmark-preflight.json`。路径 A 只补充 benchmark CLI
能力检查，并复用部署报告指定的执行上下文；路径 B 在同一上下文中完成环境、CLI 和输出
目录检查。后续服务验收和正式 benchmark 不得切换 shell、网络命名空间或环境快照。

## 统一隔离检测与默认提权

路径 B 只在 attempt 开始时执行一次下列只读检测；路径 A 使用部署 skill 已生成的同名
artifact，不重复执行：

```text
set +e
test -r /dev/davinci_manager
device_node_rc=$?
npu_smi_output="$(npu-smi info 2>&1)"
npu_smi_rc=$?
ss_output="$(ss -H -ltn 2>&1)"
ss_rc=$?
set -e
```

上面的 `npu-smi info` 仅属于路径 B 的一次沙箱隔离检测。路径 A 完全复用部署报告，不再
采集 NPU；路径 B 的后续服务验收、数据集校验和正式 benchmark 也不得再次调用 `npu-smi`。

将 `device_node_rc`、`npu_smi_rc`、`ss_rc`、命令摘要和输出尾部写入
`sandbox-detection.json`。满足下列任一条件时判定为 `sandbox_isolation_detected` 或
`sandbox_isolation_unknown`：

- 驱动设备节点不可读，且错误是 `EPERM`、`EACCES` 或命名空间不可见；
- `npu-smi info` 被拒绝、看不到设备，或输出显示设备节点不可访问；
- 只读端口检查无法访问当前网络命名空间，不能确认监听状态。

路径 A 中命令缺失、驱动缺失和普通依赖缺失分别记录为部署环境预检失败；路径 B 不要求
本地存在 Ascend 驱动，缺失的 `davinci`/`npu-smi` 只记录为 `not_applicable`，不得把它
解释成远程服务或本地端口空闲。任一路径都不得把沙箱内的零设备数或 `EPERM` 当作 Provider
资源结论。

路径 A 由部署 skill 在检测到 `detected` 或 `unknown` 时默认通过执行平台的
`sandbox_permissions=require_escalated` 请求启动提权 Bash；路径 B 由本 skill 执行相同
的默认提权。不得先发送人工确认消息或停止 attempt。平台弹窗负责授权，授权成功后，
benchmark-preflight、路径 B 的 HTTP 服务验收、CLI 检查和正式 `vllm bench serve` 必须复用
同一个提权上下文。

平台拒绝、提权失败或状态无法确认时，写出 `sandbox-detection.json` 和
`benchmark-preflight.json`，报告 `sandbox_approval_denied`、`sandbox_escalation_failed`
或 `sandbox_isolation_unknown`，并停止正式 benchmark。路径 A 还必须引用部署报告中的
失败状态；不得再次请求提权或创建第二份沙箱快照。

artifact 至少包含：

```json
{
  "status": "not_detected|detected|unknown",
  "execution_context": "current|elevated|unknown",
  "approval": "not_required|requested|granted|denied",
  "provider_path": "managed_deployment|predeployed",
  "failure_type": "<type or null>",
  "commands": {},
  "stdout_tail": {},
  "stderr_tail": {}
}
```

## Benchmark Bash 探测

路径 B 在沙箱上下文确定后解析 Python、vLLM 和 benchmark CLI：

```text
set -euo pipefail
PYTHON_BIN="$(command -v python3 || command -v python)"
VLLM_BIN="$(command -v vllm)"
test -x "$PYTHON_BIN" && test -x "$VLLM_BIN"
"$PYTHON_BIN" -c 'import vllm'
"$VLLM_BIN" bench serve --help=all
```

路径 A 不重新探测部署环境、NPU 或 vLLM serve；只在部署报告标记环境和资源通过后，于同一
`execution_context` 中执行一次 `vllm bench serve --help=all`，并将结果写入
`benchmark-preflight.json`。Provider 不需要提供 Python、vLLM 或 HTTP 客户端路径；路径
由当前 Bash 的 `PATH` 解析，命令不存在时产生结构化失败。

版本记录使用 `importlib.metadata.version("vllm")` 和
`importlib.metadata.version("vllm-ascend")`；editable install 的源码位置使用模块
`__file__` 或 `importlib.util.find_spec`，不得读取 `__version__` 或在 `site-packages`
下执行 `find`。

## 必须检查

两条路径都必须确认：

- `vllm bench serve --help=all` 暴露 workload、percentile、goodput 和 raw-result 所需
  参数；
- attempt 输出目录存在、可写且不包含本次命令之前遗留的同名 raw result；
- 用于服务验收的 HTTP 客户端可在同一执行上下文访问 Provider 服务 URL；
- 路径 A 的部署报告为 `service_ready`，且 `selected_devices`、`selected_port` 与部署
  artifact 一致；路径 B 只记录 URL/端口可达性，不执行本地 NPU 或端口资源分配。

服务 URL 的 `/health`、`/v1/models`、served model identity 和最小请求属于服务验收阶段，
失败时不执行正式 `vllm bench serve`。benchmark 不读取 Provider 未授权的设备、端口、
进程列表或全机资源快照。

## 结果与失败

`benchmark-preflight.json` 至少记录探测时间、Provider 路径、继承的沙箱 artifact、解析到
的 executable、CLI help 摘要、环境状态、执行上下文和输出目录。允许的主要失败类型包括：

- `python_missing`、`vllm_missing`、`http_client_missing`；
- `benchmark_cli_unavailable`、`benchmark_cli_incompatible`；
- `python_dependency_missing`、`attempt_dir_invalid`、`attempt_dir_not_writable`；
- `sandbox_approval_denied`、`sandbox_escalation_failed`、`sandbox_isolation_unknown`。

预检失败时 benchmark attempt 使用 `status=blocked` 或 `status=error`，保留命令和 stderr
尾部，不执行正式 `vllm bench serve`。性能字段保持 `null`，清理状态与性能状态分开记录。
