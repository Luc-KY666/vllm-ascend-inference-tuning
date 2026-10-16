# Ascend 环境预检

环境预检不分配设备、不绑定 Provider 端口、不启动模型服务。每次 attempt 先生成一次
`<attempt-dir>/sandbox-detection.json`，再生成环境检查用的
`<attempt-dir>/ascend-preflight.json`。后续资源检查和服务启动只读取沙箱检测结果，不
重复执行隔离检测。

## 统一隔离检测与默认提权

skill 只在 attempt 开始时执行本节一次。检测只使用当前环境可见的信息，不修改 Provider
的设备、端口或受保护环境变量：

```text
set +e
test -r /dev/davinci_manager
device_node_rc=$?
npu_smi_output="$(npu-smi info 2>&1)"
npu_smi_rc=$?
set -e
```

同时使用 agent 选择的、当前上下文中可用的只读端口探测方式检查网络命名空间，并将其
退出码、命令摘要和输出尾部分别记录为 `port_probe_rc`、`port_probe_command` 和
`port_probe_output`。将 `device_node_rc`、`npu_smi_rc`、端口探测结果、命令摘要和输出尾部写入
`sandbox-detection.json`。满足
下列任一条件时判定为 `sandbox_isolation_detected` 或
`sandbox_isolation_unknown`：

- 驱动设备节点不可读，且错误是 `EPERM`、`EACCES` 或命名空间不可见；
- `npu-smi info` 被拒绝、看不到设备，或输出显示设备节点不可访问；
- 端口探测无法访问当前网络命名空间，不能确认监听状态。

命令不存在、驱动本身缺失和普通依赖缺失要分别记录为环境预检失败，不能伪装成“无空闲
设备”。隔离状态为 `not_detected` 时继续在当前上下文执行；状态为 `detected` 或
`unknown` 时默认立即使用工具的 `sandbox_permissions=require_escalated` 启动提权 Bash。
该调用由执行平台负责显示授权弹窗，不要求 skill 先发送人工确认消息或中断 attempt。
一次提权覆盖本次 attempt 的环境预检、资源检查和服务启动。平台拒绝、提权失败或超时都
在 `sandbox-detection.json` 和部署报告中记录，并写出 `environment_preflight_failed`，
其中包含 `failure_type=sandbox_approval_denied`、`sandbox_escalation_failed` 或
`sandbox_isolation_unknown`，不得把沙箱内的零设备或 `EPERM` 当成宿主资源事实。

提权后的上下文必须复用同一份 attempt 目录和受保护环境快照。不得在后续资源阶段重新
执行本节检测或再次请求相同的提权；所有阶段沿用 `sandbox-detection.json` 中的
`execution_context` 和 `approval`。

该 artifact 至少包含 `status`（`not_detected`、`detected` 或 `unknown`）、
`execution_context`（`current` 或 `elevated`）、`approval`（`not_required`、`requested`、
`granted` 或 `denied`）、检测命令摘要和输出尾部。资源报告与服务报告引用同一份 artifact。

## Bash 探测

在统一隔离检测确定的同一个 shell 中解析 Python 和 vLLM。Provider 不需要提供这些路径：

```text
set -euo pipefail
PYTHON_BIN="$(command -v python3 || command -v python)"
VLLM_BIN="$(command -v vllm)"
test -x "$PYTHON_BIN" && test -x "$VLLM_BIN"
PROTECTED_ENV_JSON='{"ASCEND_RT_VISIBLE_DEVICES":"<provider-value>","TMPDIR":"<attempt-tmp>"}'
"$PYTHON_BIN" <skill-root>/scripts/ascend_preflight.py \
  --protected-env "$PROTECTED_ENV_JSON" \
  --timeout 30 \
  --output <attempt-dir>/ascend-preflight.json
```

`--protected-env` 必须接收上例这样的内联 JSON object 字符串；`<attempt-dir>/protected-env.json`
之类的文件路径不会被脚本读取。变量值必须在同一个 shell/session 中序列化并原样传入，JSON
解析失败时直接记录结构化输入错误，不先尝试路径再重试。空对象 `{}` 仅表示让脚本从当前
进程读取默认受保护变量，不能用来绕过 Provider 的环境快照校验。

`ascend_preflight.py` 通过该 shell 的当前 `PATH` 解析 `vllm`。命令不存在时产生结构化
的 `vllm_executable_missing`，而不是从 Provider 输入中读取替代路径。Python 解析顺序
为 `python3`、`python`；两者都不可用时环境预检失败。

## 检查内容

预检依次检查：

- `torch`、`torch_npu`、`vllm` 和 `vllm_ascend` 可导入；
- `torch.npu.is_available()` 为真且设备数量大于零；
- `libatb.so` 可由 `ctypes.CDLL` 加载；
- 当前 PATH 中的 `vllm` 成功执行一次 `vllm serve --help=all`；不得追加裸 `--help` 或重复
  调用。CLI 输出 artifact 供后续部署阶段复用。
- `vllm` 和 `vllm-ascend` 版本通过 `importlib.metadata.version` 读取，模块源码位置通过
  `module.__file__` 或 `importlib.util.find_spec` 读取。editable install 可能位于
  `/vllm-workspace`，不得在 `site-packages` 下执行 `find`，也不得访问模块 `__version__`。

预检脚本不得改写可见设备、临时目录或其他受保护变量，也不读取固定的外部调度器或
资源池变量。

## 状态

成功 artifact 使用 `status=pass`。失败 artifact 至少包含 `failure_type`、失败阶段、
隔离检测结果、命令摘要和 stdout/stderr 尾部。允许的主要失败类型包括：

`protected_environment_changed`、`sandbox_approval_denied`、`sandbox_escalation_failed`、
`sandbox_isolation_unknown`、`atb_loader_failed`、`ascend_runtime_unavailable`、
`ascend_runtime_probe_failed`、`vllm_executable_missing`、`vllm_serve_unavailable` 和
`vllm_serve_probe_failed`。

依赖缺失、Ascend runtime 不可用、ATB 加载失败或 vLLM help 失败时，部署报告状态为
`environment_preflight_failed`，不得生成资源分配或启动服务。
