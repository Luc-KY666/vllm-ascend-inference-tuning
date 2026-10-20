# 环境探测合同

## 探测范围

本合同只确认当前 shell 能否运行 vLLM-Ascend 部署和官方 benchmark 所需的环境能力。它不
绑定 Provider 端口、不分配设备、不启动服务，也不替代 deployment 的 NPU/端口资源预检。

所有 provider path 都执行相同的严格 CANN 检查：

- `npu-smi info` 能执行；
- `torch`、`torch_npu`、`vllm`、`vllm_ascend` 能导入；
- `torch.npu.is_available()` 为真且设备数量大于零；
- `libascendcl.so` 和 `libatb.so` 可由 `ctypes.CDLL` 加载；
- 当前 PATH 中的 `vllm serve --help=all` 和 `vllm bench serve --help=all` 成功；
- Python 标准库 HTTP 客户端可导入；
- 至少有一个端口探测工具能在当前网络命名空间执行。

版本只使用 `importlib.metadata.version("vllm")` 和
`importlib.metadata.version("vllm-ascend")`。源码位置使用已导入模块的 `__file__`；不访
问模块 `__version__`，不在 `site-packages` 下执行 `find`。

## 端口工具发现

脚本按固定顺序验证以下候选。候选列表只保存名称和状态；失败候选另存退出码及有限长度的
`diagnostic_tail`，不保存成功命令的 stdout/stderr、argv 或可执行文件路径：

1. `ss -H -ltn`
2. `netstat -ltn`
3. `lsof -nP -iTCP -sTCP:LISTEN`
4. Python socket/proc fallback

对每个候选只记录端口工具名称和状态；失败候选可附带退出码和有限长度的诊断尾部。选定
工具的唯一记录位于 `environment_probe.json.port_probe_tool` 顶层，包含 `name`、`argv`、
`mode`、`executable` 和 `verified`；候选记录不重复这些字段。外部命令仍使用 argv 数组
执行，禁止通过 shell 字符串插值。Python fallback 使用当前 Python 进程检查本地 socket/proc
能力，后续资源阶段可用它对 Provider 端口执行监听和绑定判断。

环境阶段只选择工具，不把环境阶段的输出解释为某个 Provider 端口空闲。资源阶段读取
`port_probe_tool` 并在同一 shell、同一网络命名空间中复用；另行对 Provider 候选端口执行
绑定确认。

## Canonical artifact

`environment_probe.json` 使用 schema `vllm-ascend-environment-probe-v2`，至少包含：

```json
{
  "schema_version": "vllm-ascend-environment-probe-v2",
  "status": "pass|failed|escalation_required",
  "provider_path": "managed_deployment|predeployed",
  "execution_context": "current|elevated|unknown",
  "shell_fingerprint": {},
  "python": {},
  "sandbox": {
    "status": "not_detected"
  },
  "cann": {},
  "dependencies": {},
  "vllm_serve": {},
  "vllm_bench_serve": {},
  "http_client": {},
  "port_probe_tool": {},
  "artifacts": {
    "serve_help": "<path>",
    "bench_help": "<path>",
    "sandbox_detection": "<path>"
  },
  "failure": null
}
```

Canonical `environment_probe.json` 的 `sandbox` 对象只包含 `status`，表示当前是否检测到沙箱隔离；设备节点、runtime 和端口探测详情仅写入独立的 `sandbox-detection.json`，端口工具在 canonical 文件中只保留顶层 `port_probe_tool`。

`sandbox-detection.json` 是沙箱证据，包含设备节点检查、`npu-smi info` 和选定端口工具的
结构化结果、失败诊断、初始/提权 probe 摘要和授权状态。`environment_probe.json.probe_history`
（如果存在）也只保存每次执行上下文、状态、沙箱状态、shell 指纹和失败类型，不嵌套完整
probe 报告。deployment 和 benchmark 报告只引用 `environment_probe.json`；它们不得各自产生
另一份 deployment 或 benchmark 环境报告。

## 状态和提权

- 当前 shell 检测到明确的 `EPERM`、`EACCES` 或网络/设备命名空间不可见时，返回
  `escalation_required`，不把该结果当作资源失败。
- 当前 shell 状态不确定且无法排除隔离时，同样返回 `escalation_required`。
- 初始 probe 的普通依赖缺失、CANN 库缺失、命令缺失或 help 失败直接返回 `failed`，不
  为普通软件缺失请求提权。
- 提权 probe 仍检测到隔离时返回 `failed`，`failure.type=sandbox_isolation_persisted`。
- 提权被拒绝、提权 shell 无法启动或句柄丢失时返回对应的
  `sandbox_approval_denied`、`sandbox_escalation_failed` 或 `execution_context_lost`。

探测脚本使用固定超时和单次调用：普通命令为 30 秒，两个 vLLM help 命令为 180 秒。
完整 help 输出只写入指定的 help artifact，不进入 JSON；成功命令的 stdout/stderr 不写入
JSON，失败时只保留退出码和有限长度的 `diagnostic_tail`。不得追加同类命令或更换环境入口。
