# NPU 和端口资源预检

资源预检只消费 `input.normalized` 中 Provider 明确提供的设备和端口，不扫描候选范围之外
的资源，也不创建资源租约。每个 attempt 只执行一次；结构化结果直接写入报告的
`stages.resources`，随后作为服务启动的输入。

资源检查使用当前执行上下文中的一次 `npu-smi info` 和
`environment_probe.json.port_probe_tool` 已验证的端口探测工具。端口工具的发现和能力验证
由 [vllm-ascend-probes](../../vllm-ascend-probes/SKILL.md) 完成；本阶段不得重新选择工具、
执行工具发现或切换网络命名空间。工具必须能在当前网络命名空间中提供可解释的监听证据，
并配合绑定确认判断 Provider 端口状态。

## 隔离和提权

资源检查只读取 attempt 开始时生成的 `environment_probe.json`。其中
`execution_context`、`shell_fingerprint` 和 `port_probe_tool` 选定字段决定本阶段使用的
Bash、网络命名空间和端口工具。检测到隔离或状态不确定时，probe 阶段已完成唯一一次提权
请求；本阶段不重新执行沙箱检测、端口工具发现或再次请求提权。平台拒绝、提权失败、
shell 句柄丢失或选定工具失效时写出 `resource_preflight_failed`，并保留授权状态和命令尾部。

## NPU 检查

沙箱隔离检测已经执行过一次 `npu-smi info`；本阶段只在确认即将启动服务后再执行一次，
并把 stdout/stderr 捕获在内存中。除这两次外不得在预检、服务验收或报告整理阶段重复采集。
不要额外重定向到 `npu-smi-info.txt`；判定完成后将原始输出摘要（或满足报告大小限制的
完整证据）保留在内存中，必要诊断尾部写入 `stages.resources.diagnostics_tail`；部署侧不
创建 NPU 资源 JSON。

agent 根据实际设备的输出格式、表头、状态字段、进程信息及其他可核验内容，逐张判定
Provider `visible_devices` 中的设备为 `free`、`busy` 或 `unknown`，并在 artifact 中记录
所依据的原始片段和判定说明。不得预设某一种 `npu-smi` 文本格式，也不得用其他设备版本
的输出猜测当前设备状态。无法把输出可靠归属于某张 Provider 设备、命令失败或证据互相
矛盾时，该设备为 `unknown`，整个资源预检失败关闭。不读取 `visible_devices` 之外的卡，
不自动换卡或扩大 TP。

将 `visible_devices`、原始输出摘要、每张卡的 `free`/`busy`/`unknown`、PID、解析规则版本和
所需数量 `TP * DP` 写入 `stages.resources.devices`。只有根据实际输出明确确认的前
`TP * DP` 张 free 卡可以写入 `selected_devices`；数量不足时使用
`failure_type=insufficient_available_devices`。

本次探测是一次性的时间点判定，其结论对当前 attempt 是最终的：当 Provider 可见卡中
free 数量不足 `TP * DP`（含判定为 busy 或 unknown 的卡）时，立即写出
`resource_preflight_failed` 失败报告并退出当前 attempt。禁止等待、轮询、重新探测或
以任何形式挂起以等待设备释放——busy/unknown 的卡不会被重新探测，本流程中不存在
“等待释放后继续”的阶段。

## 端口检查

只按 Provider 提供的 `port_candidates` 顺序检查；`preferred_port` 如果存在则优先检查，
但它不在候选列表时立即失败。检查必须发生在将要启动服务的同一个网络命名空间中，并复用 `environment_probe.json.port_probe_tool.argv`。对每个候选保存该工具的参数、退出码、
原始输出和绑定确认结果。若选定工具无法运行，直接使用 `failure_type=port_probe_tool_unavailable`
失败关闭，不在当前 attempt 中重新发现或替换工具。

探测证据应覆盖 wildcard、loopback 和 IPv6 等实际监听地址（例如
`0.0.0.0:<PORT>`、`127.0.0.1:<PORT>`、`[::]:<PORT>` 或等价表示）。没有任何监听且
探测确认可以绑定 Provider 要求的服务地址时，候选状态为 `available`；存在冲突监听或
绑定失败时为 `occupied`；网络命名空间不确定、输出无法解释或证据互相矛盾时为 `unknown`。

将候选顺序、探测命令/机制及其原始输出摘要、探测结果和选中的唯一端口写入
`stages.resources.ports`。没有可用候选时使用 `failure_type=port_unavailable` 或
`port_probe_failed`；不自动换到 Provider 未提供的端口。

端口确认后应立即在同一个 Provider 生命周期边界内启动服务。端口检查不替代外部租约，
也不允许并行启动多个服务争抢同一端口。
