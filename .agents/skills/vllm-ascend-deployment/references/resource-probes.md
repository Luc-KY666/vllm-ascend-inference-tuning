# NPU 和端口资源预检

资源预检只消费内存中规范化输入里的 Provider 明确提供的设备和端口，不扫描候选范围之外
的资源，也不创建资源租约。初次选择和必要的重选结果保存在内存中，最终报告只写入实际
选中的 NPU 和端口。

资源选择最多两轮：第一轮按用户指定范围，或按探测到的空闲候选选择；若候选在启动前后
发生占用或不可用，第二轮只重新探测、选择和启动，复用环境 probe、输入校验、静态 OOM
和 TP 结果。第二轮仍失败就结束当前 attempt；不等待释放、不扩大范围，agent 不得自行
创建新的 attempt。探测工具不可用或状态无法确认时不重试，直接失败。

资源检查使用当前执行上下文中的 `npu-smi info` 和
`environment_probe.json.port_probe_tool` 已验证的端口探测工具；服务启动发生资源竞态时，
`npu-smi info` 和端口工具最多各额外使用一次。端口工具的发现和能力验证
由 [vllm-ascend-probes](../../vllm-ascend-probes/SKILL.md) 完成；本阶段不得重新选择工具、
执行工具发现或切换网络命名空间。工具必须能在当前网络命名空间中提供可解释的监听证据，
并配合绑定确认判断 Provider 端口状态。

## 隔离和提权

资源检查只读取 attempt 开始时生成的 `environment_probe.json`。其中
`execution_context`、`shell_fingerprint` 和 `port_probe_tool` 选定字段决定本阶段使用的
Bash、网络命名空间和端口工具。检测到隔离或状态不确定时，probe 阶段已完成唯一一次提权
请求；本阶段不重新执行沙箱检测、端口工具发现或再次请求提权。平台拒绝、提权失败、
shell 句柄丢失或选定工具失效时在内存中记录 `resource_preflight_failed`，并保留授权状态和
命令尾部；最终报告统一在清理完成后一次性写出。

## NPU 检查

沙箱隔离检测已经执行过一次 `npu-smi info`；本阶段在确认即将启动服务前执行一次，
并把 stdout/stderr 捕获在内存中。只有服务启动发生设备竞态时，才允许在同一 attempt
内额外重探一次；该重探只用于资源重选，不重跑输入校验、静态 OOM 或 TP 决策。不要额外
重定向到 `npu-smi-info.txt`；判定完成后将原始输出摘要（或满足报告大小限制的完整证据）
保留在内存中；部署侧不创建 NPU 资源 JSON，也不把诊断尾部写入最终报告。

agent 根据实际设备的输出格式、表头、状态字段、进程信息及其他可核验内容，逐张判定
Provider `visible_devices` 中的设备为 `free`、`busy` 或 `unknown`，并在 artifact 中记录
所依据的原始片段和判定说明。不得预设某一种 `npu-smi` 文本格式，也不得用其他设备版本
的输出猜测当前设备状态。无法把输出可靠归属于某张 Provider 设备、命令失败或证据互相
矛盾时，该设备为 `unknown`，不能使用该设备。不读取 `visible_devices` 之外的卡，
不自动扩大 TP。用户指定设备时只在该范围内选择；未指定时只从探测到的空闲设备中选择。

将 `visible_devices`、原始输出摘要、每张卡的 `free`/`busy`/`unknown`、PID、解析规则版本和
所需数量 `TP * DP` 保存在内存中。只有根据实际输出明确确认的前 `TP * DP` 张 free 卡可以
作为最终 `resources.npu`；数量不足时使用 `failure_type=insufficient_available_devices`。

资源探测是时间点判定：不得等待、轮询或挂起以等待设备释放。若服务启动时发现已选
设备发生竞态占用，可在当前 attempt 内重新探测并重选一次；重选仍只能使用 Provider
允许的设备范围，并复用前置阶段结果。第二次仍不足 `TP * DP` 时记录
`insufficient_available_devices`，清理并退出当前 attempt。

## 端口检查

只按 Provider 提供的 `port_candidates` 顺序检查；`preferred_port` 如果存在则优先检查，
但它不在候选列表时立即失败。检查必须发生在将要启动服务的同一个网络命名空间中，并复用 `environment_probe.json.port_probe_tool.argv`。对每个候选保存该工具的参数、退出码、
原始输出和绑定确认结果。若选定工具无法运行，直接使用 `failure_type=port_probe_tool_unavailable`
失败关闭，不在当前 attempt 中重新发现或替换工具。

探测证据应覆盖 wildcard、loopback 和 IPv6 等实际监听地址（例如
`0.0.0.0:<PORT>`、`127.0.0.1:<PORT>`、`[::]:<PORT>` 或等价表示）。没有任何监听且
探测确认可以绑定 Provider 要求的服务地址时，候选状态为 `available`；存在冲突监听或
绑定失败时为 `occupied`；网络命名空间不确定、输出无法解释或证据互相矛盾时为 `unknown`。

将候选顺序、探测命令/机制及其原始输出摘要、探测结果保存在内存中，最终只把选中的唯一端口
写入 `resources.port`。没有可用候选时使用 `failure_type=port_unavailable` 或
`port_probe_failed`。服务启动时若发生端口竞态占用，可在当前 attempt 内从允许的候选中
重选一次；不自动换到 Provider 未提供的端口，也不等待端口释放。第二次仍失败则退出当前
attempt。

端口确认后应立即在同一个 Provider 生命周期边界内启动服务。端口检查不替代外部租约，
也不允许并行启动多个服务争抢同一端口。
