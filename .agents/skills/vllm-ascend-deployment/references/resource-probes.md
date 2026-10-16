# NPU 和端口资源预检

资源预检只消费 `deployment.json` 中 Provider 明确提供的设备和端口，不扫描候选范围之外
的资源，也不创建资源租约。每个 attempt 只执行一次；检查结果写入 attempt 目录，随后
直接作为服务启动的输入。

资源检查使用当前执行上下文中的一次 `npu-smi info` 和由 agent 自行选择的端口探测方式。
探测方式必须能在当前网络命名空间中提供可解释的监听或绑定证据；agent 不得假定某个
具体工具存在，必要时可组合多种可用机制。artifact 必须记录实际使用的工具或机制、完整
命令摘要和原始输出。没有可用探测方式、输出无法确认或权限被拒绝时必须失败关闭，不能
把未知状态解释为空闲。

## 隔离和提权

资源检查使用 [ascend-probes.md](ascend-probes.md) 在 attempt 开始时生成的
`sandbox-detection.json`。其中 `execution_context` 和 `approval` 决定本阶段使用当前
Bash 还是提权 Bash；检测到隔离或状态不确定时，环境预检阶段已默认发起一次提权请求，
`npu-smi info`、端口探测和随后 `vllm serve` 必须保持在同一上下文中。本阶段不重新执行
沙箱检测或再次请求提权；平台拒绝、提权失败或隔离状态无法确认时写出
`resource_preflight_failed`，并保留授权状态和命令输出尾部。

## NPU 检查

沙箱隔离检测已经执行过一次 `npu-smi info`；本阶段只在确认即将启动服务后再执行一次，
并保存完整输出。除这两次外不得在预检、服务验收或报告整理阶段重复采集：

```text
npu-smi info > <attempt-dir>/npu-smi-info.txt 2>&1
```

agent 根据实际设备的输出格式、表头、状态字段、进程信息及其他可核验内容，逐张判定
Provider `visible_devices` 中的设备为 `free`、`busy` 或 `unknown`，并在 artifact 中记录
所依据的原始片段和判定说明。不得预设某一种 `npu-smi` 文本格式，也不得用其他设备版本
的输出猜测当前设备状态。无法把输出可靠归属于某张 Provider 设备、命令失败或证据互相
矛盾时，该设备为 `unknown`，整个资源预检失败关闭。不读取 `visible_devices` 之外的卡，
不自动换卡或扩大 TP。

将 `visible_devices`、原始输出路径、每张卡的 `free`/`busy`/`unknown`、PID、解析规则
版本和所需数量 `TP * DP` 写入 `npu-resource-probe.json`。只有根据实际输出明确确认的前
`TP * DP` 张 free 卡可以写入 `selected_devices`；数量不足时使用
`failure_type=insufficient_available_devices`。

## 端口检查

只按 Provider 提供的 `port_candidates` 顺序检查；`preferred_port` 如果存在则优先检查，
但它不在候选列表时立即失败。检查必须发生在将要启动服务的同一个网络命名空间中。由
agent 选择当前上下文中可用、能够证明端口状态的监听检查、bind 检查或其他等价机制，
并为每个候选保存实际命令/机制、参数、退出码和原始输出。若一种机制不足以区分
`available` 与 `occupied`，应追加另一种可用机制；仍无法确认时标记为 `unknown`，不得
选择该端口。

探测证据应覆盖 wildcard、loopback 和 IPv6 等实际监听地址（例如
`0.0.0.0:<PORT>`、`127.0.0.1:<PORT>`、`[::]:<PORT>` 或等价表示）。没有任何监听且
探测确认可以绑定 Provider 要求的服务地址时，候选状态为 `available`；存在冲突监听或
绑定失败时为 `occupied`；网络命名空间不确定、输出无法解释或证据互相矛盾时为 `unknown`。

将候选顺序、探测命令/机制及其输出路径、探测结果和选中的唯一端口写入
`port-resource-probe.json`。没有可用候选时使用 `failure_type=port_unavailable` 或
`port_probe_failed`；不自动换到 Provider 未提供的端口。

端口确认后应立即在同一个 Provider 生命周期边界内启动服务。端口检查不替代外部租约，
也不允许并行启动多个服务争抢同一端口。
