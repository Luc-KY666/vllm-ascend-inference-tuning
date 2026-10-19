---
name: vllm-ascend-deployment
description: 在外部 Provider 冻结的模型、设备、端口和输出合同内，完成 vLLM-Ascend 环境预检、静态 OOM 检查、服务启动、健康验收和部署报告。
---

# vLLM-Ascend 部署技能

本 skill 负责一次独立部署 attempt。Provider 提供模型、可见 NPU、端口候选、运行参数
和唯一输出目录；本 skill 不下载模型、不扫描未授权资源、不创建租约，
也不负责搜索调度或性能评估。

推荐阶段不启动服务、不占用 NPU 或端口。得到可用的最小 TP 后，skill 将推荐值原子写回
同一个 `deployment.json`，重新校验并继续部署。

## Provider 输入

Provider 只需提供字段，skill 在 attempt 目录内组装并写出唯一的 `deployment.json`，至少包含：

- `model.model_path` 和 `model.served_model_name`；
- `resources.visible_devices`、`resources.port_candidates` 和可选 `preferred_port`；
- 可选的 `runtime.tp`、`runtime.dp`、`runtime.max_num_seqs` 和
  `runtime.max_num_batched_tokens`；缺少 TP 时进入最小 TP 推荐；
- `outputs.attempt_dir`，且该目录只能属于本次 attempt。

Provider 不提供 Python、vLLM、ATB 或 activation wrapper 路径；这些内容由 skill 在当前
Bash 环境中探测和执行。

输入字段、默认值和状态边界见 [input-contract.md](references/input-contract.md)。

## 执行权限

Ascend 驱动设备节点和 wildcard TCP bind 可能不暴露给默认沙箱。每个 attempt 在执行
任何环境、资源或服务动作前，只按 [ascend-probes.md](references/ascend-probes.md) 做一次
统一隔离检测，并生成 `<attempt-dir>/sandbox-detection.json`。检测到隔离或状态不确定时，
skill 必须默认通过执行平台的 `sandbox_permissions=require_escalated` 请求启动提权 Bash，
不先发送人工确认消息或停止 attempt。平台弹窗负责授权；授权成功后，后续环境预检、NPU
资源预检、端口资源预检和 `vllm serve` 均复用同一个提权上下文，不重新检测或再次请求提权。
未检测到隔离时，后续阶段复用当前上下文执行。平台拒绝、提权失败或状态无法确认时写出
结构化 preflight failure，不把沙箱内的 `EPERM`、驱动错误或零设备数当作真实资源结论。

预检参数和运行时事实必须按以下兼容性规则处理：

- `ascend_preflight.py --protected-env` 的值是**内联 JSON object 字符串**，不是 JSON 文件路径、
  `@file` 引用或目录。先在同一 shell 中生成 JSON 字符串，再以 `--protected-env "$JSON"` 传入；
  JSON 解析失败时记录为输入错误，不先尝试路径再重试内联值。
- vLLM CLI 能力只检查一次 `vllm serve --help=all`。禁止先调用裸 `--help` 再因输出不完整重试；
  后续阶段复用这份 help artifact，完整输出落盘为 `<attempt-dir>/vllm-serve-help-all.txt`。
- 版本使用 `importlib.metadata.version("vllm")` 和
  `importlib.metadata.version("vllm-ascend")` 获取，缺失 metadata 时记录 `null/unknown` 并继续
  诊断；禁止访问 `vllm.__version__` 或 `vllm_ascend.__version__`。源码位置使用已导入模块的
  `__file__` 或 `importlib.util.find_spec`，不要在 `site-packages` 下执行 `find`；可编辑安装可能
  将源码解析到 `/vllm-workspace` 等目录。

## 执行流程

1. 在 attempt 开始时调用 [ascend-probes.md](references/ascend-probes.md) 完成唯一一次
   沙箱检测并确定后续执行上下文；检测到隔离或状态不确定时，立即使用执行平台的提权
   Bash 执行环境预检。环境预检探测 Python 和 vLLM、Ascend runtime、Python 依赖、ATB 和
   一次 `vllm serve --help=all`。失败时写出 `environment_preflight_failed`，不启动服务；不再
   追加裸 `--help` 调用或重复加载 serve CLI。
2. 调用 [input-contract.md](references/input-contract.md) 规定的
   `input_contract.py`，把 Provider 字段解析为规范输入，检查模型路径、`config.json`、
   设备、端口、输出目录和 vLLM 硬约束，生成唯一的 `deployment.json`。
3. 输入状态为 `invalid` 时记录具体错误并停止；状态为 `missing_parameters` 时进入
   recommendation mode；状态为 `pass` 时直接使用规范化配置。
4. 从 `deployment.json` 指向的 `config.json`、Provider 硬件事实和运行参数派生
   `static-oom-input.json`，按 [static-oom.md](references/static-oom.md) 执行静态检查。
   缺少参数的推荐模式显式使用 `DP=1`、`max_num_seqs=32` 和
   `max_num_batched_tokens=4096`。静态 OOM 检查不再以
   `hidden_size == num_attention_heads * head_dim` 作为前置硬约束；按模型配置提供的
   独立维度直接计算 KV Cache 风险。
5. 静态结果为 `reject` 时写出 `static_oom_rejected`；结果为 `unknown` 或推荐为
   `unavailable` 时写出 `recommendation_only`，保留原因，不启动服务。
6. 推荐得到可用 TP 后，使用 `input_contract.py --resolve-tp` 将 TP 写回
   `deployment.json`，记录原始状态、推荐值和报告路径，然后重新执行输入硬约束检查。
   推荐 TP 不满足 attention head、设备数或 Provider 上限时写出 `input_invalid`。
7. 按 [resource-probes.md](references/resource-probes.md) 在第 1 步确定的同一执行上下文中检查
   Provider 可见设备和端口。沙箱检测阶段的 `npu-smi info` 只用于隔离判定；本步骤在即将
   调用 `vllm serve` 前**仅再执行一次** `npu-smi info` 作为可用卡探测，并复用其 artifact，
   不在其他阶段重复采集。资源检查记录由 agent 自行选择的可用端口探测方式和必要的标准
   shell 命令输出，由 agent 根据实际输出判定设备与端口状态。只有
   明确判定为空闲的设备和可用的端口才能用于启动。空闲设备不足 `TP * DP`、端口不可用
   或状态无法确认时写出 `resource_preflight_failed`，不自动换卡、换端口或扩大 TP。
   该判定是一次性的时间点结论：只要 Provider 可见卡中空闲数量不足 `TP * DP`（含判定为
   busy 或 unknown 的卡），必须**立即写出失败报告并退出当前部署 attempt**，不得等待、
   轮询或以任何形式挂起以等待其他工作负载释放设备；设备释放后的重试只能通过新的
   唯一输出目录发起新的 attempt。
8. 资源检查完成后，在同一上下文中直接使用第 7 步返回的 `selected_devices` 和
   `selected_port` 执行 `vllm serve`。若第 1 步触发了提权，服务必须留在该提权上下文；
   记录命令、日志、PID 和清理边界。vLLM API server 启动时会在运行时 TMPDIR 下创建
   ZMQ IPC socket，完整路径（`TMPDIR + "/" + UUID`）不得超过 107 字符；attempt 目录
   路径过长时优先在 attempt 目录之外创建短路径临时目录运行，并在部署报告中记录
   （约束细节见 [ascend-probes.md](references/ascend-probes.md)）。
9. 依次验收进程、健康接口、`/v1/models`、served model identity 和最小请求。健康、
   身份或最小请求失败时按阶段写结构化 failure；启动或最小请求阶段 OOM 使用
   `service_startup_oom` 分类。
10. 按 [report-schema.md](references/report-schema.md) 写出 `deployment_report.json`，
    保留 `deployment.json`、静态 OOM、环境、资源、服务和清理 artifact。服务成功时将
    URL、模型身份、设备、端口和进程交还 Provider。

## 参考资料

- [输入契约](references/input-contract.md)：Provider 输入、默认值、校验和 deployment.json。
- [Ascend 环境预检](references/ascend-probes.md)：环境、依赖、ATB 和 vLLM CLI 检查。
- [资源预检](references/resource-probes.md)：空闲 NPU 和 Provider 端口检查。
- [静态 OOM 规则](references/static-oom.md)：容量判定和最小 TP 推荐。
- [部署报告结构](references/report-schema.md)：一次部署 attempt 的输出合同。
- [输入契约脚本](scripts/input_contract.py)：生成和更新 `deployment.json`。
- [环境预检脚本](scripts/ascend_preflight.py)：检查当前 Bash 中的 Ascend 环境。

## 状态与边界

- 推荐未知、没有可行 TP、环境预检失败、静态 OOM reject、资源预检失败和服务验收失败
  都必须保留现场并停止当前 attempt。
- 资源预检发现空闲卡不足 `TP * DP` 时必须立即写出 `resource_preflight_failed` 并退出：
  不等待设备释放、不轮询、不挂起，“等待释放”不构成本技能的任何阶段，也不得把等待
  时间计入部署耗时；设备释放后的重试属于使用新输出目录的新 attempt。
- 不因 OOM、启动失败或健康检查失败自动扩大 TP/DP、替换设备、降低上下文或改用未提供
  的端口。
- `service_ready` 只表示部署层健康、模型身份和最小请求通过，不表示压测成功或参数最优。
- 不实现外部 Provider 的资源租约、搜索调度、candidate 管理或独立 verifier。
