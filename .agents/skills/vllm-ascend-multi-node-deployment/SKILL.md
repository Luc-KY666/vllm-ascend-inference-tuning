---
name: vllm-ascend-multi-node-deployment
description: 在 Provider 提供的镜像、节点（SSH 地址和数据面 IP）、模型路径和 served_model_name 基础上，通过 SSH 在多个目标宿主机上完成 Docker 化 vLLM-Ascend 多机部署（普通多机与 PD 分离）、启动后状态检查、对话服务验收和部署报告。
---

# vLLM-Ascend 多机部署技能

本 skill 通过 SSH 在多个目标宿主机上完成 Docker 化 vLLM-Ascend 多机部署。本 skill
不下载模型、不扫描未授权资源、不创建租约，也不负责搜索调度或性能评估。部署方式分为两类：

- `multi-node deployment`：模型服务分布在多个目标主机，采用普通 vLLM-Ascend DP/TP
  部署且不进行 Prefill/Decode 分离；先启动 Master 再并行启动 Worker，启动后统一
  检查分布式状态并对 Master 做对话服务验收。
- `multi-node PD-disaggregated deployment`：Prefill、Decode、KV Transfer/Mooncake
  和 Proxy 服务分布在多个目标主机；按服务依赖顺序启动，并对完整的
  Prefill-to-Decode 请求链路进行 readiness 和业务验收。

## Provider 输入

多机部署由 Provider 提供镜像、节点（SSH 地址和数据面 IP）、模型路径、
`served_model_name` 和可选的端口、运行参数或指定卡；最终启动命令、挂载和设备由
Skill 按多机公共规则和对应部署说明生成，不由 Provider 提供。

## 执行流程

1. **模式判定**：仅说“多机部署”判定为普通多机，读取
   [multi-node-deployment.md](references/multi-node-deployment.md)；
   出现“Prefill/Decode”“PD 分离”“Mooncake”或“KV Transfer”判定为 PD 分离，读取
   [multi-node-PD-disaggregated-deployment.md](references/multi-node-PD-disaggregated-deployment.md)。
2. **公共流程**：按 [common.md](references/common.md) 依次执行输入检查、
   宿主机运行时预检、部署、服务验收与报告记录；SSH/Docker 边界、挂载模板、启动
   顺序、清理和报告规则均以 common.md 为准，前一阶段失败时停止。
3. **模式专属规则**：并行策略、配置参数、启动命令、启动后检查和验收策略等模式
   专属规则按第 1 步选定的模式文档执行。
4. **结果交还**：服务成功后，`service.url`、模型身份和公共入口交还 Provider；
   部分启动、rank 不完整、Proxy 不可用或清理失败必须分别报告，最终状态按
   common.md 第 4 节报告 `ready`、`failed` 或 `partially_started`。

## 执行权限

- Agent 必须具备到所有目标节点的非交互式 SSH 能力。
- Agent 直接运行在所在目标节点的宿主机上时，该节点本地直接执行、不经 SSH；Agent
  运行在该节点的容器或沙箱中时，该节点与其他目标节点同等对待；其余目标节点的
  路径、设备、Docker、镜像、容器和日志操作都通过对应 SSH 连接执行。
- Agent 所在节点上的所有 Docker 操作（镜像拉取、容器创建）都在宿主机上执行，
  不得在 Agent/Pi 容器内嵌套拉取镜像或创建容器，也不得启动或使用隔离的 `dockerd`。
- 控制端的本地 NPU 探测不能替代其他目标节点的远端探测。

## 状态与边界

- 远端资源结论只来自目标节点上的预检；不等待设备释放、不轮询，也不回退到控制端
  或 Agent 容器内的 Docker daemon。
- 多机容器清理只处理与本次部署容器名精确匹配的对象，禁止使用 `pkill`、宽范围
  `docker rm` 或通过端口占用者推断所有权。

## 参考资料

- [多机公共规则](references/common.md)：SSH、Docker、挂载、部署、验收和清理。
- [multi-node deployment](references/multi-node-deployment.md)：普通多机 vLLM
  启动与验证。
- [multi-node PD-disaggregated deployment](references/multi-node-PD-disaggregated-deployment.md)：
  Prefill/Decode、KV Transfer 和 Proxy。
