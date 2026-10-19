# 多机 readiness、业务验证和失败处理

本文件只适用于主 Skill 的 **multi-node deployment** 和
**multi-node PD-disaggregated deployment**。
它定义远端容器的验证和清理，不替代 single-node deployment 的本地进程验收规则。

本文件定义两种部署方式共用的结果判定。具体服务启动顺序和模式参数分别读取
`multi-node-deployment.md` 或 `multi-node-PD-disaggregated-deployment.md`。

## Readiness 类型

每个服务必须在计划中声明一种有界检查：

- `process`：容器运行且目标 vLLM、Mooncake 或 Proxy 进程存在；
- `tcp`：目标端口在超时时间内可以建立连接；
- `http`：健康接口返回预期状态和必要字段；
- `log`：限定时间内出现明确的启动完成日志。

容器创建成功不等于服务 ready。所有检查必须有超时，记录开始时间、结束时间、实际结果和必要的日志摘要。

在服务 readiness 之前，先执行运行时配置检查：使用 `docker inspect` 核对当前 deployment 容器实际的七项 Ascend 固定挂载、计划中的 `/dev/davinci<device_id>` 和三个共享设备。缺少任何一项时立即判定失败，不继续等待 vLLM readiness。

启动观察必须同时匹配已知失败签名，至少包括参数解析错误、模型路径错误、NPU/HCCL 错误，以及 `rtStreamEndCapture`、`stream not joined to original stream`、`stream that was not joined to the original stream` 等 graph capture 错误。发现失败签名时立即结束当前 readiness 等待，进入模式文档规定的失败或兼容回退流程。

## 普通 DP/TP 验证

读取 `multi-node-deployment.md` 后，只在 Master 和全部 Worker 都已启动后执行集群 readiness。DP RPC 和 HTTP API 不得作为 Worker 启动前的门禁。至少检查：

1. Master 和每个 Worker 的容器名称、标签、运行状态；
2. 每个 Worker 的 vLLM headless 进程、`--headless` 参数和 `data_parallel_start_rank`；
3. Master 的 DP RPC 端口，此项在所有 rank 启动后检查，并确认 Worker 可以访问；
4. 所有 DP rank 是否覆盖、唯一且连续；
5. Master 的 `/v1/models` 返回预期模型名称；
6. Master 或计划中的公共入口是否可以发送最小 completion；
7. 日志中没有未处理的 HCCL、GLOO、NPU 或模型加载错误。

Worker 使用 `--headless` 时没有 OpenAI HTTP 服务；不要请求 Worker 的 `/v1/models`。只检查 Master 的 HTTP 200 仍然不够，必须同时完成 Worker 进程、rank、RPC 可达性和日志检查。

如果 DP/TP 日志匹配 graph capture 失败，按照 `multi-node-deployment.md` 对全部 rank 一致增加 `--enforce-eager`，并最多执行一次完整重启。不得让 readiness 轮询一直等待已经无法完成的 `capture_model`。

## PD 验证

读取 `multi-node-PD-disaggregated-deployment.md` 后，至少检查：

1. KV Transfer/Mooncake 容器、端口和 readiness；
2. 所有 Prefill 容器的角色标签、rank 和对应的 readiness；提供 HTTP API 的入口检查模型身份，headless worker 检查进程、DP 通信和模型加载日志；
3. 所有 Decode 容器按相同规则检查；
4. 按 `multi-node-PD-disaggregated-deployment.md` 的“KV Connector 检查”完成配置、端口和启动日志门禁；该门禁不单独宣称 connector 已完成实际 KV 传输；
5. Proxy 容器、监听端口和后端列表；后端只包含计划中提供 API 的 P/D 入口，不包含 headless worker；
6. 通过 Proxy 的最小 completion 成功，并确认验证时间窗口内 P/D 日志没有 KV 获取、发送、远端 engine 查询或等待 P side ready 失败；以此作为 connector 实际可用的端到端证据。

Prefill、Decode、Mooncake 分别健康，但 Proxy 不能转发完整请求时，不能报告成功。

## 最小业务请求

请求必须使用计划中声明的模型名称和公共入口。例如：

```bash
curl --fail-with-body http://<public_endpoint>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<served_model_name>",
    "messages": [{"role": "user", "content": "请用一句话说明服务已启动。"}],
    "max_tokens": 16
  }'
```

验证响应包含可解析的选择结果、非空文本和正确模型标识。不要把单纯 HTTP 200 当作业务成功。

## 结果状态

- `ready`：容器、模式相关 readiness、模型身份和完整最小业务请求全部通过；
- `failed`：没有形成可使用的服务，并且清理完成或清理失败已明确记录；
- `partially_started`：部分服务已启动，但仍有服务失败、超时、rank 不完整、Proxy 不可用或业务请求未通过。

结果保存到：

```text
<attempt-dir>/deployment_report.json
```

至少记录状态、服务映射、容器名、实际挂载和设备校验、readiness 证据、提供 API 的服务的模型身份、业务请求结果、失败原因和清理动作。

## 精确清理

清理前在每个节点重新检查：

```bash
docker ps -a \
  --filter "label=boom.skill.owner=standalone-skill" \
  --filter "label=boom.skill.deployment_id=<deployment_id>" \
  --format '{{.Names}}\t{{.Status}}\t{{.Labels}}'
```

只处理同时匹配当前 deployment label 和计划中容器名的容器。按模式逆依赖顺序清理：

```text
multi-node deployment：Worker → Master
multi-node PD-disaggregated deployment：Proxy → Decode/Prefill → KV Transfer/Mooncake
```

禁止使用 `pkill`、宽范围 `docker rm`、猜测容器 ID 或把端口占用者当作本次部署的服务。

## 后续 status/stop

后续操作必须先读取：

```text
deployment-plan.json
deployment_report.json
```

再重新检查容器标签、名称和运行状态。文件缺失、标签不匹配或所有权无法确认时，停止操作并报告原因。
多机部署没有自动重试、断线恢复或资源租约；重试必须创建新的 attempt 目录。
