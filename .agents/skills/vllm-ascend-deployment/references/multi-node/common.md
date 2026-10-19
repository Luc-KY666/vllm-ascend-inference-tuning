# 多机 Docker 部署共用规则

本文件是 `vllm-ascend-deployment` 的多机部署共用参考资料，适用于 **multi-node deployment**
和 **multi-node PD-disaggregated deployment**。它复用主 Skill 的 attempt 目录、Provider
生命周期和 `deployment_report.json`；本目录不再作为独立 Skill 入口，也不调用单机部署的
本地 NPU/端口探测结论。

本文件适用于 **multi-node deployment** 和 **multi-node PD-disaggregated deployment**。它只定义公共输入、计划文件、SSH/Docker 安全边界和结果记录；具体命令必须读取对应的部署说明。

## 公共计划格式

在任何远端变更前写入当前 attempt 目录：

```text
<attempt-dir>/deployment-plan.json
```

计划是当前 attempt 的派生 JSON artifact，不是 Provider 输入，也不写入 Boom Lifecycle 数据库。示例：

```json
{
  "deployment_id": "<skill-generated-instance-id>",
  "deployment_type": "multi-node deployment",
  "runtime": {
    "image": "registry.example/vllm-ascend:tag",
    "network_mode": "host",
    "model_mount": "/data/models/qwen:/models/qwen:ro"
  },
  "nodes": [
    {
      "id": "node-0",
      "host": "10.0.0.1",
      "user": "root",
      "ssh_port": 22,
      "execution_transport": "ssh",
      "network_ip": "10.0.0.1",
      "network_interface": "eth0",
      "npu_devices": [0, 1, 2, 3],
      "role": "master"
    }
  ],
  "services": [
    {
      "id": "vllm-master",
      "node_id": "node-0",
      "role": "master",
      "container_name": "boom-skill-deploy-qwen-001-vllm-master",
      "argv": [],
      "environment": {},
      "mounts": [],
      "devices": [],
      "ports": [],
      "depends_on": [],
      "readiness": {}
    }
  ],
  "validation": {
    "endpoint": "http://10.0.0.1:8000",
    "model": "qwen",
    "request": {"prompt": "hello", "max_tokens": 8}
  }
}
```

计划中必须有最终展开的命令、节点、容器、挂载、设备、端口和 readiness。密码、私钥、token 和其他凭据不能写入计划。

## 当前 MVP 的固定 Ascend 主机运行时挂载

普通 DP/TP 和 PD 容器都使用同一组固定的 Ascend 主机运行时挂载。当前 MVP 不从镜像读取挂载清单，也不创建临时容器读取清单；Skill 必须直接使用下面七项，并在每个节点启动前检查源路径：

| 宿主机源 | 容器目标 | 类型 | 检查 |
|---|---|---|---|
| `/usr/local/dcmi` | `/usr/local/dcmi` | 目录 | `test -d` 且可读 |
| `/usr/local/bin/npu-smi` | `/usr/local/bin/npu-smi` | 文件 | `test -x` |
| `/usr/local/Ascend/driver/lib64` | `/usr/local/Ascend/driver/lib64` | 目录 | `test -d` 且可读 |
| `/usr/local/Ascend/driver/version.info` | `/usr/local/Ascend/driver/version.info` | 文件 | `test -f` 且可读 |
| `/etc/ascend_install.info` | `/etc/ascend_install.info` | 文件 | `test -f` 且可读 |
| `/usr/local/Ascend/driver/tools/hccn_tool` | `/usr/local/Ascend/driver/tools/hccn_tool` | 文件 | `test -x` |
| `/etc/hccn.conf` | `/etc/hccn.conf` | 文件 | `test -f` 且可读 |

每项都以只读方式挂载。`/home`、`/data`、模型目录和日志目录不是驱动挂载，不能用它们替代上表中的路径；模型和日志作为服务自己的挂载单独加入计划。

每个使用 NPU 的服务还必须加入当前节点选中的 `/dev/davinci<device_id>`，以及：

```text
/dev/davinci_manager
/dev/devmm_svm
/dev/hisi_hdc
```

启动前对每个节点执行等价检查：固定源路径、模型宿主机目录、选中的 `/dev/davinci<device_id>` 和三个共享设备都必须存在。任一检查失败时报告 `runtime_incomplete`，不得启动 Master 或 Worker。

启动后对每个新建容器执行 `docker inspect`，确认 `HostConfig.Binds` 包含七项固定挂载，`HostConfig.Devices` 包含计划中的 NPU 和共享设备。实际配置与计划不一致时停止当前 deployment 的容器并报告失败。

## 公共输入检查

执行 `docker pull` 或 `docker run` 前，逐项检查：

- `deployment_id` 由 Skill 根据当前 attempt 生成，用户不需要提供；
- 节点 ID、服务 ID 和容器名唯一；
- 每个节点的宿主机 SSH 地址、用户、端口和数据面 IP 已确认，`execution_transport` 为 `ssh`；
- 所有节点使用已确认的 Docker 镜像；
- 模型宿主机目录存在，挂载目标是绝对路径；
- 七项固定 Ascend 挂载源和当前节点 NPU 设备存在；
- 每个服务需要的 NPU Device 存在且未重复分配；
- 同一节点没有重复绑定宿主机端口；
- 通信网卡处于 UP 状态且节点之间可访问；
- 所有命令参数都有来源：用户输入、只读检测或模式参考文档；
- 每个容器都有有界的 process、TCP、HTTP 或日志 readiness；
- 最终业务请求的公共入口、模型名称和请求体已确认。

不能用占用端口的进程猜测服务归属，也不能用默认 rank、默认网卡或默认模型目录替代缺失输入。

## SSH 规则

Agent 的运行环境与部署目标宿主机是两个不同概念。当前 Skill 对所有节点固定使用 SSH，包括运行 Pi/Agent 容器的那台物理机：

- `nodes[].host` 必须是 Pi/Agent 容器能够访问的物理宿主机 SSH 地址，不能填写 `localhost`、`127.0.0.1` 或 `::1`；
- 宿主机 SSH 地址与 `network_ip` 可以不同；前者用于控制面登录，后者用于 vLLM/HCCL 数据面通信；
- 所有宿主机路径和设备检查，以及 `docker version`、`docker image inspect`、`docker run`、`docker inspect`、`docker logs`、`docker stop` 和 `docker rm`，都必须通过对应节点的 SSH 连接执行；
- 不得在 Pi/Agent 容器内启动 `dockerd`，不得把容器内的 Docker daemon 当作部署目标，也不得使用 `/var/run/docker.sock`、`DOCKER_HOST` 或本地 Docker context 绕过本 Skill 的 SSH 边界。

使用已经配置的 SSH key 或 SSH agent，并使用非交互式检查：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> "echo SSH_OK"
ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> \
  "hostname && docker info --format '{{.Name}}' && test -e /dev/davinci_manager"
```

第二条检查用于确认实际执行位置是计划中的物理宿主机，并确认该 SSH 用户能够访问宿主机 Docker 和 NPU 设备。SSH 不通或 Docker 权限不足时报告节点和原因，要求配置凭据后再继续。不要回退到 Pi/Agent 容器内执行，不要把密码写在命令行，不要把私钥复制到计划目录，不要使用 `sshpass`，也不要把敏感信息写入日志。

## Docker 容器规则

每个容器使用稳定名称和三类标签。标签用于本次 attempt 的精确所有权边界，
不代表 Boom Lifecycle 的资源租约：

```text
容器名：boom-skill-<deployment_id>-<service_id>
boom.skill.owner=standalone-skill
boom.skill.deployment_id=<deployment_id>
boom.skill.service_id=<service_id>
```

下面的 `docker run` 是在目标宿主机上执行的远端 payload，不是在 Pi/Agent 容器内直接执行的本地命令：

```bash
docker run --detach \
  --name boom-skill-<deployment_id>-<service_id> \
  --label boom.skill.owner=standalone-skill \
  --label boom.skill.deployment_id=<deployment_id> \
  --label boom.skill.service_id=<service_id> \
  --network host \
  --shm-size <shared_memory> \
  --env KEY=value \
  --volume /usr/local/dcmi:/usr/local/dcmi:ro \
  --volume /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  --volume /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  --volume /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  --volume /etc/ascend_install.info:/etc/ascend_install.info:ro \
  --volume /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
  --volume /etc/hccn.conf:/etc/hccn.conf:ro \
  --volume /approved/host/model:/models/model:ro \
  --volume /approved/host/logs:/data/logs \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  <image> <mode-specific-command>
```

`--privileged`、host IPC、共享内存大小和昇腾运行时挂载只能由镜像和模式文档明确要求时使用。不要复制其他部署的宿主机控制参数。

执行前展示最终命令摘要。执行后保存脱敏的命令、时间戳和结果，不保存完整密码或 token。

## 启动顺序

启动顺序由模式文档决定：

- **multi-node deployment**：先启动 Master，只确认容器和主进程没有立即退出、日志没有明显的参数或环境错误，然后立即并行启动全部 Worker。不要在 Worker 启动前等待 DP RPC、HTTP API 或模型 readiness；这些检查必须在所有节点都启动后统一执行；
- **multi-node PD-disaggregated deployment**：先启动 KV Transfer/Mooncake，再并行启动 Prefill 和 Decode，最后启动 Proxy。

**multi-node deployment** 的 Master 和 Worker 属于同一个分布式启动组，不适用逐层 readiness 门禁。**multi-node PD-disaggregated deployment** 的依赖层仍然要求上一层全部通过 readiness 后才能启动下一层。发生明确失败时停止后续启动，只清理当前 deployment label 匹配的容器。

## 本地记录

每个部署至少保存：

```text
<attempt-dir>/
├── deployment-plan.json
├── commands.log
└── deployment_report.json
```

`commands.log` 记录脱敏命令和观察时间。`deployment_report.json` 至少记录：

- Skill 生成的 `deployment_id`；
- `deployment_type`；
- `execution.deployment_type` 和 `transport=ssh`；
- 节点和服务映射；
- 容器名和容器标签摘要；
- 每个服务的 readiness 结果；
- 模型身份；
- 最小业务请求结果；
- 失败服务和清理动作；
- 最终状态 `ready`、`failed` 或 `partially_started`。

主 Skill 的报告结构是唯一结果合同。旧版 `result.json` 和 `.boom-skill/deployments/`
路径只作为迁移历史，不再创建。
