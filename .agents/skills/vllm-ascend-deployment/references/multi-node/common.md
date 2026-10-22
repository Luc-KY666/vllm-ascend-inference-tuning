# 多机 Docker 部署共用规则

本文件是 `vllm-ascend-deployment` 的多机部署共用参考资料，适用于 **multi-node deployment**
和 **multi-node PD-disaggregated deployment**。它复用主 Skill 的 Provider
生命周期和 `deployment_report.json` 报告结构，定义公共输入、部署与验收流程、SSH/Docker 安全边界和结果记录；
本目录不再作为独立 Skill 入口，也不调用单机部署的本地 NPU/端口探测结论。具体服务命令必须
读取对应的部署说明。

## 执行流程

多机部署必须按以下顺序执行，前一阶段失败时停止，不得跳到远端变更或服务启动。对目标节点的操作通过对应 SSH 连接执行，当前 Agent 所在节点本身就是目标节点时例外——该节点直接在宿主机本地执行、不经 SSH：

1. **输入检查**：识别部署意图（**multi-node deployment** 或 **multi-node
   PD-disaggregated deployment**），并通过 SSH 验证每个输入节点存在、节点间可连通、
   模型路径存在，探测各节点 NPU 卡。
2. **宿主机运行时预检**：在每个节点检查 Docker 镜像可用、容器执行权限和七项固定 Ascend
   挂载源路径；任一节点失败时报告 `runtime_incomplete`，不启动任何容器。
3. **部署**：按部署类型读取对应模式文档，获得具体部署配置和部署策略，按其启动顺序通过
   SSH 在目标宿主机执行 `docker pull`/`docker run`，不得在 Agent/Pi 容器内操作 Docker daemon。
4. **服务验收与报告记录**：按部署类型执行对应模式文档定义的验收策略；验收完成后
   将实际部署配置和验收结果写入部署目录 `/data/boom/deployment`。

## 1. 输入检查

SSH 连接规则：端口使用用户明确指定的值，否则默认 `22`；使用已配置的 SSH key 或 SSH agent 做非交互式登录，不把密码写在命令行，不使用 `sshpass`。宿主机 SSH 地址与 `network_ip` 可以不同：前者用于控制面登录，后者用于 vLLM/HCCL 数据面通信。

1. **意图识别**：仅说“多机部署”判定为 **multi-node deployment**；出现“Prefill/Decode”“PD 分离”“Mooncake”或“KV Transfer”判定为 **multi-node PD-disaggregated deployment**。
2. **节点存在**：`host` 不能是 `localhost`/`127.0.0.1`/`::1`；对每个节点执行非交互式 SSH 登录，返回的 hostname 与输入一致：

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> "hostname"
   ```

3. **节点间可连通**：从每个节点 ping 其余节点的数据面 IP：

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> \
     "ping -c 1 -W 2 <other_node_network_ip>"
   ```
4. **模型路径存在**：模型目录必须位于每个加载模型节点的本地磁盘（普通多机为全部节点，PD 分离为全部 Prefill/Decode 节点，Proxy/Mooncake 不需要）。对每个加载模型的节点通过 SSH 检查：

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> \
     "test -d <model_host_path> && test -f <model_host_path>/config.json && echo MODEL_OK"
   ```

5. **NPU 卡探测**：探测每个节点的 NPU 卡数量；用户指定了节点上的可用卡时只能使用指定卡，未指定时默认全部可用：

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> "npu-smi info"
   ```

任一项失败时报告节点和原因，不继续后续预检。

## 2. 宿主机运行时预检

在第 1 节建立 SSH 通道后，对每个节点确认用户指定的 Docker 镜像可用：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> \
  "docker image inspect <image> --format '{{.RepoTags}}'"
```

每个节点还检查已有容器执行权限：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> \
  "docker ps -a --format '{{.Names}}\t{{.Status}}'"
```

每个节点检查七项固定 Ascend 运行时挂载源路径，缺哪个报哪个：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 -p <ssh_port> <user>@<host> \
  'for p in /usr/local/dcmi /usr/local/bin/npu-smi /usr/local/Ascend/driver/lib64 \
            /usr/local/Ascend/driver/version.info /etc/ascend_install.info \
            /usr/local/Ascend/driver/tools/hccn_tool /etc/hccn.conf; do \
     test -e "$p" || echo "MISSING: $p"; \
   done'
```

所有 Docker 操作都在目标宿主机上执行，通过对应节点的 SSH 连接；当前 Agent 所在节点本身就是目标节点时，该节点在宿主机本地执行、不经 SSH。不得在 Pi/Agent 容器内直接执行本地命令或启动 `dockerd`，也不得使用 `/var/run/docker.sock`、`DOCKER_HOST` 或本地 Docker context 绕过 SSH 边界。普通多机部署与PD分类多机部署对于docker容器的拉起可以采用相同的格式，8卡样例挂载命令模版如下：

```bash
docker run --detach \
  --name <node-name> \
  --network host \
  --shm-size 250g \
  --env KEY=value \
  --volume /usr/local/dcmi:/usr/local/dcmi:ro \
  --volume /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  --volume /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  --volume /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  --volume /etc/ascend_install.info:/etc/ascend_install.info:ro \
  --volume /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
  --volume /etc/hccn.conf:/etc/hccn.conf:ro \
  --volume <model_host_path>:/models/model:ro \
  --volume /data/boom/deployment:/models/deployment \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  <image> <mode-specific-command>
```

其中：
- `<node-name>`：容器名称，默认使用当前机器的 IP 地址；
- `--volume <model_host_path>:/models/model:ro`：`<model_host_path>` 为该节点本地磁盘上的实际模型目录，只读挂载；每个加载模型的服务节点都必须有此挂载（普通多机为全部节点，PD 分离为全部 Prefill/Decode 节点，Proxy/Mooncake 不需要），且所有节点挂载到相同的容器内路径 `/models/model`；
- `--volume /data/boom/deployment:/models/deployment`：宿主机日志目录，用于在宿主机读取容器的执行记录；服务日志写入容器内 `/models/deployment`；
- `--device=/dev/davinci<N>`：挂载第 1 节在该节点探测确认的可用 NPU 卡；用户指定卡号时只挂载指定卡，未指定时挂载全部可用卡，模板中的 `davinci0`-`davinci7` 仅为 8 卡样例；
- `<image>`：本节已确认可用的用户指定镜像；
- `<mode-specific-command>`：容器内执行的服务启动命令，按部署类型读取对应模式文档生成的实际部署参数命令。

任一检查失败时报告节点、原因和 `runtime_incomplete`，不启动任何容器。

## 3. 部署

在第 2 节预检通过后，按部署类型读取对应模式文档（普通多机见 [multi-node-deployment.md](multi-node-deployment.md)，PD 分离多机见 [multi-node-PD-disaggregated-deployment.md](multi-node-PD-disaggregated-deployment.md)），获得具体部署配置和部署策略，并按其启动顺序通过 SSH 执行部署：

- **multi-node deployment**：先启动 Master，只确认容器和主进程没有立即退出、日志没有明显的参数或环境错误，然后立即并行启动全部 Worker。Master 和 Worker 同属一个分布式启动组，Worker 启动前不等待 DP RPC、HTTP API 或模型 readiness，这些检查在全部节点启动后统一执行；
- **multi-node PD-disaggregated deployment**：先启动 KV Transfer/Mooncake，等待其 readiness 通过后并行启动全部 Prefill 和 Decode，全部 P/D 就绪后再启动 Proxy。

发生明确失败时停止后续启动，进入精确清理。清理前在每个节点重新检查容器：

```bash
docker ps -a --format '{{.Names}}\t{{.Status}}'
```

只清理本次部署创建的容器，按容器名精确匹配，按逆依赖顺序执行：

```text
multi-node deployment：Worker → Master
multi-node PD-disaggregated deployment：Proxy → Decode/Prefill → KV Transfer/Mooncake
```

禁止使用 `pkill`、宽范围 `docker rm`、猜测容器 ID 或把端口占用者当作本次部署的服务。

## 4. 服务验收与报告记录

服务验收按部署类型执行对应模式文档定义的验收策略：普通多机和 PD 分离的模式文档各自规定验收入口和验收内容；并行策略等内部配置不纳入验收。验收失败时按第 3 节的精确清理处理，报告 `failed` 或 `partially_started`。

验收完成后，将实际部署配置和验收结果写入部署目录 `/data/boom/deployment`（容器内 `/models/deployment`）：

```text
/data/boom/deployment/
├── commands.log
└── deployment_report.json
```

`commands.log` 记录脱敏命令和观察时间。`deployment_report.json` 是当前部署的派生 JSON artifact，至少记录：
- `deployment_type`；
- `execution.deployment_type` 和 `transport=ssh`；
- 节点和服务映射；
- 容器名；
- 每个服务最终展开的命令、挂载、设备、端口；
- 模型身份；
- 最小业务请求结果；
- 失败服务和清理动作；
- 最终状态：`ready`（验收全部通过）、`failed`（没有形成可使用的服务，且清理完成或清理失败已明确记录）或 `partially_started`（部分服务已启动，但验收未全部通过）。

密码、私钥、token 和其他凭据不能写入报告。主 Skill 的报告结构是唯一结果合同。
