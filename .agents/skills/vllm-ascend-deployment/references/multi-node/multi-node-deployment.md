# multi-node deployment

本文件定义 **multi-node deployment** 的普通
vLLM-Ascend 多机推理：每个节点上的多张 NPU 组成一个 TP worker，多个节点之间组成
DP worker。当前第一版不自动支持 PP、EP、跨节点纯 TP 或其他未审核的并行方式。
最终命令以当前 attempt 的 `deployment-plan.json` 为准，并通过主 Skill 的统一报告交付。

## 拓扑定义

标准拓扑如下：

```text
node-0 / Master：TP worker，DP rank 0
node-1 / Worker：TP worker，DP rank 1
node-2 / Worker：TP worker，DP rank 2
```

如果有 `N` 个节点：

```text
data_parallel_size = N
data_parallel_size_local = 1
data_parallel_start_rank = 0（Master）或节点在 Worker 列表中的 rank（Worker）
tensor_parallel_size = 每个节点使用的 NPU 数量
```

这表示：

```text
节点内部：TP
节点之间：DP
```

不要把 `tensor_parallel_size` 当成节点数量，也不要把物理 NPU 编号当成 DP rank。

## 必填参数

启动前必须确认：

| 参数 | 规则 |
|---|---|
| `nodes` | 节点数量至少为 2，每个节点有唯一 ID、SSH 地址和数据面 IP |
| `image` | 已确认的 vLLM-Ascend 镜像 |
| `model_path` | 每个节点都存在且挂载到相同容器路径 |
| `served_model_name` | 所有节点一致 |
| `npu_per_node` | 实际要使用的 NPU 数量 |
| `tensor_parallel_size` | 必须等于每个 worker 实际使用的 NPU 数量，除非有已审核模板明确说明不同 |
| `data_parallel_size` | 必须等于节点数 |
| `master_node_ip` | 必须是 Master 的数据面 IP，Worker 可访问 |
| `data_parallel_rpc_port` | Master 上可用，Worker 可访问 |
| `network_interface` | 每个节点的实际物理通信网卡 |
| `port` | 每个节点上的 vLLM API 端口；使用 host network 时允许各节点使用相同端口 |
| `max_model_len` | 已确认的模型上下文长度 |
| `max_num_seqs` | 已确认的并发序列数 |
| `gpu_memory_utilization` | 已确认的 NPU 内存利用率 |

如果用户只说“每台机器 8 卡”，还必须确认实际使用的 Device 编号、节点 IP、镜像、模型目录和通信网卡。

## 启动前环境检查

在每个节点执行只读检查：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 <user>@<ip> "echo SSH_OK"
ssh <user>@<ip> "docker version --format '{{.Server.Version}}'"
ssh <user>@<ip> "docker image inspect <image>"
ssh <user>@<ip> "npu-smi info"
ssh <user>@<ip> "ip -o link"
ssh <user>@<ip> "test -r <model_host_path> && echo MODEL_OK"
```

Docker 镜像内部的 NPU 检查应在实际运行镜像中执行。如果镜像不包含 `npu-smi`，必须确认宿主机设备映射和昇腾运行时挂载，不要直接认为设备可用。

### NPU Device 选择

`ASCEND_RT_VISIBLE_DEVICES` 使用 `npu-smi info` 输出中的 Device 列，不使用 NPU 物理编号或 Chip 列。例如选择 Device 0、1 时：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1
```

对于 `npu-smi info` 显示一张物理卡对应两个 Device 的 310P 机型，必须先按相同 NPU ID 或 Bus-Id 确认 Device 配对：

- 使用 1 个 Device 时，选择一个已确认空闲的 Device；
- 使用 2 个 Device（例如 TP=2）时，选择同一张物理卡的完整 Device 对，例如实际拓扑中的 `0,1` 或 `2,3`，不要选择分属两张卡的组合；
- 使用 4 个或 8 个 Device 时，可以组合多张物理卡，但每张卡的 Device 对必须完整，例如实际拓扑允许时使用 `0,1,2,3`；
- 上述编号只是示例，必须以当前节点的 `npu-smi info` 映射为准。910 系列不适用双 Device 物理卡配对规则。

选中的 Device 上有其他进程时，先报告占用情况，不要自动停止其他进程。

## Ascend 主机运行时挂载

普通 DP/TP 的 Master 和 Worker 都使用下面七项固定宿主机挂载，全部只读：

```text
/usr/local/dcmi:/usr/local/dcmi:ro
/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
/usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
/etc/ascend_install.info:/etc/ascend_install.info:ro
/usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro
/etc/hccn.conf:/etc/hccn.conf:ro
```

启动前在每个节点检查上述宿主机源路径、模型目录和计划中的 `/dev/davinci<device_id>`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`。不要从镜像创建临时容器读取运行时清单，也不要在路径检查失败时猜测替代路径。

### 通信环境变量

每个节点根据自身数据面 IP 和网卡生成：

```bash
HCCL_IF_IP=<本节点数据面 IP>
GLOO_SOCKET_IFNAME=<通信网卡>
TP_SOCKET_IFNAME=<通信网卡>
HCCL_SOCKET_IFNAME=<通信网卡>
OMP_PROC_BIND=false
OMP_NUM_THREADS=1
HCCL_BUFFSIZE=200
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
VLLM_ASCEND_BALANCE_SCHEDULING=1
ASCEND_RT_VISIBLE_DEVICES=<已确认的 Device 列表>
```

不要把 Master 的 IP 填入所有节点的 `HCCL_IF_IP`；每个节点使用自己的数据面 IP，只有 `--data-parallel-address` 使用 Master IP。

### 图模式兼容与 `--enforce-eager`

910 系列默认可以先使用图模式，但以下任一条件成立时必须在该次部署的全部 DP rank 上一致增加 `--enforce-eager`：

- 已审核的模型/镜像运行模板明确要求 eager；
- 目标硬件为 310P；
- 启动日志出现 `rtStreamEndCapture`、`stream not joined to original stream`、`stream that was not joined to the original stream` 或其他明确的 graph capture 失败。

启动观察必须把这些字符串作为失败签名，而不是只等待 readiness 超时。首次发现图捕获失败时，停止并删除当前 deployment label 精确匹配的 Master/Worker 容器，将 `--enforce-eager` 写回计划，并用所有 rank 一致的参数重新启动一次。兼容重试最多一次；重试仍失败时报告 `failed` 或 `partially_started`，不要无限重试。

`--enforce-eager` 会关闭运行时图捕获，可能降低吞吐或增加延迟。结果文件必须记录它是用户指定、模板要求还是由图捕获失败触发。

## vLLM 命令门禁

每个节点的最终 argv 必须包含以下参数：

```text
vllm serve <model_container_path>
--host 0.0.0.0
--port <api_port>
--tensor-parallel-size <tensor_parallel_size>
--data-parallel-size <data_parallel_size>
--data-parallel-size-local 1
--data-parallel-start-rank <rank>
--data-parallel-address <master_node_ip>
--data-parallel-rpc-port <data_parallel_rpc_port>
--served-model-name <served_model_name>
--max-num-seqs <max_num_seqs>
--max-model-len <max_model_len>
--gpu-memory-utilization <gpu_memory_utilization>
--trust-remote-code
```

命令还必须满足：

- Master 的 `data_parallel_start_rank` 为 `0`；
- Worker 的 rank 从 `1` 开始且连续；
- Master 不添加 `--headless`；
- Worker 必须添加 `--headless`；
- 命令保存日志：`> /data/logs/vllm_deploy.log 2>&1`；
- 310P 必须增加 `--enforce-eager --dtype float16`；已确认的图捕获失败也必须增加 `--enforce-eager`；
- 量化模型只有在运行时确认支持时才增加 `--quantization ascend`；
- 不得同时出现互相冲突的参数；
- 不得把用户未提供且无法只读确认的值擅自填入命令。

## Docker 启动模板

使用 host network 时，每个节点的容器命令遵循以下结构，具体路径和参数必须来自计划：

### Master

```bash
docker run --detach \
  --name boom-skill-<deployment_id>-vllm-master \
  --label boom.skill.owner=standalone-skill \
  --label boom.skill.deployment_id=<deployment_id> \
  --label boom.skill.service_id=vllm-master \
  --network host \
  --shm-size <shared_memory> \
  --env HCCL_IF_IP=<master_ip> \
  --env GLOO_SOCKET_IFNAME=<network_interface> \
  --env TP_SOCKET_IFNAME=<network_interface> \
  --env HCCL_SOCKET_IFNAME=<network_interface> \
  --env VLLM_ASCEND_BALANCE_SCHEDULING=1 \
  --env ASCEND_RT_VISIBLE_DEVICES=<devices> \
  --volume /usr/local/dcmi:/usr/local/dcmi:ro \
  --volume /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  --volume /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  --volume /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  --volume /etc/ascend_install.info:/etc/ascend_install.info:ro \
  --volume /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
  --volume /etc/hccn.conf:/etc/hccn.conf:ro \
  --volume <model_host_path>:<model_container_path>:ro \
  --volume <log_host_path>:/data/logs \
  --device /dev/davinci<device_id> \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  <image> bash -lc 'source /root/.bashrc && vllm serve <model_container_path> \
    --host 0.0.0.0 --port <api_port> \
    --tensor-parallel-size <tensor_parallel_size> \
    --data-parallel-size <data_parallel_size> \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address <master_ip> \
    --data-parallel-rpc-port <rpc_port> \
    --served-model-name <served_model_name> \
    --max-num-seqs <max_num_seqs> --max-model-len <max_model_len> \
    --gpu-memory-utilization <gpu_memory_utilization> \
    --trust-remote-code <hardware_or_model_flags> \
    > /data/logs/vllm_deploy.log 2>&1'
```

### Worker

Worker 命令与 Master 相同，但必须使用当前 Worker 的 IP、当前 Worker 的 Device 和连续的 rank，并增加：

```text
--headless
```

禁止在 Master 和 Worker 两侧都添加 `--headless`。

## 启动顺序

1. 在所有节点确认端口、模型目录、镜像和 NPU；
2. 在 Master 创建并启动容器；
3. 对 Master 做一次有界的启动检查：确认容器仍为 running、主进程没有立即退出，并检查最近日志中没有参数无法识别、模型目录不存在、NPU 不可用、`rtStreamEndCapture` 或 stream join 等立即致命错误；
4. 不等待 Master 的 DP RPC、HTTP API、模型加载或完整 readiness，立即在所有 Worker 上创建容器，并行启动；
5. Master 和全部 Worker 都已启动后，再统一轮询 DP RPC、容器状态、日志和 vLLM readiness；
6. 对 Master 的 `/v1/models` 检查模型身份；Worker 使用 `--headless`，不检查 `/v1/models`，改为检查进程、rank、RPC 可达性和日志；
7. 从 Master 或计划中的公共入口发送最小 completion 请求。

Worker 启动前禁止持续轮询 Master 的 DP RPC 或 `/v1/models`。Master 的分布式初始化可能需要其他 rank 加入，把 RPC 或 API readiness 作为 Worker 前置门禁可能形成循环等待。只有 Master 容器已经退出或日志出现明确致命错误时，才停止 Worker 启动并进入精确范围的失败处理。

如果图捕获错误只在所有 rank 启动后出现，仍要把它视为启动失败，并按“图模式兼容与 `--enforce-eager`”规则对整个 DP 部署统一重启，不能只给单个 rank 增加参数。

## 成功判定

只有以下条件全部满足，普通 DP/TP 部署才是 `ready`：

- 所有节点上的容器都带有当前 deployment label；
- Master 和所有 Worker 都在运行；
- 所有 rank 都已覆盖且没有重复；
- Master RPC 端口可访问；
- Master 的 `/v1/models` 返回预期模型；
- 每个 Worker 的 headless 进程、连续 rank 和到 Master 的 DP RPC 可达性检查通过；
- 公共入口的最小 completion 成功；
- 日志没有未处理的通信、NPU 或模型加载错误。

只有部分节点启动或只有单个节点 API 可用时，状态为 `partially_started`，不能报告普通多机部署成功。
