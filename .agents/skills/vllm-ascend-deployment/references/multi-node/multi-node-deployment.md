# multi-node deployment

本文件定义普通 vLLM-Ascend 多机部署的模式专属规则。公共输入检查、SSH/Docker
边界、宿主机运行时预检、通用挂载、容器清理、业务验收和报告记录统一按
[common.md](common.md) 执行；本文件只负责以下内容：

- 每个节点空闲 NPU 的确认规则和通信环境变量；
- 普通多机的并行策略；
- 普通多机的专用参数和 Master/Worker 的具体拉起命令；
- 启动后的分布式状态检查。

本文件由主 Skill 在普通多机模式下与 `common.md` 共同作为参考；公共规则和模式专属
规则分别以对应文档为准。

## 部署流程（总览）

普通多机部署由 `common.md` 负责公共输入检查、宿主机运行时预检、Docker 边界、
通用失败处理、清理和公共业务验收。本节只描述普通多机部署的专属启动流程：

1. **确认每节点空闲卡**：按第 1 节，以 `common.md` 公共预检获得的 `npu-smi info`
   为准确认每个节点的空闲 NPU 及其最大内存，生成该节点选中的 Device 列表和
   通信环境变量。
2. **确定并行策略和参数**：按第 2 节采用节点内 TP、节点间 DP，不自动选择 PP、EP 或跨节点
   纯 TP；以每个节点选中的 Device 数量确定 `tensor_parallel_size`，要求各节点一致；
   以节点数确定 `data_parallel_size`，设置 `data_parallel_size_local=1`，Master 使用
   rank `0`，Worker 使用从 `1` 开始的连续 rank。按第 3 节确认专用参数，用户未提供且
   无法确认的值不得猜测。TP 需求或 rank 校验失败时停止。
3. **生成启动命令**：按第 3 节展开每个节点的 vLLM argv 和容器命令。
4. **启动 Master**：按第 3 节创建并启动 DP rank `0` 容器，做一次有界启动检查，确认
   容器和主进程没有立即退出，日志没有参数、模型、NPU、HCCL 或明确 graph capture 错误。
5. **并行启动 Worker**：按第 3 节在 Master 通过启动检查后，不等待 DP RPC、HTTP API
   或模型加载完成，立即在所有 Worker 上创建容器并行启动。只有 Master 已退出或出现
   明确致命错误时，才停止后续 Worker 启动。
6. **统一检查分布式状态**：按第 4 节在 Master 和全部 Worker 都启动后，统一检查容器
   状态、DP RPC、进程、rank、模型身份和日志。全部通过后进入 `common.md` 第 4 节的
   业务验收；graph capture 错误若在所有 rank 启动后出现，按 `common.md` 第 3 节精确
   清理后对全部 rank 一致增加 `--enforce-eager` 完整重启一次，不能只给单个 rank
   增加参数。

本文件后续章节分别定义上述流程使用的空闲卡和通信环境变量、并行策略、专用参数和拉起命令、启动后检查。

## 1. 空闲卡确认和通信环境变量

### 1.1 空闲卡确认

`common.md` 公共预检对每个节点执行 `npu-smi info`：节点上有几张 NPU 卡，该节点
就有几张**可用卡**；可用卡中卡上无进程的才是**空闲卡**。判定以该节点的探测结果
为准，不重复探测，不等待其他任务释放设备，也不用控制端可见卡替代远端事实。

每张空闲卡的最大内存从同一份 `npu-smi info` 输出读取：Memory-Usage 列显示
已用/总量（单位 MB），总量即该卡的最大内存。

`ASCEND_RT_VISIBLE_DEVICES` 使用 `npu-smi info` 输出中的 **Device** 列，不使用物理
NPU 编号或 Chip 列。例如：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1
```

### 1.2 通信环境变量

每个节点使用自己的数据面 IP 和通信网卡生成以下通信环境变量：

```bash
HCCL_IF_IP=<本节点数据面 IP>
GLOO_SOCKET_IFNAME=<本节点通信网卡>
TP_SOCKET_IFNAME=<本节点通信网卡>
HCCL_SOCKET_IFNAME=<本节点通信网卡>
OMP_PROC_BIND=false
OMP_NUM_THREADS=1
HCCL_BUFFSIZE=200
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
VLLM_ASCEND_BALANCE_SCHEDULING=1
ASCEND_RT_VISIBLE_DEVICES=<本节点已确认的 Device 列表>
```

不能把 Master IP 填入所有节点的 `HCCL_IF_IP`；只有 vLLM 的
`--data-parallel-address` 使用 Master IP。通信网卡必须来自该节点实际物理接口，不能
用 SSH 控制面地址代替数据面配置。

## 2. 并行策略

vLLM 支持多种并行策略（TP、DP、PP、EP 及其组合），本文件只采用其中一种：**节点内
TP、节点间 DP**——每个节点内选中的全部 NPU 构成一个 TP 组，每个节点是一个 DP rank，
不自动选择 PP、EP、跨节点 TP 或其他并行方式。

参数按此确定：`tensor_parallel_size` = 单节点使用的 NPU 数（各节点一致），
`data_parallel_size` = 节点数，`data_parallel_size_local` = 1；Master 为 DP rank `0`，
Worker 依次为 `1..N-1`，rank 按节点顺序分配，与物理 NPU 编号无关。所有节点使用
相同的 `data_parallel_size`。

以 3 个节点、每节点 4 张卡为例：`tensor_parallel_size=4`、`data_parallel_size=3`，
Master 为 rank `0`，两个 Worker 为 rank `1`、`2`。

## 3. 专用参数和拉起命令

以下值必须在生成启动命令前确认，不能从未提供或无法只读确认的环境中猜测：

| 参数 | 普通多机规则 |
|---|---|
| `nodes` | 至少 2 个节点；每个节点有唯一 ID、SSH 地址和数据面 IP |
| `image` | 已确认的 vLLM-Ascend 镜像，所有节点使用同一镜像 |
| `model_path` | 每个节点本地存在，按 `common.md` 挂载到相同容器路径 |
| `served_model_name` | Master 和所有 Worker 一致 |
| `tensor_parallel_size` | 等于每个节点实际使用的 NPU 数量，除非已有审核模板明确规定不同 |
| `data_parallel_size` | 等于节点数 |
| `master_node_ip` | Master 的数据面 IP，所有 Worker 可访问 |
| `data_parallel_rpc_port` | Master 上可监听、所有 Worker 可访问 |
| `network_interface` | 每个节点实际用于 HCCL/Gloo/TP 的物理通信网卡 |
| `port` | 每个节点的 vLLM API 端口；host network 下不同节点可以复用同一端口 |
| `max_model_len` | 模型上下文长度；用户未指定时默认 `4096` |
| `max_num_seqs` | 并发序列数；用户未指定时默认 `64` |
| `gpu_memory_utilization` | NPU 内存利用率；用户未指定时默认 `0.9` |

Docker 的公共网络、Ascend 挂载、模型挂载、日志目录和设备参数沿用
`common.md` 第 2 节模板。命令中的 `<common-docker-args>` 必须替换为该模板的公共参数：
`--network host`、共享内存、七项只读 Ascend 挂载、`/models/model` 模型挂载、
`/models/deployment` 日志目录挂载，以及本节点第 1 节选中的 `/dev/davinci<device_id>`、
`/dev/davinci_manager`、`/dev/devmm_svm` 和 `/dev/hisi_hdc`；不能在 Agent/Pi 容器内
直接执行。

### 3.1 Master

```bash
docker run --detach \
  --name <node-name> \
  <common-docker-args> \
  --env HCCL_IF_IP=<master_node_ip> \
  --env GLOO_SOCKET_IFNAME=<master_network_interface> \
  --env TP_SOCKET_IFNAME=<master_network_interface> \
  --env HCCL_SOCKET_IFNAME=<master_network_interface> \
  --env OMP_PROC_BIND=false \
  --env OMP_NUM_THREADS=1 \
  --env HCCL_BUFFSIZE=200 \
  --env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --env VLLM_ASCEND_BALANCE_SCHEDULING=1 \
  --env ASCEND_RT_VISIBLE_DEVICES=<master_devices> \
  <image> bash -lc 'source /root/.bashrc && exec vllm serve /models/model \
    --host 0.0.0.0 --port <master_api_port> \
    --tensor-parallel-size <tensor_parallel_size> \
    --data-parallel-size <data_parallel_size> \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address <master_node_ip> \
    --data-parallel-rpc-port <data_parallel_rpc_port> \
    --served-model-name <served_model_name> \
    --max-num-seqs <max_num_seqs> --max-model-len <max_model_len> \
    --gpu-memory-utilization <gpu_memory_utilization> \
    --trust-remote-code \
    <optional_hardware_or_model_flags> \
    > /models/deployment/vllm-master.log 2>&1'
```

### 3.2 Worker

每个 Worker 使用自己的数据面 IP、通信网卡、空闲 Device 列表和连续 rank，
`--data-parallel-address` 仍指向 Master：

```bash
docker run --detach \
  --name <node-name> \
  <common-docker-args> \
  --env HCCL_IF_IP=<worker_node_ip> \
  --env GLOO_SOCKET_IFNAME=<worker_network_interface> \
  --env TP_SOCKET_IFNAME=<worker_network_interface> \
  --env HCCL_SOCKET_IFNAME=<worker_network_interface> \
  --env OMP_PROC_BIND=false \
  --env OMP_NUM_THREADS=1 \
  --env HCCL_BUFFSIZE=200 \
  --env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --env VLLM_ASCEND_BALANCE_SCHEDULING=1 \
  --env ASCEND_RT_VISIBLE_DEVICES=<worker_devices> \
  <image> bash -lc 'source /root/.bashrc && exec vllm serve /models/model \
    --host 0.0.0.0 --port <worker_api_port> \
    --tensor-parallel-size <tensor_parallel_size> \
    --data-parallel-size <data_parallel_size> \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank <rank> \
    --data-parallel-address <master_node_ip> \
    --data-parallel-rpc-port <data_parallel_rpc_port> \
    --served-model-name <served_model_name> \
    --max-num-seqs <max_num_seqs> --max-model-len <max_model_len> \
    --gpu-memory-utilization <gpu_memory_utilization> \
    --trust-remote-code \
    <optional_hardware_or_model_flags> \
    --headless \
    > /models/deployment/vllm-worker-<rank>.log 2>&1'
```

命令约束：

- Master 的 `data_parallel_start_rank` 为 `0`，Worker 的 rank 从 `1` 开始且连续；
- Master 不添加 `--headless`，Worker 必须添加 `--headless`；
- 启动日志保存到 `/models/deployment` 日志目录；
- 出现 graph capture 失败时至少使用 `--enforce-eager`；
- 量化模型只有在运行时确认支持时才增加 `--quantization ascend`；
- 不得同时出现互相冲突的参数，也不得擅自填入未知值。

## 4. 启动后分布式状态检查

本节对应部署流程第 6 步，只检查普通多机启动后的模式专属状态，不重复公共输入检查、
Docker 预检、清理或公共业务验收：

- 所有 Master/Worker 容器处于运行状态；
- DP rank 覆盖 `0..N-1`，没有重复或缺失；
- Master 的 DP RPC 端口可访问；
- Master 的 `/v1/models` 返回预期 `served_model_name`；
- Worker 使用 `--headless`，其进程存在、rank 正确连续，并且可以访问 Master 的 DP RPC；
- 日志没有未处理的通信、NPU、模型加载或参数错误。

Worker 不检查 `/v1/models`；Worker 的启动状态以 headless 进程、rank、DP RPC 可达性和日志
为准。以上检查全部通过后，进入 `common.md` 第 4 节的业务验收。
