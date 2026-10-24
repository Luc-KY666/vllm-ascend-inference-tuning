# multi-node deployment

本文件定义普通 vLLM-Ascend 多机部署的模式专属规则。公共输入检查、SSH/Docker
边界、宿主机运行时预检、通用挂载、容器清理和报告记录统一按
[common.md](common.md) 执行；本文件只负责以下内容：

- 每个节点空闲 NPU 的确认规则和通信环境变量；
- 普通多机的并行策略；
- 普通多机的配置参数和 Master/Worker 的部署命令；
- 启动后的分布式状态检查和对话服务验收。

本文件由主 Skill 在普通多机模式下与 `common.md` 共同作为参考；公共规则和模式专属
规则分别以对应文档为准。

## 部署流程（总览）

普通多机部署由 `common.md` 负责公共输入检查、宿主机运行时预检、Docker 边界、
通用失败处理、清理和报告记录。本节只描述普通多机部署的专属部署流程：

1. **空闲卡确认和通信环境变量**：确认每个节点的空闲 NPU 及其 Device 列表，
   生成通信环境变量。
2. **并行策略**：采用节点内 TP、节点间 DP，确定并行参数和 rank 分配。
3. **配置参数和部署**：确认配置参数并生成启动命令，先启动 Master，
   通过有界启动检查后并行启动全部 Worker。
4. **启动后检查和服务验收**：统一检查 Master 和全部 Worker 的分布式状态，
   通过后只对 Master 做对话服务验收。

## 1. 空闲卡确认和通信环境变量

### 1.1 空闲卡确认

`common.md` 公共预检对每个节点执行 `npu-smi info`，可得到每张卡的 HBM 大小和
是否存在进程。节点上有几张 NPU 卡，该节点就有几张**可用卡**；可用卡上无进程
的才是**空闲卡**。判定以该节点的探测结果为准，不重复探测，不等待其他任务释放
设备，也不用控制端可见卡替代远端事实。

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

vLLM 支持多种并行策略（TP、DP、PP、EP 及其组合），当前 skill 暂时仅支持其中一种：
**节点内 TP、节点间 DP**——每个节点内选中的全部 NPU 构成一个 TP 组，每个节点是一个 DP rank，
不自动选择 PP、EP、跨节点 TP 或其他并行方式。

参数按此确定：`tensor_parallel_size` = 单节点使用的 NPU 数（各节点一致，除非已有
审核模板明确规定不同），
`data_parallel_size` = 节点数，`data_parallel_size_local` = 1；Master 为 DP rank `0`，
Worker 依次为 `1..N-1`，rank 按节点顺序分配，与物理 NPU 编号无关。所有节点使用
相同的 `data_parallel_size`。

以 3 个节点、每节点 4 张卡为例：`tensor_parallel_size=4`、`data_parallel_size=3`，
Master 为 rank `0`，两个 Worker 为 rank `1`、`2`。

在确认并行参数后，若任一节点的空闲卡数小于 `tensor_parallel_size` 时，明确判定为**不可部署**：停止执行
后续部署流程，并向用户报告不可部署的节点和原因。

## 3. 配置参数和部署

Docker 的公共网络、Ascend 挂载、模型挂载、日志目录和设备参数沿用
`common.md` 第 2 节模板。命令中的 `<common-docker-args>` 必须替换为该模板的公共参数：
`--name <node-name>`、`--network host`、共享内存、七项只读 Ascend 挂载、`/models/model` 模型挂载、
`/models/deployment` 日志目录挂载，以及本节点第 1 节选中的 `/dev/davinci<device_id>`、
`/dev/davinci_manager`、`/dev/devmm_svm` 和 `/dev/hisi_hdc`；不能在 Agent/Pi 容器内
直接执行。

容器按 `common.md` 第 2 节的输入分支确认：用户提供镜像且目标容器不存在时，按上述
模板创建新容器，容器名默认为该节点的 IP 地址；用户提供了容器名称、或同名容器已
存在时，直接复用现有容器，不执行 `docker run`。复用时按 `common.md` 第 2 节的
`docker exec` 模板执行，`--env` 环境变量和 `bash -lc` 启动命令使用本节对应角色
（Master/Worker）的相同内容。

以下值必须在生成启动命令前确认，不能从未提供或无法只读确认的环境中猜测：

| 参数 | 普通多机规则 |
|---|---|
| `image` | 用户提供，每个节点使用用户指定镜像；仅在创建新容器时使用 |
| `model_path` | 容器内统一路径，固定为 `/models/model`；各节点的宿主机模型路径只出现在 `<common-docker-args>` 的挂载参数中 |
| `served_model_name` | Master 和所有 Worker 一致 |
| `master_node_ip` | Master 的数据面 IP；所有 Worker 可访问 |
| `data_parallel_rpc_port` | DP 节点间 RPC 通信端口；Master 上可监听，默认 `51216`，端口被占用时切换其他端口，所有 Worker 可访问 |
| `network_interface` | 每个节点用于 HCCL/Gloo/TP 通信的物理网卡名，填入第 1.2 节的 `*_SOCKET_IFNAME` 环境变量 |
| `port` | 每个节点的 vLLM API 端口；默认 `8899`，所有节点采用相同端口号，端口被占用时切换其他端口 |
| `max_model_len` | 模型上下文长度；用户未指定时默认 `4096` |
| `max_num_seqs` | 并发序列数；用户未指定时默认 `64` |
| `gpu_memory_utilization` | NPU 内存利用率；用户未指定时默认 `0.9` |

命令约束：

- 启动日志保存到 `/models/deployment` 日志目录，文件名带部署时间戳 `<timestamp>`（格式 `YYYYMMDD-HHMM`，精确到分钟），同一次部署所有节点使用同一时间戳；
- 部署量化后的模型时，启动命令必须加上 `--quantization ascend`；
- 部署若出现 graph capture 失败时，使用 `--enforce-eager`；
- 命令行参数不得擅自填入未知值；本节未说明的参数均使用默认值，不出现在部署命令中。

### 3.1 Master

```bash
docker run --detach \
  <common-docker-args> \
  --env HCCL_IF_IP=<master_node_ip> \
  --env GLOO_SOCKET_IFNAME=<network_interface> \
  --env TP_SOCKET_IFNAME=<network_interface> \
  --env HCCL_SOCKET_IFNAME=<network_interface> \
  --env OMP_PROC_BIND=false \
  --env OMP_NUM_THREADS=1 \
  --env HCCL_BUFFSIZE=200 \
  --env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --env VLLM_ASCEND_BALANCE_SCHEDULING=1 \
  --env ASCEND_RT_VISIBLE_DEVICES=<master_devices> \
  <image> bash -lc 'source /root/.bashrc && exec vllm serve <model_path> \
    --host 0.0.0.0 --port <port> \
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
    > /models/deployment/vllm-master-<timestamp>.log 2>&1'
```

Master 约束：

- `data_parallel_start_rank` 为 `0`；
- 不添加 `--headless`。

### 3.2 Worker

```bash
docker run --detach \
  <common-docker-args> \
  --env HCCL_IF_IP=<worker_node_ip> \
  --env GLOO_SOCKET_IFNAME=<network_interface> \
  --env TP_SOCKET_IFNAME=<network_interface> \
  --env HCCL_SOCKET_IFNAME=<network_interface> \
  --env OMP_PROC_BIND=false \
  --env OMP_NUM_THREADS=1 \
  --env HCCL_BUFFSIZE=200 \
  --env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --env VLLM_ASCEND_BALANCE_SCHEDULING=1 \
  --env ASCEND_RT_VISIBLE_DEVICES=<worker_devices> \
  <image> bash -lc 'source /root/.bashrc && exec vllm serve <model_path> \
    --host 0.0.0.0 --port <port> \
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
    --headless \
    > /models/deployment/vllm-worker-<rank>-<timestamp>.log 2>&1'
```

Worker 约束：

- rank 从 `1` 开始且连续；
- 必须添加 `--headless`；
- 每个 Worker 使用自己的数据面 IP、通信网卡、空闲 Device 列表，`--data-parallel-address` 仍指向 Master。

## 4. 启动后检查和服务验收

只做普通多机启动后的模式专属状态检查和对话服务验收，不重复公共输入检查、Docker预检或清理。

整体：
- 所有 Master/Worker 容器处于运行状态；
- DP rank 覆盖 `0..N-1`，没有重复或缺失；
- 日志没有未处理的通信、NPU、模型加载或参数错误。

Master：
- 节点进程存在，`/v1/models` 返回预期 `served_model_name`。

Worker：
- 节点进程存在、rank 正确连续，不检查 `/v1/models`。

分布式状态检查全部通过后，只对 Master 做对话服务验收；Worker 不做对话验收。向
Master 的 API 端口发送最小对话请求，验证请求成功完成、返回的模型名称与
`served_model_name` 一致：

```bash
curl --fail-with-body http://<master_node_ip>:<port>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<served_model_name>",
    "messages": [{"role": "user", "content": "请用一句话说明服务已启动。"}],
    "max_tokens": 256
  }'
```

不要把单纯 HTTP 200 当作业务成功，响应必须包含可解析的选择结果和非空文本。验收通过后按 `common.md` 第 4 节记录部署报告。
