# multi-node PD-disaggregated deployment

本文件定义多机 Prefill/Decode（PD）分离的部署方法。SSH、Docker 宿主机边界、Ascend
挂载、节点预检、公共清理、业务请求和报告格式遵循上层 Skill 与 common.md；本文件只负责
PD 的 connector 选择、P/D 拓扑、KV 服务、Proxy 命令和启动后 PD 状态检查。

本文件的模型→connector 映射、connector JSON 配置模板、并行参数、环境变量、Proxy 方向和
Mooncake 服务要求以 vLLM-Ascend v0.26 官方教程、特性文档和 e2e 配置为基准。

## 部署流程（总览）

先按 `common.md` 完成节点、镜像、模型目录、挂载、设备、网卡和端口预检，再按以下
顺序执行 PD 分离部署的专属流程：

1. **服务图与 connector 选择**：按 1.2 映射表和 1.3 选择规则确定 connector，由
   connector 确定 Proxy 方向和是否需要 Mooncake KV Pool 服务。
2. **并行策略**：采用节点内 TP、节点间 DP，确定每个角色的节点数、TP 和 rank 分配。
3. **通信环境变量**：按第 3 节为每个 P/D 容器生成环境变量，实际值按所在节点和
   角色展开。
4. **Connector 配置与 KV 服务要求**：按第 4 节生成 P/D 两侧共用的
   `--kv-transfer-config` JSON（仅 `kv_role` 不同）；池化组合另需 `mooncake.json`。
5. **配置参数和部署命令**：按第 5 节参数表确定取值并生成启动命令（所有占位符必须
   解析，不能原样发送）；仅池化组合先启动 Mooncake Master 并通过 readiness；然后
   按依赖顺序启动全部 Prefill 和 Decode，逐个确认容器存活、vLLM 启动完成、日志无
   connector / KV 传输相关报错；全部就绪后按 5.4 为选定 connector 启动Proxy。
6. **启动后 PD 状态检查**：统一检查全部 P/D、Proxy 和池化组合的 Mooncake Master，
   在最小业务请求的时间窗口内确认请求实际经过 Prefill 和 Decode；

任一本地检查失败时，停止生成或启动命令，报告节点、检查项和原因。

## 1. 服务图与 connector 选择（v0.26 基准）

### 1.1 服务图

```text
除池化组合外的 connector（无独立 KV 服务）:
  Prefill workers ──┐
  Decode workers ───┼── Proxy ── 对外 OpenAI API

池化组合:
  Mooncake Master ──┬── Prefill workers ──┐
                    └── Decode workers ───┼── Proxy ── 对外 OpenAI API
```

每个 Prefill / Decode 角色按第 2 节展开一个或多个容器。

默认节点分配：agent 所在节点运行 Prefill、Proxy 和 Mooncake Master（仅池化组合），
Decode 运行在其他节点（单机部署时 P/D 同节点）；agent 所在节点不在本次部署的节点
列表中时，由 agent 按节点资源和用户需求自行分配各容器所在节点，并在部署计划中
记录。

### 1.2 模型 → connector 映射

下表是 connector 选择的唯一依据；P/D 拓扑（DP/TP/节点数）按第 2 节计算。量化规则见第 5 节参数表。

| connector | 模型权重 |
|---|---|
| `MooncakeConnectorV1` | DeepSeek-V3 / V3.1、DeepSeek-V3.2、GLM-5、GLM-5.1、GLM-5.2（A3）、Kimi-K2.6、MiniMax-M2、MiniMax-M2.5、Qwen3-235B-A22B、Qwen3-VL-235B、Qwen3.5-397B-A17B、Qwen3.5-27B、Qwen3.6-27B、Qwen2.5-VL-7B（单机 1P1D） |
| `MooncakeHybridConnector` | DeepSeek-V4-Pro、DeepSeek-V4-Flash |
| `MooncakeLayerwiseConnector` | DeepSeek-V3.1-Terminus |
| `MultiConnector` = `MooncakeConnectorV1` + `AscendStoreConnector(backend=mooncake)` | GLM-5.2（8×Atlas 800 A2） |

### 1.3 connector 选择规则

按以下优先级确定 connector：

1. 用户明确指定 connector（包括明确要求 MultiConnector 或某个池化组合）时，采用用户所指定的connector。
2. 用户未指定、模型在 1.2 表中时，选择表对应行的 connector；同一模型多行时按硬件
   档位确定一行。
3. 用户未指定、模型不在 1.2 表中时，停止并报告。

用户需求映射：

| 用户表达 | connector |
|---|---|
| “普通 P/D”“先跑通 P 到 D” | 按 1.2 表选择 |
| “layerwise”“逐层传输” | `MooncakeLayerwiseConnector` |
| “需要 Mooncake KV Pool”“前缀 KV 复用并使用 Mooncake” | `MultiConnector` = `MooncakeConnectorV1` + `AscendStoreConnector(backend=mooncake)` |

只有带 `AscendStoreConnector` 的组合才是 KV Pool。connector 名称包含 Mooncake 不等于需要
KV Pool，不能因此自动增加 `AscendStoreConnector`、Mooncake Master、`mooncake.json`
或 `MOONCAKE_CONFIG_PATH`——这些只属于池化组合。

## 2. 并行策略

Prefill 和 Decode 是两个独立的并行组，各自确定 TP 和节点数，两组拓扑可以不同。本文件
只采用**节点内 TP、节点间 DP**：每个节点内选中的全部 NPU 构成一个 rank，
`--tensor-parallel-size` 为该节点的选卡数；每个节点是一个 DP rank，不使用 PP、EP 或
跨节点 TP。

- 某角色只有 1 个节点（单机 1P1D 或 1P1D 双机）时，命令行只保留
  `--tensor-parallel-size`，不加任何 `--data-parallel-*` 参数；
- 某角色有多个节点时，每个节点一个 `vllm serve`，在 `--tensor-parallel-size` 后追加
  `--data-parallel-size <该角色节点数> --data-parallel-rank <本节点序号> --data-parallel-address <该角色 rank 0 节点 IP> --data-parallel-rpc-port <端口>`
  （rpc-port 同角色一致，默认 `12321`）；
- connector 配置中 `prefill` / `decode` 段的 `dp_size` 填该角色节点数（单节点为
  1），`tp_size` 填该节点的 NPU 数（与 `--tensor-parallel-size` 一致），必须与 CLI
  实际拓扑一致。

## 3. 通信环境变量

Prefill 和 Decode 使用相同的通信环境变量结构，实际值按所在节点和角色展开：

```bash
--env ASCEND_RT_VISIBLE_DEVICES=<role_devices>
--env HCCL_IF_IP=<role_node_ip>
--env HCCL_SOCKET_IFNAME=<role_network_interface>
--env GLOO_SOCKET_IFNAME=<role_network_interface>
--env TP_SOCKET_IFNAME=<role_network_interface>
--env OMP_PROC_BIND=false
--env OMP_NUM_THREADS=1
--env HCCL_BUFFSIZE=256
--env HCCL_OP_EXPANSION_MODE=AIV
--env TASK_QUEUE_ENABLE=1
--env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## 4. Connector 配置与 KV 服务要求

`--kv-transfer-config` 的值是一段 JSON；本节给出各 connector 的 JSON 模板，`<...>`
占位符由部署计划解析。P 和 D 两侧用同一个 JSON，差异只有 `kv_role`（P 侧
`kv_producer`，D 侧 `kv_consumer`）和池化组合的 `lookup_rpc_port`。JSON 原文内联在命令的
`--env KV_TRANSFER_CONFIG='...'` 行里，不生成文件。

### 4.1 单 connector：`MooncakeConnectorV1` / `MooncakeHybridConnector` / `MooncakeLayerwiseConnector`

```json
{
  "kv_connector": "MooncakeConnectorV1 或 MooncakeHybridConnector 或 MooncakeLayerwiseConnector",
  "kv_role": "kv_producer 或 kv_consumer",
  "kv_port": "<role_kv_port>",
  "engine_id": "<unique_engine_id>",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {
      "dp_size": <prefill_dp>,
      "tp_size": <prefill_tp>
    },
    "decode": {
      "dp_size": <decode_dp>,
      "tp_size": <decode_tp>
    }
  }
}
```

三种 connector 共用这个 JSON，只有 `kv_connector` 的取值不同。

### 4.2 `MultiConnector` 池化组合（KV Pool）

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer 或 kv_consumer",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "<same_role>",
        "kv_port": "<role_kv_port>",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {
            "dp_size": <prefill_dp>,
            "tp_size": <prefill_tp>
          },
          "decode": {
            "dp_size": <decode_dp>,
            "tp_size": <decode_tp>
          }
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "<same_role>",
        "kv_connector_extra_config": {
          "backend": "mooncake",
          "lookup_rpc_port": "<lookup_rpc_port>"
        }
      }
    ]
  }
}
```

- 子 connector 的 `kv_role` 与外层一致；`MooncakeConnectorV1` 负责实时 KV 传输，
  `AscendStoreConnector` 负责前缀 KV 池；`lookup_rpc_port` 是 `AscendStoreConnector`
  的内部 RPC 端口，P 侧填 `"0"`，D 侧填 `"1"`。

### 4.3 Mooncake 传输引擎与 KV Pool 服务

`mooncake.json` 是镜像里没有的配置文件，需要自己创建，只读挂载进
容器并用 `MOONCAKE_CONFIG_PATH` 指向容器内路径，且只有池化组合需要。

**仅池化组合（`MultiConnector` 带 `AscendStoreConnector(backend=mooncake)`）追加**：

1. 所有 P/D 节点统一 `--env PYTHONHASHSEED=0`（官方要求全节点一致）；
2. `mooncake.json` 只读挂载并 `--env MOONCAKE_CONFIG_PATH=<容器内路径>`，内容固定
   如下（`master_server_address` 填 Mooncake Master 的 IP:端口，默认 `50088`，与
   5.1 的 `--port` 一致）：

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "<mooncake_master_ip>:<mooncake_service_port>",
  "global_segment_size": "<per-card 注册内存，默认 1GB，须 1GB 对齐>",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true
}
```

3. Mooncake Master 单节点运行（启动命令见 5.1），`master_server_address` 与 Master
   监听端口对齐。

如果更换 connector 或镜像，先重新确认 connector 及其服务依赖，再生成本次部署命令；
不能在启动过程中临时删除或替换选定 connector 的环境变量。

## 5. 配置参数和部署命令

下面的代码块都是完整的 `docker run` 模板。`<common-pd-docker-args>` 表示 common.md
中的模型服务外壳，包括网络、Ascend 挂载、模型挂载、部署目录挂载和 NPU 设备参数，
必须在执行前替换，不能原样发送给 Docker。Mooncake Master 和 Proxy 容器不继承模型
服务外壳，只加 `--network host`。

P/D 模板的 env 唯一例外是池化组合（`MultiConnector` = V1 +
`AscendStoreConnector(backend=mooncake)`）：其 P/D 容器在 env 块后追加
`--env PYTHONHASHSEED=0` 和 `--env MOONCAKE_CONFIG_PATH=<容器内路径>`，并把
`mooncake.json` 只读挂载到该路径（见 4.3）。

生成启动命令前先按下表确定各参数取值：

| 参数 | 含义和默认值 |
|---|---|
| prefill_kv_port / decode_kv_port | KV 传输端口；未指定时默认 30000 / 30200。必须避开 AscendDirectTransport 随机端口区（从 20000 起、每卡预留 1000 个端口）：8 卡节点选 ≥28000，16 卡节点选 ≥36000 |
| engine_id | 引擎标识，字符串；同一节点上多个进程时互不相同，跨节点允许重复。默认按节点序号填 `"0"`、`"1"`、`"2"`… |
| prefill_api_port / decode_api_port | Prefill / Decode 对外 API 端口；必须在部署计划中指定 |
| max_model_len | P/D 两侧一致；用户指定时用用户值，未指定时默认 `4096`，不超过模型上下文上限 |
| max_num_seqs | 用户指定时用用户值，未指定时默认 `64` |
| max_num_batched_tokens | 默认值，用户明确指定时才写入对应侧命令 |
| gpu_memory_utilization | 用户指定时用用户值，未指定时默认 `0.9` |
| quantization | 用户权重为 Ascend 量化格式（w8a8 / w4a8 等）时，P/D 两侧加 `--quantization ascend` |
| Mooncake 服务参数 | 仅池化组合；按 4.3 和 5.1 的固定值 |

计划必须记录：

- Prefill 和 Decode 的节点、rank、NPU、TP、DP、API 和 KV 端口；
- 只有选定 connector 需要的 KV 服务节点、容器、监听端口和配置文件；
- Proxy 节点、监听端口、Proxy 方向（P-first / D-first）和 P/D 后端列表；
- 模型路径、served model name、镜像 digest；
- 池化组合才记录：`mooncake.json` 宿主机路径和容器路径、Mooncake Master
  节点与端口；
- connector 配置和必要环境变量；
- 最小 completion 请求。

### 5.1 Mooncake Master（仅池化组合）

非池化组合不生成本节容器。v0.26 官方命令模板：

```bash
docker run --detach \
  --name mooncake-master-<node-name> \
  --network host \
  <image> bash -lc 'exec mooncake_master \
    --port <mooncake_service_port> \
    --eviction_high_watermark_ratio 0.9 \
    --eviction_ratio 0.1 \
    --default_kv_lease_ttl 11000 \
    --enable_offload=false \
    --client_ttl 120'
```

- Master 只需在一个节点运行；`--port` 必须与 `mooncake.json` 的
  `master_server_address` 一致；
- 上述数值为 v0.26 官方示例值；`--default_kv_lease_ttl` 应大于
  `ASCEND_CONNECT_TIMEOUT` 和 `ASCEND_TRANSFER_TIMEOUT`；
- 只有池化组合需要先通过 TCP `<mooncake_service_port>` readiness；P/D connector 配置
  继续使用各自的 `prefill_kv_port` / `decode_kv_port`，不能把 KV 传输端口改成
  Mooncake Master 的控制端口。

### 5.2 Prefill

```bash
docker run --detach \
  --name prefill-<rank>-<node-name> \
  <common-pd-docker-args> \
  --env ASCEND_RT_VISIBLE_DEVICES=<role_devices> \
  --env HCCL_IF_IP=<role_node_ip> \
  --env HCCL_SOCKET_IFNAME=<role_network_interface> \
  --env GLOO_SOCKET_IFNAME=<role_network_interface> \
  --env TP_SOCKET_IFNAME=<role_network_interface> \
  --env OMP_PROC_BIND=false \
  --env OMP_NUM_THREADS=1 \
  --env HCCL_BUFFSIZE=256 \
  --env HCCL_OP_EXPANSION_MODE=AIV \
  --env TASK_QUEUE_ENABLE=1 \
  --env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --env KV_TRANSFER_CONFIG='<prefill_connector_config>' \
  <image> bash -lc 'source /root/.bashrc && exec vllm serve /models/model \
    --host 0.0.0.0 --port <prefill_api_port> \
    --tensor-parallel-size <prefill_tp> \
    <prefill_dp_args> \
    --served-model-name <served_model_name> \
    --max-model-len <prefill_max_model_len> --max-num-seqs <prefill_max_num_seqs> \
    --gpu-memory-utilization <gpu_memory_utilization> \
    --trust-remote-code \
    --kv-transfer-config "$KV_TRANSFER_CONFIG"'
```

`<prefill_connector_config>` 替换为 4.1 / 4.2 中选定 connector 的 JSON 原文；
`<prefill_dp_args>` 按第 2 节生成：单节点角色为空，多节点角色为完整
`--data-parallel-*` 参数串。

### 5.3 Decode

```bash
docker run --detach \
  --name decode-<rank>-<node-name> \
  <common-pd-docker-args> \
  --env ASCEND_RT_VISIBLE_DEVICES=<role_devices> \
  --env HCCL_IF_IP=<role_node_ip> \
  --env HCCL_SOCKET_IFNAME=<role_network_interface> \
  --env GLOO_SOCKET_IFNAME=<role_network_interface> \
  --env TP_SOCKET_IFNAME=<role_network_interface> \
  --env OMP_PROC_BIND=false \
  --env OMP_NUM_THREADS=1 \
  --env HCCL_BUFFSIZE=256 \
  --env HCCL_OP_EXPANSION_MODE=AIV \
  --env TASK_QUEUE_ENABLE=1 \
  --env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --env KV_TRANSFER_CONFIG='<decode_connector_config>' \
  <image> bash -lc 'source /root/.bashrc && exec vllm serve /models/model \
    --host 0.0.0.0 --port <decode_api_port> \
    --tensor-parallel-size <decode_tp> \
    <decode_dp_args> \
    --served-model-name <served_model_name> \
    --max-model-len <decode_max_model_len> --max-num-seqs <decode_max_num_seqs> \
    --gpu-memory-utilization <gpu_memory_utilization> \
    --trust-remote-code \
    --kv-transfer-config "$KV_TRANSFER_CONFIG"'
```

`<decode_connector_config>` 替换为 4.1 / 4.2 中选定 connector 的 JSON 原文；
`<decode_dp_args>` 按第 2 节生成：单节点角色为空，多节点角色为完整
`--data-parallel-*` 参数串。

### 5.4 Proxy

v0.26 官方提供两个 Proxy 脚本（位于 vllm-ascend 仓库 `examples/disaggregated_prefill_v1/`），
参数结构完全相同，**请求路由方向不同**，必须按选定 connector 选择：

| connector | Proxy 脚本 | 路由行为 |
|---|---|---|
| `MooncakeConnectorV1`、`MooncakeHybridConnector`、池化组合 `MultiConnector` | `load_balance_proxy_server_example.py` | 请求先路由到 Prefill 节点完成预填充，再转交 Decode 节点续写 |
| `MooncakeLayerwiseConnector` | `load_balance_proxy_layerwise_server_example.py` | 请求先路由到 Decode 节点，Decode 按需从 Prefill 逐层拉取 KV |

```bash
docker run --detach \
  --name proxy-<node-name> \
  --network host \
  --env http_proxy= --env https_proxy= \
  <proxy_script_mount_if_needed> \
  <proxy_image> bash -lc 'exec python <proxy_script> \
    --host <proxy_host> \
    --port <proxy_port> \
    --prefiller-hosts <prefill_backend_hosts> \
    --prefiller-ports <prefill_backend_ports> \
    --decoder-hosts <decode_backend_hosts> \
    --decoder-ports <decode_backend_ports>'
```

- `<proxy_script>` 按上表选择；镜像已内置脚本时 `<proxy_script_mount_if_needed>` 为空，
  脚本在宿主机时才以只读方式挂载；
- Proxy 官方指南默认部署在 Prefill 节点，也可用任何能访问全部 P/D 后端的节点；启动前
  必须清空 `http_proxy` / `https_proxy` 环境变量，避免请求转发经过外部代理；

## 6. 启动后 PD 状态检查

本节对应部署流程第 6 步，只检查 PD 分离部署的链路专属状态，不重复公共输入检查、
Docker 预检、清理或公共业务验收：

- 所有 Prefill / Decode 容器处于运行状态，vLLM 启动完成，日志无 connector / KV
  传输相关报错；
- Proxy 容器处于运行状态，返回成功；
- 在 `common.md` 第 4 节最小业务请求（PD 部署的公共入口是 Proxy）的时间窗口内，
  Prefill 和 Decode 两侧日志均有该请求的处理记录——这是请求真实经过 P→D 链路的
  证据，仅 Proxy 返回成功不能证明；
- 池化组合的 Mooncake Master 容器处于运行状态，控制端口可访问，并可重发一次
  相同前缀的请求验证 KV Pool 复用。

只有带 `AscendStoreConnector` 的池化组合才执行最后一条，其余 connector 跳过。

以上检查全部通过后才报告 `ready`；任一项失败按 `common.md` 第 4 节报告
`partially_started` 或 `failed`。