# multi-node PD-disaggregated deployment

本文件定义 **multi-node PD-disaggregated deployment**。当前版本使用固定的
PD 部署方案，不在运行时自动选择其他 connector。最终服务计划属于当前 attempt，
并由主 Skill 统一写入 `deployment_report.json`。

## 固定服务结构

```text
Mooncake master
      │ TCP readiness
      ├──────────────────┐
Prefill workers      Decode workers
      └─────────┬────────┘
                ↓
              Proxy ── 对外 OpenAI API
```

- Mooncake master 是独立容器；Skill 必须取得已确认的镜像、argv、配置和监听端口，不能根据 connector 名称猜测启动命令。
- Prefill 和 Decode 是 vLLM-Ascend 容器，使用本文件规定的参数和 KV Transfer 配置。
- Proxy 是独立容器；Skill 必须取得已确认的 argv 和 Prefill/Decode 后端地址。
- Prefill/Decode 依赖 Mooncake master，Proxy 依赖全部 Prefill/Decode worker。

Skill 必须把 Mooncake master、Prefill、Decode 和 Proxy 四类服务都展开到部署计划中。

## 启动前必须确认的输入

| 输入 | 要求 |
|---|---|
| `deployment_id` | 由 Skill 根据当前 attempt 自动生成，用户不需要提供 |
| `image` | 所有 P/D 节点使用内容一致的 vLLM-Ascend 镜像，且支持本文件规定的三个 connector；Mooncake/Proxy 可按服务单独覆盖 |
| `model_path` | 所有 P/D 节点可访问，容器内路径一致 |
| `served_model_name` | Prefill、Decode 和 Proxy 使用同一名称 |
| `prefill_nodes` | 每项包含节点、rank、数据面 IP、NPU、TP 和 API 端口 |
| `decode_nodes` | 每项包含节点、rank、数据面 IP、NPU、TP 和 API 端口 |
| `mooncake_master` | 镜像、argv、配置文件、监听地址、端口和 TCP readiness |
| `proxy` | 镜像、argv、监听地址、端口、P/D 后端和 HTTP readiness |
| `network_interface` | 所有通信节点的实际数据面网卡 |
| `request` | 最终通过 Proxy 发送的最小 completion 请求 |

下面列出的默认值由 Skill 自动写入部署计划，不要求用户逐项输入。只有用户明确覆盖、端口冲突或已确认运行模板要求其他值时才修改。

## 服务图

计划至少包含：

```json
{
  "services": [
    {
      "id": "mooncake",
      "role": "mooncake_master",
      "depends_on": []
    },
    {
      "id": "prefill-0",
      "role": "prefill",
      "depends_on": ["mooncake"]
    },
    {
      "id": "decode-0",
      "role": "decode",
      "depends_on": ["mooncake"]
    },
    {
      "id": "proxy",
      "role": "proxy",
      "depends_on": ["prefill-0", "decode-0"]
    }
  ]
}
```

存在多个 Prefill 或 Decode rank 时，Proxy 必须依赖全部 P/D 服务；Proxy 的 HTTP 后端只包含所选模式实际提供 API 的服务。

## 固定默认值

| 参数 | Prefill | Decode |
|---|---:|---:|
| API port | `7100` | `7100` |
| KV port | `30000` | `30200` |
| DP RPC port | `12321` | `12321` |
| `data_parallel_size_local` | `1` | `1` |
| `max_model_len` | `4096` | `4096` |
| `max_num_seqs` | `32` | `32` |
| `max_num_batched_tokens` | `4096` | `256` |
| `gpu_memory_utilization` | `0.9` | `0.9` |
| `no_disable_hybrid_kv_cache_manager` | `true` | `true` |
| `enforce_eager` | `true` | 不设置 |
| `compilation_config` | 不设置 | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` |

同一个 API 或 RPC 端口可以在不同节点复用；同一节点存在冲突时必须显式覆盖。默认不增加下列模型专属参数：

```text
--trust-remote-code
--quantization
--tokenizer-mode
--tool-call-parser
--reasoning-parser
```

只有用户输入、模型元数据或已确认运行模板提供依据时才加入这些参数。

## DP 入口选择与拓扑参数

Pi 分别计算 Prefill 和 Decode 的 `role_dp_size`（该角色的 rank 数量），结合 Proxy 实际支持的后端形式及目标镜像的已验证启动模板，为两个角色分别选择入口模式：

| 条件 | 模式 | Proxy 使用的后端 |
|---|---|---|
| `role_dp_size = 1` | `single` | 该角色唯一的 API 地址 |
| `role_dp_size > 1`，Proxy 只接该角色一个 API 入口，且镜像支持内部 DP | `internal` | 该角色对外提供 API 的入口；其他 rank 为 headless worker |
| `role_dp_size > 1`，Proxy 能管理每个 rank，且镜像和模型支持外部 DP LB | `external` | 该角色所有 rank 的 API 地址 |

用户明确指定模式时先检查上述条件；条件不成立时报告冲突，不为满足模式要求虚增 DP size。没有明确指定时，Pi 根据已确认的 Proxy 模板和镜像模板选择，并在 `deployment-plan.json` 的 `runtime.pd_endpoint_modes` 中分别记录 Prefill/Decode 的模式、依据和 API 后端；每个服务的最终 argv 写入对应的 `services[].argv`。模板信息不足时先补充证据，不猜测 rank 参数。

每个 P/D 服务都按实际拓扑设置 `--tensor-parallel-size`。其余 DP 参数按所选模式展开：

- `single`：默认省略所有 DP 参数；若已确认的镜像模板要求，可显式设置 `--data-parallel-size 1 --data-parallel-size-local 1`。不加 `--data-parallel-rank`、`--data-parallel-start-rank`、`--headless`，也不启用 external LB。除非镜像模板要求，不加 DP address/RPC 参数。
- `internal`：设置该角色的 `--data-parallel-size <role_dp_size>`、`--data-parallel-size-local 1`、rank 0 的 `--data-parallel-address` 和同组一致的 `--data-parallel-rpc-port`。入口进程提供 API；其他 rank 按镜像模板使用 `--data-parallel-start-rank <rank> --headless`。rank 0 是否显式写 `--data-parallel-start-rank 0` 也以目标镜像模板为准，不能仅凭 rank 号推断。
- `external`：仅当 `role_dp_size > 1` 且镜像、模型和 Proxy 均确认支持时使用；每个 rank 设置 `--data-parallel-size <role_dp_size> --data-parallel-size-local 1 --data-parallel-rank <rank>`，同组使用 rank 0 的 DP address 和一致的 RPC port，每个 rank 提供独立 API。不得加 `--headless`。

1P1D 的两组都选 `single`；例如 2P1D 可以为 Prefill 选择 `internal`、为 Decode 选择 `single`。Prefill 与 Decode 各自计算 DP size，不能把两种角色相加来满足 external LB 的 `DP size > 1` 要求。

表中的 DP RPC 默认端口仅在所选模式或已确认的镜像模板需要 RPC 参数时使用；`single` 不因存在该默认值而自动生成 RPC 参数。

生成前必须校验：

1. Prefill 和 Decode 都至少有一个服务；
2. 每个角色的 rank 从 `0` 连续递增，不能重复或缺失；
3. rank 对应节点的数据面 IP 与该角色地址列表一致；
4. 当前 MVP 中 Prefill/Decode 的 TP 大小一致，且不超过对应节点 NPU 数量；
5. 同一节点上的 API、KV、DP RPC、Mooncake 和 Proxy 端口没有冲突；
6. Proxy 能访问两种角色在部署计划中选定的 API 后端；headless worker 不得作为 HTTP 后端。

## 固定 KV Transfer 配置

当前版本固定使用：

```text
MultiConnector
├── MooncakeHybridConnector
└── AscendStoreConnector
```

每个 P/D 服务的 `--kv-transfer-config` 按下面的模板生成：

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "<kv_role>",
  "engine_id": "<engine_id>",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": "<kv_role>",
        "kv_port": "<kv_port>",
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": {
            "dp_size": "<prefill_dp_size>",
            "tp_size": "<tp_size>"
          },
          "decode": {
            "dp_size": "<decode_dp_size>",
            "tp_size": "<tp_size>"
          }
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "<kv_role>",
        "kv_connector_extra_config": {
          "lookup_rpc_port": "<lookup_rpc_port>",
          "backend": "mooncake"
        }
      }
    ]
  }
}
```

角色字段固定映射如下：

| 角色 | `kv_role` | `engine_id` | 默认 `kv_port` | `lookup_rpc_port` |
|---|---|---|---:|---:|
| Prefill | `kv_producer` | `PREFILL_ENGINE` | `30000` | `0` |
| Decode | `kv_consumer` | `DECODE_ENGINE` | `30200` | `1` |

`prefill.dp_size`、`decode.dp_size` 和 `tp_size` 必须来自本次已校验的拓扑，不能从示例复制。

启动前必须确认镜像同时支持：

```text
MultiConnector
MooncakeHybridConnector
AscendStoreConnector
```

如果镜像只支持 `MooncakeConnectorV1` 或 `MooncakeLayerwiseConnector`，报告镜像与当前 Skill 模板不兼容；不要自动替换 connector 后继续部署。

## KV Connector 检查

检查分为两层：

1. 启动 Proxy 前，对每个 P/D 容器执行启动门禁：
   - 使用 `docker inspect` 核对实际 argv 中的 `--kv-transfer-config` 与计划一致，包括 connector 名称、`kv_role`、`engine_id`、P/D 的 DP/TP 大小、`kv_port`、`lookup_rpc_port` 和 Mooncake backend；
   - 确认 Mooncake master 的计划端口可从 P/D 节点访问；
   - 在有界时间内确认 P/D 进程未退出，并检查启动日志中没有 connector 创建失败、Mooncake/AscendStore 初始化失败、无法连接或发现远端 engine、KV 传输失败等错误；
   - 如果已确认的镜像模板提供 connector 健康接口或明确的初始化成功日志，再把它加入计划并检查。没有该证据时，不得自行猜测成功日志；部分实现会在首个业务请求时才完成初始化。
2. Proxy ready 后，通过 Proxy 发送计划中的最小 completion。请求必须实际经过计划中的 Prefill 和 Decode 后端，不能用直接请求单个 P/D 服务代替。检查请求成功、响应非空，并确认请求期间 P/D 日志没有 KV 获取、发送、远端 engine 查询或等待 P side ready 失败。

第一层只证明配置正确且启动阶段没有已知错误；第二层通过完整 P→D 请求验证实际 KV 传输。两层都通过后，才能认定 connector 链路可用。若镜像日志包含可关联的请求 ID，将 P/D 两侧对应日志摘要写入 `deployment_report.json`；没有稳定请求 ID 时，至少保存验证时间窗口内的 P/D 日志摘要和 Proxy 请求结果。

## vLLM 命令

每个 P/D 容器的最终 argv 必须完整写入计划，基本形状为：

```text
vllm serve <model_container_path>
--host 0.0.0.0
--port <api_port>
--served-model-name <served_model_name>
--tensor-parallel-size <tp_size>
<按该角色的 single/internal/external 模式展开 DP 参数>
--max-model-len 4096
--max-num-seqs 32
--max-num-batched-tokens <role_default>
--gpu-memory-utilization 0.9
--no-disable-hybrid-kv-cache-manager
<prefill_enforce_eager_or_decode_compilation_config>
--kv-transfer-config <role_json>
```

每个容器使用稳定名称：

```text
boom-skill-<deployment_id>-mooncake
boom-skill-<deployment_id>-prefill-<rank>
boom-skill-<deployment_id>-decode-<rank>
boom-skill-<deployment_id>-proxy
```

Docker 挂载、设备、标签和所有权规则读取 [common.md](common.md)。使用 NPU 的 Prefill/Decode 必须映射计划中的 NPU 和共享设备；Mooncake master、Proxy 是否使用 NPU 由其已确认的运行模板决定。

## 启动顺序

1. 检查所有节点的 SSH、Docker、镜像、模型挂载、Ascend 挂载、NPU 和端口；
2. 启动 Mooncake master；
3. 等待 Mooncake master TCP readiness；
4. 并行启动全部 Prefill 和 Decode worker；
5. 按选定模式等待全部 P/D 服务就绪：API 入口检查端口和模型身份；headless worker 检查进程、rank、DP 通信和模型加载日志；按“KV Connector 检查”的第一层完成启动门禁；
6. 启动 Proxy；
7. 等待 Proxy HTTP readiness；
8. 通过 Proxy 发送最小 completion。

Mooncake master 未 ready 时不得启动 P/D，任一 P/D 未 ready 时不得启动 Proxy。

## 验证

只有以下检查全部通过才能报告 `ready`：

- Mooncake master 容器、进程和 TCP 端口正常；
- 所有 Prefill/Decode 容器带正确标签且处于运行状态；
- P/D rank 覆盖完整，实际参数与计划一致；
- Prefill 使用 `kv_producer`，Decode 使用 `kv_consumer`；
- 两侧均使用 `MultiConnector`、`MooncakeHybridConnector` 和 `AscendStoreConnector`；
- KV connector 的实际配置与计划一致，启动日志没有未处理的 connector、Mooncake 或 AscendStore 错误；
- 提供 API 的 Prefill/Decode 入口返回预期模型身份；headless worker 的模型加载、rank 和 DP 通信检查通过；
- Proxy 后端列表与计划中所选模式的 API 地址一致，且没有 headless worker；
- 通过 Proxy 的最小 completion 成功且返回非空结果，请求期间 P/D 日志没有 KV 传输错误。

只验证容器和端口不能证明 PD 链路成功。Mooncake、P/D 或 Proxy 单独健康，但最终 completion 失败时必须报告 `partially_started`。

## 失败处理

失败时记录最后一份 readiness、相关日志摘要和失败命令。只停止和删除同时满足以下标签的容器：

```text
boom.skill.owner=standalone-skill
boom.skill.deployment_id=<deployment_id>
```

按逆依赖顺序清理：

```text
Proxy → Decode/Prefill → Mooncake master
```

禁止使用 `pkill`、宽范围 `docker rm`、猜测容器 ID 或通过端口占用者推断所有权。
