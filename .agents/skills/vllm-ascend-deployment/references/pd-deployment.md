# 单机 PD 专属部署规则

本文件将多机 PD 文档中的角色规则收敛到单台主机。多机文档中的 connector 映射、
JSON 模板和 Proxy 脚本名称仍是唯一依据；本文件只补充单机的设备、端口和生命周期
约束。

## 节点输入与预检

PD 拓扑按节点列表处理。现有单实例输入可以由 agent 包装为一个元素：

```text
pd.prefill -> [p1, ...]
pd.decode  -> [d1, ...]
```

每个节点必须由 agent 根据最终启动命令生成一份 Ordinary 输入文件：

```text
<attempt-dir>/deployment_input_p1.json
<attempt-dir>/deployment_input_p2.json
<attempt-dir>/deployment_input_d1.json
```

每份文件的 `serve.argv` 是该节点真实的 `vllm serve` 参数，
`resources.visible_devices` 只包含该节点本地设备，`resources.role_device_count` 包含
同一 Prefill 或 Decode DP 角色的总设备数，`resources.card_memory_bytes` 填写该节点设备池
对应卡型的单卡显存（来自 Provider 冻结硬件事实或环境 probe）。所有节点使用 probe 生成的同一份
`vllm-serve-help-all.txt`，逐个调用：

```text
python <skill-root>/scripts/input_contract.py \
  --input <attempt-dir>/deployment_input_<role><rank>.json \
  --serve-help-file <attempt-dir>/vllm-serve-help-all.txt \
  --runtime-output <attempt-dir>/.work/<role><rank>/runtime.json
```

输入脚本只负责该节点的 argv、模型 config、TP/DP 数值和本地资源边界。以下关系由 agent
在脚本调用前或调用后校验，不放入 PD 专用脚本：

- 节点 ID、rank、角色 DP size 和 `--data-parallel-*` 参数的一致性；
- P/D 设备池互斥、端口唯一性、端口候选和 AscendDirectTransport 端口区避让；
- connector、KV transfer、engine ID、Proxy、Mooncake 和后端列表；
- 所有节点的模型身份、served model name 和共享运行参数；
- 同一角色的最终 TP 必须一致。

每个节点的规范化结果只保存在内存，同时输出一个临时 `runtime.json`。PD 静态 OOM 逐节点
调用 Ordinary `static_oom_precheck.py`：

```text
python <skill-root>/scripts/static_oom_precheck.py \
  --deployment-input <attempt-dir>/deployment_input_<role><rank>.json \
  --runtime <attempt-dir>/.work/<role><rank>/runtime.json
```

脚本从 deployment input 和 runtime artifact 合并输入，并在 `single_node_pd` 模式使用本地
`dp=1`；agent 不再手工拼装 static OOM payload。缺失 TP 时只对当前节点推荐和写回 TP，
不重复输入校验或静态 OOM。所有节点通过后，agent 才生成最终 connector/KV/Proxy 配置并
拉起服务。

## 设备和并行

Prefill 与 Decode 是两个独立的 DP 角色组，每个节点在自己的设备集合内解析 TP：

```text
prefill_tp[p_i] = min(static_oom_pass_candidates(p_i.devices))
decode_tp[d_i]  = min(static_oom_pass_candidates(d_i.devices))
prefill_dp_size = len(prefill.nodes)
decode_dp_size  = len(decode.nodes)
```

未显式指定节点 TP 时，使用该节点的静态 OOM 最小推荐。每个实例都加载完整模型，所以
每个节点都要单独通过静态容量检查；不能以 P/D 的卡数或 TP 相加替代节点级检查。
同一角色的节点最终 TP 必须一致；若推荐结果不一致，agent 在启动前报告拓扑不一致并
停止部署。输入中的 `devices` 是节点本地设备池，两组角色的设备池必须不相交。

角色包含多个节点时，每个节点的命令追加：

```text
--data-parallel-size <role_dp_size>
--data-parallel-rank <node_rank>
--data-parallel-address <role_rank0_data_ip>
--data-parallel-rpc-port <role_rpc_port>
```

单节点角色不追加这些参数。`role_dp_size`、rank、address 和 rpc port 的跨节点关系由
agent 校验，Ordinary input contract 只负责识别 help 中的 option，并校验
`--data-parallel-size` 与 `role_device_count` 的数值边界。

## 通信环境

P/D 使用同一个本机数据面 IP 和通信网卡，但使用不同的
`ASCEND_RT_VISIBLE_DEVICES`：

```text
HCCL_IF_IP=<local_data_ip>
HCCL_SOCKET_IFNAME=<network_interface>
GLOO_SOCKET_IFNAME=<network_interface>
TP_SOCKET_IFNAME=<network_interface>
OMP_PROC_BIND=false
OMP_NUM_THREADS=1
HCCL_BUFFSIZE=256
HCCL_OP_EXPANSION_MODE=AIV
TASK_QUEUE_ENABLE=1
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## P/D 与 Proxy 联合启动

Prefill、Decode 和 Proxy 是同一次 PD deployment 的统一启动组，不能把 P/D 当作独立服务
分别验收，也不能只启动 P/D 后把 Proxy 留给后续流程。所有角色的输入、设备、端口和
connector 配置确定后，按以下依赖顺序在同一个 attempt 内启动：

1. 池化 connector 先启动唯一的 Mooncake Master；非池化 connector 跳过这一步；
2. 启动全部 Prefill 和 Decode 实例；同一角色的多个实例可以并行启动；
3. 只确认 P/D 进程存活、API 端口可访问且 `/health` 返回 `200`，不向 P/D 发送业务请求；
4. 使用完整且顺序对应的 P/D 后端列表启动 Proxy；Proxy 不请求 `/health`，只通过后续两个
   生成接口请求判定自身健康；
5. Proxy 进程和端口可访问后，所有业务验收请求只能发送到 Proxy，并且必须同时通过
   `/v1/completions` 和 `/v1/chat/completions`。

任一 P/D、Mooncake Master 或 Proxy 启动失败，必须停止同一启动组中已经启动的全部进程，
不能把部分启动状态报告为成功。

## P/D 启动命令

Prefill 和 Decode 使用同一命令结构，角色值由部署输入展开：

```bash
vllm serve /models/model \
  --host 0.0.0.0 --port <role_api_port> \
  --tensor-parallel-size <role_tp> \
  <role_dp_args> \
  --served-model-name <served_model_name> \
  --max-model-len <max_model_len> \
  --max-num-seqs <role_max_num_seqs> \
  --gpu-memory-utilization <gpu_memory_utilization> \
  --trust-remote-code \
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
```

启动命令需要同时注入本节通信环境变量，并设置：

```text
ASCEND_RT_VISIBLE_DEVICES=<selected_role_devices>
KV_TRANSFER_CONFIG=<role-specific-json>
```

Prefill 的 JSON 使用 `kv_role=kv_producer`，Decode 使用 `kv_role=kv_consumer`。
P/D 的 `dp_size` 按实际角色实例数量填写；只有单实例时才是 `1`。`tp_size` 必须与最终
`--tensor-parallel-size` 一致，且同一角色的所有节点必须使用同一个最终 TP。量化模型增加
`--quantization ascend`；只有确认
graph capture 失败时才增加 `--enforce-eager`。

`<role_dp_args>` 在单节点角色中为空；多节点角色中展开为本节点的
`--data-parallel-size`、`--data-parallel-rank`、`--data-parallel-address` 和
`--data-parallel-rpc-port` 参数。

## Connector 配置

单 connector 的 `KV_TRANSFER_CONFIG`：

```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer",
  "kv_port": "30000",
  "engine_id": "prefill-0",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": {"dp_size": <prefill_dp_size>, "tp_size": <prefill_tp_size>},
    "decode": {"dp_size": <decode_dp_size>, "tp_size": <decode_tp_size>}
  }
}
```

Decode 侧只把 `kv_role` 改为 `kv_consumer`、`kv_port` 改为 Decode 端口并使用不同
`engine_id`。实际 connector 名称必须按模型映射替换。

池化 `MultiConnector` 继续使用多机文档的嵌套 JSON，并为 P/D 挂载同一个只读
`mooncake.json`。Mooncake Master 只启动一个，默认控制端口 `50088`；P/D KV 端口
不能改成该控制端口。

## 端口

单机至少需要每个 P/D 节点的 API/KV 端口，以及一个 Proxy 端口：

```text
prefill_api_ports[...]
decode_api_ports[...]
proxy_port
prefill_kv_ports[...]
decode_kv_ports[...]
```

池化组合再增加 `mooncake_service_port`。所有端口必须来自 Provider 候选并通过端口
探针确认可用，所有 P/D/API/KV/Proxy/Mooncake 端口必须全局唯一。KV 端口要避开
AscendDirectTransport 随机端口区；不能因为端口在本机而跳过检查。

## Proxy

Proxy 脚本位于 vLLM 源码树的 `examples/disaggregated_prefill_v1/`：

- `load_balance_proxy_server_example.py`：V1/Hybrid/MultiConnector；
- `load_balance_proxy_layerwise_server_example.py`：Layerwise。

查找脚本：

```bash
find /vllm-workspace -name "load_balance_proxy_server_example.py" 2>/dev/null
find /vllm-workspace -name "load_balance_proxy_layerwise_server_example.py" 2>/dev/null
```

优先使用镜像内脚本；否则只读挂载查找到的脚本。启动前清空 `http_proxy` 和
`https_proxy`。

Proxy 的宿主机/容器内启动命令模板如下：

```bash
env http_proxy= https_proxy= \
python <proxy_script> \
  --host <proxy_host> \
  --port <proxy_port> \
  --prefiller-hosts <prefill_backend_hosts> \
  --prefiller-ports <prefill_backend_ports> \
  --decoder-hosts <decode_backend_hosts> \
  --decoder-ports <decode_backend_ports>
```

四个后端参数使用 `nargs='+'`，多个值用空格分隔，不能用逗号；hosts 和 ports
数量相同且顺序对应：

```bash
--prefiller-hosts 192.168.0.171 192.168.0.171 \
--prefiller-ports 18080 18081 \
--decoder-hosts 192.168.0.171 192.168.0.171 \
--decoder-ports 19080 19081
```

主机和端口按实际后端实例展开；同一主机可以重复，但端口必须不同。容器模式在 `--network host` 容器内执行，宿主机
模式直接执行该命令。

## P/D 健康检查

`/health` 只检查 HTTP 状态码，返回 `200` 才算就绪，不检查响应体：

```bash
health_code=$(curl -sS -o /dev/null -w "%{http_code}" \
  --connect-timeout 2 --max-time 5 \
  "http://<service_host>:<service_port>/health")
[ "$health_code" = "200" ]
```

状态码不是 `200` 时继续轮询，直到截止时间；日志出现启动错误、OOM 或进程退出时
立即失败。

P/D 后端的直接 HTTP 检查只允许用于 `/health` readiness；不得直接向 Prefill 或 Decode
发送 `/v1/completions`、`/v1/chat/completions` 或其它业务请求来代替 Proxy 链路验收。

以上 `/health` 检查只适用于 Prefill 和 Decode。Proxy 不提供可用的 `/health` 健康判断，
不得向 Proxy 请求该路径；Proxy 健康必须由下一节中的 `/v1/completions` 和
`/v1/chat/completions` 实际响应共同确认。

## 链路验收

普通部署的公共服务验收还必须检查 `/v1/models`、`/v1/completions` 和
`/v1/chat/completions`。PD Proxy 不实现 `/v1/models`，因此只向 Proxy 发送下面两个
生成请求。每个请求携带唯一 marker；请求前记录 P/D 日志偏移，请求后在有界窗口检查：

Completion 使用 `model`、marker、`max_tokens=1`、`stream=false`；chat 使用同一 `model`、
带 marker 的单条 user message、`max_tokens=1`、`stream=false`。

1. Proxy 对 `/v1/completions` 和 `/v1/chat/completions` 均返回正确模型名和非空文本；
2. Prefill 节点集合至少一个节点出现该请求的处理记录；
3. Decode 节点集合至少一个节点出现同一请求窗口的处理记录；
4. connector/KV 日志没有失败；
5. 池化模式 Mooncake Master 的全部 P/D 客户端均已注册，状态为 serving/ready。

日志优先匹配 marker；若后端日志不回显 marker，则使用请求前后的 endpoint/access-log
记录和无其它业务请求干扰的时间窗口进行关联。只要任一角色没有处理证据，即使 Proxy
返回成功也不能通过链路验收。
