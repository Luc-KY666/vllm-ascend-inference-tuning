# 单机 PD 专属部署规则

本文件将多机 PD 文档中的角色规则收敛到单台主机。多机文档中的 connector 映射、
JSON 模板和 Proxy 脚本名称仍是唯一依据；本文件只补充单机的设备、端口和生命周期
约束。

## 设备和并行

Prefill 与 Decode 是两个独立实例，分别在自己的设备集合内解析 TP；单实例 DP=1：

```text
prefill_tp = min(static_oom_pass_candidates(prefill.devices))
decode_tp  = min(static_oom_pass_candidates(decode.devices))
```

未显式指定角色 TP 时，使用上述默认最小 TP。只有 Provider 明确指定角色 TP 时才
使用显式值；显式值仍需通过 attention heads、设备数和静态 OOM 校验。输入中的
`devices` 是角色设备池，TP 决策完成后按输入顺序选择 TP 所需设备。两组设备池必须
不相交。每个实例都加载完整模型，所以每一侧都要单独通过静态容量检查；不能以
`prefill_tp + decode_tp` 的总 TP 替代两次角色检查。

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

## P/D 启动命令

Prefill 和 Decode 使用同一命令结构，角色值由部署输入展开：

```bash
vllm serve /models/model \
  --host 0.0.0.0 --port <role_api_port> \
  --tensor-parallel-size <role_tp> \
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
`--tensor-parallel-size` 一致。量化模型增加 `--quantization ascend`；只有确认
graph capture 失败时才增加 `--enforce-eager`。

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

单机至少需要：

```text
prefill_api_port
decode_api_port
proxy_port
prefill_kv_port
decode_kv_port
```

池化组合再增加 `mooncake_service_port`。所有端口必须来自 Provider 候选并通过端口
探针确认可用。KV 端口要避开 AscendDirectTransport 随机端口区；不能因为端口在本机
而跳过检查。

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

## 健康检查

`/health` 只检查 HTTP 状态码，返回 `200` 才算就绪，不检查响应体：

```bash
health_code=$(curl -sS -o /dev/null -w "%{http_code}" \
  --connect-timeout 2 --max-time 5 \
  "http://<service_host>:<service_port>/health")
[ "$health_code" = "200" ]
```

状态码不是 `200` 时继续轮询，直到截止时间；日志出现启动错误、OOM 或进程退出时
立即失败。

## 链路验收

向 Proxy 发送带唯一标记的最小 completion 请求，并在有界时间窗口内检查：

1. Proxy 返回正确模型名和非空文本；
2. Prefill 日志出现该请求的处理记录；
3. Decode 日志出现同一请求窗口的处理记录；
4. connector/KV 日志没有失败；
5. 池化模式 Mooncake Master 的全部 P/D 客户端均已注册，状态为 serving/ready。
