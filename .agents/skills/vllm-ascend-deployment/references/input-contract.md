# 部署输入契约

Provider 输入由 `scripts/input_contract.py` 一次规范化。脚本只返回 stdout JSON，不创建
目录、不写 `deployment.json`，agent 将返回对象保存在内存中，供静态 OOM、TP 决策和服务
启动复用，避免阶段之间重复读写。

部署接口可选声明 `service.backend`、`service.endpoint` 和 `service.tokenizer`；缺省值分别为
`vllm`、`/v1/chat/completions` 和模型路径。部署完成后只把最终接口写入报告的 `service`，
供 benchmark 直接复用。

## Provider 输入

```text
python <skill-root>/scripts/input_contract.py \
  --model-path /absolute/model/path \
  --served-model-name model-name \
  --visible-devices 0 1 \
  --port-candidates 8117 8118 \
  --preferred-port 8117 \
  --attempt-dir /absolute/attempt-dir \
  [--backend vllm --endpoint /v1/chat/completions --tokenizer /absolute/model/path] \
  [runtime options]
```

`--config-path`、`--tp`、`--dp`、`--max-num-seqs`、`--max-num-batched-tokens`、
`--gpu-memory-utilization`、`--min-required-context-len`、`--max-model-len`、`--block-size`、
`--dtype`、`--quantization`、`--extra-arg`、`--card-memory-bytes`、`--device-count`、
`--max-tp-dp` 和 `--port` 为可选字段。`attempt-dir` 只作为报告和外置 artifact 的目标
位置校验，不由脚本创建。

缺少 `tp` 时保持 `null` 并返回 `input_status=missing_parameters`；其它默认值写入
`resolution.defaulted_fields`。`pass` 表示输入完整，`invalid` 表示硬约束失败。配置文件
原始字节只读取一次，同时完成 JSON 解析和 SHA256 计算；返回对象只携带静态 OOM 所需的
架构摘要，不复制完整配置文本。

## TP 解析

静态 OOM 返回推荐值后，agent 在内存中调用：

```python
updated = resolve_recommended_tp(normalized, recommended_tp, "static_oom")
```

或通过 stdin CLI：

```text
python <skill-root>/scripts/input_contract.py \
  --resolve-tp <recommended-tp> \
  --resolution-source static_oom \
  < normalized-input.json
```

该命令只输出更新后的规范化对象，不读取或写入任何部署文件，也不执行静态 OOM。它只依据
第一次校验缓存的 `allowed_tp`、设备数量、DP 和 attention heads 做轻量检查，不重新执行完整
输入契约或读取 `config.json`。调用方将前后对象和 `static_oom_reused=true`、
`recheck_performed=false` 保存在内存上下文中，复用原有 OOM 结果；最终报告只写入解析后的
`launch.tp`。

## PD 模式扩展

当 `deployment.mode=single_node_pd` 时，调用 `scripts/pd_input_contract.py`，输入在
普通 Provider 字段之外增加 P/D 设备池和端口：

```json
{
  "deployment": {"mode": "single_node_pd", "connector": "MooncakeConnectorV1"},
  "pd": {
    "prefill": {"devices": [0, 1], "api_port": 18080, "kv_port": 30000},
    "decode": {"devices": [2, 3], "api_port": 18081, "kv_port": 30200},
    "proxy": {"port": 18000}
  }
}
```

P/D 设备池必须互不重叠且属于 `resources.visible_devices`；API、KV、Proxy 和池化
控制端口必须唯一。未指定角色 TP 时，由 PD 静态 OOM 阶段在该角色设备池中选择
最小可行 TP，再按设备池顺序选择对应设备。
