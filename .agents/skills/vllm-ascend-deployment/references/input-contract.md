# 部署输入契约

Provider 为每次部署提供字段。`scripts/input_contract.py` 是本契约的唯一可执行入口，
skill 在输出目录中组装并写出 `deployment.json`；之后所有阶段只消费该规范输入。

## Provider 输入

Provider 字段由 skill 通过命令行参数传入，不要求 Provider 预先创建 JSON 文件：

```text
python <skill-root>/scripts/input_contract.py \
  --model-path /absolute/model/path \
  --served-model-name model-name \
  --visible-devices 0 1 \
  --port-candidates 8117 8118 \
  --preferred-port 8117 \
  --attempt-dir /absolute/attempt-dir \
  [runtime options]
```

`--config-path`、`--tp`、`--dp`、`--max-num-seqs`、`--max-num-batched-tokens`、
`--gpu-memory-utilization`、`--min-required-context-len`、`--max-model-len`、`--block-size`、
`--dtype`、`--quantization`、`--extra-arg`、`--card-memory-bytes`、`--device-count`、
`--max-tp-dp` 和 `--port` 为可选字段。工具路径、activation wrapper
和环境变量值不属于 Provider 合同，由 skill 在环境预检阶段处理。

`config_path` 可以省略，默认使用 `model_path/config.json`。远程模型标识、认证和缓存
不是本 skill 的输入来源；本 skill 要求本地模型路径和配置可读。

## 默认值与状态

缺少字段时脚本写入规范化 `runtime` 并记录 `resolution.defaulted_fields`：

| 字段 | 默认值 |
| --- | ---: |
| `dp` | `1` |
| `max_num_seqs` | `32` |
| `max_num_batched_tokens` | `4096` |
| `gpu_memory_utilization` | `0.9` |
| `min_required_context_len` | `4096` |
| `block_size` | `128` |

`tp` 缺失时保持 `null`，并返回 `input_status=missing_parameters`。缺失的其他启动字段
也会被记录在 `resolution.missing_fields`，但会先使用上述值进行最小 TP 推荐。

- `pass`：模型、配置、资源和启动参数完整且满足硬约束。
- `missing_parameters`：模型和资源输入有效，但需要最小 TP 推荐后才能启动。
- `invalid`：路径、JSON、类型、数值关系、设备、端口或输出目录违反契约。

`invalid` 必须写出包含 `diagnostics.errors` 的 `deployment.json`，但不得进入静态部署、
资源分配或服务启动。

## 硬约束

- `model_path` 是可读目录，且 `config.json` 是可读 JSON object。
- `served_model_name` 是非空字符串，不能从路径推导。
- 设备 ID 必须为不重复的十进制字符串；端口必须位于 `1..65535` 且不能重复。
- `port` 或 `preferred_port`（如果提供）必须属于 `port_candidates`；端口占用由资源
  probe 动态确认，不能由本脚本伪造可用。
- `dp`、`max_num_seqs` 和 `max_num_batched_tokens` 为正整数，且后者不小于前者。
- `gpu_memory_utilization` 必须是 `(0, 1]` 内的有限数。
- 若 `tp` 已确定，则 `tp * dp` 不得超过 Provider 可见设备数或 `max_tp_dp`，并且
  `num_attention_heads % tp == 0`（配置可确认时）。
- `attempt_dir` 必须是绝对路径；每次 attempt 必须使用独立目录和
  `deployment.json`。

## 规范文件和推荐写回

生成规范文件：

```text
python <skill-root>/scripts/input_contract.py \
  --model-path <model-path> \
  --served-model-name <served-model-name> \
  --visible-devices <device-id> [<device-id> ...] \
  --port-candidates <port> [<port> ...] \
  --attempt-dir <attempt-dir>
```

脚本固定将规范文件写为 `<attempt-dir>/deployment.json`。输出目录必须是本次 attempt
专属且不能已经存在同名规范文件。

静态 OOM 报告给出可用 TP 后，原子写回同一文件：

```text
python <skill-root>/scripts/input_contract.py \
  --input <attempt-dir>/deployment.json \
  --resolve-tp <recommended-tp> \
  --resolution-source <attempt-dir>/static-oom-report.json
```

写回后的结构保留原始状态和推荐来源：

```json
{
  "input_status": "pass",
  "resolution": {
    "mode": "auto_recommended",
    "recommended_tp": 2,
    "source": "static-oom-report.json",
    "original_status": "missing_parameters"
  },
  "runtime": {
    "tp": 2,
    "dp": 1,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096
  }
}
```

写回后必须再次执行本契约的硬约束检查。推荐值不满足约束时报告
`input_invalid`；不得自动尝试更大的 TP、替换设备或修改端口。
