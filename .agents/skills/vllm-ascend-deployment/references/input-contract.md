# 部署输入契约

Provider 输入由 `scripts/input_contract.py` 一次规范化。脚本只返回 stdout JSON，不创建
目录、不写 `deployment.json`，agent 将返回对象直接放入 `input.normalized` 和
`stages.input_validation`，避免阶段之间重复读写。

## Provider 输入

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

该命令只输出更新后的规范化对象，不读取或写入任何部署文件，也不执行静态 OOM。调用方
将前后对象和 `static_oom_reused=true`、`recheck_performed=false` 写入报告的
`stages.tp_resolution`，复用原有 OOM 结果。
