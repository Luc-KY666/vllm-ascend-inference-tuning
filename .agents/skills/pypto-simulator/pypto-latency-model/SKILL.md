---
name: pypto-latency-model
description: 从规范化 PyPTO profile 构建可审计的 task、subslot 和 physical-block 延迟模型。
---

# PyPTO 延迟模型

本 skill 只负责把 `pypto-profile` 产出的 native evidence 转成 latency model。它不认识具体模型，
也不负责启动 serving、替换 child 或执行 bracket。

## 输入与输出

输入是一个或多个 `profile_manifest.json`。重复传入 `--manifest` 合并独立 capture：

```bash
python .agents/skills/pypto-simulator/pypto-latency-model/scripts/fit_latency_models.py \
  --manifest /tmp/capture-a/profile_manifest.json \
  --manifest /tmp/capture-b/profile_manifest.json \
  --manifest /tmp/capture-c/profile_manifest.json \
  --min-samples 3 --min-workloads 3 \
  --output /tmp/pypto-model
```

输出：

- `model_manifest.json`：完整 group、signature、sample、range、confidence、provenance 和 block model；
- `latency_lookup.json`：runtime 使用的紧凑 lookup 表。

## Signature 与 confidence

默认 latency signature 必须至少区分：

- model family/model id；
- task name、phase、kind、resource shape；
- block count、active subslot、kernel/callable name；
- 所有有证据表明会改变延迟的动态 feature。

`sample_count` 表示 raw task observation，`workload_count` 表示独立 native capture。二者必须分开
统计；同一 graph 的重复 layer/task 不能冒充独立 workload。

只有两个计数都达到命令阈值，point 才能成为 `baseline`；否则为 `low_confidence`。生产 strict
model 默认只接受 exact point 或测量域内、categorical feature 完全一致的插值，禁止外推。

## Block 与 subslot

对 AIC/AIV/MIX task 保留：

- active subslot；
- subslot completion duration；
- `block_num`；
- 每个 epoch 排序后的 physical-block duration distribution。

block rank 只表示排序后的 occupancy distribution，不恢复原始 physical block identity。vector 长度
必须精确等于 `block_num`，不能补齐、截断、跨 block count 拼接或用另一个 workload 猜测。

## 模型要求

模型必须保留：

- raw observation 和 source workload identity；
- observed feature range；
- per-group MAE/MAPE 和 critical-path-weighted error；
- low-confidence、unsupported 和 extrapolation 状态；
- compute、communication、memory、runtime-overhead 的独立边界。

禁止按 ACLNN operation name 代替 outlined simpler task identity。一个 MIX/SPMD outlined task 即使
内部包含多个 primitive operation，仍是一个 runtime scheduling node。

## 新模型接入

新模型的 feature extractor、program role、tensor role 和 control-output metadata 放入模型 reference，
不要在 fitter 中增加 `if model == ...`。参见 `../references/model-adapter-contract.md` 和
`../references/manifest-contract.md`。

## 前置条件

- Python 3.10+；
- 不需要 PyPTO、CANN 或 NPU；
- fixture 只用于 schema 和 fitter contract 测试，不能作为硬件性能证据。
