# Qwen Adapter 契约

## 初始范围

首个 runtime 目标是固定或受控 batch/shape 的单卡 Qwen3 decode。只有 decode task coverage
完成后，才扩展 prefill 和 distributed-window communication。

## Tensor Role

Compiler/runner adapter 至少必须标记：

- 静态模型 tensor：`weight`；
- key/value page pool：`kv_cache`；
- distributed communication buffer：`window`；
- 适用时将 `token_ids`、`sampled_ids`、sequence length、block table、slot mapping 标为 `control`；
- 除非 device sampling 明确省略，否则 full logits 标为 `activation`。

分类依据是语义，不能只按 allocation size 判断 tensor 是否虚拟。

## Sampled ID

Qwen decode 读取 input token buffer，并写 output sampled-ID buffer。ABI 每行可能 padding 到八个
`int32`，但 serving 只消费第 0 列。模拟时：

```text
sampled_ids[row][0] = configured_constant_token
sampled_ids[row][1:] = 0
```

默认 scaffold 值为 `1`，runtime integration 必须允许修改，并保持在 vocabulary 范围内。该值
无需从 logits 推导，生成文本没有数值意义。

Timing-only 运行应走 device-greedy path，使 serving 只消费小型 sampled-ID 控制输出。Host
sampling 需要物化大型 synthetic logits，不在首阶段范围内。

## 必需查找 Feature

Qwen task metadata 至少应按实际影响暴露：

- prefill/decode phase；
- kernel batch 与 active batch；
- active token count；
- sequence/KV length 或有边界 bucket；
- SPMD block count；
- active AIC/AIV subslot 和 MIX shape；
- dtype 与 quantization variant；
- tensor-parallel world size 与 rank；
- distributed task 的 communication bytes。

不得静默默认会改变 task structure 的 feature。缺失此类 feature 是 profile/export coverage 失败。
没有 args dump 或其他 dispatch-correlated 证据时，不得从固定 tensor shape 推断动态 prompt/KV 值。
