# 静态 OOM 容量检查

本节定义输入合法性检测中的显存静态检查。检查只使用模型权重和最低上下文 KV Cache 容量，不估算真实负载 KV Cache、activation、graph capture、sampler buffer、运行时工作区或通信开销。静态检查用于提前拒绝权重或最低上下文容量明显不可能运行的配置；真实 benchmark 负载是否触发 OOM 由后续 vLLM 服务启动、健康检查和 benchmark 验证。不得再使用 `max_num_seqs * config_max_model_len` 作为静态拒绝依据。

## 1. 执行位置

在完成以下检查后执行本节：

- 七个必填输入存在且通过基础约束。
- 本地 `model_path` 的配置和 tokenizer 检查已完成；`random` 场景不要求真实 checkpoint，非 `random` 场景的真实权重可用性仍由主入口的模型路径检查负责。只要模型配置可静态读取，就按本节从配置推导权重显存需求；配置不可读取或关键结构字段缺失时，将本节结果设为 `unknown`。
- 已确定本次任务实际使用的 NPU 卡号。

本节执行后：

- `status=reject`：停止流程，不启动服务，不运行 benchmark。
- `status=pass`：继续启动服务并运行 benchmark。
- `status=unknown`：继续启动服务并运行 benchmark，不把未知当作通过。

## 2. 选卡与显存容量来源

选卡必须满足：

- 只在 `ASCEND_RT_VISIBLE_DEVICES` 限定的可见设备范围内选择 NPU。
- 需要选择 `tp * dp` 张 NPU。
- 如果能够检测到某张 NPU 上已经存在其他计算任务，不选择该 NPU。
- 如果排除已占用设备后不足 `tp * dp` 张 NPU，停止流程，按环境或资源不足失败处理。

静态 OOM 计算只使用所选 NPU 的总 HBM 容量，不扣减其他任务占用。最终 JSON 只能记录本次任务实际使用的 `model_config.npu_devices`，不得记录 NPU 健康状态、HBM 占用、运行进程、启动前资源快照或清理后资源快照。

显存预算使用所选 NPU 中最小的单卡总 HBM：

```text
CARD_MEMORY_BYTES = min(total_hbm_bytes(selected_npu_devices))
GPU_MEMORY_UTILIZATION = 显式覆盖值；未覆盖时使用 vLLM 默认值 0.9
MEMORY_BUDGET_BYTES = floor(CARD_MEMORY_BYTES * GPU_MEMORY_UTILIZATION)
```

`GPU_MEMORY_UTILIZATION` 仅用于静态预算比较；不得因为静态检查失败或未知而自动修改 vLLM 启动参数。

## 3. 模型权重需求

模型权重需求优先从模型配置推导，不以 checkpoint 文件大小作为主依据。`random` 场景使用 `--load-format dummy` 时，vLLM 仍会先根据模型配置构造同一模型结构，再按各参数 tensor 的形状随机初始化权重；因此无论本次是否加载真实 checkpoint，都应使用同一套配置推导规则估算权重显存需求。

本节只估算常驻模型参数本身，不估算 activation、graph capture、通信 buffer、编译缓存或临时加载峰值。真实 checkpoint 文件存在时，可以把文件大小作为人工排查或实现自检参考，但不得因为未加载真实 checkpoint 就跳过权重显存估算，也不得把 safetensors/bin 文件大小作为唯一计算来源。

### 3.1 配置字段和 dtype

从模型配置顶层或 `text_config` 中读取以下字段；字段命名可按当前 vLLM/Hugging Face 配置的等价字段解析：

- `architectures`、`model_type`
- `vocab_size`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`；缺失时使用 `num_attention_heads`
- `head_dim`；缺失时使用 `hidden_size / num_attention_heads`
- `intermediate_size`
- `tie_word_embeddings`；缺失时按 `false` 保守估算输出层权重
- `torch_dtype`、`dtype` 或显式服务 dtype

权重 dtype 字节数按实际服务会创建的权重 dtype 处理：显式 `--dtype` 或等价覆盖优先；未覆盖时使用模型配置 dtype。非量化或 dummy 权重按以下规则计算：

```text
float32 = 4
float16 = 2
bfloat16 = 2
float8 / int8 = 1
其他未知 dtype = unknown
```

如果存在 `quantization_config` 或显式量化配置，且能够从配置可靠确定常驻权重 bit 数和必要元数据开销，则按该量化格式估算；无法可靠确定时，设置 `status=unknown`、`risk_level=unknown`，继续进入启动加载阶段验证。

### 3.2 Dense Decoder 权重参数量

对常见 decoder-only dense Transformer，按配置推导参数量。实现可根据已知架构精确选择 gated 或非 gated MLP；无法确定时使用 gated MLP 上界。

```text
ATTN_PARAMS_PER_LAYER =
    hidden_size * num_attention_heads * head_dim      # q_proj
  + hidden_size * num_key_value_heads * head_dim      # k_proj
  + hidden_size * num_key_value_heads * head_dim      # v_proj
  + num_attention_heads * head_dim * hidden_size      # o_proj

DENSE_MLP_PARAMS_PER_LAYER =
    3 * hidden_size * intermediate_size               # gated MLP 上界

NORM_PARAMS_PER_LAYER =
    2 * hidden_size

EMBED_PARAMS =
    vocab_size * hidden_size

LM_HEAD_PARAMS =
    0, if tie_word_embeddings = true
    vocab_size * hidden_size, otherwise

DENSE_PARAMS_TOTAL =
    EMBED_PARAMS
  + LM_HEAD_PARAMS
  + num_hidden_layers * (
        ATTN_PARAMS_PER_LAYER
      + DENSE_MLP_PARAMS_PER_LAYER
      + NORM_PARAMS_PER_LAYER
    )
```

如果架构明确使用非 gated MLP 且实现可确认，可把 `DENSE_MLP_PARAMS_PER_LAYER` 改为 `2 * hidden_size * intermediate_size`。bias、额外 norm、logit scale 等小参数在实现可确认时应计入；无法确认时可以忽略这些远小于主干矩阵的参数，但 `message` 中应说明权重需求来自配置估算。

### 3.3 MoE Decoder 权重参数量

对常见 MoE decoder，所有 expert 权重都常驻显存，`num_experts_per_tok`、`top_k` 或等价路由激活专家数不减少权重显存。MoE 字段可使用当前模型配置中的等价名称，例如 `num_experts`、`n_routed_experts`、`moe_intermediate_size`、`moe_ffn_hidden_size`、`shared_expert_intermediate_size` 或 `n_shared_experts`。

```text
ROUTER_PARAMS_PER_LAYER =
    hidden_size * num_experts

EXPERT_PARAMS_PER_LAYER =
    num_experts * 3 * hidden_size * moe_intermediate_size

SHARED_EXPERT_PARAMS_PER_LAYER =
    3 * hidden_size * shared_expert_intermediate_size
```

当模型每层都是 MoE FFN 时：

```text
MOE_PARAMS_TOTAL =
    EMBED_PARAMS
  + LM_HEAD_PARAMS
  + num_hidden_layers * (
        ATTN_PARAMS_PER_LAYER
      + ROUTER_PARAMS_PER_LAYER
      + EXPERT_PARAMS_PER_LAYER
      + SHARED_EXPERT_PARAMS_PER_LAYER
      + NORM_PARAMS_PER_LAYER
    )
```

如果模型配置包含 dense/MoE 混合层、共享 expert 数量、expert bias 或架构专用 FFN 变体，必须按该架构的实际层分布和字段修正上述公式；无法可靠推导时设置 `status=unknown`、`risk_level=unknown`。

### 3.4 单卡权重需求

配置推导出总参数量后，按实际权重 dtype 转成字节数，再按 TP 估算单张 NPU 的常驻权重需求：

```text
WEIGHT_PARAMETER_COUNT = DENSE_PARAMS_TOTAL 或 MOE_PARAMS_TOTAL
WEIGHT_BYTES_TOTAL_CONFIG = WEIGHT_PARAMETER_COUNT * WEIGHT_DTYPE_BYTES
WEIGHT_BYTES_PER_NPU = ceil(WEIGHT_BYTES_TOTAL_CONFIG / tp)
```

DP 不增加单个 replica 内单张 NPU 的权重显存需求。若当前模型启用了 expert parallel、非均匀 TP 切分、无法识别的量化格式、远端模型 ID 且配置不可静态读取，或必要配置字段缺失，设置 `status=unknown`、`risk_level=unknown`，继续进入启动加载阶段验证。

## 4. KV Cache 容量检查

从模型配置中读取以下字段：

- `num_hidden_layers`
- `num_key_value_heads`；缺失时使用 `num_attention_heads`
- `head_dim`；缺失时使用 `hidden_size / num_attention_heads`
- `torch_dtype` 或等价 dtype；未明确指定 KV dtype 时，KV Cache dtype 按模型 dtype 处理

字段缺失且无法可靠推导 `KV_BYTES_PER_TOKEN_PER_NPU` 时，设置 `status=unknown`、`risk_level=unknown`，继续进入启动加载阶段验证。

dtype 字节数按以下规则处理：

```text
float32 = 4
float16 = 2
bfloat16 = 2
float8 / int8 = 1
其他未知 dtype = unknown
```

KV head 在单个 TP rank 上的数量使用保守上界：

```text
KV_HEADS_PER_TP_RANK = max(1, ceil(num_key_value_heads / tp))
```

单 token 在单张 NPU 上的 KV Cache 需求：

```text
KV_BYTES_PER_TOKEN_PER_NPU =
    2 * num_hidden_layers * KV_HEADS_PER_TP_RANK * head_dim * KV_DTYPE_BYTES
```

如果能够确定 vLLM block size，则按每条序列单独向上对齐；不能确定时使用默认 block size `16`：

```text
BLOCK_SIZE = 显式覆盖值；未覆盖时使用 16
```

### 4.1 最低上下文容量门禁

最低上下文门槛使用 `MIN_REQUIRED_CONTEXT_LEN`，默认值为 `4096`。用户显式覆盖时必须写入 `overrides`。该门禁用于判断服务是否至少能在 `max_num_seqs` 并发槽位下容纳指定的业务最低上下文，不使用模型配置中的 `max_position_embeddings`、`model_max_length`、`seq_length` 或 `n_positions` 作为乘数。

```text
MIN_REQUIRED_CONTEXT_LEN = 显式覆盖值；未覆盖时使用 4096
MIN_CONTEXT_TOKENS_PER_SEQ =
    ceil(MIN_REQUIRED_CONTEXT_LEN / BLOCK_SIZE) * BLOCK_SIZE
MIN_CONTEXT_KV_TOKENS =
    max_num_seqs * MIN_CONTEXT_TOKENS_PER_SEQ
MIN_CONTEXT_KV_BYTES_PER_NPU =
    MIN_CONTEXT_KV_TOKENS * KV_BYTES_PER_TOKEN_PER_NPU
```

`MIN_REQUIRED_CONTEXT_LEN * max_num_seqs` 是可行的静态硬门禁：它不会像 `max_num_seqs * config_max_model_len` 那样假设所有请求都达到模型声明的超长上下文，同时仍能保证当前服务配置至少拥有一组明确的并发上下文容量下限。若该门禁超过 KV 预算，说明当前输入配置连最低业务上下文都无法静态容纳，必须 `reject`。

`max_num_batched_tokens` 不作为 KV Cache 总容量上界。它只用于服务调度、profile buffer 和 graph/编译范围；本节不使用它替代最低上下文容量门禁。

## 5. 判定规则

剩余 KV 预算：

```text
KV_BUDGET_BYTES =
    MEMORY_BUDGET_BYTES - WEIGHT_BYTES_PER_NPU
```

最低上下文静态需求显存由以下两部分构成：

```text
MIN_CONTEXT_STATIC_REQUIRED_BYTES_PER_NPU =
    WEIGHT_BYTES_PER_NPU + MIN_CONTEXT_KV_BYTES_PER_NPU
```

判定顺序：

1. `WEIGHT_BYTES_PER_NPU >= MEMORY_BUDGET_BYTES` 时，设置 `status=reject`、`risk_level=capacity_exceeded`。
2. `KV_BUDGET_BYTES <= 0` 时，设置 `status=reject`、`risk_level=capacity_exceeded`。
3. `MIN_CONTEXT_KV_BYTES_PER_NPU > KV_BUDGET_BYTES` 时，设置 `status=reject`、`risk_level=capacity_exceeded`。
4. 以上检查均能可靠计算且未触发拒绝时，设置 `status=pass`、`risk_level=none`。

如果模型权重、KV Cache 或设备容量中的任何关键字段无法可靠计算，设置 `status=unknown`、`risk_level=unknown`，并在 `message` 中说明原因，继续进入启动加载阶段验证。

DP 不增加单张 NPU 的显存需求。每个 DP replica 都必须拥有一组满足上述预算的 `tp` 张 NPU；因此只用 `dp` 检查所需设备数量和分组，不把上述 per-NPU 显存需求再乘以 `dp`。

## 6. 结果写入

最终 `benchmark_result.json` 必须包含 `static_oom_check`：

```json
{
  "static_oom_check": {
    "status": "pass",
    "risk_level": "none",
    "message": null
  }
}
```

字段规则：

- `status` 只能是 `pass`、`reject` 或 `unknown`。
- `risk_level` 只能是 `none`、`capacity_exceeded` 或 `unknown`。
- `message` 必须用一句明确文本说明判定原因；`status=pass` 时可以为 `null`。
- 可以额外记录 `minimum_context_check` 等纯计算字段，例如 token 数、字节数、`min_required_context_len`、`block_size` 和估计方法。
- 不在该字段记录 NPU 健康状态、HBM 占用、运行进程、启动前资源快照或清理后资源快照。

`status=reject` 时：

- 不启动服务。
- 不运行 benchmark。
- `Valid=false`。
- 顶层 `status=input_invalid`。
- `error.stage=input`。
- `error.type=static_oom_capacity_exceeded`。
- `error.message` 必须说明显存需求由模型权重和最低上下文 KV Cache 容量构成，且 `MIN_CONTEXT_KV_BYTES_PER_NPU` 已经超过 `KV_BUDGET_BYTES`，或权重本身已经超过预算。
- 因为服务未启动，最终 JSON 的 `model_config.npu_devices` 必须为空数组；静态检查阶段选出的候选卡号不得记作实际使用卡号。
- `performance` 中所有指标为 `null`。

`status=pass` 时：

- 继续启动服务和执行 benchmark。
- 后续如果仍发生 OOM，按实际 OOM 失败处理，不得修改静态检查结论。

`status=unknown` 时：

- 继续启动服务和执行 benchmark。
- `message` 必须说明无法静态判定的原因，例如非本地模型 ID 的配置不可读取、模型配置字段缺失、参数量无法可靠推导、量化格式无法静态估算、dtype 无法确定、KV Cache 关键字段缺失或设备容量无法可靠读取。
