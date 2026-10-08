---
name: vllm-ascend-benchmark-skill
description: 在昇腾 NPU 上使用 vLLM-Ascend 执行可验证的模型压测。执行时必须显式提供模型名称、模型路径、负载场景、TP、DP、max_num_seqs 和 max_num_batched_tokens，最终输出单个 benchmark_result.json。
---

# vLLM-Ascend 压测 Verifier

用于验证指定模型、并行配置和负载场景下模型服务是否能够拉起，并在服务可用时执行 `vllm bench serve`。执行时必须提供完整的模型、并行和场景输入，不能依赖主入口中的隐式默认值；最终以一个 JSON 文件作为契约输出。

数据集默认值和覆盖规则见[数据集配置](references/dataset-config.md)，输入显存合法性与 OOM 容量静态检查见[静态 OOM 容量检查](references/static-oom-precheck.md)，服务启动、动态诊断和性能结果提取见[模型拉起与压测](references/model-launch-and-benchmark.md)。

本 Skill 支持四种负载场景，`scenario` 是必填输入：

- `long_prefill`：`hf` 数据集 `likaixin/InstructCoder`，真实代码长输入、短输出。
- `long_decode`：`hf` 数据集 `AI-MO/NuminaMath-CoT`，数学题短输入、CoT 长输出。
- `prefill_decode_balance`：`sharegpt` 数据集，真实多轮对话分布。
- `random`：vLLM random 数据集，使用可控的合成输入和输出长度。

## 一、输入契约

### 1. 必填输入

以下七个字段没有默认值，执行时必须显式提供：

```json
{
  "model_name": "输入的模型名称",
  "model_path": "输入的模型路径或模型 ID",
  "scenario": "long_prefill、long_decode、prefill_decode_balance 或 random",
  "tp": "输入的正整数",
  "dp": "输入的正整数",
  "max_num_seqs": "输入的正整数",
  "max_num_batched_tokens": "输入的正整数"
}
```

字段语义：

- `model_name`：服务对外暴露的模型名称，直接用于 `vllm serve --served-model-name`，并作为在线 HTTP benchmark 的 `vllm bench serve --model` 值。
- `model_path`：本地模型目录、模型配置路径或模型 ID。它用于服务加载模型，并作为 `vllm bench serve --tokenizer` 的 tokenizer 来源。对于可解析为本地路径的输入执行文件静态检查；对于模型 ID 按配置的权重来源在启动加载阶段验证。它不由 Skill 猜测，也不能用默认模型替代。
- `scenario`：负载场景，必须是 `long_prefill`、`long_decode`、`prefill_decode_balance` 或 `random` 之一；场景直接决定 benchmark 数据集。
- `tp`、`dp`：tensor parallel 和 data parallel 配置。
- `max_num_seqs`、`max_num_batched_tokens`：服务启动配置。

参数约束：

- `model_name` 和 `model_path` 必须是非空字符串。
- `scenario` 必须是 `long_prefill`、`long_decode`、`prefill_decode_balance` 或 `random` 之一。
- `tp` 和 `dp` 必须是正整数，且都为 2 的幂次。
- `max_num_seqs` 和 `max_num_batched_tokens` 必须是正整数，且都为 2 的幂次。
- `max_num_seqs` 必须大于等于 `tp`，`max_num_batched_tokens` 必须大于等于 `max_num_seqs`。
- `tp * dp` 不得超过动态检查确认可用的 NPU 数量。
- 如果设置了 `ASCEND_RT_VISIBLE_DEVICES`，必须正确解析其设备列表，只在可见卡范围内检查、统计和选择 NPU。
- 通过基础参数、模型路径和设备选择检查后，必须按[静态 OOM 容量检查](references/static-oom-precheck.md)对模型权重和最低上下文 KV Cache 容量执行显存合法性检查。权重或最低上下文容量远超当前卡显存能力的配置不得继续启动服务或执行 benchmark；无法静态确定的配置必须标记为 unknown 并继续通过运行时验证；检查通过的配置直接继续压测。不得再使用 `max_num_seqs * config_max_model_len` 作为静态拒绝依据。
- 必须确定服务实际使用的 NPU 卡号，并仅记录本次任务实际使用的 NPU 卡号。不得在最终 JSON 或必需的用户可见诊断工件中记录 NPU 健康状态、HBM 占用、运行进程、启动前资源快照或清理后资源快照。

除非显式覆盖，否则不得修改这七个输入。

### 2. 其他配置的来源

除上述七个必填输入外，其他配置不属于主入口的必填输入：

- 数据集参数默认值、数据集路径和场景对应关系由[数据集配置](references/dataset-config.md)提供；所有支持的模式默认 `num-prompts=200`，仅在显式覆盖时改变。
- 静态 OOM 容量检查的最低上下文门槛 `MIN_REQUIRED_CONTEXT_LEN` 默认使用 `4096`；仅在显式覆盖时改变，并必须写入 `overrides`。
- host、端口、启动超时、轮询间隔、权重来源、远程代码开关、HTTP backend/endpoint、结果目录和本机环境画像缓存路径由[模型拉起与压测](references/model-launch-and-benchmark.md)提供默认值；仅在显式覆盖时改变。
- 默认配置文档不能覆盖或替代上述七个必填输入。

## 二、主入口职责和流程

主入口负责：

1. 检查七个必填输入是否全部存在且非空；缺少任一项时不启动服务，生成 `status=input_invalid` 的结果 JSON。
2. 先静态检查 `scenario` 枚举、数值、TP/DP 幂次、`max_num_seqs` 和 `max_num_batched_tokens` 的幂次及大小关系。
3. 如果任一输入参数约束违反，不进行实际环境测试、服务启动或 benchmark，直接生成 `Valid=false`、`status=input_invalid` 的结果 JSON，并在 `error.message` 中记录具体字段和违反的约束。
4. 参数约束通过后，按[模型拉起与压测](references/model-launch-and-benchmark.md)读取或生成本机环境画像文件。命令路径、vLLM/vLLM-Ascend/Torch 版本、CLI 参数兼容性和本地数据集路径校验等可复用探测只在画像缺失、失效或显式刷新时执行；后续运行读取画像，不重复执行耗时探测。端口、设备占用、服务健康、模型路径和本次输入相关检查仍按每次运行执行，不得用画像缓存替代。
5. 静态检查本地 `model_path` 的模型配置、tokenizer 和必要文件；`random` 以外的场景还检查本地真实权重或权重索引，非本地模型 ID 的来源可用性在启动加载阶段验证。
6. 模型路径静态检查通过后，选择本次任务实际使用的 NPU 卡号；选卡时排除已能检测到有其他计算任务占用的 NPU，但静态 OOM 计算不扣减其他任务的显存占用。
7. 调用[静态 OOM 容量检查](references/static-oom-precheck.md)执行显存合法性检查，并生成 `static_oom_check` 结果。检查结果为 `reject` 时，不启动服务或 benchmark，直接写出失败 JSON；检查结果为 `pass` 或 `unknown` 时继续后续流程，并把检查结果写入最终 JSON。
8. 模型路径和静态 OOM 检查通过或为 unknown 后，按 `scenario` 检查对应的数据集路径和其他数据集资源；本地数据集目录或文件的可复用校验结果优先来自本机环境画像，画像缺失该路径或路径指纹变化时才重新校验并更新画像。
9. 创建独立运行目录和最终输出文件路径。
10. 调用[数据集配置](references/dataset-config.md)生成 `vllm bench serve` 所需的数据集参数。
11. 调用[模型拉起与压测](references/model-launch-and-benchmark.md)执行服务启动、健康检查和 benchmark；静态 OOM 容量检查为 unknown 时，后续启动和 benchmark 结果即为实际验证结果。
12. 无论成功还是失败，都写出最终 `benchmark_result.json`；不要只输出终端文本。

主入口不得把 TP/DP 或 NPU 资源检查委托给数据集配置文档，也不得因为 benchmark 失败而丢弃模型启动结果。

## 三、输出契约

### 1. 唯一接口输出

本 Skill 对 `/goal` 的唯一契约输出是一个 JSON 文件：

```text
<result_root>/<run_id>/benchmark_result.json
```

日志、服务命令、benchmark 原始 JSON 和诊断文件可以保留在同一运行目录中，但它们是诊断工件，不替代最终 JSON。

### 2. JSON 最低结构

以下是结构示例，示例中的数值不是默认值，实际值必须来自七个必填输入和最终执行配置：

```json
{
  "model_config": {
    "model_name": "输入的 model_name",
    "model_path": "输入的 model_path",
    "tp": 0,
    "dp": 0,
    "max_num_seqs": 0,
    "max_num_batched_tokens": 0,
    "npu_devices": [],
    "port": 0
  },
  "dataset": {
    "name": "random",
    "path": null,
    "num_samples": 0,
    "requested_num_prompts": 0,
    "max_concurrency": 0,
    "num_warmups": 0,
    "parameters": {}
  },
  "Valid": true,
  "static_oom_check": {
    "status": "pass",
    "risk_level": "none",
    "message": null
  },
  "performance": {
    "unit": "ms",
    "TTFT": {"P90": 0.0, "P99": 0.0},
    "TPOT": {"P90": 0.0, "P99": 0.0},
    "E2EL": {"P90": 0.0, "P99": 0.0}
  },
  "status": "success",
  "error": null,
  "overrides": {},
  "artifacts": {
    "summary_json": ".../benchmark_result.json",
    "server_log": ".../vllm_server.log",
    "benchmark_log": ".../benchmark.log",
    "raw_benchmark_json": "..."
  }
}
```

字段语义：

- `model_config`：必须记录模型名称、模型路径、TP、DP、`max_num_seqs`、`max_num_batched_tokens`、实际端口和 `npu_devices`；`npu_devices` 只记录本次任务实际使用的 NPU 卡号，不记录资源快照或进程信息。负载场景记录在 `dataset.scenario` 中。
- `dataset.name`：实际使用的 `random`、`hf` 或 `sharegpt`；`dataset.path` 记录数据集目录、文件或数据集 ID；HF 数据集同时记录 `dataset.hf_name`；`dataset.num_samples` 是实际处理的数据条数，无法执行 benchmark 时为 `null`；`requested_num_prompts` 保留最终请求值。
- `dataset.scenario`：当 `scenario` 为 `long_prefill`、`long_decode` 或 `prefill_decode_balance` 时必须记录对应值；当 `scenario` 为 `random` 时省略该字段，不写入 `null`。
- `dataset.max_concurrency`：实际使用的最大并发。
- `static_oom_check`：记录[静态 OOM 容量检查](references/static-oom-precheck.md)的结论。`status` 只能是 `pass`、`reject` 或 `unknown`；`risk_level` 使用 `none`、`capacity_exceeded` 或 `unknown`。该字段不得记录 NPU 健康状态、HBM 占用、运行进程、启动前资源快照或清理后资源快照。
- `Valid`：模型服务是否成功拉起并通过 `/health`，且完成服务身份验证（优先通过 `/v1/models` 与 `model_name` 匹配；无法获取时通过最小接口预检）。模型进程退出、健康检查超时、服务身份验证失败、OOM 或主入口静态检查失败时为 `false`；OOM 必须为 `false`。
- `performance`：成功完成 benchmark 后填写 TTFT、TPOT、E2EL 的 P90 和 P99，单位为毫秒。未完成 benchmark 时对应值为 `null`，不要填 `0`。
- 服务已通过 `/health` 且服务身份验证成功，但 benchmark 阶段失败时，`Valid` 保持 `true`，`status` 为 `benchmark_failed`，性能字段为 `null`，并在 `error` 中记录原因。
- 原始 benchmark JSON 中存在失败请求、完成请求数少于本次请求数，或记录的请求数与本次配置不一致时，不能判为 `success`；按 `benchmark_failed` 记录具体计数，性能字段为 `null`。
- `artifacts.raw_benchmark_json`：只能记录本次运行中已解析、已校验且唯一确定的原始 benchmark JSON 路径；无法唯一确定时写为 `null`，不得用未经验证的文件名或宽泛 glob 结果代替。性能字段只能从该文件提取。
- `overrides` 只记录显式覆盖的非必填配置，不要把数据集和服务默认值误记为输入；不得在 `overrides` 中重复记录 NPU 可见设备环境变量、资源快照或运行进程，NPU 信息仅通过 `model_config.npu_devices` 表达。

`TTFT`、`TPOT`、`E2EL` 的命名必须保持一致，不得写成 `tpop` 或 `esel`。

## 四、失败时的 JSON 规则

即使缺少必填输入、启动失败、OOM、健康检查超时或 benchmark 失败，也必须尽力写出 `benchmark_result.json`。缺少输入时示例：

```json
{
  "model_config": {
    "model_name": null,
    "model_path": null,
    "tp": null,
    "dp": null,
    "max_num_seqs": null,
    "max_num_batched_tokens": null,
    "npu_devices": []
  },
  "dataset": null,
  "Valid": false,
  "static_oom_check": null,
  "performance": {
    "unit": "ms",
    "TTFT": {"P90": null, "P99": null},
    "TPOT": {"P90": null, "P99": null},
    "E2EL": {"P90": null, "P99": null}
  },
  "status": "input_invalid",
  "error": {
    "stage": "input",
    "type": "missing_required_input",
    "message": "缺少一个或多个必填输入"
  }
}
```

违反输入参数约束（包括 `scenario` 不在允许枚举内）时，不启动服务或运行 benchmark，`Valid` 必须为 `false`，`status` 必须为 `input_invalid`，`performance` 中所有指标必须为 `null`，并设置 `error.stage=input`、`error.type=invalid_parameter_constraint`；`error.message` 必须明确指出具体字段、实际值和违反的约束。

本地 `model_path` 或其必要文件静态检查失败时，不启动服务或运行 benchmark，`Valid` 必须为 `false`，`status` 必须为 `input_invalid`，并设置 `error.stage=input`、`error.type=invalid_model_path`；`error.message` 必须明确指出缺失、不可读或无法解析的具体路径。非本地模型 ID 的来源、认证或下载失败在启动加载阶段记录为服务启动失败，并保留具体原因。

静态 OOM 容量检查结果为 `reject` 时，不启动服务或运行 benchmark，`Valid` 必须为 `false`，`status` 必须为 `input_invalid`，并设置 `error.stage=input`、`error.type=static_oom_capacity_exceeded`；`error.message` 必须说明当前配置的显存需求由模型权重和最低上下文 KV Cache 容量构成，且已经超过[静态 OOM 容量检查](references/static-oom-precheck.md)定义的拒绝阈值。检查结果为 `unknown` 时不得改写为失败，必须保留 unknown 标记并继续执行服务启动与 benchmark。

不要通过修改 TP、DP、`max_num_seqs` 或 `max_num_batched_tokens` 来掩盖 OOM。保留原始输入和日志供 `/goal` 判断。
