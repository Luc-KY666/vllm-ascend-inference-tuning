# 部署输入契约

本文件是部署 skill 的唯一输入规范。它规定任务 Main 或 Goal Plus provider 如何提供
一次部署 attempt 的规范输入；它不是一个模型目录扫描器，也不要求额外维护输入校验
脚本。

## 文件关系

| 文件 | 生成者 | 用途 |
| --- | --- | --- |
| `deployment-input.json` | 任务 Main 或 Goal Plus provider | 本次部署唯一的外部规范输入 |
| `static-oom-input.json` | 部署流程 | 从规范输入、官方配置和硬件事实派生的内部预检输入 |
| `static-oom-report.json` | `static_oom_precheck.py` | 静态容量和最小 TP 结果 |
| `deployment_report.json` | 部署流程 | 一次部署 attempt 的最终事实记录 |

这些文件位于本次 attempt 的输出目录。它们各自只有一个规范名称，不能互相替代。
`static-oom-input.json` 不是用户需要额外填写的第三份合同，而是部署流程从唯一规范
输入派生出的 artifact。

## 规范输入结构

下面的代码块是字段形状示意，尖括号内容必须由任务或 provider 替换，不能作为实际
默认值直接执行：

```json
{
  "schema_version": "vllm-ascend-deployment-input-v1",
  "model": {
    "model_path": "<任务提供的模型路径或允许的模型标识>",
    "served_model_name": "<服务对外暴露的模型身份>",
    "config_path": "<任务或 provider 解析出的官方 config.json 路径>"
  },
  "runtime": {
    "tp": "<正整数或 null>",
    "dp": "<正整数>",
    "max_num_seqs": "<正整数>",
    "max_num_batched_tokens": "<正整数>",
    "gpu_memory_utilization": "<(0, 1] 内的有限数>",
    "min_required_context_len": "<正整数>",
    "max_model_len": "<正整数>",
    "dtype": "<任务冻结的 dtype>",
    "quantization": null,
    "extra_args": []
  },
  "resources": {
    "provider": "<当前 provider 标识>",
    "authorized_devices": ["<GP 授权设备>"],
    "visible_devices": ["<provider 实际注入的设备>"],
    "visibility_env": {
      "<设备可见性环境变量>": "<provider 冻结值>"
    },
    "device_groups": [["<可选的 provider 设备分组>"]],
    "port": "<provider 分配的端口>",
    "activation_script": "<任务提供的绝对环境 wrapper>",
    "python": "<任务提供的绝对 Python>",
    "output_root": "<provider 允许写入的绝对输出 surface>"
  },
  "outputs": {
    "attempt_dir": "<provider 允许本次 attempt 使用的目录>",
    "deployment_report": "<attempt-dir>/deployment_report.json"
  }
}
```

`model.model_path` 和 `model.served_model_name` 始终必填，且不能相互推断。
`config_path` 必须指向本轮实际使用的官方模型配置来源；远程模型标识的解析、认证和
缓存由任务或 provider 负责。

真实启动时，`runtime` 中的部署参数和 `resources` 中的资源事实必须完整。`tp` 可以
在推荐模式下为 `null`，但不能在启动命令中保持未解析状态。`extra_args` 只能包含任务
合同允许且当前 vLLM CLI 已确认支持的参数。示例中的字段值只是类型标记，不是公共
默认值。

## 字段来源与冻结关系

- 任务合同、`workflow.md`、`inputs.md` 和 SearchSpec 决定模型身份、服务身份、固定
  运行参数、可调参数白名单和输出要求。
- Goal Plus provider 决定授权设备、可见性环境、端口、wrapper、Python、资源锁和
  attempt 输出 surface。skill 不从全机设备枚举结果扩大授权范围。
- 官方模型 `config.json` 提供架构字段和默认 dtype 等模型事实。部署流程在派生静态
  输入时读取它；静态预检脚本不扫描模型目录，也不从权重文件大小推断权重需求。
- 每个 candidate 或 attempt 都必须重新生成自己的 `deployment-input.json`，不得复用
  其他 candidate 的设备、端口、输出目录或预检报告。

任务应明确每个字段属于以下哪一类：

| 类别 | 规则 |
| --- | --- |
| 固定字段 | 本轮所有 attempt 必须一致，例如模型身份或 workload 约束 |
| 可搜索字段 | 只能来自 SearchSpec 白名单，由 Goal Plus 分配到当前 candidate |
| provider 字段 | 由 provider 注入，skill 只能核对和消费 |
| 派生字段 | 由部署流程从规范输入和官方事实生成，不能由用户另填一份冲突值 |

## 契约检查

部署流程按本规范执行检查，不维护第二套 Python 输入校验器。至少确认：

- 必填身份是非空字符串；路径、配置来源、wrapper、Python 和输出 surface 与当前
  provider 合同一致。
- 并行和批处理参数是正整数，数值关系满足任务合同；其中
  `runtime.max_num_batched_tokens` 不小于 `runtime.max_num_seqs`；显存比例是
  `(0, 1]` 内的有限数，上下文要求不超过任务允许的模型长度。
- 部署流程从 `model.config_path` 指向的官方 `config.json` 读取
  `num_attention_heads`；当 `runtime.tp` 已确定时，必须满足
  `num_attention_heads % runtime.tp == 0`，使 attention heads 能够被 TP 均匀切分。
  配置来源或该关系暂时无法确认时保留 `unknown`。
- 设备可见性是授权资源的子集，`tp * dp` 不超过本次 provider 交付的设备数量；
  分组存在时，分组数量、大小和重复分配关系一致。
- 端口属于有效端口范围且只能使用 provider 分配的值；端口可用性和 CLI 参数支持
  必须由真实环境事实确认，不能用缺失信息伪造 `pass`。
- 本地模型的配置、tokenizer、权重和权限问题由部署流程根据任务合同动态确认；
  远程来源不能静态确认时保留 `unknown`。

契约检查只回答输入是否满足部署前提，不替代静态容量计算、服务启动或健康验收。

对于 vLLM 或 vLLM Ascend 针对特定模型、硬件、后端版本或测试环境给出的推荐值，
部署流程记录 `warning`，默认不因此阻断部署，包括：

- Ascend 后端常见的 `block_size=128` 推荐值；
- 模型支持矩阵或模型指南推荐的 TP/DP 组合；
- 特定硬件环境下推荐的并发数、批处理 token 数、显存比例或上下文长度；
- 特定模型、量化格式、dtype 或并行方式的性能推荐。

推荐值必须结合其适用的模型、硬件和软件版本解释；只有任务合同或 provider 将某项
推荐值明确冻结为强制约束时，才按违反合同处理。

## 缺省与状态边界

- `model_path` 或 `served_model_name` 缺失：输入不完整，不能进入部署。
- 身份存在但 TP/DP、批处理、上下文或 provider 资源等启动字段缺失：进入
  `recommendation_only`；可以在静态输入足够时输出最小 TP 建议，但不启动服务。
- 字段类型、数值关系、授权资源或已知 CLI 事实违反合同：`input_invalid`，不启动服务。
- 外部来源、端口占用或环境事实尚未能确认：`unknown`，必须完成动态确认后才能继续
  相应阶段，不能直接写成 `service_ready`。
- 静态预检明确容量不足：`reject`，不启动服务。静态预检未知不等于通过，真实启动
  后仍要记录动态结果。
- 服务启动、健康、模型身份和最小请求分别使用部署报告中的独立状态，不用一个
  `Valid` 字段覆盖前述结论。

## 派生 `static-oom-input.json`

部署流程在 attempt 目录中按以下关系生成内部 artifact：

1. 读取当前 `deployment-input.json` 和任务/provider 提供的官方 `config_path`。
2. 将官方配置解析后的对象写入 `model.config`，保留标准 `text_config` 结构。
3. 仅在任务或 provider 提供了明确、可信的权重大小事实时写入
   `model.weight_bytes` 或 `model.weight_gib`；不能在静态脚本中扫描权重目录或按文件
   大小推断。
4. 将 provider 提供的单卡容量、设备数量以及当前 candidate 的静态运行参数映射到
   `hardware` 和 `runtime`。
5. 将派生结果写为 `static-oom-input.json`，并把路径记录到部署报告。

该派生输入只保留静态预检所需字段，例如模型配置、权重大小、单卡容量、设备数量、
TP/DP、并发、显存比例、最低上下文、block size、允许的 TP 和官方最小 TP。它不包含
端口、tokenizer、CLI 帮助输出或设备选择逻辑。
