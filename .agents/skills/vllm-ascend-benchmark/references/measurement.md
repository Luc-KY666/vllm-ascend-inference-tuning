# 官方测量方法

测量合同规定当前环境安装的官方 `vllm bench serve` 的使用方式。TTFT、TPOT、E2EL 的
P50/P95 以及 goodput 必须由 vLLM benchmark 计算并写入其 raw result；执行流程负责命令
构造、执行、核验和归档。

## 1. 服务与 CLI 能力

在允许当前 provider 管理服务的路径，先调用部署 skill 完成服务启动、`/health`、
`/v1/models`、最小请求和 served model identity 验收。预启动服务路径的操作范围为读取
已交付的 base URL。

每次 attempt 使用目标环境的：

```text
vllm bench serve --help=all
```

确认当前版本支持任务冻结的所有参数。至少核对：

```text
--backend
--model
--tokenizer
--base-url
--endpoint
--dataset-name
--dataset-path
--hf-name / --hf-split / --hf-subset / --hf-output-len
--hf-text-column（如当前版本支持）
--sharegpt-output-len
--custom-output-len
--custom-skip-chat-template 或 --skip-chat-template
--random-input-len / --random-output-len / --random-range-ratio
--num-prompts
--max-concurrency
--num-warmups
--request-rate
--burstiness
--percentile-metrics
--metric-percentiles
--save-result / --result-dir 或等价结果输出选项
```

启用 goodput 时，额外确认当前版本支持 `--goodput`，并确认输出包含
`request_goodput` 或版本文档明确规定的等价字段。CLI 能力不完整时，正式测量状态为
`blocked`。

## 2. 测量入口门禁

执行 `vllm bench serve` 前，必须使用目标环境的完整帮助和实际参数解析能力确认：

- backend、model、tokenizer、base URL 和 endpoint；
- 由[负载契约](workload.md)派生的数据集模式、路径、split、长度和采样参数；
- 请求数、并发、warmup、到达率和 burstiness；
- `--percentile-metrics ttft,tpot,e2el`；
- `--metric-percentiles 50,95`；
- 任务需要 goodput 时的 `--goodput`；
- raw JSON 保存路径或结果目录。

测量入口门禁包括固定 workload 重放、P50/P95 输出、启用的 `request_goodput` 输出和
raw result 唯一性。任一门禁不满足时，本次 attempt 状态为 `blocked`；测量后端保持为
官方 `vllm bench serve`，结果有效范围限定为当前 workload 合同。

## 3. 命令构造

由任务合同生成一条唯一命令。benchmark backend 使用已交付的 HTTP 服务；`--backend vllm`
由任务合同按目标环境能力确定：

```text
vllm bench serve \
  --backend <task-defined-http-backend> \
  --model <served-model-name> \
  --tokenizer <task-defined-tokenizer> \
  --base-url <task-defined-base-url> \
  --endpoint <task-defined-endpoint> \
  --num-prompts <task-defined-number> \
  --max-concurrency <task-defined-concurrency> \
  --num-warmups <task-defined-warmups> \
  --request-rate <task-defined-rate> \
  --burstiness <task-defined-burstiness> \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,95 \
  --save-result \
  --result-dir <attempt-dir>
```

数据集参数根据[负载契约](workload.md)的 `selection` 和 `dataset.mode` 追加。
公共命令不包含数据集参数；适配器命令只能使用与当前数据集模式对应的参数分支：

```text
# selection=preset, scenario=random
--dataset-name random
--random-input-len <frozen-input-length>
--random-output-len <frozen-output-length>
--random-range-ratio <frozen-range-ratio>

# selection=preset, scenario=long_prefill|long_decode
--dataset-name hf
--dataset-path <resolved-dataset-path>
--hf-name <derived-hf-name>
--hf-split <derived-split>
[--hf-output-len <derived-output-length>]

# selection=preset, scenario=prefill_decode_balance
--dataset-name sharegpt
--dataset-path <resolved-dataset-path>
[--sharegpt-output-len <frozen-output-length>]

# selection=hf
--dataset-name hf
--dataset-path <contract-dataset-path>
--hf-name <contract-hf-name>
--hf-split <contract-split>
[--hf-subset <contract-subset>]
[--hf-output-len <contract-output-length>]
[--hf-text-column <contract-text-column>]

# selection=custom
--dataset-name custom
--dataset-path <contract-custom-jsonl>
[--custom-output-len <contract-output-length>]
[--custom-skip-chat-template | --skip-chat-template]
```

`selection=custom` 时，命令不包含 `--hf-name`、`--hf-split`、`--hf-subset`、
`--hf-output-len` 或 `--sharegpt-output-len`。`skip_chat_template` 的命令参数名由
当前 `vllm bench serve --help=all` 决定，两个别名不能同时传递。CLI 不支持合同冻结的
适配器参数时，attempt 状态为 `blocked`。

在命令执行前，数据集准备阶段必须已经完成并记录：

- `selection`、`scenario` 和 `dataset.mode`；
- 解析后的 dataset path、dataset name、split、revision/hash 和有效样本数；
- 数据集是复用、下载、缓存解析还是本地 custom 文件；
- 适配器专用参数和格式校验结果。

Hugging Face 数据集的下载缓存位于合同指定的 `dataset_root` 或 provider 冻结的缓存
根目录下；命令使用解析后的本地路径或合同指定的远程数据集路径。Custom 命令使用
本地 JSONL 路径，不执行远程下载。数据集准备未完成时不执行 benchmark 命令。

任务提供 goodput SLO 时，按 vLLM 当前 CLI 语法传递已冻结的约束：

```text
--goodput ttft:<milliseconds> tpot:<milliseconds> e2el:<milliseconds>
```

goodput 命令只包含任务实际提供的约束；未提供的约束保持缺省缺失状态。自然语言中的
`TPOP` 统一按官方字段 `tpot` 处理。未提供 goodput SLO 时，命令不包含
`--goodput`，最终报告状态为 `not_requested`。

## 4. 执行与结果选择

1. 创建 provider 允许的独立 attempt 目录，记录服务接口、配置摘要、workload 摘要和
   运行前该目录内已有的 JSON 文件集合。
2. 记录完整 benchmark command、vLLM/vLLM-Ascend 版本、CLI 能力检查摘要和开始时间。
3. 执行一次 `vllm bench serve`。诊断复测由任务合同单独授权，并使用新的 attempt。
4. 从本次命令开始后新建或更新的结果中筛选可读、非空、可解析的 JSON；最终报告、临时
   文件和日志不属于 raw result 候选。
5. raw result 候选数为一，并且模型身份、请求数、完成数、失败数、数据集摘要、到达率、
   并发和结果时间等可核对字段与本次冻结输入一致。
6. 使用官方 raw JSON 中的 P50/P95 字段填充报告。指标来源限定为官方 raw JSON；日志、
   SSE、单调时间戳、请求事件数、吞吐和其他 percentile 不参与缺失指标推导。
7. 启用 goodput 时，raw JSON 提供合法的 `request_goodput`；缺失、为 `null` 或无法证明
   来自本次命令时，报告 `goodput.status=missing` 并将 attempt 判为失败。
8. 未启用 goodput 时，报告 `goodput.status=not_requested`，`request_goodput` 保持
   `null`；请求吞吐和 token 吞吐不映射为 goodput。

## 5. 样本与失败规则

- warmup 与正式请求分离，正式统计只覆盖合同定义的正式请求。
- 失败、超时、服务断流和请求数不足保留在 raw result 和最终报告中。
- 每次测量使用一次合同定义的请求集合；失败样本保留，缺失 percentile 保持缺失。
- 固定多轮任务使用官方工具生成合同要求的结果，百分位口径沿用官方输出。
- OOM 只在 `vllm bench serve` 已进入正式请求执行后进行分类。可识别的证据包括
  `out of memory`、`OutOfMemoryError`、`MemoryError`、`Cannot allocate memory`、`ACLRT`
  内存不足及目标运行时明确表示显存不足的等价信息；只记录当前 benchmark attempt
  产生的服务日志、benchmark 日志或进程 artifact 路径和证据摘要。
- 正式请求期间发生 OOM 时，报告 `failure.stage=benchmark`、
  `failure.type=runtime_oom`、`oom.phase=benchmark` 和 `oom.type=runtime_oom`。该分类
  优先于通用的 `benchmark_incomplete` 或 `service_error`。
- 静态预检、服务启动和最小请求验收构成部署阶段。正式 benchmark 的前置条件为部署阶段
  完成；部署阶段 OOM 的分类和证据写入 `deployment_report.json`。
- OOM attempt 终止后不得通过修改 TP、DP、并发、上下文或 workload 自动重试；如任务合同
  授权诊断复测，必须创建新的 attempt。
- OOM 失败时，TTFT、TPOT、E2EL 和吞吐字段全部为 `null`；启用 goodput 但没有可验证的
  官方结果时使用 `goodput.status=missing`。请求计数、OOM 证据和 raw result（若存在）
  仍然保留，清理结果单独写入 `cleanup.status`。
- 未发现可验证 OOM 证据时不得标记为 `runtime_oom`；按实际阶段报告通用服务、环境或
  benchmark 错误。

## 6. 指标门禁

最终报告只接受官方工具本次输出的：

- TTFT：P50、P95；
- TPOT：P50、P95；
- E2EL：P50、P95；
- goodput：任务启用 SLO 时的 `request_goodput`，单位 `req/s`。

必需指标缺失、数值非法、请求未完成或 raw result 归属不唯一时，`status` 映射为非
`success` 状态；缺失或不可信的性能字段写 `null`。请求计数沿用官方结果中的完整
统计，不移除失败样本，也不以 `0` 补齐缺失值。

## 7. 生命周期与清理

- git_worktree/direct provider 可在同一持锁 verifier 中依次完成部署、benchmark 和清理；
- ThinkThread 或预启动 provider 只能访问已交付服务接口；
- benchmark 结束后按部署/provider 合同清理或交付服务；
- 清理失败保留 raw result 和日志，并在报告中写明 `cleanup.status`；性能状态与清理
  状态分别记录。

profile、编译、服务启动和 warmup 归入独立诊断或准备阶段，不计入官方正式 benchmark
指标。
