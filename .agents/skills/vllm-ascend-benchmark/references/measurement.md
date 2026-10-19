# 官方测量方法

正式指标必须来自当前环境安装的官方 `vllm bench serve`。本文件规定 CLI 门禁、命令
构造、raw result 归属、指标核验、OOM 分类和清理；不实现自定义压测器。

## CLI 门禁

每次 attempt 仅在环境预检阶段使用目标环境执行一次（测量阶段复用该 artifact）：

```text
vllm bench serve --help=all
```

至少确认当前版本支持：

```text
--backend --model --tokenizer --base-url --endpoint
--dataset-name --dataset-path
--num-prompts --max-concurrency --num-warmups
--request-rate --burstiness
--temperature（Provider 显式提供或按默认值 0 传入时）
--percentile-metrics --metric-percentiles
--save-result --result-dir 或等价结果输出参数
```

还必须确认 workload 适配器要求的参数。Provider 提供 goodput SLO 时，必须确认
`--goodput` 和 raw JSON 中的 `request_goodput`（或当前版本明确等价字段）。能力缺失时
使用 `status=blocked`，不得用替代 benchmark 或自定义指标计算。

## 命令构造

由已解析的服务和 workload 输入构造一条唯一命令。服务已经由路径 A 的部署 skill 或
路径 B 的 Provider 交付；benchmark 不重新推导模型、端口或设备：

```text
vllm bench serve \
  --backend <provider-defined-http-backend> \
  --model <served-model-name> \
  --tokenizer <resolved-tokenizer> \
  --base-url <base-url> \
  --endpoint <endpoint> \
[--temperature <provider-temperature>] \
  --num-prompts <num-prompts> \
  --max-concurrency <max-concurrency> \
  --num-warmups <num-warmups> \
  --request-rate <request-rate> \
  --burstiness <burstiness> \
  --percentile-metrics ttft,tpot,e2el \
  --metric-percentiles 50,95 \
  --save-result \
  --result-dir <attempt-dir>
```

`--base-url` 必须是 scheme、host 和 port；`--endpoint` 必须是以 `/` 开头且与 backend 匹配的
HTTP 路径（completions 以 `completions` 结尾，chat 以 `chat/completions` 结尾），并由服务
验收阶段用最终拼接 URL 校验。temperature 按[负载契约](workload.md)解析：Provider 显式
冻结时按冻结值追加 `--temperature`；未显式冻结时按默认值追加 `--temperature 0`；仅显式
冻结为 `null` 时不追加该参数，且不要把 vLLM CLI 的未设置值推断为 `0`。实际传入值与
来源记录在 workload artifact 中。

追加 workload 参数必须严格来自 [workload.md](workload.md) 的同一分支。命令不得同时
传递两个 skip-chat-template 别名，不得把请求吞吐当作 goodput。

Provider 只提供部分 goodput 约束时，仅传递实际提供的字段，例如：

```text
--goodput ttft:<milliseconds> tpot:<milliseconds> e2el:<milliseconds>
```

未提供任何 goodput 约束时不传 `--goodput`，最终报告为
`goodput.status=not_requested`。

## 执行上下文、执行和 raw result

1. 只在 [ascend-probes.md](ascend-probes.md) 确定的执行上下文中运行命令。路径 A 复用
   部署报告的提权上下文和已验收服务；路径 B 复用本次 benchmark 的提权上下文。不得在
   测量阶段重新探测沙箱、NPU 或 Provider 端口。
2. 在执行前建立独立 attempt 目录，并记录目录中已有的 JSON 文件集合。
3. 记录完整命令、vLLM/vLLM-Ascend 版本、CLI 能力摘要、服务身份和开始时间。
4. 正式执行一次 `vllm bench serve`。warmup 不计入正式请求统计；诊断复测必须使用新的
   attempt 目录。
5. 只从本次命令开始后新建或更新的 JSON 中选择 raw result；`benchmark_result.json`、
   临时文件和日志不是 raw result 候选。
6. raw result 候选必须唯一、非空、可解析，并且模型身份、请求数、完成/失败数、数据集
   摘要、并发和到达率与冻结输入一致。零个候选报告 `raw_result_not_found`，多个候选
   报告 `raw_result_ambiguous`。
7. 保存 raw JSON 路径和 SHA256；不覆盖既有 attempt 的 artifact。

## 指标门禁

最终报告只接受 raw JSON 已计算的：

- TTFT P50/P95；
- TPOT P50/P95；
- E2EL P50/P95；
- Provider 请求 goodput SLO 时的 `request_goodput`，单位 `req/s`。

指标缺失、非有限、请求不完整、请求计数不一致或 raw result 无法归属当前命令时，报告
使用非 `success` 状态，性能字段为 `null`。不得从请求事件、日志、SSE、部分样本或自定义
公式补齐百分位。

goodput 状态规则：

- `reported`：至少一个 SLO 已传入，raw JSON 有合法 `request_goodput`；
- `not_requested`：未提供 SLO，命令未传 `--goodput`；
- `missing`：要求 goodput，但字段缺失、为 `null`、不可解析或无法证明属于本次命令。

## OOM 和其他失败

只在 `vllm bench serve` 已进入正式请求阶段后判定 runtime OOM。当前 attempt 的 benchmark
日志、服务日志或进程 artifact 中出现下列证据时可分类：`out of memory`、
`OutOfMemoryError`、`MemoryError`、`Cannot allocate memory`、`ACLRT` 显存不足或运行时明确
表示显存不足的等价信息。

正式请求期间 OOM 必须报告：

```text
status=error
failure.stage=benchmark
failure.type=runtime_oom
oom.detected=true
oom.phase=benchmark
oom.type=runtime_oom
```

OOM 证据至少包含当前 attempt artifact 路径、匹配摘要和时间信息。未找到可验证证据时不
得标记 `runtime_oom`。OOM attempt 的 TTFT、TPOT、E2EL、吞吐和 goodput 全部保持 `null`；
不得通过修改 TP、DP、并发、上下文或 workload 自动重试。

服务健康、模型身份和最小请求阶段的 OOM 归入部署报告；环境、CLI、workload、服务断流、
请求不完整和清理问题按实际阶段记录，并保留已有 raw artifact。

## 生命周期和清理

- 路径 A 的服务由部署/provider 合同决定清理时机；当前 skill 至少记录清理结果。
- 路径 B 的服务不由当前 skill 停止或修改。
- 清理失败写入 `cleanup.status=failed` 和错误 artifact，但不把已完成的性能结果改写为
  `success` 以外的状态，除非清理本身使报告无法交付。
- 无论命令成功或失败，都必须尝试写出可解析的 `benchmark_result.json`。
