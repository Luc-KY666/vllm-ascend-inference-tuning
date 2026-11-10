# benchmark 场景下的负载画像采集

调优迭代每次 benchmark 复测时，随压测记录当次负载画像：输入/输出长度分布、Waiting/Running 队列深度、请求排队等待时间。画像只记录事实，作为当次压测的场景特征；`/metrics` 采集是只读旁路，不影响 benchmark 合同。

## 采集项与数据源

| 画像项 | 指标/数据源 | 类型与语义 | 采集方式 |
|---|---|---|---|
| 输入/输出长度 P50/P95/P99 | benchmark 保留的 `raw_benchmark_result.json` 中 `input_lens`、`output_lens` 逐请求数组 | 精确值 | 压测后直接读取 |
| Waiting/Running 队列深度 | `vllm:num_requests_waiting` / `vllm:num_requests_running` | Gauge 瞬时值：当前等待调度的请求数 / 当前在模型执行批中的请求数 | 窗口内轮询采样 |
| 排队等待时间 P50/P95/P99 | `vllm:request_queue_time_seconds` | Histogram：请求在 WAITING 阶段停留的秒数，仅请求完成时观测、进程内累计 | 前后快照差值 |

三者均为 vLLM v1 引擎标准指标（定义在 `vllm/v1/metrics/loggers.py`），vllm-ascend 经同一 `/metrics` 端点暴露；该端点由 `vllm serve` 默认挂在 API 端口（`vllm/entrypoints/serve/instrumentator/metrics.py`），无需额外启动参数。

## 采集操作

窗口纪律与 [benchmark-profiling.md](benchmark-profiling.md) 一致：warmup 在窗口外，从首个正式请求开始，全部正式请求完成后结束。

```bash
# 1) 基线快照：benchmark 命令启动前
curl -s <service_url>/metrics > <iter_dir>/metrics_start.txt

# 2) 队列轮询：压测进行期间后台采样
( while true; do
    echo "# $(date +%s.%N)"
    curl -s <service_url>/metrics | grep -E '^vllm:num_requests_(running|waiting)[{ ]'
    sleep 5
  done ) > <iter_dir>/queue_series.txt &
POLLER_PID=$!

# 3) 正式压测：benchmark 命令本身不变

# 4) benchmark 命令返回后：停止轮询，取结束快照
kill "$POLLER_PID"
curl -s <service_url>/metrics > <iter_dir>/metrics_end.txt
```

- 轮询间隔默认 5 秒，可按压测时长调整；
- 指标行带 `model_name` 等 label（较新版本 DP 多 engine 时另有 engine 标签），解析时按 label 匹配本次服务实例；多 engine 时分别统计并在画像中写明口径；
- 请求级直方图只含已完成请求，结束快照必须等全部请求完成；快照间隔内不得发送其他请求，否则计入差值；
- 上述命令把快照与轮询绑定在 benchmark 命令整体上，工具内部 warmup 请求计入差值与序列；要剥离 warmup 须改用与 profile 采集相同的首个正式请求 gate 对齐窗口，做不到时在画像中记录该事实。

## 计算口径

分位数统一线性插值（与 numpy.percentile 默认一致）。

- **长度分位数**：对 raw JSON 的 `input_lens`、`output_lens` 数组计算：

  ```python
  import json, statistics

  r = json.load(open("<attempt_dir>/raw_benchmark_result.json"))
  for key in ("input_lens", "output_lens"):
      q = statistics.quantiles(r[key], n=100, method="inclusive")
      print(key, "P50/P95/P99:", q[49], q[94], q[98])
  ```

  请求数少于 2 时 `statistics.quantiles` 无法计算，直接记录原始序列。长度数组覆盖全部请求（含失败请求，`output_lens` 对失败请求记 0），计算前核对 raw JSON 的 completed/failed 计数，失败请求的处理口径写入画像。`/metrics` 的 `vllm:request_prompt_tokens`、`vllm:request_generation_tokens` 直方图只用于交叉核对长度量级：桶是按 max_model_len 的 1-2-5 序列，粒度粗，不出分位数。
- **队列深度**：解析 `queue_series.txt` 得到每个采样点的 running/waiting 值，分别取 max、P50、P95。
- **等待时间分位数**：对 `vllm:request_queue_time_seconds_bucket{le="..."}` 取 end−start 的桶计数，分母用 `_count` 差值（纳入统计的完成请求数），在目标分位所在桶内按桶宽线性插值（与 Prometheus `histogram_quantile` 语义一致）。`_count` 差值与 benchmark 完成请求数不一致时在画像中标记。
- 直方图分位数精度受桶边界限制：当前 vLLM main 的等待时间桶从 0.3s 起（`vllm/v1/metrics/buckets.py`），低于首桶的等待全部落在首桶，只能报上界；实际桶边界以运行版本源码为准。

## 输出与失败处置

每次迭代在迭代目录产出 `workload_profile.json`：记录上文三项画像的结果值，另加采集事实——采样间隔与样本数、快照与轮询起止时间、warmup 是否计入、数据缺失项。采集失败或数据缺失时记录缺失项，不伪装成服务或 benchmark 失败，不阻断调优迭代。
