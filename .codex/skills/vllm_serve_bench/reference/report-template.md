# 压测报告模板

> 报告写入 `./running-time/<served_model_name>/bench/`，文件名建议 `<场景>_report.md`。

## 单场景性能报告

```markdown
## vLLM-Ascend 在线服务性能报告 — {场景}

### 压测配置
| 项目 | 值 |
|------|-----|
| 服务地址 | {base_url} |
| 模型 | {model} |
| served_model_name | {served_model_name} |
| 负载场景 | {long_prefill / long_decode / balanced} |
| 数据集 | {hf:likaixin/InstructCoder / hf:AI-MO/NuminaMath-CoT / sharegpt} |
| 输入/输出长度 | {input_len} / {output_len} tokens（真实数据集填"真实分布"） |
| 请求数 | {num_prompts} |
| 并发度 | {max_concurrency} |
| 请求速率 | {request_rate} |
| 温度 | {temperature} |

### 执行命令
{full_command}

### 请求统计
| 指标 | 值 |
|------|-----|
| 成功请求数 | {successful_requests} |
| 失败请求数 | {failed_requests} |
| 压测总耗时 | {duration} s |
| 峰值并发 | {peak_concurrent_requests} |

### 吞吐量
| 指标 | 值 |
|------|-----|
| 请求吞吐量 | {request_throughput} req/s |
| 输出 Token 吞吐量 | {output_throughput} tok/s |
| 总 Token 吞吐量 | {total_token_throughput} tok/s |
| 峰值输出吞吐 | {peak_output_throughput} tok/s |

### 延迟
| 指标 | 平均 | 中位数 | P99 |
|------|------|--------|-----|
| TTFT | {mean_ttft_ms} ms | {median_ttft_ms} ms | {p99_ttft_ms} ms |
| TPOT | {mean_tpot_ms} ms | {median_tpot_ms} ms | {p99_tpot_ms} ms |
| ITL | {mean_itl_ms} ms | {median_itl_ms} ms | {p99_itl_ms} ms |

### 异常说明
{失败请求原因，无则填"无"}

---
*压测时间: {timestamp}*
*结果文件: {result_file}*
```

---

## 多组并发对比报告

```markdown
## 并发对比性能报告 — {场景}

### 压测配置
| 项目 | 值 |
|------|-----|
| 服务地址 | {base_url} |
| 模型 | {model} |
| 负载场景 | {scenario} |
| 数据集 | {dataset} |
| 输入/输出长度 | {input_len} / {output_len} |
| 请求数 | {num_prompts}（各组一致） |

### 对比结果
| 并发度 | 请求吞吐 | 输出吞吐 | 平均 TTFT | P99 TTFT | 平均 TPOT | 平均 ITL | 失败数 |
|--------|----------|----------|-----------|----------|-----------|----------|--------|
| {c_1} | {req_tp_1} | {out_tp_1} | {ttft_1} | {p99_ttft_1} | {tpot_1} | {itl_1} | {fail_1} |
| {c_2} | {req_tp_2} | {out_tp_2} | {ttft_2} | {p99_ttft_2} | {tpot_2} | {itl_2} | {fail_2} |
| {c_3} | {req_tp_3} | {out_tp_3} | {ttft_3} | {p99_ttft_3} | {tpot_3} | {itl_3} | {fail_3} |

---
*压测时间: {timestamp}*
```

---

## 三场景综合报告

```markdown
## vLLM-Ascend 服务综合性能报告

### 服务信息
| 项目 | 值 |
|------|-----|
| 服务地址 | {base_url} |
| 模型 | {model} |
| 部署 TP | {tp} |
| max_model_len | {max_model_len} |

### 三场景指标汇总
| 指标 | 长 prefill | 长 decode | 均衡负载 |
|------|------------|-----------|----------|
| 数据集 | InstructCoder | NuminaMath-CoT | sharegpt |
| 输入/输出长度 | 8192/128 | 真实解答 | 真实分布 |
| 请求数 | {n1} | {n2} | {n3} |
| 请求吞吐 (req/s) | {r1} | {r2} | {r3} |
| 输出吞吐 (tok/s) | {o1} | {o2} | {o3} |
| 平均 TTFT (ms) | {t1} | {t2} | {t3} |
| P99 TTFT (ms) | {p1} | {p2} | {p3} |
| 平均 TPOT (ms) | {tp1} | {tp2} | {tp3} |
| 平均 ITL (ms) | {i1} | {i2} | {i3} |
| 失败请求数 | {f1} | {f2} | {f3} |

---
*压测时间: {timestamp}*
```
