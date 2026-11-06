/goal 针对模型 `qwen3.8-27b`、`prefill_decode_balance` 负载场景和 `production` 模式，寻找一组模型参数配置，使得 SLO TTFT<500ms, TPOT<25ms 下的 goodput 最大。gootput 指标需要通过模型部署压测得到，参考 $vllm-ascend-deployment 和 $vllm-ascend-benchmark 两个 SKILL。其中通用环境探测 $vllm-ascend-probes 全局只做一次，并保留结果进行复用。可用输出路径为 qwen3.8-27b-search/，请先在该目录下建立一个 search_path.md，并在每一次搜索结束后更新搜索内容，包括当前搜索配置、选择该配置的原因和搜索结果，搜索失败的案例不计入最大搜索次数的限制。

通用参数
| 模型路径 | `/data/disk/models/Qwen3.8-27B` |
| 最大搜索次数 | `30` |
| 可用 NPU 设备 | `["6", "7"]` |

负载参数
| `num_prompts` | `200` |
| `max_concurrency` | `10` |
| `request_rate` | `2` |

固定参数
| `trust_remote_code` | 启用，传递 `--trust-remote-code` |
| `additional_config` | `{"enable_cpu_binding": true}` |
| `gpu_memory_utilization` | `0.90` |
| `max_model_len` | `133000` |
| `async_scheduling` | 启用，传递 `--async-scheduling` |

可调节参数
| `tp` | `1, 2` |
| `dp` | `1, 2` |
| `max_num_seqs` | `16-256` |
| `max_num_batched_tokens` | `2048-32768` |
| `enable_prefix_caching` | `true|false` |
| `compilation_config` | `{"cudagraph_mode":"NONE"}`; `{"cudagraph_mode":"FULL_DECODE_ONLY"}`; `{"cudagraph_mode":"PIECEWISE"}`; `{"cudagraph_mode":"FULL_AND_PIECEWISE"}` |
其中 `enable_prefix_caching` 为 `false` 时传递 `--no-enable-prefix-caching`。

具体提升幅度参考 baseline
| CLI 参数 | baseline 值 |
| --- | --- |
| `--data-parallel-size` | `1` |
| `--tensor-parallel-size` | `2` |
| `--max-num-seqs` | `32` |
| `--max-model-len` | `133000` |
| `--max-num-batched-tokens` | `16384` |
| `--no-enable-prefix-caching` | 启用 |
| `--compilation-config` | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` |
| `--additional-config` | `{"enable_cpu_binding":true}` |
| `--gpu-memory-utilization` | `0.90` |
| `--trust-remote-code` | 启用 |
| `--async-scheduling` | 启用 |
