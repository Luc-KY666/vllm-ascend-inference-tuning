1. Qwen3-Coder-30B-A3B-Instruct 使用最小 TP 进行推荐部署
- prompt:
    ```
    使用 [$vllm-ascend-deployment](.agents/skills/vllm-ascend-deployment/SKILL.md) 进行模型服务拉起。
    模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。
    可用 NPU 为 0, 1, 2 , 3，port 使用 8117，如果已被占用则不进行模型服务拉起。
    在 qwen3-coder-30b-deployment/ 目录下建立一个唯一标识目录作为本次部署的输出目录。
    模型部署成功后终止拉起的服务，不要影响任何其他不是由你拉起的服务。
    记录本次模型部署每个阶段的总耗时，包含设备侧执行耗时和 agent 侧的思考、文件读写、脚本编写耗时。
    ```
- 实际部署参数 `TP=2, DP=1, max_num_seqs=32, max_num_batched_token=4096`
- Pi + Qwen3.7-Plus high 模型部署耗时：
    | **阶段** | **耗时** | **说明** |
    | :---: | :---: | :---: |
    | 初始化 + 环境探测 | 110.82 s | Agent 思考/文件读写 + 环境 probe 脚本执行 |
    | 输入校验 + 静态 OOM + 最小 TP 推荐 | 55.5 s | Agent 思考/脚本编写 + input_contract.py + static_oom_precheck.py + 最小 TP 推荐 |
    | 资源检查 | 12.52 s | Agent 思考 + npu-smi info + netstat 端口检查 |
    | 服务启动 + 验收 | 224.32 s | 设备侧：模型加载+图捕获+启动；Agent 侧：轮询等待+curl 健康检查 |
    | 清理 | 29.69 s | Agent 思考 + SIGTERM 发送 + 进程/端口释放验证 |
    | **总计** | 432.87 s | |

- 蓝区 GLM 和 DeepSeek 都会出现思维链混乱，不按照 SKILL 给定步骤执行的问题，执行时间比 Qwen3.7-Plus 长

2. Qwen3-Coder-30B-A3B-Instruct 模型压测
- prompt:
    ```
    使用 [$vllm-ascend-benchmark](.agents/skills/vllm-ascend-benchmark/SKILL.md) 进行模型压测。
    模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。
    模型参数配置 `TP=2, DP=1, max_num_seqs=32, max_num_batched_tokens=4096`。
    可用 NPU 为 0，1，2，3，port 使用 8117，如果已被占用则不进行模型服务拉起。
    在 qwen3-coder-30b-benchmark/ 目录下建立一个唯一标识目录作为本次部署的输出目录。
    使用长 Prefill 负载，共发送 500 条请求，最大并发 64，请求速率 4 req/s。
    记录本次模型部署压测每个阶段的总耗时，每个阶段的总耗时为其在设备侧执行耗时加上 agent 侧的思考、文件读写、脚本编写耗时。
    ```
- Pi + Qwen3.7-Plus high 模型压测耗时：
    | **阶段** | **耗时** |
    | :---: | :---: |
    | 环境探测 | 65 s |
    | 服务部署+健康验收 | 403 s(包含首次卡被占用) |
    | 数据集校验 | 62 s |
    | 正式压测(warmup+500请求) | 225 s |
    | 清理+报告生成 | 32 s |
    | **总计** | 787 s | |

3. Qwen3-Coder-30B-A3B-Instruct 推理部署调优
- 目前指定了 `TP=2, DP=1`，调节 `max_num_seqs, max_num_batched_tokens`，使用预设的 long prefill 负载
- 使用方法：
    - 将 `.agents` 目录拷贝到 `goal-plus-playground/scenarios/boom-new/` 下
    - 将 `tasks/qwen3-coder-30b-long-prefill-search` 拷贝到 `goal-plus-playground/tasks/boom/` 下，随后修改其中的 `input.md`
    - 按照 playground 的流程创建 workspace，然后在 workspace 中启动 Pi 并复制 `goal-pi.md`
- 目前使用 Pi + DeepSeek-V4-Pro 能够完成串行推理部署调优的流程，执行报告见 [search_report](./SEARCH_REPORT.md)。本次执行由于未固定 `vllm bench serve` 的 `--seed` 参数，导致针对同一个参数配置出现了 `P95 TTFT` 测量的浮动，现已在 SKILL 中修复。现在执行中遇到的大部分问题都是 agent 与 goal-plus 交互时的小问题，均可由 agent 自行修复，可能是上下文遗忘导致的。