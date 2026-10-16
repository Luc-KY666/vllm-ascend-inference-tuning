1. Qwen3-Coder-30B-A3B-Instruct 使用最小 TP 进行推荐部署
- prompt:
    ```
    使用 [$vllm-ascend-deployment](/root/vllm-ascend-inference-tuning-repo/.agents/skills/vllm-ascend-deployment/SKILL.md) 进行模型服务拉起。
    模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。
    可用 NPU 为 4, 5, 6, 7，port 使用 8117，如果已被占用则不进行模型服务拉起。
    在 @qwen3-coder-30b-deployment/ 目录下建立一个唯一标识目录作为本次部署的输出目录。
    模型部署成功后终止拉起的服务，不要影响任何其他不是由你拉起的服务，并记录本次模型部署的总耗时。
    ```
- 实际部署参数 `TP=2, DP=1, max_num_seqs=32, max_num_batched_token=4096`
- Pi + GLM-5.3 high 模型部署耗时：
    | 任务                | 耗时  |
    | :---:              | :---: |
    | 预检+配置+TP推荐    | 250 s |
    | vllm serve 服务启动 | 189 s |
    | 验收检查            | 33s   |
    | 完整任务            | 7 min 52 s |

2. Qwen3-Coder-30B-A3B-Instruct 模型压测
- prompt:
    ```
    使用 [$vllm-ascend-benchmark](/root/vllm-ascend-inference-tuning-repo/.agents/skills/vllm-ascend-bechmark/SKILL.md) 进行模型压测。
    模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。
    模型参数配置 `TP=2, DP=1, max_num_seqs=32, max_num_batched_tokens=4096`。
    可用 NPU 为4, 5, 6, 7 四张卡，port 使用 8117，如果已被占用则不进行模型服务拉起。
    在 @qwen3-coder-30b-benchmark/ 目录下建立一个唯一标识目录作为本次部署的输出目录。
    使用长 Prefill 负载，共发送 500 条请求，最大并发 64，请求速率 4 req/s。
    数据集在 '/data/disk/liangkaiyi/hf_cache' 下寻找。
    请分别记录整个模型部署流程和模型压测流程的耗时。
    ```
- Pi + GLM-5.3 high 模型压测耗时：
    | 任务                | 耗时  |
    | :---:              | :---: |
    | 压测预检(CLI 门禁+数据集准备+负载合同冻结)    | 81 s  |
    | 压测命令准备                                | 18 s  |
    | vllm bencn serve 测量(500请求)              | 188 s |
    | raw result 检验+服务清理+生成最终报告        | 93 s  |
    | 完整任务                                   | 6 min 20 s |