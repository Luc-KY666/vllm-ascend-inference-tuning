使用 $vllm_ascend_benchmark_skill 进行晟腾上的模型压测，模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。设置模型参数为 `TP=2, DP=1, max_num_seqs=64, max_num_batched_tokens=4096`，使用长 Decode 负载进行压测，本地数据集在目录 `/data/disk/liangkaiyi/hf_cache/` 下寻找，其余配置保持为默认值。

使用 $vllm_ascend_benchmark_skill 进行晟腾上的模型压测，模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。设置模型参数为 `TP=2, DP=2, max_num_seqs=64, max_num_batched_tokens=4096`，使用长 Prefill 负载进行压测，本地数据集在目录 `/data/disk/liangkaiyi/hf_cache/` 下寻找，其余配置保持为默认值。

使用 $vllm_ascend_benchmark_skill 进行晟腾上的模型压测，模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。设置模型参数为 `TP=4, DP=1, max_num_seqs=32, max_num_batched_tokens=2048`，使用长 Decode 负载进行压测，本地数据集在目录 `/data/disk/liangkaiyi/hf_cache/` 下寻找，其余配置保持为默认值。

/goal 
任务目标：为 qwen3-coder-30b 模型寻找最优部署配置，使长 Prefill 负载下的 P99 E2EL 延迟最优，至多可以使用 4 张 NPU 卡；
可调节模型参数：`TP, DP, max_num_seqs, max_num_batched_tokens`；
评估方法：调用 $vllm_ascend_benchmark_skill 进行 ascend 环境下的模型压测，得到参数是否合法(输出 json 文件中的 Valid 字段)以及对应配置下的 P99 E2EL 延迟(输出 json 文件中的 performence 字段)；
基线：使用 `TP=2, DP=1, max_num_seqs=64, max_num_batched_tokens=4096` 作为性能基线。
其他：模型路径为`/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`，本地数据集在目录 `/data/disk/liangkaiyi/hf_cache/` 下寻找，其余配置保持为默认值。