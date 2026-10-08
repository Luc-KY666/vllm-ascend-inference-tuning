使用 $vllm_ascend_benchmark_skill 进行晟腾上的模型压测，模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。设置模型参数为 `TP=2, DP=1, max_num_seqs=64, max_num_batched_tokens=4096`，使用长 Decode 负载进行压测，本地数据集在目录 `/data/disk/liangkaiyi/hf_cache/` 下寻找，其余配置保持为默认值。

使用 $vllm_ascend_benchmark_skill 进行晟腾上的模型压测，模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。设置模型参数为 `TP=2, DP=2, max_num_seqs=64, max_num_batched_tokens=4096`，使用长 Prefill 负载进行压测，本地数据集在目录 `/data/disk/liangkaiyi/hf_cache/` 下寻找，其余配置保持为默认值。

使用 $vllm_ascend_benchmark_skill 进行晟腾上的模型压测，模型路径为 `/data/disk/models/Qwen3-Coder-30B-A3B-Instruct`，模型名称为 `qwen3-coder-30b`。设置模型参数为 `TP=4, DP=1, max_num_seqs=32, max_num_batched_tokens=2048`，使用长 Decode 负载进行压测，本地数据集在目录 `/data/disk/liangkaiyi/hf_cache/` 下寻找，其余配置保持为默认值。