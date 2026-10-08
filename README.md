# vLLM-Ascend Inference Tuning

用于在单机昇腾 NPU 上完成 vLLM-Ascend 推理服务部署、验收和在线服务压测。

## Skills

- `.codex/skills/minimal-tp-vllm-ascend-deploy/`
  - 根据模型、显存和上下文要求计算最小 TP
  - 启动 vLLM-Ascend OpenAI 兼容服务
  - 完成进程、启动日志、模型 API 和对话 API 验收
- `.codex/skills/vllm_serve_bench/`
  - 对已运行的 vLLM 服务执行 `vllm bench serve`
  - 支持长 prefill、长 decode 和均衡负载场景
  - 保存压测结果并生成报告

- `.codex/skills/vllm_ascend_benchmark_skill/`
  - 按显式的模型、TP、DP 和批处理参数执行昇腾模型压测
  - 支持 `random`、Hugging Face 和本地 custom 数据集
  - 以单个 `benchmark_result.json` 作为最终结果契约

## Recommended workflow

1. 使用 `minimal-tp-vllm-ascend-deploy` 部署并验收服务。
2. 需要三类在线服务场景或真实数据集对比时，使用 `vllm_serve_bench`。
3. 需要固定输入契约、验证 TP/DP 配置并生成标准结果 JSON 时，使用 `vllm_ascend_benchmark_skill`。

三个 Skill 都保留了各自的配置、脚本和参考文档，可从项目根目录启动 Codex 后使用。
