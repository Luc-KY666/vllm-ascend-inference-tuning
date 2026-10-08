# vLLM-Ascend Inference Tuning

用于在单机昇腾 NPU 上完成 vLLM-Ascend 推理服务部署、验收和在线服务压测。

## Skills

- `.codex/skills/minimal-tp-vllm-ascend-deploy/`
  - 根据模型、显存和上下文要求计算最小 TP
  - 启动 vLLM-Ascend OpenAI 兼容服务
  - 完成进程、启动日志、模型 API 和对话 API 验收
- `.codex/skills/vllm_ascend_benchmark_skill/`
  - 按显式的模型、TP、DP 和批处理参数执行昇腾模型压测
  - 已融合原 `vllm_serve_bench` 的在线服务压测能力
  - 支持 `long_prefill`、`long_decode`、`prefill_decode_balance` 和 `random` 场景
  - 内置服务启动、健康检查、模型身份校验、静态 OOM 预检和数据集路径校验
  - 以单个 `benchmark_result.json` 作为最终结果契约

## Recommended workflow

1. 使用 `minimal-tp-vllm-ascend-deploy` 部署并验收服务。
2. 使用 `vllm_ascend_benchmark_skill` 执行统一压测；它覆盖长 prefill、长 decode、prefill/decode 均衡和 random 场景，并生成标准结果 JSON。

当前保留的 Skill 都有独立配置和参考文档，可从项目根目录启动 Codex 后使用。
