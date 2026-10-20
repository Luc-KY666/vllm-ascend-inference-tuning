# vLLM-Ascend Inference Tuning

本仓库维护可复用的 vLLM-Ascend Codex skills，用于环境探测、服务部署和独立性能压测。
部署、压测和环境探测分别遵循独立的 v2 合同；本仓库不提供统一的顶层调度 skill。

## Skills

- [`vllm-ascend-probes`](.agents/skills/vllm-ascend-probes/)
  - 在固定执行上下文中探测 CANN、Ascend runtime、Python 依赖、vLLM CLI、HTTP 和端口工具
  - 生成唯一的 `environment_probe.json`，供部署和 benchmark 复用
- [`vllm-ascend-deployment`](.agents/skills/vllm-ascend-deployment/)
  - 校验 Provider 输入和资源合同
  - 执行静态 OOM 检查、服务启动、健康检查、模型身份验收和最小请求验收
  - 交付模板驱动的 `deployment_report.json`
- [`vllm-ascend-benchmark`](.agents/skills/vllm-ascend-benchmark/)
  - 在 Provider 冻结的服务和 workload 合同上执行官方 `vllm bench serve`
  - 支持 Path A 的本次部署服务和 Path B 的 Provider 预部署服务
  - 交付单个 `benchmark_report.json`，并保留官方 `raw_benchmark_result.json` 作为测量证据

## Recommended workflow

1. 使用 `vllm-ascend-probes` 完成一次环境预检，并复用唯一 probe artifact。
2. Path A 使用 `vllm-ascend-deployment` 部署并验收服务；Path B 直接使用 Provider 交付的服务。
3. 使用 `vllm-ascend-benchmark` 执行独立压测，读取 deployment v2 service-ready checkpoint，
   并将报告和 raw result 保存在独立 attempt 目录。

每个 skill 都有独立的 `SKILL.md`、参考合同和测试，可从项目根目录启动 Codex 后使用。
