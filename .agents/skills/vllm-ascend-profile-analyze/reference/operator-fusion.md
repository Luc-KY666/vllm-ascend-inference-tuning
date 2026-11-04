# Profile 驱动的小算子融合

本 reference 集中算子融合分析所需的规则，来源于 vLLM-Ascend 融合流程并适配为当前仓库的独立文档。
目标是把 profile 中有证据的重复小算子，与 vLLM/vllm-ascend 源码中的真实连续计算块对应起来，设计可验证的融合接口；
不是凭算子名称批量重写源码，也不依赖运行时读取其他仓库的 skill。

## Profile 类型要求

- **单请求 profile 是候选定位的首选输入**：只保留一个正式请求，停止并发请求，warmup 放在采集窗口外；窗口应覆盖请求处理到完成，或明确标注只覆盖 prefill/decode 等阶段。它用于确认算子顺序、shape/dtype、临时张量和模块边界。
- **benchmark profile 只作为聚合证据**：可用于发现高频小算子、统计总耗时和观察并发场景影响，但多个请求的算子可能交错，不能用来确认某个请求的完整连续算子序列。
- 只有 benchmark profile 时，可以输出热点候选和待补证据；没有单请求调用链证据时，不提交确定的融合替换方案。
- 融合候选确定后，仍必须回到固定 benchmark 做吞吐、延迟、质量、显存和多 rank 复测；单请求 profile 不能替代收益验证。

## 候选筛选

优先检查以下 profile 信号：

- `op_statistic.csv` 中高 Count、低 Avg 的 `AI_CORE`/`AI_VECTOR_CORE` 小算子；
- `AI_CPU` fallback、host `launch` 或同步 API 的累计开销；
- advisor 的 Fusible Operator/dispatch 结果；
- 相邻且无数据依赖断点的 `Add/Mul/Sub/Div`、Norm、RoPE、激活、量化、MatMul 后处理和 MoE 路由序列。

如果 `Free%` 主要来自请求间隔，或 profile 只显示大 kernel/通信瓶颈，先不要做小算子融合。融合收益估算应使用 `Count × Avg`、dispatch 次数、临时张量读写和 profile 的 host/device 时间，不把总算子时间直接等同于可节省时间。

## Profile、源码和模型配置联合分析

1. 从 profile 读取候选算子的 Count、Avg、Total/Ratio、Core Type、host dispatch 和临时张量迹象；只有 device 忙时段受其影响时才进入源码追踪。
2. 确认 vLLM 和 vllm-ascend 源码根目录，以及模型 `config.json`；读取模型名、hidden size、head 数、head_dim、rotary_dim、专家数、top-k、dtype、TP/DP/EP 等 shape/并行属性。
3. 在 `vllm/model_executor/models` 中定位模型实现，必要时跟随 import 到多个文件；重点阅读 decoder layer 的 `forward`，记录从输入到输出的真实调用顺序、条件分支和模块边界。
4. 只有当被融合算子在同一可控模块、形状和 dtype 约束明确、输出语义可保持时才建立候选。禁止把一个模块的一部分偷偷移入另一个融合模块，禁止把尚未存在的 NPU 接口写成已实现接口。
5. 常见候选模式包括：
   - `RMSNorm/LayerNorm + RoPE`、Norm + Quant；
   - `MatMul/Linear + Bias/Activation`、Transpose + MatMul、Add + MatMul；
   - `SiLU/GELU + Mul`、连续 vector element-wise；
   - QKV split + Norm + RoPE、mask + softmax；
   - Gate + TopK、router 下沉、shared/routed expert 合并。

## 替换报告格式

每个确定候选都必须输出：

1. 原始代码路径、逻辑位置/行号和真实代码片段；
2. 替换后的 Python 调用或最小 patch，标出每个输入 Tensor 的 shape、dtype、layout 和 TP/EP 属性；
3. 算子接口表，至少包含算子名称、输入 Tensor 规格、关键属性（TPN/TP、top-k、rotary_dim 等）；
4. 可行性、预期节省的 kernel/dispatch/内存访问、数值风险、版本依赖和回退方式；
5. 需要实现的 C++/ACLNN/torch.ops 接口与当前仓库中可复用实现的位置。若接口不存在，报告必须标记为“需实现”，不能直接提交替换。

## 实施和验证

上层调优 skill 负责保存最小 patch，并通过导入、静态检查、相关单测和最小请求；之后使用同一 benchmark 配置执行 profile。验证输出数值/质量、成功率、p95/p99、显存峰值、rank 间行为、算子 Count/Avg 和 host dispatch。若没有可运行的融合 kernel 或接口实现，只交付报告和待实现清单，不修改业务源码。任何融合 patch 都必须能一键回退，并记录源码版本和 patch 摘要。
