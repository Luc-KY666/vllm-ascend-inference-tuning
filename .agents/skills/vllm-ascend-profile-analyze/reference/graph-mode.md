# 图模式分析与推荐

基于 profile 证据和部署形态推荐 vllm-ascend 的 cudagraph_mode，只产候选不执行。当前模式从实际启动命令读（`--compilation-config`）；profile 的作用是判断收益——free-analysis §4 命中模型侧下发压力或 Host Bound 时，图模式直接对症（capture/replay 消除逐算子下发）。

## 1. 模式语义（编译期 / 运行期）

| 模式 | 编译期 | 运行期 |
|---|---|---|
| NONE | 不编译 | eager，逐算子下发 |
| PIECEWISE | 基础 FX 融合 pass（npugraph_ex 关闭） | ACLGraph capture + replay |
| FULL / FULL_DECODE_ONLY | npugraph_ex 优化 FX 图（`force_eager=True`，仅编译期不 capture），FULL_DECODE_ONLY 只对 decode 图 | 优化后的 callable 由 ACLGraph capture + replay |
| FULL_AND_PIECEWISE（默认） | PIECEWISE 编译 | 均匀 decode 批次可走全图行为 |

## 2. 推荐矩阵

| 部署形态 | 推荐 | 理由 |
|---|---|---|
| 单节点 | FULL | capture/replay 消除逐算子下发 |
| PD 分离 · P 节点 | FULL_AND_PIECEWISE | prefill shape/batch 多变，piecewise 编译适配多变输入，全图 capture 不适配 |
| PD 分离 · D 节点 | FULL_DECODE_ONLY | decode 批次均匀，整图 capture/replay 收益最大 |

## 3. 配置格式与联动

- `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`：必须 JSON object，不能直接传字符串；
- `cudagraph_capture_sizes`：指定 capture 的 batch 尺寸列表（如 [1,2,4,8,16,32,64,128,256,512]），默认不设置；
- `TASK_QUEUE_ENABLE`：图模式下官方示例设为 1；设为 2 仅 eager 下有效，是留在 eager 时缓解下发压力的手段——当前 NONE 时推荐首先推荐使用图模式，如果用户强调只能使用eager时推荐设置为 2。