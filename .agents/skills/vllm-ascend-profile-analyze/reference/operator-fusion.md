# 算子融合替换与模型代码优化

两类工作：把源码中命中 torch_npu 融合算子清单的算子序列替换为已实现的融合算子；把已验证的优化方法和思路应用到模型代码。只产候选不执行，落地由 `vllm-ascend-profiling` 调优迭代编排；不凭算子名批量重写源码，不自行设计新 kernel。

## 1. 融合算子替换

以 torch_npu 官方融合算子清单为起点，对照当前采集的 profile、模型 `config.json` 和源码，判断小算子序列能否替换为已支持的融合算子；只产候选不执行，不自行设计新 kernel。

1. **拉融合算子清单**：
   - 数据源：`https://www.hiascend.com/doc_center/source/zh/Pytorch/2610/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu_list.md`
   - 解析：读表格的 API 名称、说明两列，按 fused/fusion/融合 关键词筛出融合类算子；
   - 记录：每个算子的功能（可替换的计算序列）、接口签名（输入 Tensor 规格、dtype/shape 约束、关键属性）；
2. **profile 定热点**：高 Count 低 Avg 小算子、host launch/dispatch 开销、`AI_CPU` fallback、advisor Fusible Operator 结果；确认 device 忙时段受其影响；
3. **源码找序列**：在 `vllm/model_executor/models` 定位模型实现，必要时跟随 import 到多个文件，读 decoder layer `forward`，找到热点算子对应的连续序列、模块边界和条件分支；
4. **config 核约束**：`config.json` 的 hidden size、head/head_dim、rotary、top-k、专家数、dtype、TP/EP 是否满足融合算子输入规格；
5. **匹配判定**（对清单里每个融合算子，拿它覆盖的计算去源码热点路径找对应序列）：
   - 功能对应：融合算子说明列出的被融合计算（算子种类、顺序）与源码热点序列逐项一致，只覆盖其中一部分的不算；
   - 数据流可行：热点序列的中间张量没有被序列外的代码消费（融合后中间结果不再单独产出），序列内没有依赖运行时状态的分支；
   - 约束满足：输入 dtype/shape/layout 和 config 并行属性满足接口签名要求；
   - 本地验证：在运行环境的 torch_npu 中确认算子可用（`hasattr(torch_npu, '<算子名>')`，必要时按接口签名跑最小输入）——清单来自最新版文档，本地版本可能未包含；
   - 四项都过 → 产出替换 patch 候选：在原调用位置替换为融合算子调用，不跨模块搬代码，标注 shape/dtype/layout/TP/EP、数值风险（融合算子内部计算顺序/精度可能不同）、版本依赖、回退方式；任一不过或清单无对应算子跳过，不自行设计新 kernel；
   - 收益估算：`Count × Avg`、dispatch 次数减少量、临时张量读写减少量，不把总算子时间直接等同于可节省时间。

## 2. 模型代码优化方法

已验证的优化及使用规范：

- **零拷贝——构造路径**
  - 适用 pattern：大批量 host 数据（如批量 token ids）从 Python list 逐元素构造 tensor（`torch.tensor([...])`）；
  - 改法：`np.asarray` 走 C 层连续内存后 `torch.from_numpy` 零拷贝转 tensor，缩短 H2D 拷贝前的准备时间；
  - 改法示例：
    ```python
    # 修改前
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(device)

    # 修改后
    def _ids_tensor_to_device(list_of_ids, device):
        arr = np.asarray(list_of_ids, dtype=np.int64)  # C 层连续内存
        return torch.from_numpy(arr).to(device)        # 零拷贝转 tensor
    ```
  - 限制：
    - 数据须规整等长，ragged list `np.asarray` 失败；
    - `from_numpy` 与 numpy 数组共享 host 内存，数组被改写会污染 tensor（`.to(device)` 拷贝后不受影响）。
- **零拷贝——物理复制改视图**
  - 适用 pattern：同一份张量物理复制多份、副本内容完全相同（`unsqueeze(0).repeat(B, 1, 1)`）；
  - 改法：改 `expand(B, -1, -1)` 视图，省 B 份内存分配；
  - 改法示例：
    ```python
    # 修改前
    query = self.query.unsqueeze(0).repeat(B, 1, 1)    # 物理复制 B 份数据

    # 修改后
    query = self.query.unsqueeze(0).expand(B, -1, -1)  # 零拷贝视图，语义等价
    ```
  - 限制：
    - expand 只能扩 size-1 维，源张量该维不为 1（各份数据本来就不同）时不适用；
    - 视图 B 份共享存储，下游 in-place 写或各副本需独立改写时须保留物理复制；
    - 下游要求 contiguous（如 `.view()`）会隐式拷贝，收益消失。

应用流程：在模型和输入准备代码中扫描命中适用 pattern 的写法，核对条目内限制后产出 patch 候选。

## 3. 候选输出

每个候选输出：原始源码路径、逻辑位置/行号和真实代码片段；替换或修改后的最小 patch，标注每个输入 Tensor 的 shape、dtype、layout 和 TP/EP 属性；所用接口（torch_npu 融合算子接口名或优化方法名）；可行性、预期收益、数值风险、版本依赖和回退方式。
