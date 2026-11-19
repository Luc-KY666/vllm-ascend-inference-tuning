# 模型适配契约

本文件定义新模型接入 `pypto-simulator` 时必须提供的边界。通用脚本不应加入模型名判断；模型
差异应通过 profile 字段、adapter reference 和 manifest metadata 表达。

## 适配层职责

模型适配层可以提供：

- serving command 和 request workload 构造；
- tokenizer 或输入 token 精确构造；
- program/phase/capture role 命名；
- task 到 semantic tensor role 的映射；
- weight、KV、window 和 runtime scratch 的 exact allocation 分类；
- producer control output 的 ABI 描述；
- feature extractor，例如 active batch、active tokens、KV length、dtype、world size；
- 多 program 或多生命周期之间的 feature selection policy。

适配层不应改变：

- task-level interception boundary；
- profile/model/plan schema；
- strict coverage 和 extrapolation policy；
- sleep、timewarp、device-delay 的 launcher 隔离；
- native-A/B bracket 和 5% acceptance 规则。

新模型的 serving runner 不应被通用 hook 直接 import。设置：

```bash
export PYPTO_MODEL_RUNNER_MODULE=pypto_serving.model.<family>.npu_runner
export PYPTO_MODEL_RUNNER_CLASS=<RunnerClass>
```

Qwen 仍有兼容默认；其他模型缺少这两个变量时 fail closed。

## 最小 metadata

每个模型至少需要定义：

```json
{
  "model": {"family": "example", "model_id": "example-1"},
  "programs": [
    {
      "name": "decode_program",
      "phase": "decode",
      "capture_roles": ["request_decode"],
      "features": ["active_batch", "kv_length", "dtype"]
    }
  ],
  "tensor_roles": ["weight", "kv_cache", "control"],
  "control_outputs": [
    {"name": "sampled_ids", "abi": "model-specific", "provider": "adapter"}
  ],
  "memory": {
    "weight_bytes": "exact runtime tensor sum",
    "kv_bytes": "exact page_count * page_bytes"
  }
}
```

字段名可以由模型 reference 细化，但不能用模型参数量、日志四舍五入值或尺寸阈值猜测 exact
memory。allocation 分类必须由 one-shot token 或等价的精确调用身份完成；未知 allocation 默认
真实执行或 fail closed，不能误分类成 weight/KV。

## Workload 矩阵

适配层应声明实际覆盖的 workload axes，而不是只给一个端到端样本。至少区分：

- prompt/input token 数；
- output token 数；
- active batch/concurrency；
- KV length 或 page count；
- prefill、decode、sampling 和初始化生命周期；
- dtype、resource shape、block count；
- world size、rank 和通信模式。

每个需要作为模型 baseline 的 feature point 至少使用三个独立 native workload。同一 dispatch 内的
重复 task 只增加 raw observation，不增加 workload confidence。

## Control output

模拟器只保证控制流可继续，不保证模型数值正确。适配层必须明确：

- producer task；
- output tensor descriptor；
- 有效字段和 padding；
- simulator provider 写入方式；
- terminal/非 terminal output 语义。

不明确的 control output 必须阻止 `runtime_stub_ready=true`。

## 推荐文件布局

```text
references/
|-- <model>-adapter.md              模型特有 ABI、feature、memory、命令
|-- <model>-validation.md           模型特有 bracket 和证据
examples/
`-- <model>-*.json                  contract fixture，不是硬件测量
```

模型 reference 可以引用通用的 `manifest-contract.md` 和四个子 skill，但不应复制通用流程全文。
