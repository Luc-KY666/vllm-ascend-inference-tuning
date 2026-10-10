# 部署输入契约

本文件描述部署 skill 的通用输入形状。它不是任务的完整 SearchSpec；任务可以增加字段，
但不能降低设备授权、输出隔离和独立验收要求。

## 必需身份

```json
{
  "model_path": "<模型目录、配置来源或任务允许的模型 ID>",
  "served_model_name": "<服务对外模型名>"
}
```

`model_path` 与 `served_model_name` 不能互相推断。前者用于加载模型和 tokenizer，后者
用于服务参数、请求体和 `/v1/models` 身份核对。

## 运行参数

真实启动通常需要以下字段；任务必须明确哪些字段固定、哪些字段允许搜索：

```json
{
  "tp": 1,
  "dp": 1,
  "max_num_seqs": 1,
  "max_num_batched_tokens": 1,
  "gpu_memory_utilization": 0.9,
  "min_required_context_len": 4096,
  "dtype": "auto",
  "quantization": null,
  "max_model_len": 4096,
  "extra_args": []
}
```

示例中的数字不是公共默认值。缺少任务明确要求的字段时，部署 skill 只能进入
`recommendation_only` 或 `input_invalid`，不能把示例值静默写入真实启动命令。

## Goal Plus 资源上下文

设备和输出信息必须来自宿主或任务，而不是用户自由输入后直接信任：

```json
{
  "provider": "git_worktree",
  "authorized_devices": ["<GP 分配的设备标识>"],
  "visibility_env": {
    "ASCEND_RT_VISIBLE_DEVICES": "<GP 冻结值>"
  },
  "port": "<GP 分配端口>",
  "activation_script": "<任务提供的绝对 wrapper>",
  "python": "<任务提供的绝对 Python>",
  "output_root": "<GP verifier 或任务允许的输出目录>"
}
```

skill 只核对这些值的一致性，不创建租约、不发现未声明设备、不修改锁和端口。

## 状态规则

部署输入至少应能映射到以下状态：

| 状态 | 含义 |
| --- | --- |
| `recommendation_only` | 模型身份已知，但缺少启动参数，只输出预检和最小 TP 建议 |
| `input_invalid` | 输入、模型路径、CLI 兼容性或静态容量门禁失败 |
| `startup_failed` | 服务进程未启动、异常退出或运行时 OOM |
| `healthcheck_failed` | 服务未在任务时限内健康 |
| `service_identity_invalid` | 健康接口通过但模型身份或最小请求不匹配 |
| `service_ready` | 部署层全部独立验收通过 |
| `cleanup_failed` | 本次服务未能按合同清理或移交 |
