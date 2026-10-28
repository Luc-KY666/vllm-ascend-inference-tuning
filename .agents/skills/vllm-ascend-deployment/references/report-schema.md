# 部署最终报告结构

`deployment_report.json` 是部署完成后一次性写出的 v3 最终事实记录。执行期间的输入校验、
静态 OOM、资源探测、服务日志、PID、probe/help、清理和诊断结果不得进入报告；需要保留的
中间结果只写入内存或 `.work/`，清理后删除。

## 顶层字段

报告只允许包含：

- `schema_version`：固定为 `vllm-ascend-deployment-report-v3`；
- `status`：`success`、`failed` 或 `preflight_failed`；
- `model`：`name` 和 `weights_path`；
- `resources`：实际使用的 `npu` 数组和 `port`；
- `service`：`backend`、`base_url`、`endpoint`、`health_endpoint`、`models_endpoint` 和
  `tokenizer`；
- `launch`：只包含 `parameters`，保存实际传给 `vllm serve` 的参数；
- `failure`：成功时为 `null`，失败时包含 `stage`、`type` 和 `reason`；
- `errors`：按发生顺序追加的错误数组。

报告不得包含 `attempt`、`input`、`environment`、`stages`、`artifacts`、`preflight`、`oom`、
`cleanup`、`source`、`timings` 或其它阶段区块；也不得包含阶段执行耗时或时间戳字段。

## 启动参数

`launch.parameters` 必须保存本次实际传给 `vllm serve` 的所有参数，使用稳定的 flag 名称；
TP、DP、`max_num_seqs` 和 `max_num_batched_tokens` 等参数只在此处记录一次。未使用的可选
参数可以省略，不得填入宿主默认值。

## 错误

`errors` 每项至少包含 `stage`、`type`、`message`、`recoverable` 和 `resolved`，可选
`repair_action`。错误消息只保留有限原因摘要，不嵌入 stdout、stderr、服务日志、临时路径或
任何时间戳。失败终态还必须在 `failure` 中提供最终失败阶段和原因。

## 写出和校验

报告对象在内存中组装，清理 `.work/` 后通过 `scripts/report_io.py` 的
`atomic_write_report` 写出一次。写出完成后只调用一次 `scripts/report_schema.py`；中间阶段不
写报告、不运行 schema 校验，校验失败不重跑 attempt。

## PD 模式

PD 仍使用本文件定义的 v3 最终报告，不恢复旧版阶段报告结构。`service.base_url` 指向
Proxy；`resources.npu` 保存实际使用的设备；P/D、Proxy 和 Mooncake 的展开命令、角色
端口及链路结果放入 `launch.parameters`。普通模式和 PD 的角色级 TP/DP 均使用
`launch.parameters` 中的对应参数记录。
