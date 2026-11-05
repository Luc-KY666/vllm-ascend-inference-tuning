# benchmark 场景下的 profile 采集范式

本范式规定 benchmark 调用方（含调优迭代复测）如何编排 profile 采集窗口，不改变
vllm-ascend-profiling 的通用采集接口。完成服务验收和 warmup 后，把 profile
窗口绑定到首个正式请求开始处理的事件：

```text
完成 warmup（不采集）
  -> 首个正式请求处理 gate
  -> 调用 /start_profile
  -> 立即放行/发送首个正式请求
  -> 按用户指定时长持续采集（未指定默认 3 秒）
  -> 调用 /stop_profile
```

- 不能以压测线程启动作为触发点，不能在线程启动后固定等待 3 秒再调用 `/start_profile`。
- 采集时长由调用方传入，未指定时使用 3 秒；以单调时钟从 `/start_profile` 成功返回开始计时，
  达到该时长后调用 `/stop_profile`。允许的等待只能发生在 `/start_profile` 成功之后，
  不能用“先 sleep 3 秒再开始采集”代替请求触发。
- 如果无法观测服务端内部的“开始处理”事件，使用请求发送前的同步 gate 作为近似触发，并在
  benchmark artifact 中标记该事实；不要把它描述成精确的服务端处理时间。
- warmup 必须在 profile 外完成；多卡必须保留所有 rank。
- 将触发事件、实际开始/停止时刻、配置的持续时长和首个正式请求 ID 写入迭代 artifact。
- 同一调优会话的复测必须复用相同的触发范式和采集时长，避免 profile 窗口变化导致结论不可比。
- 采集失败不伪装成服务或 benchmark 失败。
