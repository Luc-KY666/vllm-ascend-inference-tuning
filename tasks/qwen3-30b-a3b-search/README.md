# Qwen3-30B-A3B 参数搜索

本任务提供 serial 和 parallel 两种 Goal Plus 入口。启动 Pi 前，必须先在启动 Pi 的同一个终端中
加载任务资源环境脚本，使环境变量能够继承到 Pi、Goal Plus host 和 verifier。

## 启动前设置环境

先确认共享锁目录已经存在，并且当前用户具备读写权限：

```bash
test -d /root/goal-plus-playground/gp-shared-locks
test -r /root/goal-plus-playground/gp-shared-locks && test -w /root/goal-plus-playground/gp-shared-locks
```

在同一个终端执行：

```bash
source tasks/boom/qwen3-30b-a3b-search/set-goal-plus-env.sh
printf '%s\n' "$GOAL_PLUS_DEVICE_POOLS"
printf '%s\n' "$GOAL_PLUS_VERIFIER_RESOURCE_LOCK_DIR"
```

脚本默认授权 Ascend NPU `0` 到 `7`，并使用 `/root/goal-plus-playground/gp-shared-locks` 作为跨任务共享锁目录。
如果本轮授权设备或锁目录不同，先修改对应脚本中的值，再执行 `source`。

必须使用 `source`（或 `. set-goal-plus-env.sh`），不能使用 `sh set-goal-plus-env.sh`；后者只会在
子 shell 中设置变量，变量不会回到当前终端。完成设置后，保持该终端不变，在同一 shell 中构建
workspace 并启动 Pi。parallel 示例：

```bash
playground build --name qwen3-30b-a3b-search --scenario boom-new \
  --task-dir tasks/boom/qwen3-30b-a3b-search/parallel
cd /path/to/returned/workspace
source ../../tasks/boom/qwen3-30b-a3b-search/set-goal-plus-env.sh
pi
```

不要在另一个终端、未继承当前环境的 IDE 任务或独立后台 shell 中启动 Pi；否则 Pi 及其 Goal Plus
host 可能读取不到上述两个变量。
