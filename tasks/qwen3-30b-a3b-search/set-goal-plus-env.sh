#!/bin/sh

export GOAL_PLUS_DEVICE_POOLS='{"ascend-npu":{"visibility_env":"ASCEND_RT_VISIBLE_DEVICES","devices":["0","1","2","3","4","5","6","7"]}}'
export GOAL_PLUS_VERIFIER_RESOURCE_LOCK_DIR='/root/goal-plus-playground/gp-shared-locks'
