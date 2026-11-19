/* Zero-work graph used to compare a2a3 and a2a3sim control-plane latency. */

#include <cstdint>

#include "orchestration_api.h"

namespace {
constexpr int kChainProduce = 0;
constexpr int kChainConsume = 1;
constexpr int kSpmd = 2;
constexpr int kMixAic = 3;
constexpr int kMixAiv0 = 4;
constexpr int kMixAiv1 = 5;
constexpr int kSingle = 6;
}  // namespace

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig overhead_benchmark_config(
    const ChipTaskArgs &orch_args
) {
    (void)orch_args;
    return OrchestrationConfig{.expected_arg_count = 0};
}

__attribute__((visibility("default"))) void overhead_chain(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    uint32_t shape[1] = {1};
    TensorCreateInfo intermediate(shape, 1, DataType::INT32);

    CoreTaskArgs producer;
    producer.add_output(intermediate);
    TaskOutputTensors outputs = rt_submit_aiv_task(kChainProduce, producer);

    CoreTaskArgs consumer;
    consumer.add_input(outputs.get_ref(0));
    rt_submit_aiv_task(kChainConsume, consumer);
}

__attribute__((visibility("default"))) void overhead_spmd8(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    CoreTaskArgs task;
    task.launch_spec.set_block_num(8);
    rt_submit_aiv_task(kSpmd, task);
}

__attribute__((visibility("default"))) void overhead_single(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    CoreTaskArgs task;
    rt_submit_aiv_task(kSingle, task);
}

__attribute__((visibility("default"))) void overhead_mix(const ChipTaskArgs &orch_args) {
    (void)orch_args;
    MixedKernels kernels;
    kernels.aic_kernel_id = kMixAic;
    kernels.aiv0_kernel_id = kMixAiv0;
    kernels.aiv1_kernel_id = kMixAiv1;
    CoreTaskArgs task;
    rt_submit_task(kernels, task);
}

}  // extern "C"
