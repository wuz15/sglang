#include <torch/extension.h>

#include <cstdint>
#include <iostream>
#include <memory>

#include "communicator.h"
#include "cpuinfer.h"
#include "engine_config.h"
#include "logger.h"
#include "moe.h"
#include "numa_launcher.h"
#include "numa_tasks.h"

TaskQueue* CPUInfer::task_queue_ = nullptr;
class MOEBindings {
   public:
    class ForwardBindings {
       public:
        struct Args {
            CPUInfer *cpuinfer;
            MOE *moe;
            const void *input;
            void *logits;
            int *topkIds;
            float *topkWeights;
            void *output;
            int ntoks;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&MOE::forward, args_->moe, args_->input, args_->logits, args_->output, args_->ntoks);
        }
        static void inner1(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&MOE::forwardExperts, args_->moe, args_->input, args_->topkIds, args_->topkWeights,
                args_->output, args_->ntoks);
        }
        static void dispatch_task(MOE &moe, NumaTaskType taskType, size_t bufferSize, size_t paramSize, size_t extra) {
            // Create a task for forward operation
            NumaTask task;
            task.type = taskType;
            task.bufferSize = bufferSize;
            task.paramSize = paramSize;
            task.masterMOE = &moe;
            task.extraInfo = extra;

            // Broadcast the task to all NUMA nodes
            //Communicator::getInstance(0).broadcast(&task, sizeof(NumaTask));
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MOE &moe, intptr_t input, intptr_t logits,
                                                                intptr_t output, int ntoks) {
            // Dispatch task to NUMA nodes if enabled
            if (EngineConfig::getInstance().numaEnabled()) {
                size_t inputSize = ntoks * moe.getConfig().hidden_size * sizeof(bfloat16_t);
                size_t logitsSize = ntoks * moe.getConfig().expert_num * sizeof(float);
                size_t outputSize = inputSize;
                dispatch_task(moe, ForwardMOE_Logits, inputSize + logitsSize + outputSize, inputSize + logitsSize,
                              ntoks);
            }
            Args *args = new Args{nullptr, &moe, (const void *)input, (void *)logits, nullptr, nullptr, (void *)output, ntoks};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface1(MOE &moe, intptr_t input, intptr_t topkIds,
                                                                intptr_t topkWeights, intptr_t output, int ntoks) {
            // Dispatch task to NUMA nodes if enabled
            if (EngineConfig::getInstance().numaEnabled()) {
                size_t inputSize = ntoks * moe.getConfig().hidden_size * sizeof(bfloat16_t);
                size_t topkIdsSize = ntoks * moe.getConfig().num_experts_per_tok * sizeof(int);
                size_t topkWeightsSize = ntoks * moe.getConfig().num_experts_per_tok * sizeof(float);
                size_t outputSize = inputSize;
                dispatch_task(moe, ForwardMOE_Topk, inputSize + topkIdsSize + topkWeightsSize + outputSize,
                              inputSize + topkIdsSize + topkWeightsSize, ntoks);
            }
            Args *args = new Args{nullptr, &moe, (const void *)input, nullptr, (int *)topkIds, (float *)topkWeights,
                (void *)output, ntoks};
            return std::make_pair((intptr_t)&inner1, (intptr_t)args);
        }
    };
};

PYBIND11_MODULE(moe_cpu_engine, m) {
    if (EngineConfig::getInstance().numaEnabled()) {
        NumaLauncher launcher;
    }

    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit", &CPUInfer::submit)
        .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
        .def("sync", &CPUInfer::sync)
        .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream);

    auto moe_module = m.def_submodule("moe");
    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int layer_id, int expert_num, int num_experts_per_tok, int hidden_size, int intermediate_size,
                bool normTopKProb, int nGroup, int topKGroup) {
            return MOEConfig(layer_id, expert_num, num_experts_per_tok, hidden_size, intermediate_size, normTopKProb,
                    nGroup, topKGroup);
        }));
    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("set_weights", &MOE::setWeights)
        .def("forward", &MOEBindings::ForwardBindings::cpuinfer_interface)
        .def("forward_experts", &MOEBindings::ForwardBindings::cpuinfer_interface1);
}
