#include "moe.h"

#include "engine_config.h"
#include "numa_tasks.h"
#include "communicator.h"
#include "logger.h"
#include <chrono>

MOE::MOE(const MOEConfig& cfg) : config(cfg) {
    // Broadcast to all NUMA nodes to initialize the MOE
    // This is only done by the master node (rank 0), slave nodes receive it in task dispatch thread
    EngineConfig& engineCfg = EngineConfig::getInstance();
    if (engineCfg.numaEnabled() && Communicator::getInstance(0).getRank() == 0) {
        // Initialize NUMA tasks
        NumaTask task;
        task.type = InitMOE;
        task.bufferSize = sizeof(MOEConfig);
        task.paramSize = sizeof(MOEConfig);
        task.masterMOE = this;
        
        Communicator::getInstance(0).broadcast(&task, sizeof(NumaTask));
        Communicator::getInstance(0).broadcast((void *)&cfg, sizeof(MOEConfig));
    }
}

void MOE::setWeights(intptr_t gateUpWeights, intptr_t downWeights, intptr_t gateUpScales, intptr_t downScales,
        intptr_t gatingCorrBias) {
    int tpRank = 0, tpSize = 1;

    // Broadcast to all NUMA nodes to set weights
    EngineConfig& engineCfg = EngineConfig::getInstance();
    if (engineCfg.numaEnabled()) {
        if (Communicator::getInstance(0).getRank() == 0) {
            // dispatch set weights task
            NumaTask task;
            task.type = SetWeights;
            task.bufferSize =
                getGateUpSize() + getDownSize() + getGateUpScaleSize() + getDownScaleSize() + getGatingBiasSize();
            task.paramSize = task.bufferSize;
            task.masterMOE = this;
            Communicator::getInstance(0).broadcast(&task, sizeof(NumaTask));

            // Weights broadcasting
            void *params[5] = {(void *)gateUpWeights, (void *)downWeights, (void *)gateUpScales, (void *)downScales,
                               (void *)gatingCorrBias};
            size_t sizes[5] = {getGateUpSize(), getDownSize(), getGateUpScaleSize(), getDownScaleSize(),
                               getGatingBiasSize()};
            Communicator::getInstance(0).broadcast(params, sizes, 5);
            // no model slice on rank 0
            if (engineCfg.exclCpuMaster()) { return; }
        }

        tpRank = Communicator::getInstance(0).getRank();
        tpSize = Communicator::getInstance(0).getSize();
        if (engineCfg.exclCpuMaster()) {
            // Exclude the framework master from the TP ranks
            tpRank -= 1;
            tpSize -= 1;
        }
    }

    xft::Logger::debug("MOE setWeights: tpRank=%d, tpSize=%d, gateUpWeights[100]=%d", tpRank, tpSize,
                       ((uint8_t *)gateUpWeights)[100]);

    moe = xft::createDeepSeekMoE(config.layer_id, config.expert_num, config.num_experts_per_tok, config.hidden_size, config.intermediate_size,
        config.normTopKProb, config.nGroup, config.topKGroup, (const void *)gateUpWeights, (const void *)downWeights, (const void *)gateUpScales,
            (const void *)downScales, (const void *)gatingCorrBias, 128, tpRank, tpSize);
}

void MOE::forward(const void *input, void *logits, void *output, int n_toks) {
    // Broadcast parameters
    EngineConfig& engineCfg = EngineConfig::getInstance();
    if (engineCfg.numaEnabled() && Communicator::getInstance(0).getRank() == 0) {
        NumaTask task;
        task.type = ForwardMOE_Logits;
        task.bufferSize =
            getInputSize(n_toks) + getLogitsSize(n_toks) + getOutputSize(n_toks);
        task.paramSize = task.bufferSize;
        task.masterMOE = this;
        task.extraInfo = n_toks;
        Communicator::getInstance(0).broadcast(&task, sizeof(NumaTask));
        void *params[2] = {(void *)input, (void *)logits};
        size_t sizes[2] = {getInputSize(n_toks), getLogitsSize(n_toks)};
        Communicator::getInstance(0).broadcast(params, sizes, 2);
    }

    memset(output, 0, getOutputSize(n_toks));
    xft::forwardDeepSeekMoE(moe, (void *)input, (void *)output, n_toks, (void *)logits);
    bfloat16_t *output_tensor = static_cast<bfloat16_t *>(output);

    // xft::Logger::debug("rank=%d, MOE forward: n_toks=%d, input[0, 1]=%f, %f, logits[0, 1]=%f, %f, output[0, 1]=%f, %f",
    //                    Communicator::getInstance(0).getRank(), n_toks,
    //                    (float)((bfloat16_t *)input)[0], (float)((bfloat16_t *)input)[1],
    //                    (float)((float *)logits)[0], (float)((float *)logits)[1],
    //                    (float)output_tensor[0], (float)output_tensor[1]);

    if (engineCfg.numaEnabled()) {  // reduce the result
        Communicator::getInstance(0).reduceAdd(output_tensor, output_tensor, n_toks * config.hidden_size);
    }
}

void MOE::forwardExperts(const void *input, int *topkIds, float *topkWeights, void *output, int n_toks) {
    // Broadcast parameters
    EngineConfig &engineCfg = EngineConfig::getInstance();
    if (engineCfg.numaEnabled() && Communicator::getInstance(0).getRank() == 0) {
        NumaTask task;
        task.type = ForwardMOE_Topk;
        task.bufferSize =
            getInputSize(n_toks) + getTopkIdSize(n_toks) + getTopkWeightSize(n_toks) + getOutputSize(n_toks);
        task.paramSize = task.bufferSize;
        task.masterMOE = this;
        task.extraInfo = n_toks;
        Communicator::getInstance(0).broadcast(&task, sizeof(NumaTask));
        void *params[3] = {(void *)input, (void *)topkIds, (void *)topkWeights};
        size_t sizes[3] = {getInputSize(n_toks), getTopkIdSize(n_toks), getTopkWeightSize(n_toks)};
        Communicator::getInstance(0).broadcast(params, sizes, 3);
    }
    //int node;
    //engineCfg.getMemNuma(node, (void *)input);
    //printf("[%d] input alloc at %p on node %d\n", Communicator::getInstance(0).getRank(), input, node);
    //engineCfg.getMemNuma(node, (void *)output);
    //printf("[%d] output alloc at %p on node %d\n", Communicator::getInstance(0).getRank(), output, node);
    //engineCfg.getMemNuma(node, (void *)topkIds);
    //printf("[%d] topkIds alloc at %p on node %d\n", Communicator::getInstance(0).getRank(), topkIds, node);
    //engineCfg.getMemNuma(node, (void *)topkWeights);
    //printf("[%d] topkWeights alloc at %p on node %d\n", Communicator::getInstance(0).getRank(), topkWeights, node);

    //auto start = std::chrono::high_resolution_clock::now();
    memset(output, 0, getOutputSize(n_toks));
    if (!engineCfg.exclCpuMaster() || Communicator::getInstance(0).getRank() != 0)
        xft::forwardDeepSeekMoE(moe, (void *)input, (void *)output, n_toks, topkIds, topkWeights);
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    //std::cout << Communicator::getInstance(0).getRank() << " forwardMoE time: " << duration << " us" << std::endl;
    bfloat16_t *output_tensor = static_cast<bfloat16_t *>(output);

    // xft::Logger::debug("rank=%d, MOE forwardExperts: n_toks=%d, input[0, 7167]=%f, %f, topkIds[0]=%d, topkWeights[0]=%f, output[0, 7167]=%f, %f",
    //                    Communicator::getInstance(0).getRank(), n_toks,
    //                    (float)((bfloat16_t *)input)[0], (float)((bfloat16_t *)input)[7167],
    //                    topkIds[0], topkWeights[0],
    //                    (float)output_tensor[0], (float)output_tensor[7167]);

    if (engineCfg.numaEnabled()) {  // reduce the result
        //auto start = std::chrono::high_resolution_clock::now();
        Communicator::getInstance(0).reduceAdd(output_tensor, output_tensor, n_toks * config.hidden_size);
        //auto end = std::chrono::high_resolution_clock::now();
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout << Communicator::getInstance(0).getRank() << " reduceAdd Execution time: " << duration << " us" << std::endl;
    }
}
