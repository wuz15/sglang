#pragma once

#include <cstdint>
#include "bfloat16.h"
#include "layers_mlp.h"

struct MOEConfig {
    int layer_id;
    int expert_num;  // routed expert num
    int num_experts_per_tok;  // num experts per token
    int hidden_size;
    int intermediate_size;
    bool normTopKProb;
    int nGroup;
    int topKGroup;

    MOEConfig() {}

    MOEConfig(int layer_id, int expert_num, int num_experts_per_tok, int hidden_size, int intermediate_size,
        bool normTopKProb = false, int nGroup = 1, int topKGroup = 1)
        : layer_id(layer_id),
          expert_num(expert_num),
          num_experts_per_tok(num_experts_per_tok),
          hidden_size(hidden_size),
          intermediate_size(intermediate_size),
          normTopKProb(normTopKProb),
          nGroup(nGroup),
          topKGroup(topKGroup) {}
};

class MOE {
   public:
    MOE(const MOEConfig &cfg);
    ~MOE() {}

    const MOEConfig &getConfig() const { return config; }
    size_t getInputSize(int ntokens) const {
        return ntokens * config.hidden_size * sizeof(bfloat16_t);
    }
    size_t getLogitsSize(int ntokens) const {
        return ntokens * config.expert_num * sizeof(float);
    }
    size_t getTopkIdSize(int ntokens) {
        return ntokens * config.num_experts_per_tok * sizeof(int);
    }
    size_t getTopkWeightSize(int ntokens) {
        return ntokens * config.num_experts_per_tok * sizeof(float);
    }
    size_t getOutputSize(int ntokens) const {
        return ntokens * config.hidden_size * sizeof(bfloat16_t);
    }
    size_t getGateUpSize() const {
        return (size_t)config.expert_num * config.hidden_size * config.intermediate_size * 2;  // 2 means gate and up
    }
    size_t getDownSize() const {
        return (size_t)config.expert_num * config.hidden_size * config.intermediate_size;
    }
    size_t getGateUpScaleSize() const {
        return (size_t)config.expert_num * ((config.hidden_size + 127) / 128) *
               ((config.intermediate_size + 127) / 128) * 2 * sizeof(float);
    }
    size_t getDownScaleSize() const {
        return (size_t)config.expert_num * ((config.hidden_size + 127) / 128) *
               ((config.intermediate_size + 127) / 128) * sizeof(float);
    }
    size_t getGatingBiasSize() const {
        return config.expert_num * sizeof(float);
    }

    void setWeights(intptr_t gateUpWeights, intptr_t downWeights, intptr_t gateUpScales,
        intptr_t downScales, intptr_t gatingCorrBias = 0);
    void forward(const void *input, void *logits, void *output, int ntokens = 1);
    void forwardExperts(const void *input, int *topkIds, float *topkWeights, void *output, int ntokens = 1);

   private:
    MOEConfig config;
    void *moe;
};
