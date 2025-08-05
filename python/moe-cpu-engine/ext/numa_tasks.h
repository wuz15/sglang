#pragma once
#include <cstddef>
#include <string>

enum NumaTaskType {
    InitMOE,
    SetWeights,
    ForwardMOE_Logits,
    ForwardMOE_Topk,
    DONE,
};

struct NumaTask {
    NumaTaskType type;
    void *masterMOE;
    // Buffer size needed for the task (parameters/input_size + output_size)
    size_t bufferSize;
    // Size of the parameters from the master
    size_t paramSize;
    // Extra information for the task (like tokens)
    size_t extraInfo;
};

inline std::string TaskTypeToString(NumaTaskType type) {
    switch (type) {
        case InitMOE: return "InitMOE";
        case SetWeights: return "SetWeights";
        case ForwardMOE_Logits: return "ForwardMOE_Logits";
        case ForwardMOE_Topk: return "ForwardMOE_Topk";
        case DONE: return "DONE";
        default: return "Unknown";
    }
}