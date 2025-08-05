/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <cstdio>

#include "task_queue.h"
#include "vendors/cuda.h"

class CPUInfer {
   public:
    CPUInfer(int thread_num) {
        if (task_queue_ == nullptr) {
            task_queue_ = new TaskQueue();
        }
    }

    //~CPUInfer() { delete task_queue_; }
    ~CPUInfer() {  }

    template <typename Func, typename Obj, typename... Args>
    void enqueue(Func f, Obj* obj, Args... args) {
        task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
    }

    void submit(std::pair<intptr_t, intptr_t> params) {
        void (*func)(void*) = (void (*)(void*))params.first;
        void* args = (void*)params.second;
        *((CPUInfer**)args) = this;
        func(args);
    }

    void sync() { task_queue_->sync(); }

    void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
        void (*func)(void*) = (void (*)(void*))params.first;
        void* args = (void*)params.second;
        *((CPUInfer**)args) = this;
        cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
    }

    static void sync_(void* cpu_infer_ptr) {
        CPUInfer* cpuinfer = (CPUInfer*)cpu_infer_ptr;
        cpuinfer->sync();
    }

    void sync_with_cuda_stream(intptr_t user_cuda_stream) {
        cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)this);
    }

   public:
    static TaskQueue* task_queue_;
};

#endif
