/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"
#include "engine_config.h"
#include "numa_launcher.h"
#include <chrono>

TaskQueue::TaskQueue() {
    worker = std::thread(&TaskQueue::processTasks, this);
    sync_flag.store(true, std::memory_order_seq_cst);
    exit_flag.store(false, std::memory_order_seq_cst);
    // set busy_wait_us based on env variable or default to 20us
    const char* env_busy_wait_us = std::getenv("BUSY_WAIT_US");
    if (env_busy_wait_us) {
        busy_wait_us = std::atoi(env_busy_wait_us);
    } else {
        busy_wait_us = 20; // default to 20 microseconds
    }
}

TaskQueue::~TaskQueue() {
    {
        mutex.lock();
        exit_flag.store(true, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void TaskQueue::enqueue(std::function<void()> task) {
    {
        mutex.lock();
        tasks.push(task);
        sync_flag.store(false, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_one();
}

inline void busyWaitUs(unsigned int microseconds) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::microseconds(microseconds))
        ;
}

void TaskQueue::sync() {
    while (!sync_flag.load(std::memory_order_seq_cst)) {
        busyWaitUs(busy_wait_us);
    }
}

void TaskQueue::processTasks() {
    EngineConfig &engineCfg = EngineConfig::getInstance();
    bool exclCpuMaster = engineCfg.exclCpuMaster();
    if (exclCpuMaster) {
        int numReservedCores = engineCfg.numReservedCores();
        std::vector<int> cores = NumaDetector().getCoresForNode(0);
        cores.erase(cores.begin() + numReservedCores, cores.end());
        engineCfg.setAffinity(cores);
    }
    engineCfg.setMemAffinity(0, NumaDetector().getNumaNodes());

    while (true) {
        std::function<void()> task;
        {
            mutex.lock();
            cv.wait(mutex, [this]() { return !tasks.empty() || exit_flag.load(std::memory_order_seq_cst); });
            if (exit_flag.load(std::memory_order_seq_cst) && tasks.empty()) {
                return;
            }
            task = tasks.front();
            tasks.pop();
            mutex.unlock();
        }
        task();
        {
            mutex.lock();
            if (tasks.empty()) {
                sync_flag.store(true, std::memory_order_seq_cst);
            }
            mutex.unlock();
        }
    }
}
