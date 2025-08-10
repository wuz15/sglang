#pragma once
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include "logger.h"

class EngineConfig {
public:
    static EngineConfig& getInstance() {
        static EngineConfig instance;
        return instance;
    }

    bool numaEnabled() const {
        return numa_enabled;
    }

    bool exclCpuMaster() const {
        return excl_cpu_master;
    }

    int numReservedCores() const {
        return num_reserved_cores;
    }

    int get_system_thread_num(void) {
        pthread_t thread = pthread_self();
        cpu_set_t cpuset;

        CPU_ZERO(&cpuset);
        int result = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        if (result == 0) {
            return CPU_COUNT(&cpuset);
        }
        else {
            return 0;
        }
    }

    void setAffinity(const std::vector<int> &cores) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int core : cores) {
            CPU_SET(core, &cpuset);
        }
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
            xft::Logger::warning("Failed to set CPU affinity.");
        }
    }

    void setMemAffinity(int subNumaId, int nSubNumas ) {
        unsigned long nodemask = (1L << subNumaId);
        set_mempolicy(MPOL_BIND, &nodemask, nSubNumas);
    }

    void getMemNuma(int &node, void *ptr) {
	get_mempolicy(&node, NULL, 0, ptr, MPOL_F_NODE | MPOL_F_ADDR);
    }

private:
    EngineConfig() {
        const char* env = std::getenv("ENABLE_NUMA");
        if (env && std::strcmp(env, "1") == 0) {
            numa_enabled = true;
        } else {
            numa_enabled = false;
        }

        const char* env_master = std::getenv("EXCL_CPU_MASTER");
        if (env_master && std::strcmp(env_master, "0") == 0) {
            excl_cpu_master = false;
        } else {
            excl_cpu_master = true; // Default to true if not set
        }

        const char* env_reserved = std::getenv("NUM_RESERVED_CORES");
        if (env_reserved) {
            num_reserved_cores = std::atoi(env_reserved);
        } else {
            num_reserved_cores = 2; // Default value if not set
        }
    }

    ~EngineConfig() = default;

    EngineConfig(const EngineConfig&) = delete;
    EngineConfig& operator=(const EngineConfig&) = delete;

    bool numa_enabled;
    bool excl_cpu_master;
    int num_reserved_cores;
};
