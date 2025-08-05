#include "numa_launcher.h"

#include "moe.h"
#include "numa_tasks.h"
#include "engine_config.h"

// Initialize the static member
bool NumaLauncher::initialized = false;

// Global variable to store child PIDs
std::vector<pid_t> child_pids;

// Signal handler to terminate child processes
void handleShutdown(int signal) {
    for (pid_t pid : child_pids) {
        if (pid > 0) {
            xft::Logger::info("[signal] kill child %d", pid);
            kill(pid, SIGTERM); // Send termination signal to child
        }
    }
    // Optionally wait for all children to exit
    for (pid_t pid : child_pids) {
        if (pid > 0) {
            xft::Logger::info("[signal] wait for child %d", pid);
            waitpid(pid, nullptr, 0);
        }
    }
    exit(0); // Exit parent process
}

void on_exit_handler(int status, void *arg) {
    xft::Logger::info("[on_exit] Exited with status %d; arg=%s\n", status, (char*)arg);
    handleShutdown(9);
}

void NumaLauncher::initialize() {
    // Register signal handler
    struct sigaction sa;
    sa.sa_handler = handleShutdown;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);

    on_exit(on_exit_handler, (void *)"shutdown");

    NumaDetector detector;
    int numa_nodes = detector.getNumaNodes();
    EngineConfig &engineCfg = EngineConfig::getInstance();
    bool exclCpuMaster = engineCfg.exclCpuMaster();
    int nranks = exclCpuMaster ? numa_nodes + 1 : numa_nodes;
    int cpuMaster = exclCpuMaster ? 1 : 0;
    int numReservedCores = engineCfg.numReservedCores();

    if (numa_nodes <= 1) {
        xft::Logger::info("Only one NUMA node detected, no need to fork processes.");
        return;
    }

    int commInfo[2];
    Communicator &comm = Communicator::getInstance(0, nranks, commInfo);

    // Get the Python package install directory for moe-cpu-engine (.so parent directory)
    FILE* pipe = popen("python3 -c 'import importlib.util; spec = importlib.util.find_spec(\"moe_cpu_engine\"); \
                        print(spec.origin.rsplit(\"/\", 1)[0] if spec and spec.origin else \"\")'", "r");
    if (!pipe) {
        xft::Logger::error("Failed to run python command to get installDir");
        exit(1);
    }
    char buffer[512];
    std::string installDir;
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        installDir = std::string(buffer);
        // Remove trailing newline
        if (!installDir.empty() && installDir.back() == '\n') {
            installDir.pop_back();
        }
    }
    pclose(pipe);
    if (installDir.empty()) {
        xft::Logger::error("Could not determine installDir for moe_cpu_engine");
        exit(1);
    }

    for (int i = 1 - cpuMaster; i < numa_nodes; ++i) {
        pid_t pid = vfork();
        if (pid == 0) {
            xft::Logger::info("Child process for NUMA node %d started.", i);
            std::vector<int> cores = detector.getCoresForNode(i);
            cores.erase(cores.begin() + cores.size() / 2, cores.end());
            // for cpu master, erase the first two cores
            if (exclCpuMaster && cores.size() > numReservedCores && i == 0) {
                cores.erase(cores.begin(), cores.begin() + numReservedCores);
            }
            engineCfg.setAffinity(cores);
            engineCfg.setMemAffinity(i, numa_nodes);
#ifndef FALLBACK
            // Prepare arguments for exec
            std::string workerPath = installDir + "/bin/worker_entry";
            std::string nodeStr = std::to_string(i);
            std::string rankStr = std::to_string(i + cpuMaster);
            std::string nranksStr = std::to_string(nranks);
            std::string commInfo1 = std::to_string(commInfo[0]);
            std::string commInfo2 = std::to_string(commInfo[1]);

            // Build numactl command with core range and memory node
            int coreStart = cores.front();
            int coreEnd = cores.back();
            std::string coreRange = std::to_string(coreStart) + "-" + std::to_string(coreEnd);
            std::string memNode = std::to_string(i);

            // Prepare arguments for exec: numactl -C start-end -m i worker_entry ...
            std::string numactlPath = "/usr/bin/numactl";
            char* args[] = {
                const_cast<char*>(numactlPath.c_str()),
                const_cast<char*>("-C"),
                const_cast<char*>(coreRange.c_str()),
                const_cast<char*>("-m"),
                const_cast<char*>(memNode.c_str()),
                const_cast<char*>(workerPath.c_str()),
                const_cast<char*>(nodeStr.c_str()),
                const_cast<char*>(rankStr.c_str()),
                const_cast<char*>(nranksStr.c_str()),
                const_cast<char*>(commInfo1.c_str()),
                const_cast<char*>(commInfo2.c_str()),
                nullptr
            };

            execvp(numactlPath.c_str(), args);
            // If execl fails, log an error
            xft::Logger::error("Failed to exec worker_entry for NUMA node %d.", i);
            exit(1); // Exit child process if exec fails
#else
            engineCfg.setMemAffinity(i, numa_nodes);
            // Create Child process
            processTasks(i + cpuMaster, nranks, commInfo);

            exit(0); // Exit child process
#endif
        } else if (pid > 0) {
            // Parent process
            child_pids.push_back(pid);
            std::cerr << "[numa] push back " << pid << " to child list" << std::endl;
            xft::Logger::info("Forked child process with PID: %d for NUMA node %d.", pid, i);
        } else {
            // Fork failed
            xft::Logger::error("Failed to fork process for NUMA node %d.", i);
        }
    }

    std::vector<int> cores = detector.getCoresForNode(0);
    cores.erase(cores.begin() + cores.size() / 2, cores.end());
    engineCfg.setAffinity(cores);
    unsigned long nodemask = 0;
    for (int i = 0; i < numa_nodes; ++i) {
        nodemask |= (1L << i);
    }
    set_mempolicy(MPOL_INTERLEAVE, &nodemask, numa_nodes);

    int startFlag = 1;
    comm.broadcast(&startFlag, sizeof(startFlag));
}

void NumaLauncher::processTasks(int rank, int nranks, int* commInfo) {
    Communicator::resetInstance();
    Communicator& comm = Communicator::getInstance(rank, nranks, commInfo);

    // Master will broadcast a start flag to all processes
    int startFlag;
    comm.broadcast(&startFlag, sizeof(startFlag));
    if (startFlag != 1) {
        xft::Logger::error("Failed to start tasks [%d].", rank);
        return;
    } else {
        xft::Logger::info("Waiting tasks [%d].", rank);
    }

    // Map master MOE to my MOE
    std::unordered_map<void*, MOE*> moeMap;

    // Used to reveive data from master
    void* buffer = nullptr;
    size_t bufferSize = 0;

    // Wait for tasks until receive DONE
    while (true) {
        NumaTask task;
        comm.broadcast(&task, sizeof(task));

        if (task.type == DONE) {
            xft::Logger::info("Received DONE signal on rank %d.", rank);
            break;
        } else {
            xft::Logger::debug("Received task (type=%s) on rank %d, task.bufferSize=%lld.",
                               TaskTypeToString(task.type).c_str(), rank, task.bufferSize);
        }

        // Allocate buffer if needed
        if (bufferSize < task.bufferSize) {
            if (buffer) {
                free(buffer);
            }
            buffer = aligned_alloc(64, task.bufferSize);
            bufferSize = task.bufferSize;

            xft::Logger::debug("Buffer(%p) of size %zu alloced on rank %d.", buffer, task.bufferSize, rank);

            if (buffer == nullptr) {
                xft::Logger::error("Failed to allocate buffer of size %zu on rank %d.", task.bufferSize, rank);
                exit(-1);
            }
        }

        // Get messages from master by calling broadcast
        comm.broadcast(buffer, task.paramSize);

        MOE* moe = nullptr;
        auto it = moeMap.find(task.masterMOE);
        if (it != moeMap.end()) {
            moe = it->second;
        }
        if (!moe && task.type != InitMOE) {
            xft::Logger::error("MOE not found for task on rank %d.", rank);
            continue;
        }

        // Call related functions to finish the task based on taskType
        switch (task.type) {
            case InitMOE: {
                MOEConfig* cfg = static_cast<MOEConfig*>(buffer);
                moe = new MOE(*cfg);
                moeMap.emplace(task.masterMOE, moe);
                break;
            }
            case SetWeights: {
                // The buffer contains gateUpWeights, downWeights, gateUpScales, downScales, and gatingCorrBias
                intptr_t gateUpWeights = (intptr_t)buffer;
                intptr_t downWeights = (intptr_t)((char*)buffer + moe->getGateUpSize());
                intptr_t gateUpScales = (intptr_t)((char*)downWeights + moe->getDownSize());
                intptr_t downScales = (intptr_t)((char*)gateUpScales + moe->getGateUpScaleSize());
                intptr_t gatingCorrBias = (intptr_t)((char*)downScales + moe->getDownScaleSize());
                // Call the setWeights method
                moe->setWeights(gateUpWeights, downWeights, gateUpScales, downScales, gatingCorrBias);
                break;
            }
            case ForwardMOE_Logits: {
                size_t tokens = task.extraInfo;
                // The buffer contains input, logits, and output in that order
                const void* input = buffer;
                void* logits = static_cast<char*>(buffer) + moe->getInputSize(tokens);
                void* output = static_cast<char*>(logits) + moe->getLogitsSize(tokens);
                //memset(output, 0, moe->getOutputSize(tokens));
                // Call the forward method
                moe->forward(input, logits, output, tokens);
                break;
            }
            case ForwardMOE_Topk: {
                size_t tokens = task.extraInfo;
                // The buffer contains input, topkIds, topkWeights, and output in that order
                const void* input = buffer;
                int* topkIds = (int*)((char*)buffer + moe->getInputSize(tokens));
                float* topkWeights = (float*)((char*)topkIds + moe->getTopkIdSize(tokens));
                void* output = (char*)topkWeights + moe->getTopkWeightSize(tokens);
                //memset(output, 0, moe->getOutputSize(tokens));
                // Call the forward method
                moe->forwardExperts(input, topkIds, topkWeights, output, tokens);
                break;
            }
            default:
                xft::Logger::error("Unknown task type on rank %d: %d.", rank, task.type);
        }
    }
}
