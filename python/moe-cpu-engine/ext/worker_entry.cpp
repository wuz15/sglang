#include <iostream>
#include <cstdlib>
#include "numa_launcher.h"
#include "numa_detector.h"
#include "engine_config.h"

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <node> <rank> <nranks> <fd1> <fd2>" << std::endl;
        return 1;
    }
    int node = std::atoi(argv[1]);
    int rank = std::atoi(argv[2]);
    int nranks = std::atoi(argv[3]);
    int fd1 = std::atoi(argv[4]);
    int fd2 = std::atoi(argv[5]);

    NumaDetector numaDetector;
    EngineConfig& config = EngineConfig::getInstance();
    config.setMemAffinity(node, numaDetector.getNumaNodes());

    int commInfo[2]; // Construct commInfo as needed
    commInfo[0] = fd1;
    commInfo[1] = fd2;

    NumaLauncher launcher(true); // worker process
    launcher.processTasks(rank, nranks, commInfo);
    return 0;
}
