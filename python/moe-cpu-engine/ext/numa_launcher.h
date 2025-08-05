#pragma once

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <iostream>
#include <vector>
//#include <numa.h>
//#include <numaif.h>

#include "communicator.h"
#include "logger.h"
#include "numa_detector.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

// Declare child_pids and handleShutdown as extern
extern std::vector<pid_t> child_pids;
void handleShutdown(int signal);

class NumaLauncher {
   public:
    NumaLauncher(bool worker = false) {
        if (!initialized) {
            initialized = true;
            if (!worker) initialize();
        }
    }

    void processTasks(int rank, int nranks, int *commInfo);

   private:
    static bool initialized;

    void initialize();
};
