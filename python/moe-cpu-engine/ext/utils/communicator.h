#pragma once

#include <pthread.h>

#include "numa_detector.h"
#include "shm_ccl.h"

class Communicator {
   public:
    // Get the singleton instance
    // For child processes, need extra info (commInfo) to connect to the shared memory
    static Communicator& getInstance(int rank, int size = 0, int* commInfo = nullptr) {
        if (!instance) {
            if (size == 0) size = NumaDetector().getNumaNodes();
            instance = new Communicator(rank, size, commInfo);
        }
        return *instance;
    }

    // Reset the singleton instance (used after fork in the child process)
    static void resetInstance() {
        if (instance != nullptr) delete instance;
        instance = nullptr;
    }

    int getRank() const { return shmCCL.rank_; }

    int getSize() const { return shmCCL.rank_size_; }

    // Broadcast method
    void broadcast(void* data, size_t size) { shmCCL.broadcast(data, size); }
    void broadcast(void** data, size_t* sizes, int count) { shmCCL.broadcast(data, sizes, count); }

    // ReduceAdd method
    template <typename T>
    void reduceAdd(T* sendbuf, T* recvbuf, size_t size) {
        shmCCL.reduceAdd(sendbuf, recvbuf, size, shmCCL.rank_, shmCCL.rank_size_);
    }

   private:
    // Private constructor to prevent instantiation
    Communicator(int rank, int size, int* commInfo)
        : shmCCL(rank, size, [&](int* fd, size_t size) {
              for (int i = 0; i < std::min((int)size, 2); ++i) {
                  // Master process (copy fd to commInfo)
                  if (rank == 0 && commInfo != nullptr) {
                      commInfo[i] = fd[i];
                  }
                  // Child process (copy commInfo to fd)
                  else if (rank != 0 && commInfo != nullptr) {
                      fd[i] = commInfo[i];
                  }
              }
          }, size == NumaDetector().getNumaNodes() + 1 ? 1 : 0) {}

    // Delete copy constructor and assignment operator
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;

    ShmCCL shmCCL;

    static Communicator* instance;  // Singleton instance
};
